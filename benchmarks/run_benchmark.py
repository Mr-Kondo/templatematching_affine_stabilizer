#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark harness for the affine video stabiliser.

Runs the existing pipeline (``TemplateMatcher`` + ``AffineCorrector``) under
fixed comparison conditions and records stage timings, resource usage and
accuracy metrics into a timestamped run directory so results are never
overwritten and comparisons are reproducible.

Modes
-----
``synth``
    Generates a synthetic clip with known translation/rotation/scale (see
    ``benchmarks/synth.py``) and validates the stabiliser against the exact
    ground truth: matched-point errors, stabilised-coordinate residuals at
    background points that are NOT used for tracking, and affine-matrix error.
``video``
    Runs the pipeline on an existing clip with the provided template CSV and
    records timings, RANSAC inlier/outlier rates, stabilised residuals at the
    tracked points and (optionally) forward-backward LK residuals at
    background feature points outside the template ROIs.

Usage::

    uv run python benchmarks/run_benchmark.py --mode synth
    uv run python benchmarks/run_benchmark.py --mode video \
        --video 01_input/input.mp4 --csv 01_input/Input.csv --bg-eval

Every run writes ``report.json`` and ``summary.md`` into
``03_output/benchmark/run_<timestamp>[_r<i>]/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '02_code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affine_template_matching import AffineCorrector, TemplateMatcher  # noqa: E402
import synth as synth_gen  # noqa: E402


def dir_size(path: Path) -> int:
    """Total byte size of a directory tree (0 when missing)."""
    total = 0
    if path.is_dir():
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    return total


def git_commit() -> str:
    """Short commit hash of the working tree ('unknown' outside a repo)."""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def stats_of(values: List[float]) -> Dict[str, float]:
    """Mean / median / p95 / max / (std) summary of a value list."""
    a = np.asarray(values, float)
    out: Dict[str, float] = {'n': int(a.size)}
    if a.size == 0:
        return out
    out.update(
        mean=float(a.mean()),
        median=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        max=float(a.max()),
        std=float(a.std()),
    )
    return out


def to3(M: Any) -> np.ndarray:
    """2x3 affine matrix -> 3x3 homogeneous."""
    M = np.asarray(M, float)
    return np.vstack([M, [0.0, 0.0, 1.0]])


def apply(M3: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous matrix to an (n, 2) point array -> (n, 2)."""
    return (M3[:2, :2] @ pts.T + M3[:2, 2:3]).T


def eval_synth(run_dir: Path, gt: Dict[str, Any],
               estimate_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare pipeline results against the synthetic ground truth.

    Reports, kept separate:
    - tracking error of matched points vs the true displaced positions,
    - stabilisation residual at background points not used for tracking
      (``M_k @ A_{k-1} @ p - p``), which exposes errors that tracking-point
      residuals can hide,
    - affine-matrix error vs the true inverse motion.
    """
    frames = gt['frame_count']
    A = [to3(m) for m in gt['A']]
    bg = np.array(gt['bg_points'], np.float64)
    tpl_by_id = {int(r['No']): (float(r['x']), float(r['y']))
                 for r in gt['templates']}

    df = pd.read_csv(run_dir / 'Matched.csv')
    per_match_err: List[float] = []
    fail_count = 0
    per_frame_mean: List[float] = []
    for frame in range(2, frames + 1):
        rows = df[df.Frame_No == frame]
        errs = []
        for _, r in rows.iterrows():
            cx, cy = tpl_by_id[int(r.Template_No)]
            gt_xy = apply(A[frame - 1], np.array([[cx, cy]]))[0]
            e = float(np.hypot(r.mx - gt_xy[0], r.my - gt_xy[1]))
            errs.append(e)
            per_match_err.append(e)
            if e > 2.0:
                fail_count += 1
        per_frame_mean.append(float(np.mean(errs)) if errs else float('nan'))

    bg_res: List[float] = []
    mat_err: List[float] = []
    for st in estimate_stats:
        k = int(st['frame'])  # video frame k (2..K), motion index k-1
        if st.get('M') is None:
            continue
        M = np.asarray(st['M'], float)
        E = to3(M) @ A[k - 1]
        r = np.linalg.norm(apply(E, bg) - bg, axis=1)
        bg_res.extend(float(v) for v in r)
        M_gt = to3(cv2.invertAffineTransform(np.asarray(gt['A'][k - 1], np.float64)
                                             [:2]))
        mat_err.append(float(np.linalg.norm(to3(M) - M_gt)))

    return dict(
        matched_error_px=stats_of(per_match_err),
        matched_frame_mean_error_px=per_frame_mean,
        tracking_failures={'count': fail_count, 'rate': (fail_count /
                          max(1, len(per_match_err)))},
        background_residual_px_not_tracked=stats_of(bg_res),
        affine_matrix_error_frobenius=stats_of(mat_err))


def bg_residuals_lk(video_path: Path, templates: List[Dict[str, Any]],
                    estimate_stats: List[Dict[str, Any]], *,
                    max_corners: int = 200, fb_err: float = 1.0) -> Dict[str, Any]:
    """Stabilisation residual at background points not used for tracking.

    Detects Shi-Tomasi corners on frame 1 outside the template ROIs, tracks
    them to every frame with pyramid LK plus a forward-backward sanity check,
    and reports ``|M_k @ q_k - q_1|`` for the surviving points only.  This is
    independent of the matched tracking points and of the RANSAC fit.
    """
    cap = cv2.VideoCapture(str(video_path), apiPreference=cv2.CAP_FFMPEG)
    ok, f1 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('cannot read video for bg evaluation')
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    mask = np.full(g1.shape, 255, np.uint8)
    for t in templates:
        half = int(t['tpl']) // 2 + 4
        x0 = max(0, int(t['cx']) - half)
        y0 = max(0, int(t['cy']) - half)
        x1 = min(mask.shape[1], int(t['cx']) + half)
        y1 = min(mask.shape[0], int(t['cy']) + half)
        mask[y0:y1, x0:x1] = 0
    feats = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners,
                                    qualityLevel=0.01, minDistance=8,
                                    blockSize=7, mask=mask)
    if feats is None or len(feats) == 0:
        cap.release()
        return dict(error='no background features found')
    p0 = feats.astype(np.float32).reshape(-1, 2)

    M_by_frame: Dict[int, Optional[np.ndarray]] = {
        int(st['frame']): (None if st.get('M') is None
                           else np.asarray(st['M'], float))
        for st in estimate_stats}
    per_frame: List[float] = []
    survivors: List[float] = []
    k = 1
    lk_kw = dict(winSize=(25, 25), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        gk = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        pk, st_, _ = cv2.calcOpticalFlowPyrLK(g1, gk, p0, None, **lk_kw)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gk, g1, pk, None, **lk_kw)
        good = ((st_.reshape(-1) == 1) & (st_b.reshape(-1) == 1) &
                (np.linalg.norm(p0 - back.reshape(-1, 2), axis=1) < fb_err))
        if not good.any():
            continue
        qk = pk.reshape(-1, 2)[good]
        M = M_by_frame.get(k)
        if M is None:
            continue
        r = np.linalg.norm(apply(to3(M), qk) - p0[good], axis=1)
        survivors.extend(float(v) for v in r)
        per_frame.append(float(np.mean(r)))
    cap.release()
    return dict(points=len(p0), residuals_px=stats_of(survivors),
                per_frame_mean_px=per_frame,
                note='forward-backward checked LK; residual computed only on '
                     'surviving points')


def eval_video(run_dir: Path, estimate_stats: List[Dict[str, Any]],
               bg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coordinate-domain metrics for an existing clip.

    Distinguishes the per-point RANSAC fit residual at the tracked template
    points from the background-point residual computed by `bg_residuals_lk`.
    """
    df = pd.read_csv(run_dir / 'Matched.csv')
    res: List[float] = []
    for st in estimate_stats:
        k = int(st['frame'])
        if st.get('M') is None:
            continue
        rows = df[df.Frame_No == k]
        pts = rows[['mx', 'my']].to_numpy(np.float64)
        ref = rows[['x', 'y']].to_numpy(np.float64)
        r = np.linalg.norm(apply(to3(np.asarray(st['M'], float)), pts) - ref,
                           axis=1)
        res.extend(float(v) for v in r)
    inl = [st.get('inliers', 0) / max(1, st.get('points', 1)) for st in estimate_stats]
    out = dict(
        tracked_point_residual_px=stats_of(res),
        ransac_inlier_rate={'mean': float(np.mean(inl)),
                            'min': float(np.min(inl)) if inl else None,
                            'frames': len(inl)},
        outlier_point_rate=float(1.0 - np.mean(inl)) if inl else None)
    if bg is not None:
        out['background_residual_px_not_tracked'] = bg
    return out


def run_once(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """One benchmark run: prepare -> match -> split -> warp -> evaluate."""
    cv2.setRNGSeed(args.seed)
    report: Dict[str, Any] = {
        'status': 'ok',
        'run_dir': str(run_dir),
        'seed': args.seed,
    }
    timings: Dict[str, float] = {}
    timings['wall_start'] = time.perf_counter()
    gt: Optional[Dict[str, Any]] = None

    try:
        if args.mode == 'synth':
            t0 = time.perf_counter()
            gt = synth_gen.generate(
                run_dir / 'synthetic.mp4', run_dir / 'synthetic_templates.csv',
                run_dir / 'ground_truth.json',
                width=args.width, height=args.height, frames=args.frames,
                vx=args.vx, vy=args.vy, rot_deg=args.rot_deg, scale=args.scale,
                seed=args.seed, tpl_size=args.tpl_size,
                n_templates=args.templates, fps=args.fps)
            timings['prepare_s'] = time.perf_counter() - t0
            video_path = run_dir / 'synthetic.mp4'
            csv_path = run_dir / 'synthetic_templates.csv'
        else:
            video_path = Path(args.video)
            csv_path = Path(args.csv)

        t0 = time.perf_counter()
        matcher = TemplateMatcher(str(video_path), str(csv_path))
        matcher.run(str(run_dir / 'Matched.csv'))
        timings['match_s'] = time.perf_counter() - t0

        corrector = AffineCorrector(str(video_path), str(run_dir / 'Matched.csv'),
                                    str(run_dir / 'Affined.mp4'),
                                    tmp=str(run_dir / 'tmp'))
        t0 = time.perf_counter()
        corrector.split()
        timings['split_s'] = time.perf_counter() - t0
        n_input = len(list(corrector.img.glob('frame_*.png')))

        t0 = time.perf_counter()
        corrector.correct(split=False)
        timings['warp_s'] = time.perf_counter() - t0
    except Exception as e:  # keep a usable report even when the run fails
        timings['wall_end'] = time.perf_counter()
        report['status'] = f'failed: {type(e).__name__}: {e}'
        report['timings_s'] = {k: v for k, v in timings.items()
                               if k.endswith('_s')}
        return report

    pipeline_s = sum(timings[k] for k in ('match_s', 'split_s', 'warp_s'))
    timings['pipeline_total_s'] = pipeline_s
    timings['wall_end'] = time.perf_counter()

    est = corrector.estimate_stats
    est_fail = sum(1 for s in est if not s.get('ok'))

    report['timings_s'] = {
        'match': timings['match_s'],
        'split_png_dump': timings['split_s'],
        'warp_and_write': timings['warp_s'],
        'pipeline_total': pipeline_s,
        'prepare_synthetic': timings.get('prepare_s', 0.0),
    }
    report['frames'] = {'input': n_input,
                        'stabilised_written': max(0, n_input - 1)}
    report['throughput'] = {
        'frames_per_second': (n_input - 1) / pipeline_s if pipeline_s > 0 else None,
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report['resources'] = {
        'max_rss_raw': usage.ru_maxrss,
        # NB: ru_maxrss is bytes on macOS, KiB on Linux.
        'max_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin'
                         else usage.ru_maxrss * 1024,
        'temp_png_bytes': dir_size(run_dir / 'tmp'),
        'output_video_bytes': (run_dir / 'Affined.mp4').stat().st_size
                               if (run_dir / 'Affined.mp4').exists() else 0,
        'matched_csv_bytes': (run_dir / 'Matched.csv').stat().st_size
                             if (run_dir / 'Matched.csv').exists() else 0,
    }
    inl = [s.get('inliers', 0) / max(1, s.get('points', 1)) for s in est]
    report['tracking_estimation'] = {
        'estimation_frames': len(est),
        'estimation_failures': est_fail,
        'ransac_inlier_rate_mean': float(np.mean(inl)) if inl else None,
    }

    if args.mode == 'synth' and gt is not None:
        report['accuracy'] = eval_synth(run_dir, gt, est)
        # keep the ground truth and parameters with the run for reproducibility
        report['synthetic'] = {key: gt[key] for key in (
            'motion', 'max_displacement_px', 'tpl_margin_cap_px',
            'frame_count', 'width', 'height', 'fps', 'seed', 'templates')}
    else:
        tpl = pd.read_csv(csv_path).to_dict('records')
        tpl = [dict(cx=int(r['x座標']), cy=int(r['y座標']),
                    tpl=int(r['テンプレートの大きさ（正方形）'])) for r in tpl]
        bg = None
        if args.bg_eval:
            bg = bg_residuals_lk(video_path, tpl, est)
        report['accuracy'] = eval_video(run_dir, est, bg)

    shutil.rmtree(run_dir / 'tmp', ignore_errors=True)
    report['resources'].pop('temp_png_bytes', None)
    report['resources']['temp_png_bytes_after_cleanup'] = 0
    return report


def env_info(args: argparse.Namespace) -> Dict[str, Any]:
    import PIL
    import pandas
    import tqdm
    return dict(
        timestamp=datetime.now().isoformat(timespec='seconds'),
        platform=platform.platform(),
        python=sys.version.replace('\n', ' '),
        git_commit=git_commit(),
        packages=dict(numpy=np.__version__, opencv=cv2.__version__,
                      pandas=pandas.__version__, pillow=PIL.__version__,
                      tqdm=tqdm.__version__),
        parameters=vars(args))


def write_report(run_dir: Path, env: Dict[str, Any],
                 report: Dict[str, Any]) -> None:
    """Write report.json and a human-readable summary.md (never overwrite)."""
    (run_dir / 'report.json').write_text(
        json.dumps({'env': env, 'report': report}, ensure_ascii=False, indent=2),
        encoding='utf-8')

    lines = [f"# Benchmark run — {run_dir.name}", "",
             f"- status: `{report['status']}`",
             f"- mode: `{env['parameters']['mode']}`  commit: {env['git_commit']}",
             f"- platform: {env['platform']}",
             f"- packages: numpy {env['packages']['numpy']}, "
             f"opencv {env['packages']['opencv']}", "", "## Timing (s)", "",
             "| stage | seconds |", "|---|---|"]
    for k, v in report.get('timings_s', {}).items():
        if isinstance(v, (int, float)):
            lines.append(f"| {k} | {v:.3f} |")
    t = report.get('throughput', {})
    if t.get('frames_per_second'):
        lines += ["", f"throughput: **{t['frames_per_second']:.2f} frames/s**"]
    f = report.get('frames', {})
    lines += ["", f"frames: input {f.get('input')} -> stabilised "
              f"{f.get('stabilised_written')}", "", "## Accuracy", ""]
    acc = report.get('accuracy', {})
    for k, v in acc.items():
        if isinstance(v, list):
            if len(v) > 12:
                lines.append(f"- {k}: [{v[0]:.3f}, {v[1]:.3f}] .. "
                             f"({len(v)} values, see report.json)")
            else:
                lines.append(f"- {k}: {v}")
        elif isinstance(v, dict) and 'mean' in v:
            parts = [f"mean {v['mean']:.3f}"]
            for key, label in (('median', 'median'), ('p95', 'p95'),
                               ('max', 'max')):
                if key in v:
                    parts.append(f"{label} {v[key]:.3f}")
            parts.append(f"(n={v.get('n', 'n/a')})")
            lines.append(f"- {k}: " + ', '.join(parts))
        elif isinstance(v, dict):
            compact = {k2: v2 for k2, v2 in v.items()
                       if not (isinstance(v2, list) and len(v2) > 12)}
            lines.append(f"- {k}: {json.dumps(compact, ensure_ascii=False)}")
            for k2, v2 in v.items():
                if isinstance(v2, list) and len(v2) > 12:
                    lines.append(f"  - {k2}: [{v2[0]:.3f}, {v2[1]:.3f}] .. "
                                 f"({len(v2)} values, see report.json)")
        else:
            lines.append(f"- {k}: {v}")
    lines += ["", "## Resources", ""]
    r = report.get('resources', {})
    if 'max_rss_bytes' in r:
        lines.append(f"- max RSS: {r['max_rss_bytes'] / 1024 / 1024:.1f} MiB")
    lines.append(f"- temp PNG bytes (after cleanup): "
                 f"{r.get('temp_png_bytes_after_cleanup', 'n/a')}")
    lines.append(f"- output video bytes: {r.get('output_video_bytes', 'n/a')}")
    lines.append("- note: with --repeat > 1 in one process, max RSS is the "
                 "process-lifetime peak (upper bound per run).")
    (run_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def new_run_dir(args: argparse.Namespace, index: int) -> Path:
    """Create a unique run directory; fail rather than overwrite results."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(args.outdir) / f"run_{stamp}_{args.mode}_r{index:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('synth', 'video'), required=True)
    p.add_argument('--video', default='01_input/input.mp4',
                   help='(video mode) input clip')
    p.add_argument('--csv', default='01_input/Input.csv',
                   help='(video mode) template definition CSV')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--outdir', default='03_output/benchmark')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bg-eval', action='store_true',
                   help='(video mode) forward-backward LK residual at '
                        'background points')
    # synthetic clip tuning
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--vx', type=float, default=2.0)
    p.add_argument('--vy', type=float, default=1.0)
    p.add_argument('--rot-deg', type=float, default=0.2)
    p.add_argument('--scale', type=float, default=1.002)
    p.add_argument('--tpl-size', type=int, default=64)
    p.add_argument('--templates', type=int, default=6)
    p.add_argument('--fps', type=float, default=30.0)
    args = p.parse_args()

    env = env_info(args)
    print(f"[env] commit {env['git_commit']} | {env['platform']} | "
          f"opencv {env['packages']['opencv']} numpy {env['packages']['numpy']}")
    for i in range(1, args.repeat + 1):
        run_dir = new_run_dir(args, i)
        print(f"[run {i}/{args.repeat}] {run_dir}")
        report = run_once(args, run_dir)
        write_report(run_dir, env, report)
        total = report.get('timings_s', {}).get('pipeline_total')
        if total is not None:
            print(f"[run {i}] status={report['status']} pipeline={total:.2f}s")
        else:
            print(f"[run {i}] status={report['status']}")


if __name__ == '__main__':
    main()
