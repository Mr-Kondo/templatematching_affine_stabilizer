#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template-based Affine Video Stabiliser (GUI + Thread-Safe Progress + In-place Edit)
================================================================================

This script stabilises a video by matching user‑specified template regions on the
first frame and then estimating a per‑frame affine transform to correct the
motion in subsequent frames.  It provides both a command‑line interface (CLI)
and an interactive Tk GUI.  The folder layout is assumed to be:

  - `01_input/`   : contains input data such as the MP4 file and Input.csv
  - `02_code/`    : contains this script
  - `03_output/`  : generated output (Matched.csv and Affined.mp4)

Dependencies: `opencv-python-headless`, `pandas`, `numpy`, `tqdm`, `pillow`

Usage:
    # Run the GUI (requires Tk to be available)
    python affine_template_matching.py --video 01_input/sample.mp4

    # Run in batch mode without GUI
    python affine_template_matching.py --video 01_input/sample.mp4 --nogui

"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import gc
import math
import os
import queue
import threading
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Utils
# =============================================================================


@contextmanager
def output_files(paths: Sequence[Path]):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'既存出力は上書きしません: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stabilizer-', dir=paths[0].parent) as folder:
        staged = [Path(folder) / p.name for p in paths]
        published = []
        try:
            yield staged
            for source, destination in zip(staged, paths):
                os.link(source, destination)
                published.append(destination)
        except BaseException:
            for path in published:
                path.unlink()
            raise


@contextmanager
def open_video(path: str | Path):
    cap = cv2.VideoCapture(str(path), apiPreference=cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise ValueError(f'動画を開けません: {path}')
        yield cap
    finally:
        cap.release()


def verify_video(path: Path, expected: int) -> None:
    with open_video(path) as cap:
        count = 0
        while cap.read()[0]:
            count += 1
    if count != expected or count == 0:
        raise RuntimeError(f'動画の書込検証に失敗: {path} ({count}/{expected}フレーム)')


def run_pipeline(video: str, csv_path: str, outdir: str | Path, q=None):
    folder = Path(outdir)
    paths = [folder / name for name in ('Matched.csv', 'Matched_diagnostics.csv',
                                        'Affined.mp4', 'Affined_diagnostics.csv')]
    inputs = {Path(video).resolve(), Path(csv_path).resolve()}
    if any(path.resolve() in inputs for path in paths):
        raise ValueError('入力と出力を同じパスにできません')
    matcher = TemplateMatcher(video, csv_path)
    try:
        with output_files(paths) as staged:
            matcher.run(str(staged[0]), q)
            corrector = AffineCorrector(video, str(staged[0]), str(staged[2]))
            corrector.precompute()
            corrector.correct(q)
    finally:
        matcher.cap.release()
    return matcher, corrector


def _pil_to_tk_photo(pil_img, tk_module):
    """Convert PIL Image to Tkinter PhotoImage without using ImageTk.

    This is a workaround for compatibility issues between Pillow's _imagingtk
    module and certain Tcl/Tk installations (e.g., uv-managed Python on macOS).
    """
    import io

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PPM')
    return tk_module.PhotoImage(data=buffer.getvalue())


def _quad(a, b, c) -> float:
    """Compute the quadratic interpolation coefficient for sub‑pixel peak estimation."""
    d = a - 2 * b + c
    return 0.0 if d == 0 else 0.5 * (a - c) / d


def refine(mat: np.ndarray, loc: Sequence[int]) -> Tuple[float, float]:
    """Refine the integer peak location to sub‑pixel accuracy using quadratic interpolation.

    Parameters
    ----------
    mat : ndarray
        The correlation surface returned by `cv2.matchTemplate`.
    loc : sequence of int
        Two‑element sequence of (x, y) integer coordinates of the peak.

    Returns
    -------
    (dx, dy) : tuple of floats
        The sub‑pixel offsets from the integer peak.
    """
    x, y = loc[0], loc[1]
    if 0 < x < mat.shape[1] - 1 and 0 < y < mat.shape[0] - 1:
        return (
            _quad(mat[y, x - 1], mat[y, x], mat[y, x + 1]),
            _quad(mat[y - 1, x], mat[y, x], mat[y + 1, x]),
        )
    return 0.0, 0.0


def crop(src: np.ndarray, cx: int, cy: int, h: int) -> np.ndarray:
    """Crop a square patch of size 2*h centred at (cx, cy), padding with zeros if necessary."""
    return crop_lrbt(src, cx, cy, h, h, h, h)


def crop_lrbt(src: np.ndarray, cx: int, cy: int, left: int, right: int, top: int,
              bottom: int) -> np.ndarray:
    """Crop a rectangular ROI using individual pixel margins measured from the centre.

    The ROI spans from (cx-left, cy-top) to (cx+right, cy+bottom).  If the window
    crosses the image borders, the missing areas are padded with zeros.
    """
    x0, y0 = cx - left, cy - top
    x1, y1 = cx + right, cy + bottom
    H, W = src.shape[:2]
    if left <= 0 or right <= 0 or top <= 0 or bottom <= 0:
        raise ValueError('ROI: 各探索マージンは正の整数が必要です')
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        raise ValueError('ROI: 切出し領域が完全に画像外です')
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


# =============================================================================
# Matcher
# =============================================================================

# --- Confidence thresholds -------------------------------------------------------------------
# Calibrated with measured distributions on the 306-frame real clip
# (01_input/input.mp4, 1525 template matches) and synthetic degradation
# scenarios (occlusion / illumination change / repeated pattern / moving
# object).  Real-clip distributions: score min 0.9732, peak clearness min
# 0.0062, inter-frame position delta max 4.67 px, ROI-edge peaks 0/1525.
# See benchmarks/README.md "信頼度評価" for the recorded evaluation data.
MATCH_SCORE_MIN = 0.95     # peak correlation must exceed this (occlusion defect: 0.91)
MATCH_CLEAR_MIN = 0.0002   # peak - best alternative peak (repeated pattern: ~0.0001)
MATCH_DELTA_MAX = 8.0      # px/frame jumped by the same template (gross errors: >16px)
MATCH_EXPAND_FACTOR = 2.0  # ROI expansion factor for the low-confidence re-search
# Second-score alarm (TM_CCOEFF_NORMED): CCORR keeps its illumination
# invariance for the adopted position, but its normalised score stays high
# (~0.98) even when a template is fully occluded, while CCOEFF drops to
# ~0.5.  The CCOEFF peak is therefore recorded as an independent alarm
# metric; the adopted position is still chosen by the CCORR criteria so the
# match behaviour and the outputs are unchanged.
MATCH_CCOEFF_ALARM = 0.78  # real clip min 0.819, synth normal min 0.854 (fn: occluded 0.48)
# An enlarged-window candidate is only adopted as the match when it beats the
# original peak by this margin: 1e-4-level CCORR differences are noise, and
# swapping the peak on them measurably degraded accuracy in the synthetic
# benchmark (measured: matched-error mean 12.2px / background residual 25.5px
# without the margin, vs <=1px level with it).
MATCH_RESEARCH_GAIN = 0.005


TEMPLATE_FIELDS = dict(zip(
    ('No.', 'x座標', 'y座標', 'テンプレートの大きさ（正方形）', '探索左', '探索右', '探索上', '探索下'),
    ('id', 'cx', 'cy', 'tpl', 'l', 'r', 't', 'b')))


def template_integer(raw: Any, label: str) -> int:
    try:
        number = Decimal(str(raw))
    except InvalidOperation:
        raise ValueError(f'{label}: 有限の整数を指定してください') from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f'{label}: 有限の整数を指定してください')
    # Avoid converting arbitrarily large exponents into an unbounded integer.
    if abs(number) > 2**31 - 1:
        raise ValueError(f'{label}: 整数が大きすぎます')
    return int(number)


def validate_template(row: Dict[str, Any], shape: Sequence[int], line: int) -> Dict[str, int]:
    tpl = {key: template_integer(row.get(key), f'行{line} {column}')
           for column, key in TEMPLATE_FIELDS.items()}
    height, width = shape[:2]
    limits = dict(id=(1, 2**31 - 1), cx=(0, width - 1), cy=(0, height - 1),
                  tpl=(2, min(width, height)))
    for key, (low, high) in limits.items():
        if not low <= tpl[key] <= high:
            column = next(c for c, k in TEMPLATE_FIELDS.items() if k == key)
            raise ValueError(f'行{line} {column}: {low}〜{high}の範囲で指定してください')
    if tpl['tpl'] % 2:
        raise ValueError(f'行{line} テンプレートの大きさ（正方形）: 偶数を指定してください')
    for column, key in list(TEMPLATE_FIELDS.items())[4:]:
        maximum = width if key in ('l', 'r') else height
        if not tpl['tpl'] // 2 <= tpl[key] <= maximum:
            raise ValueError(f'行{line} {column}: テンプレート半辺以上、画像寸法{maximum}以下が必要です')
    return tpl


def validate_templates(rows: Sequence[Dict[str, Any]], shape: Sequence[int]) -> List[Dict[str, int]]:
    templates = [validate_template(row, shape, line) for line, row in enumerate(rows, 2)]
    ids = set()
    for line, tpl in enumerate(templates, 2):
        if tpl['id'] in ids:
            raise ValueError(f"行{line} No.: ID {tpl['id']} が重複しています")
        ids.add(tpl['id'])
    if len({(t['cx'], t['cy']) for t in templates}) < 2:
        raise ValueError('テンプレート: 相似変換の推定には異なる中心座標が2点以上必要です')
    return templates


def load_templates(path: str | Path, shape: Sequence[int]) -> List[Dict[str, int]]:
    with open(path, newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source, strict=True)
        columns = reader.fieldnames or []
        missing = set(TEMPLATE_FIELDS) - set(columns)
        if missing or len(columns) != len(set(columns)):
            raise ValueError(f'CSV列: 必須列の不足または重複があります ({", ".join(sorted(missing))})')
        rows = []
        for row in reader:
            if None in row or any(row[c] is None for c in TEMPLATE_FIELDS):
                raise ValueError(f'行{reader.line_num}: CSVの列数が一致しません')
            rows.append({key: row[column] for column, key in TEMPLATE_FIELDS.items()})
    return validate_templates(rows, shape)


class TemplateMatcher:
    """Perform template matching on a video to track template locations over time."""

    def __init__(self, video: str, csv_path: str):
        self.cap = cv2.VideoCapture(video, apiPreference=cv2.CAP_FFMPEG)
        ok, first = self.cap.read()
        if not ok:
            self.cap.release()
            raise ValueError(f'動画を読み込めません: {video}')
        self.first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        try:
            self.templates = self._load(csv_path)
        except (OSError, ValueError, csv.Error):
            self.cap.release()
            raise

    def _load(self, p: str) -> List[Dict[str, Any]]:
        templates = load_templates(p, self.first.shape)
        return [dict(id=t['id'], cx=t['cx'], cy=t['cy'], th=t['tpl'] // 2,
                     l=t['l'], r=t['r'], t=t['t'], b=t['b'],
                     patch=crop(self.first, t['cx'], t['cy'], t['tpl'] // 2))
                for t in templates]

    def _match(self, g: np.ndarray, t: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Match a single template t in the current grayscale frame g.

        Returns the adopted position ``(mx, my)`` and the confidence
        diagnostics of this match (score, peak clearness, ROI-edge flag,
        whether a re-search with an enlarged window ran, and whether the
        enlarged-window candidate was adopted).

        Low-confidence matches (score / peak clearness below the calibrated
        thresholds, or the peak sitting against the ROI border) trigger a
        re-search with an enlarged ROI; the candidate with the higher
        confidence (score, then peak clearness) is adopted.  The pipeline
        never stops on low confidence: the match is always adopted and the
        diagnostics mark it for reporting instead.
        """
        mx, my, score, clear, edge, ccoeff_peak = self._match_window(
            g, t, t['l'], t['r'], t['t'], t['b'])
        trigger = (score < MATCH_SCORE_MIN or clear < MATCH_CLEAR_MIN or edge)
        re_searched = expanded = False
        if trigger:
            H, W = g.shape[:2]
            l2 = min(max(int(t['l'] * MATCH_EXPAND_FACTOR), t['l'] + 16), max(t['cx'], t['l']))
            r2 = min(max(int(t['r'] * MATCH_EXPAND_FACTOR), t['r'] + 16), max(W - 1 - t['cx'], t['r']))
            t2 = min(max(int(t['t'] * MATCH_EXPAND_FACTOR), t['t'] + 16), max(t['cy'], t['t']))
            b2 = min(max(int(t['b'] * MATCH_EXPAND_FACTOR), t['b'] + 16), max(H - 1 - t['cy'], t['b']))
            if (l2, r2, t2, b2) != (t['l'], t['r'], t['t'], t['b']):
                mx2, my2, s2, c2, e2, cc2 = self._match_window(g, t, l2, r2, t2, b2)
                re_searched = True
                # Adopt the most confident candidate (score, then clearness),
                # but only on a clear win: noise-level score differences must
                # not swap the peak (see MATCH_RESEARCH_GAIN).
                if s2 > score + MATCH_RESEARCH_GAIN and (s2, c2) > (score, clear):
                    mx, my, score, clear, edge, ccoeff_peak = mx2, my2, s2, c2, e2, cc2
                    expanded = True
        return mx, my, dict(score=round(float(score), 6),
                            clearness=round(float(clear), 6),
                            ccoeff_peak=round(float(ccoeff_peak), 6),
                            roi_edge=bool(edge),
                            re_searched=bool(re_searched),
                            expanded_adopted=bool(expanded),
                            low_conf=bool(trigger or ccoeff_peak < MATCH_CCOEFF_ALARM))

    def _match_window(self, g: np.ndarray, t: Dict[str, Any], l: int, r: int, w: int,
                      b: int) -> Tuple[float, float, float, float, bool, float]:
        """Match template t within one specific search window (l, r, w, b margins).

        Computes both surface metrics in the same window: the CCORR peak
        (adoption criterion) and the CCOEFF peak (independent alarm metric
        which degrades under occlusion even when CCORR stays high).
        """
        roi = crop_lrbt(g, t['cx'], t['cy'], l, r, w, b)
        res = cv2.matchTemplate(roi, t['patch'], cv2.TM_CCORR_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        # Clearness: score minus the best alternative peak (search peak
        # neighbourhood suppressed so alternative peaks are counted).
        m = res.copy()
        m[max(0, loc[1] - 3):loc[1] + 4, max(0, loc[0] - 3):loc[0] + 4] = 0.0
        edge = bool(loc[0] in (0, res.shape[1] - 1) or loc[1] in (0, res.shape[0] - 1))
        dx, dy = refine(res, loc)
        mx = t['cx'] - l + loc[0] + dx + t['th']
        my = t['cy'] - w + loc[1] + dy + t['th']
        ccoeff = float(cv2.matchTemplate(roi, t['patch'], cv2.TM_CCOEFF_NORMED).max())
        return float(mx), float(my), float(score), float(score - m.max()), edge, ccoeff

    def run(self, out_csv: str = '03_output/Matched.csv', q: queue.Queue | None = None) -> None:
        """Run template matching across all frames and save results to out_csv.

        Parameters
        ----------
        out_csv : str, optional
            Path to the CSV file where match results should be saved.  The
            default points into the `03_output` folder.  Each row of the
            resulting CSV contains: Frame_No, Template_No, x, y, mx, my.
        q : Queue, optional
            If provided, progress updates will be sent to this queue as
            ('setmax', total_frames-1) once at the start and ('prog', frame)
            after each processed frame.
        """
        try:
            rec: List[List[float]] = []
            diag: List[Dict[str, Any]] = []
            previous_pos: Dict[int, Tuple[float, float]] = {
                t['id']: (float(t['cx']), float(t['cy'])) for t in self.templates}
            idx = 1
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            # Initialise progress counter; always define prog to avoid UnboundLocalError
            prog = 0
            if q:
                q.put(('setmax', total))
            with tqdm(total=total, desc='match') as bar:
                while True:
                    ok, f = self.cap.read()
                    if not ok:
                        break
                    idx += 1
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    for t in self.templates:
                        mx, my, d = self._match(g, t)
                        rec.append([idx, t['id'], t['cx'], t['cy'], mx, my])
                        # Inter-frame position jump of the same template: catches
                        # gross mismatches that per-frame scores cannot see
                        # (e.g. objects moving across the template ROI).
                        dxp = math.hypot(mx - previous_pos[t['id']][0],
                                         my - previous_pos[t['id']][1])
                        d['delta_px'] = round(float(dxp), 4)
                        d['low_conf'] = bool(d['low_conf'] or d['delta_px'] > MATCH_DELTA_MAX)
                        previous_pos[t['id']] = (mx, my)
                        diag.append([idx, t['id'], d['score'], d['clearness'],
                                     d['ccoeff_peak'], d['roi_edge'], d['delta_px'],
                                     d['re_searched'], d['expanded_adopted'], d['low_conf']])
                    bar.update(1)
                    if q:
                        prog += 1
                        q.put(('prog', prog))
            # Total number of frames actually read (useful metadata for callers;
            # the reference frame 1 is included even though it is not matched).
            self.frame_count = idx
            self.match_stats = diag
            # Ensure output directory exists
            out_path = Path(out_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if total >= 0 and idx != total + 1:
                raise RuntimeError(f'動画の途中読込に失敗: {idx}/{total + 1}フレーム')
            with output_files([out_path, out_path.with_name(out_path.stem + '_diagnostics.csv')]) as staged:
                pd.DataFrame(rec, columns=['Frame_No', 'Template_No', 'x', 'y', 'mx',
                                           'my']).to_csv(staged[0], index=False)
                # Confidence diagnostics (separate file: the Matched.csv format is
                # unchanged for existing consumers).
                pd.DataFrame(diag, columns=['Frame_No', 'Template_No', 'score', 'clearness',
                                            'ccoeff_peak', 'roi_edge', 'delta_px', 're_searched',
                                            'expanded_adopted', 'low_conf']).to_csv(
                    staged[1], index=False)
            del self.first
            gc.collect()
        finally:
            self.cap.release()


# =============================================================================
# Affine
# =============================================================================

# --- Estimation-quality thresholds ------------------------------------------------------------
# Calibrated on the 305 estimated frames of the real-clip Matched.csv
# (inlier count {3: 7, 4: 273, 5: 25}, inlier ratio min 0.600, max inlier
# residual 1.259 px, inlier spread std min 14.6 px) and on the frame-to-frame
# matrix deltas of the same clip (|Δt| max 4.65 px, |Δscale| max 0.006,
# |Δrot| max 0.243 deg).  See benchmarks/README.md "信頼度評価".
EST_INLIER_COUNT_MIN = 3    # quality threshold; two distinct pairs suffice for estimation
EST_INLIER_RATIO_MIN = 0.6  # inliers / points (real clip min = 3/5)
EST_RESIDUAL_MAX = 2.0      # px, max residual of adopted points (= RANSAC threshold)
EST_SPREAD_MIN = 10.0       # px, per-axis std of inlier positions (degenerate config)
EST_DELTA_T_MAX = 12.0      # px/frame, translation change (2x observed max 4.65)
EST_DELTA_SCALE_MAX = 0.02  # /frame relative scale change (observed max 0.006)
EST_DELTA_ROT_MAX = 1.0     # deg/frame rotation change (observed max 0.243)


class AffineCorrector:
    """Correct video frames using affine (and perspective) transformations based on matched points.

    Frames are processed one at a time straight from the input video
    (``VideoCapture -> matrix estimation -> warp -> VideoWriter``); no
    intermediate images are written to disk, so memory and disk usage stay
    constant regardless of the number of frames.

    The per-frame affine matrix and the fixed perspective transform are
    composed into a single 3x3 homography, so each frame is resampled only
    once (no double interpolation between the two warps).  The output keeps
    every input frame, including frame 1 (identity affine), so the frame
    count matches the input 1:1.
    """

    def __init__(self,
                 video: str,
                 csv: str,
                 outv: str = '03_output/Affined.mp4'):
        self.matches = pd.read_csv(csv)
        self.video = Path(video)
        self.out = Path(outv)
        # Diagnostics recorded for each affine-estimation call (frame, point count,
        # inlier count, success flag and the estimated 2x3 matrix).  Populated by
        # `_M` and consumed by the benchmark harness to compute tracking/estimation
        # failure rates and coordinate-domain accuracy metrics.
        self.estimate_stats: List[Dict[str, Any]] = []
        # Per-frame matrix cache filled by `precompute()`; `correct()` reuses it so
        # the estimation stage can be timed separately from the warp stage.
        self._mcache: Dict[int, np.ndarray] = {}
        # (frame, matrix) of the last freshly estimated frame, for the
        # frame-to-frame delta diagnostics (`d_t`, `d_scale`, `d_rot`).
        self._last_est: Tuple[int, np.ndarray] | None = None

    def precompute(self) -> None:
        """Estimate and cache the affine matrix for every frame of the matches CSV.

        Also writes the estimation-quality diagnostics next to the output
        video (``<output>_diagnostics.csv``).
        """
        for n in sorted(self.matches.Frame_No.unique()):
            self._M(int(n))
        self.write_diagnostics()

    def write_diagnostics(self, path: str | None = None) -> None:
        """Write the per-frame estimation quality diagnostics as a CSV file.

        Columns: frame, points, inliers, inlier_ratio, residual_max,
        spread_x, spread_y, delta_frame, d_t, d_scale, d_rot, est_ok,
        quality_ok.  ``est_ok`` reports estimation success while
        ``quality_ok`` reports (independently) whether the per-frame
        quality thresholds passed.
        """
        cols = ['frame', 'points', 'inliers', 'inlier_ratio', 'residual_max',
                'spread_x', 'spread_y', 'delta_frame', 'd_t', 'd_scale', 'd_rot',
                'est_ok', 'quality_ok']
        p = Path(path) if path is not None else self.out.with_name(
            self.out.stem + '_diagnostics.csv')
        p.parent.mkdir(parents=True, exist_ok=True)
        with output_files([p]) as staged:
            if self.estimate_stats:
                df = pd.DataFrame(self.estimate_stats)
                df['est_ok'] = df['ok']
                df[[c for c in cols if c in df.columns]].to_csv(staged[0], index=False)
            else:
                pd.DataFrame(columns=cols).to_csv(staged[0], index=False)


    def _M(self, n: int) -> np.ndarray:
        """Estimate the affine matrix for frame n using matched control points.

        Results are cached per frame so repeated calls (e.g. after
        `precompute()`) do not repeat the RANSAC estimation.

        In addition to the matrix, per-frame quality diagnostics are recorded
        in ``estimate_stats``: RANSAC inlier count/ratio, residuals and
        position spread of the adopted points, and the frame-to-frame matrix
        deltas (translation / scale / rotation) used to flag abrupt changes.
        ``est_ok`` means the estimation succeeded; ``quality_ok`` is the
        independent quality verdict using the calibrated thresholds.

        Raises
        ------
        RuntimeError
            If RANSAC fails entirely (estimation failure, reported
            separately from a quality failure).
        """
        cached = self._mcache.get(n)
        if cached is not None:
            return cached
        df = self.matches[self.matches.Frame_No == n]
        src = df[['mx', 'my']].to_numpy(np.float32)
        dst = df[['x', 'y']].to_numpy(np.float32)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
        mask = inliers.ravel().astype(bool) if inliers is not None else None
        n_inl = int(mask.sum()) if mask is not None else 0
        entry = dict(frame=n,
                     points=int(len(src)),
                     inliers=n_inl,
                     ok=M is not None,
                     M=None if M is None else M.astype(float).tolist())
        if M is not None:
            inp = src[mask] if mask is not None else np.empty((0, 2), np.float32)
            pred = (M[:, :2] @ inp.T + M[:, 2:3]).T
            entry['inlier_ratio'] = round(float(mask.mean()) if mask is not None else 0.0, 4)
            entry['residual_max'] = round(float(np.linalg.norm(pred - dst[mask], axis=1).max())
                                          if mask is not None and mask.any() else float('nan'), 4)
            entry['spread_x'] = round(float(inp[:, 0].std()), 2) if inp.shape[0] else float('nan')
            entry['spread_y'] = round(float(inp[:, 1].std()), 2) if inp.shape[0] else float('nan')
            # frame-to-frame matrix delta vs the previous estimated frame
            prev = self._last_est
            if prev is not None and n - prev[0] == 1:
                B = prev[1]
                entry['delta_frame'] = prev[0]
                entry['d_t'] = round(float(np.linalg.norm(M[:, 2] - B[:, 2])), 4)
                sa = float(np.linalg.svd(B[:2, :2], compute_uv=False).mean())
                sb = float(np.linalg.svd(M[:2, :2], compute_uv=False).mean())
                entry['d_scale'] = round(abs(sb / sa - 1.0), 6)
                entry['d_rot'] = round(abs(math.degrees(math.atan2(M[1, 0], M[0, 0])
                                                        - math.atan2(B[1, 0], B[0, 0]))), 4)
            # Independent quality verdict (est_ok and quality_ok are reported
            # separately; quality_ok never blocks the pipeline).
            quality = bool(
                n_inl >= EST_INLIER_COUNT_MIN
                and entry['inlier_ratio'] >= EST_INLIER_RATIO_MIN
                and entry.get('residual_max', float('nan')) <= EST_RESIDUAL_MAX
                and entry.get('spread_x', float('nan')) >= EST_SPREAD_MIN
                and entry.get('spread_y', float('nan')) >= EST_SPREAD_MIN
                and (entry.get('d_t', 0.0) <= EST_DELTA_T_MAX
                     and entry.get('d_scale', 0.0) <= EST_DELTA_SCALE_MAX
                     and entry.get('d_rot', 0.0) <= EST_DELTA_ROT_MAX))
            entry['quality_ok'] = quality
            self._last_est = (n, M)
        self.estimate_stats.append(entry)
        if M is None:
            raise RuntimeError('affine fail')
        self._mcache[n] = M
        return M

    def correct(self, q: queue.Queue | None = None) -> None:
        """Stream the input video once, warping each frame with its estimated transform.

        The output contains every input frame (1..K), so the frame count and
        the frame correspondence are 1:1 with the input:

        - frame 1 (the reference frame from which templates were extracted)
          is emitted with the identity affine composed with the fixed
          perspective transform,
        - frames 2..K are emitted with their estimated affine matrix composed
          with the fixed perspective transform.

        The tracking CSV covers frames 2..K (frame 1 is the template
        reference and needs no matching); existing CSVs remain compatible.

        Parameters
        ----------
        q : Queue, optional
            Progress queue as used by the GUI: ('setmax', total) once at the
            start and ('prog', done) after each written frame.
        """
        if self.video.resolve() == self.out.resolve():
            raise ValueError('入力動画と出力動画を同じパスにできません')
        with open_video(self.video) as cap, output_files([self.out]) as staged:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width < 2 or height < 2 or not math.isfinite(fps) or fps <= 0:
                raise ValueError('動画の寸法・FPSが不正です')
            writer = cv2.VideoWriter(str(staged[0]), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (width, height))
            try:
                if not writer.isOpened():
                    raise RuntimeError(f'動画の出力を開始できません: {self.out}')
                done = self._warp_frames(cap, writer, q)
            finally:
                writer.release()
            verify_video(staged[0], done)

    def _warp_frames(self, cap, writer, q) -> int:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if q is not None:
            q.put(('setmax', total))
        source = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        target = np.float32([[10, 5], [width-11, 10], [width-6, height-11], [5, height-6]])
        perspective = cv2.getPerspectiveTransform(source, target)
        done = 0
        with tqdm(total=total, desc='warp') as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                affine = np.eye(3) if done == 0 else np.vstack([self._M(done + 1), [0, 0, 1]])
                writer.write(cv2.warpPerspective(frame, perspective @ affine, (width, height),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
                done += 1
                bar.update(1)
                if q is not None:
                    q.put(('prog', done))
        if done == 0 or (total > 0 and done != total):
            raise RuntimeError(f'動画の途中読込に失敗: {done}/{total}フレーム')
        return done


# =============================================================================
# GUI
# =============================================================================


class TemplateGUI:
    """Interactive GUI for selecting templates and running the stabilisation pipeline."""

    def __init__(self, video: str, csv_p: str = '01_input/Input.csv'):
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

        from PIL import Image

        # For Pillow ≥10 compatibility: fallback to older API name
        try:
            self.ResamplingFilter = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:
            self.ResamplingFilter = Image.LANCZOS  # type: ignore[attr-defined]

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.video = video
        self.csv = Path(csv_p)
        self.cap = cv2.VideoCapture(video, apiPreference=cv2.CAP_FFMPEG)
        try:
            ok, first = self.cap.read()
            if not ok:
                self.cap.release()
                raise ValueError(f'動画を読み込めません: {video}')
            self.frame_shape = first.shape
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.tpls: List[Dict[str, Any]] = []
            self.hist: List[Dict[str, Any]] = []
            self.zoom = 1.0
            self.root = tk.Tk()
            self.root.title('Template Designer')
            self._build_widgets()
            self._load_csv()
            self._show(1)
            self.root.mainloop()
        finally:
            self.cap.release()

    # -------- widgets --------
    def _build_widgets(self) -> None:
        ttk = self.ttk
        ctl = ttk.Frame(self.root)
        ctl.pack(fill='x')
        ttk.Label(ctl, text='Template').pack(side='left')
        # Free‑form numeric entries for template size
        self.ent_tpl = ttk.Entry(ctl, width=6)
        self.ent_tpl.insert(0, '64')
        self.ent_tpl.pack(side='left')
        # Entries for ROI margins
        ttk.Label(ctl, text='SearchLeft').pack(side='left')
        self.ent_left = ttk.Entry(ctl, width=6)
        self.ent_left.insert(0, '128')
        self.ent_left.pack(side='left')
        ttk.Label(ctl, text='SearchRight').pack(side='left')
        self.ent_right = ttk.Entry(ctl, width=6)
        self.ent_right.insert(0, '128')
        self.ent_right.pack(side='left')
        ttk.Label(ctl, text='SearchTop').pack(side='left')
        self.ent_top = ttk.Entry(ctl, width=6)
        self.ent_top.insert(0, '128')
        self.ent_top.pack(side='left')
        ttk.Label(ctl, text='SearchBottom').pack(side='left')
        self.ent_bottom = ttk.Entry(ctl, width=6)
        self.ent_bottom.insert(0, '128')
        self.ent_bottom.pack(side='left')
        # Control buttons
        self.btn_delete = ttk.Button(ctl, text='Delete', command=self._del)
        self.btn_delete.pack(side='right')
        self.btn_undo = ttk.Button(ctl, text='Undo', command=self._undo)
        self.btn_undo.pack(side='right')
        self.btn_execute = ttk.Button(ctl, text='Execute', command=self._exec)
        self.btn_execute.pack(side='right')
        self.btn_zoom_in = ttk.Button(ctl, text='+', command=self._zoom_in)
        self.btn_zoom_in.pack(side='right')
        self.btn_zoom_out = ttk.Button(ctl, text='-', command=self._zoom_out)
        self.btn_zoom_out.pack(side='right')
        # Table for templates (with ID column)
        self.tree = ttk.Treeview(
            self.root,
            columns=('no', 'cx', 'cy', 'tpl', 'l', 'r', 't', 'b'),
            show='headings',
            height=4,
        )
        for col, text in [
            ('no', 'No.'),
            ('cx', 'x座標'),
            ('cy', 'y座標'),
            ('tpl', 'テンプレートサイズ'),
            ('l', '探索左'),
            ('r', '探索右'),
            ('t', '探索上'),
            ('b', '探索下'),
        ]:
            self.tree.heading(col, text=text)
        self.tree.bind('<Double-1>', self._edit)
        self.tree.pack(fill='x')
        # Progress bar
        self.pb = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate')
        self.pb.pack(fill='x', padx=4, pady=4)
        # Frame slider and canvas
        self.scale = ttk.Scale(self.root,
                               from_=1,
                               to=self.total,
                               orient='horizontal',
                               command=self._seek)
        self.scale.pack(fill='x')
        self.canvas = self.tk.Canvas(self.root)
        self.canvas.pack()
        # Status label showing mouse coordinates
        self.status = self.ttk.Label(self.root, text='x: -, y: -')
        self.status.pack(fill='x', pady=2)
        # Bindings
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Motion>', self._on_motion)
        self._photo = None

    # -------- data io --------
    def _load_csv(self) -> None:
        """Load existing templates from the CSV if it exists."""
        if not self.csv.exists():
            return
        try:
            templates = load_templates(self.csv, self.frame_shape)
        except (OSError, ValueError, csv.Error) as error:
            self.messagebox.showerror('入力エラー', str(error))
            return
        self.tpls.clear()
        self.tree.delete(*self.tree.get_children())
        for t in templates:
            self._add_tpl(t['cx'], t['cy'], t['tpl'], t['l'], t['r'], t['t'], t['b'],
                          existing=True, id=t['id'])

    def _add_tpl(
        self,
        cx: int,
        cy: int,
        tpl: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
        existing: bool = False,
        id: int | None = None,
    ) -> None:
        """Append a new template entry to the list and TreeView."""
        if not existing:
            id = max((t['id'] for t in self.tpls + self.hist), default=0) + 1
        d = validate_template(dict(id=id, cx=cx, cy=cy, tpl=tpl, l=left, r=right,
                                   t=top, b=bottom), self.frame_shape, len(self.tpls) + 2)
        self.tpls.append(d)
        self.tree.insert('',
                         'end',
                         iid=d['id'],
                         values=(d['id'], cx, cy, tpl, left, right, top, bottom))

    # -------- UI actions --------
    def _show(self, idx: int) -> None:
        from PIL import Image

        # Fallback to older API name if necessary
        try:
            ResamplingFilter = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:
            ResamplingFilter = Image.LANCZOS  # type: ignore[attr-defined]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Draw templates and ROIs
        for t in self.tpls:
            cx, cy = t['cx'], t['cy']
            ht = t['tpl'] // 2
            hl = t['l']
            hr = t['r']
            ht_margin = t['t']
            hb = t['b']
            cv2.rectangle(rgb, (cx - ht, cy - ht), (cx + ht, cy + ht), (0, 0, 255), 2)
            cv2.putText(
                rgb,
                str(t['id']),
                (cx - ht, cy - ht - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(rgb, (cx - hl, cy - ht_margin), (cx + hr, cy + hb), (255, 0, 0), 1)
        img = Image.fromarray(rgb)
        if self.zoom != 1.0:
            w, h = img.size
            img = img.resize((int(w * self.zoom), int(h * self.zoom)), ResamplingFilter)
        self._photo = _pil_to_tk_photo(img, self.tk)
        self.canvas.config(width=img.width, height=img.height)
        if getattr(self, '_canvas_image', None) is None:
            self._canvas_image = self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        else:
            self.canvas.itemconfigure(self._canvas_image, image=self._photo)

    def _seek(self, val: float) -> None:
        self._show(int(float(val)))

    def _click(self, e) -> None:
        try:
            self._add_tpl(int(e.x / self.zoom), int(e.y / self.zoom),
                          self.ent_tpl.get(), self.ent_left.get(), self.ent_right.get(),
                          self.ent_top.get(), self.ent_bottom.get())
        except ValueError as error:
            self.messagebox.showerror('入力エラー', str(error))
            return
        self._show(int(self.scale.get()))

    def _del(self) -> None:
        iid = self.tree.focus()
        if not iid:
            return
        self.hist.append(next(t for t in self.tpls if t['id'] == int(iid)))
        self.tpls = [t for t in self.tpls if t['id'] != int(iid)]
        self.tree.delete(iid)
        self._show(int(self.scale.get()))

    def _undo(self) -> None:
        if not self.hist:
            return
        t = self.hist.pop()
        self.tpls.append(t)
        self.tree.insert('',
                         'end',
                         iid=t['id'],
                         values=(t['id'], t['cx'], t['cy'], t['tpl'], t['l'], t['r'], t['t'],
                                 t['b']))
        self._show(int(self.scale.get()))

    def _edit(self, evt) -> None:
        iid = self.tree.focus()
        if not iid:
            return
        row = next(t for t in self.tpls if t['id'] == int(iid))
        edited = row.copy()
        for key in ('cx', 'cy', 'tpl', 'l', 'r', 't', 'b'):
            val = self.simpledialog.askinteger('Edit',
                                               f'{key} value',
                                               initialvalue=row[key],
                                               parent=self.root,
                                               minvalue=0)
            if val is None:
                return
            edited[key] = val
        try:
            edited = validate_template(edited, self.frame_shape, self.tpls.index(row) + 2)
        except ValueError as error:
            self.messagebox.showerror('入力エラー', str(error))
            return
        row.update(edited)
        self.tree.item(iid,
                       values=(row['id'], row['cx'], row['cy'], row['tpl'], row['l'], row['r'],
                               row['t'], row['b']))
        self._show(int(self.scale.get()))

    def _on_motion(self, event) -> None:
        x = int(event.x / self.zoom)
        y = int(event.y / self.zoom)
        self.status.config(text=f'x: {x}, y: {y}')

    def _zoom_in(self) -> None:
        self.zoom *= 1.2
        self._show(int(float(self.scale.get())))

    def _zoom_out(self) -> None:
        self.zoom /= 1.2
        self._show(int(float(self.scale.get())))

    # -------- execute --------
    def _exec(self) -> None:
        try:
            self.tpls = validate_templates(self.tpls, self.frame_shape)
        except ValueError as error:
            self.messagebox.showerror('入力エラー', str(error))
            return
        # Write current templates to CSV (ensure folder exists)
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['No.', 'x座標', 'y座標', 'テンプレートの大きさ（正方形）', '探索左', '探索右', '探索上', '探索下'])
            for t in self.tpls:
                w.writerow([t['id'], t['cx'], t['cy'], t['tpl'], t['l'], t['r'], t['t'], t['b']])
        # Disable buttons during processing
        self.btn_delete.config(state='disabled')
        self.btn_undo.config(state='disabled')
        self.btn_execute.config(state='disabled')
        q: queue.Queue[Tuple[str, Any]] = queue.Queue()

        self.outdir = Path('03_output') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')

        def task() -> None:
            try:
                run_pipeline(self.video, str(self.csv), self.outdir, q)
                q.put(('done', None))
            except Exception as e:
                q.put(('err', e))

        threading.Thread(target=task, daemon=True).start()
        self._poll(q)

    def _poll(self, q: queue.Queue[Tuple[str, Any]]) -> None:
        for _ in range(1000):
            try:
                typ, payload = q.get_nowait()
            except queue.Empty:
                break
            if typ == 'setmax':
                self.pb['maximum'] = payload
                self.pb['value'] = 0
            elif typ == 'prog':
                self.pb['value'] = payload
            elif typ in ('done', 'err'):
                self._finish(typ, payload)
                return
        self.root.after(0 if not q.empty() else 100, self._poll, q)

    def _finish(self, typ: str, payload: Any) -> None:
        for button in (self.btn_delete, self.btn_undo, self.btn_execute):
            button.config(state='normal')
        if typ == 'done':
            self.messagebox.showinfo('完了', f'補正完了: {self.outdir}')
        else:
            self.messagebox.showerror('Error', str(payload))
        self.root.destroy()


# =============================================================================
# main
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--video',
        required=False,
        default=None,
        help=
        'Path to the input video file (MP4). If omitted, the first MP4 under 01_input/ will be used if available.'
    )
    p.add_argument('--csv',
                   default='01_input/Input.csv',
                   help='Path to the CSV file with template definitions')
    p.add_argument('--nogui',
                   action='store_true',
                   help='Run in batch mode without launching the GUI')
    p.add_argument('--outdir', default='03_output', help='Output directory (existing files are never overwritten)')
    a = p.parse_args()
    # Auto-pick first MP4 in 01_input when --video is omitted
    if a.video is None:
        default_dir = Path('01_input')
        candidates = []
        if default_dir.exists():
            for pth in default_dir.rglob('*'):
                if pth.is_file() and pth.suffix.lower() == '.mp4':
                    candidates.append(pth)
        if not candidates:
            print(
                'Error: --video not provided and no MP4 found under 01_input/. Please specify --video 01_input/your.mp4'
            )
            raise SystemExit(2)
        # Pick the first in sorted order for determinism
        a.video = str(sorted(candidates)[0])
        print(f'[info] --video omitted; using {a.video}')
    if a.nogui:
        # Batch mode: run matching and correction directly (streamed: no
        # intermediate PNG files are produced)
        try:
            matcher, corrector = run_pipeline(a.video, a.csv, a.outdir)
        except (OSError, ValueError, csv.Error, RuntimeError) as error:
            p.error(str(error))
        # Confidence report: estimation success and quality pass are separate
        # verdicts (issue #8).  Low-confidence matches are flagged in the
        # diagnostics files, they do not stop the pipeline.
        estim = corrector.estimate_stats
        n_est = len(estim)
        n_fail = sum(1 for s in estim if not s.get('ok'))
        q_fail = [s['frame'] for s in estim if s.get('quality_ok') is False]
        low = sum(1 for d in matcher.match_stats if d[-1])
        print(f'[report] frames: {n_est}  est success: {n_est - n_fail}  '
              f'est failure: {n_fail}')
        print(f'[report] quality: pass {n_est - len(q_fail)}  fail {len(q_fail)}'
              + (f'  (frames {q_fail[:20]})' if q_fail else ''))
        print(f'[report] low-confidence matches: {low} / {len(matcher.match_stats)} '
              f'(details: {a.outdir}/Matched_diagnostics.csv, '
              f'{a.outdir}/Affined_diagnostics.csv)')
    else:
        # Interactive GUI
        TemplateGUI(a.video, a.csv)


if __name__ == '__main__':
    main()
