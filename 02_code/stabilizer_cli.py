"""Command-line arguments and reporting for both application entries."""
import argparse
import csv
from pathlib import Path

from stabilizer_core import run_pipeline
from stabilizer_gui import TemplateGUI

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
        low = matcher.low_conf_count
        print(f'[report] frames: {n_est}  est success: {n_est - n_fail}  '
              f'est failure: {n_fail}')
        print(f'[report] quality: pass {n_est - len(q_fail)}  fail {len(q_fail)}'
              + (f'  (frames {q_fail[:20]})' if q_fail else ''))
        print(f'[report] low-confidence matches: {low} / {matcher.match_count} '
              f'(details: {a.outdir}/Matched_diagnostics.csv, '
              f'{a.outdir}/Affined_diagnostics.csv)')
    else:
        # Interactive GUI
        TemplateGUI(a.video, a.csv)


