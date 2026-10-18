"""Command-line arguments and reporting for both application entries."""
import argparse
import csv
from pathlib import Path

from stabilizer_core import run_pipeline
from stabilizer_gui import TemplateGUI


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None,
                        help='Path to the input video file (MP4). If omitted, the first MP4 under 01_input/ will be used if available.')
    parser.add_argument('--csv', default='01_input/Input.csv', help='Path to the CSV file with template definitions')
    parser.add_argument('--nogui', action='store_true', help='Run in batch mode without launching the GUI')
    parser.add_argument('--outdir', default='03_output', help='Output directory (existing files are never overwritten)')
    return parser


def _video_path(requested):
    if requested is not None:
        return requested
    candidates = sorted(p for p in Path('01_input').rglob('*')
                        if p.is_file() and p.suffix.lower() == '.mp4')
    if not candidates:
        print('Error: --video not provided and no MP4 found under 01_input/. Please specify --video 01_input/your.mp4')
        raise SystemExit(2)
    video = str(candidates[0])
    print(f'[info] --video omitted; using {video}')
    return video


def _print_report(matcher, corrector, outdir):
    estimates = corrector.estimate_stats
    failures = sum(1 for entry in estimates if not entry.get('ok'))
    quality_failures = [entry['frame'] for entry in estimates if entry.get('quality_ok') is False]
    print(f'[report] frames: {len(estimates)}  est success: {len(estimates) - failures}  est failure: {failures}')
    print(f'[report] quality: pass {len(estimates) - len(quality_failures)}  fail {len(quality_failures)}'
          + (f'  (frames {quality_failures[:20]})' if quality_failures else ''))
    print(f'[report] low-confidence matches: {matcher.low_conf_count} / {matcher.match_count} '
          f'(details: {outdir}/Matched_diagnostics.csv, {outdir}/Affined_diagnostics.csv)')


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    video = _video_path(args.video)
    if not args.nogui:
        TemplateGUI(video, args.csv)
        return
    try:
        matcher, corrector = run_pipeline(video, args.csv, args.outdir)
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        parser.error(str(error))
    _print_report(matcher, corrector, args.outdir)
