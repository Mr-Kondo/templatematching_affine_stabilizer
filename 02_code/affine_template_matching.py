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
import gc
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Utils
# =============================================================================


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
    x0, y0 = cx - h, cy - h
    x1, y1 = cx + h, cy + h
    H, W = src.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


def crop_lrbt(src: np.ndarray, cx: int, cy: int, left: int, right: int, top: int,
              bottom: int) -> np.ndarray:
    """Crop a rectangular ROI using individual pixel margins measured from the centre.

    The ROI spans from (cx-left, cy-top) to (cx+right, cy+bottom).  If the window
    crosses the image borders, the missing areas are padded with zeros.
    """
    x0, y0 = cx - left, cy - top
    x1, y1 = cx + right, cy + bottom
    H, W = src.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


# =============================================================================
# Matcher
# =============================================================================


class TemplateMatcher:
    """Perform template matching on a video to track template locations over time."""

    def __init__(self, video: str, csv_path: str):
        self.cap = cv2.VideoCapture(video, apiPreference=cv2.CAP_FFMPEG)
        ok, first = self.cap.read()
        assert ok, 'cannot read video'
        # The first frame is used as the reference for template extraction
        self.first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        self.templates = self._load(csv_path)

    def _load(self, p: str) -> List[Dict[str, Any]]:
        """Load template definitions from a CSV file."""
        df = pd.read_csv(p)
        out: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            half_tpl = int(r['テンプレートの大きさ（正方形）']) // 2
            top = int(r['探索上'])
            bottom = int(r['探索下'])
            left = int(r['探索左'])
            right = int(r['探索右'])
            out.append(
                dict(
                    id=int(r['No.']),
                    cx=int(r['x座標']),
                    cy=int(r['y座標']),
                    th=half_tpl,
                    t=top,
                    b=bottom,
                    l=left,
                    r=right,
                    patch=crop(self.first, int(r['x座標']), int(r['y座標']), half_tpl),
                ))
        return out

    def _match(self, g: np.ndarray, t: Dict[str, Any]) -> Tuple[float, float]:
        """Match a single template t in the current grayscale frame g."""
        roi = crop_lrbt(g, t['cx'], t['cy'], t['l'], t['r'], t['t'], t['b'])
        res = cv2.matchTemplate(roi, t['patch'], cv2.TM_CCORR_NORMED)
        *_, loc = cv2.minMaxLoc(res)
        dx, dy = refine(res, (loc[0], loc[1]))
        mx = t['cx'] - t['l'] + loc[0] + dx + t['th']
        my = t['cy'] - t['t'] + loc[1] + dy + t['th']
        return mx, my

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
        rec: List[List[float]] = []
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
                    mx, my = self._match(g, t)
                    rec.append([idx, t['id'], t['cx'], t['cy'], mx, my])
                bar.update(1)
                if q:
                    prog += 1
                    q.put(('prog', prog))
        self.cap.release()
        # Ensure output directory exists
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rec, columns=['Frame_No', 'Template_No', 'x', 'y', 'mx',
                                   'my']).to_csv(out_path, index=False)
        del self.first
        gc.collect()


# =============================================================================
# Affine
# =============================================================================


class AffineCorrector:
    """Correct video frames using affine (and perspective) transformations based on matched points."""

    def __init__(self,
                 video: str,
                 csv: str,
                 outv: str = '03_output/Affined.mp4',
                 tmp: str = 'temp'):
        self.matches = pd.read_csv(csv)
        self.video = Path(video)
        self.out = Path(outv)
        self.tmp = Path(tmp)
        # Use a temporary directory inside the output folder for intermediate images
        self.img = self.tmp / 'images'
        self.conv = self.tmp / 'converted'
        self.img.mkdir(parents=True, exist_ok=True)
        self.conv.mkdir(parents=True, exist_ok=True)

    def split(self) -> Tuple[int, int, float]:
        """Split the input video into individual frames saved as PNGs in self.img."""
        c = cv2.VideoCapture(str(self.video), apiPreference=cv2.CAP_FFMPEG)
        fps = c.get(cv2.CAP_PROP_FPS)
        w, h = int(c.get(3)), int(c.get(4))
        idx = 1
        while True:
            ok, f = c.read()
            if not ok:
                break
            cv2.imwrite(str(self.img / f'frame_{idx:06d}.png'), f)
            idx += 1
        c.release()
        return w, h, fps

    def _M(self, n: int) -> np.ndarray:
        """Estimate the affine matrix for frame n using matched control points."""
        df = self.matches[self.matches.Frame_No == n]
        src = df[['mx', 'my']].to_numpy(np.float32)
        dst = df[['x', 'y']].to_numpy(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
        if M is None:
            raise RuntimeError('affine fail')
        return M

    def correct(self, q: queue.Queue | None = None) -> None:
        """Warp each frame using the estimated affine (plus optional perspective) transform and save the corrected video."""
        w, h, fps = self.split()
        frames = sorted(self.img.glob('frame_*.png'))
        if q:
            q.put(('setmax', len(frames) - 1))
            done = 0
        fourcc_fn = cv2.VideoWriter_fourcc if hasattr(
            cv2, 'VideoWriter_fourcc') else cv2.VideoWriter.fourcc
        # Ensure parent directory exists
        self.out.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(self.out), fourcc_fn(*'mp4v'), fps if fps > 0 else 30, (w, h))
        for fp in tqdm(frames[1:], desc='warp'):
            n = int(fp.stem.split('_')[1])
            img = cv2.imread(str(fp))
            # Apply the affine transformation
            aff = cv2.warpAffine(img,
                                 self._M(n), (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            # Apply an optional perspective warp (sample implementation)
            src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            dst_pts = np.float32([
                [10, 5],  # top-left
                [w - 11, 10],  # top-right
                [w - 6, h - 11],  # bottom-right
                [5, h - 6],  # bottom-left
            ])
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            persp = cv2.warpPerspective(aff, M_persp, (w, h))
            out.write(persp)
            cv2.imwrite(str(self.conv / f'conv_{n:06d}.png'), persp)
            if q:
                done += 1
                q.put(('prog', done))
        out.release()

    def cleanup(self) -> None:
        """Remove the temporary directory used for intermediate frames."""
        shutil.rmtree(self.tmp, ignore_errors=True)


# =============================================================================
# GUI
# =============================================================================


class TemplateGUI:
    """Interactive GUI for selecting templates and running the stabilisation pipeline."""

    def __init__(self, video: str, csv_p: str = '01_input/Input.csv'):
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

        from PIL import Image, ImageTk

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
        df = pd.read_csv(self.csv)
        self.tpls.clear()
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self._add_tpl(
                int(r['x座標']),
                int(r['y座標']),
                int(r['テンプレートの大きさ（正方形）']),
                int(r['探索左']),
                int(r['探索右']),
                int(r['探索上']),
                int(r['探索下']),
                existing=True,
                id=int(r['No.']),
            )

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
            id = len(self.tpls) + 1
        assert id is not None
        d = dict(id=id, cx=cx, cy=cy, tpl=tpl, l=left, r=right, t=top, b=bottom)
        self.tpls.append(d)
        self.tree.insert('',
                         'end',
                         iid=d['id'],
                         values=(d['id'], cx, cy, tpl, left, right, top, bottom))

    # -------- UI actions --------
    def _show(self, idx: int) -> None:
        from PIL import Image, ImageTk

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
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

    def _seek(self, val: float) -> None:
        self._show(int(float(val)))

    def _click(self, e) -> None:
        # Parse numeric values from entries
        tpl = int(self.ent_tpl.get()) + int(self.ent_tpl.get()) % 2
        left = int(self.ent_left.get())
        right = int(self.ent_right.get())
        top = int(self.ent_top.get())
        bottom = int(self.ent_bottom.get())
        # Map canvas (zoomed) coords back to original image
        orig_x = int(e.x / self.zoom)
        orig_y = int(e.y / self.zoom)
        self._add_tpl(orig_x, orig_y, tpl, left, right, top, bottom)
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
        for key in ('cx', 'cy', 'tpl', 'l', 'r', 't', 'b'):
            val = self.simpledialog.askinteger('Edit',
                                               f'{key} value',
                                               initialvalue=row[key],
                                               parent=self.root,
                                               minvalue=0)
            if val is None:
                return
            row[key] = val
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
        if len(self.tpls) < 3:
            self.messagebox.showwarning('warn', 'テンプレートは3点以上必要')
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

        def task() -> None:
            try:
                # Run template matching; save matches into the output folder
                TemplateMatcher(self.video, str(self.csv)).run('03_output/Matched.csv', q)
                # Use the matched points to correct the video
                AffineCorrector(self.video, '03_output/Matched.csv',
                                '03_output/Affined.mp4').correct(q)
                # Clean up temporary files
                AffineCorrector(self.video, '03_output/Matched.csv',
                                '03_output/Affined.mp4').cleanup()
                q.put(('done', None))
            except Exception as e:
                q.put(('err', e))

        threading.Thread(target=task, daemon=True).start()
        self._poll(q)

    def _poll(self, q: queue.Queue[Tuple[str, Any]]) -> None:
        try:
            typ, *rest = q.get_nowait()
            if typ == 'setmax':
                self.pb['maximum'] = rest[0]
                self.pb['value'] = 0
            elif typ == 'prog':
                self.pb['value'] = rest[0]
            elif typ == 'done':
                self.btn_delete.config(state='normal')
                self.btn_undo.config(state='normal')
                self.btn_execute.config(state='normal')
                self.messagebox.showinfo('完了', '補正完了')
                self.root.destroy()
                return
            elif typ == 'err':
                self.btn_delete.config(state='normal')
                self.btn_undo.config(state='normal')
                self.btn_execute.config(state='normal')
                self.messagebox.showerror('Error', str(rest[0]))
                self.root.destroy()
                return
        except queue.Empty:
            pass
        self.root.after(100, self._poll, q)


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
        # Batch mode: run matching and correction directly
        TemplateMatcher(a.video, a.csv).run('03_output/Matched.csv')
        AffineCorrector(a.video, '03_output/Matched.csv', '03_output/Affined.mp4').correct()
        AffineCorrector(a.video, '03_output/Matched.csv', '03_output/Affined.mp4').cleanup()
    else:
        # Interactive GUI
        TemplateGUI(a.video, a.csv)


if __name__ == '__main__':
    main()
