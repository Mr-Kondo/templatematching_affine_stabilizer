"""Tk template editor and main-thread progress updates."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import queue
import threading
from typing import Any, Dict, List, Tuple

import cv2

from stabilizer_core import (TEMPLATE_FIELDS, load_templates, run_pipeline, validate_template, validate_templates)

def _pil_to_tk_photo(pil_img, tk_module):
    """Convert PIL Image to Tkinter PhotoImage without using ImageTk.

    This is a workaround for compatibility issues between Pillow's _imagingtk
    module and certain Tcl/Tk installations (e.g., uv-managed Python on macOS).
    """
    import io

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PPM')
    return tk_module.PhotoImage(data=buffer.getvalue())


class TemplateGUI:
    """Interactive GUI for selecting templates and running the stabilisation pipeline."""

    # clean-code exception (function size): session lifetime must release capture after Tk exits; revisit when adding cancellation.
    def __init__(self, video: str, csv_p: str = '01_input/Input.csv'):
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

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

    # clean-code exception (function size): explicit Tk layout keeps binding order visible; split when a new panel is added.
    def _build_widgets(self) -> None:
        ttk = self.ttk
        ctl = ttk.Frame(self.root)
        ctl.pack(fill='x')
        ttk.Label(ctl, text='Template').pack(side='left')
        self.ent_tpl = ttk.Entry(ctl, width=6)
        self.ent_tpl.insert(0, '64')
        self.ent_tpl.pack(side='left')
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
        self.pb = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate')
        self.pb.pack(fill='x', padx=4, pady=4)
        self.scale = ttk.Scale(self.root,
                               from_=1,
                               to=self.total,
                               orient='horizontal',
                               command=self._seek)
        self.scale.pack(fill='x')
        self.canvas = self.tk.Canvas(self.root)
        self.canvas.pack()
        self.status = self.ttk.Label(self.root, text='x: -, y: -')
        self.status.pack(fill='x', pady=2)
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Motion>', self._on_motion)
        self._photo = None

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

    # clean-code exception (four-argument ceiling): legacy GUI argument order is retained; replace with a row object if fields change.
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

    # clean-code exception (function size): one render transaction owns the image reference; split when another view is introduced.
    def _show(self, idx: int) -> None:
        from PIL import Image

        ResamplingFilter = Image.Resampling.LANCZOS
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # clean-code exception (function size): one edit transaction keeps cancellation atomic; split when the editor becomes non-modal.
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

    def _exec(self) -> None:
        try:
            self.tpls = validate_templates(self.tpls, self.frame_shape)
        except ValueError as error:
            self.messagebox.showerror('入力エラー', str(error))
            return
        self._save_templates()
        self.btn_delete.config(state='disabled')
        self.btn_undo.config(state='disabled')
        self.btn_execute.config(state='disabled')
        q: queue.Queue[Tuple[str, Any]] = queue.Queue()

        self.outdir = Path('03_output') / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')

        threading.Thread(target=self._run_task, args=(q,), daemon=True).start()
        self._poll(q)

    def _save_templates(self):
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(TEMPLATE_FIELDS)
            for template in self.tpls:
                writer.writerow([template[key] for key in TEMPLATE_FIELDS.values()])

    def _run_task(self, q):
        try:
            run_pipeline(self.video, str(self.csv), self.outdir, q)
            q.put(('done', None))
        except Exception as error:
            # Transport every worker failure to Tk's main thread; none is treated as success.
            q.put(('err', error))

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


