#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic test video with known affine motion (benchmark ground truth).

Generates a textured base scene, applies a per-frame affine motion
(translation + rotation + scale composed about the image centre) and writes
the encoded video together with template definitions and background
evaluation points.

Ground-truth convention
-----------------------
Video frames are numbered k = 1..K (1-based) and frame 1 is the reference.
A base-scene point ``p`` appears in frame ``k`` at ``A[k-1] @ p`` where
``A[0]`` is the identity.  The stabiliser is expected to recover
``M_k ~= A[k-1]^-1`` (frame-k coordinates -> reference coordinates), so the
stabilisation residual of a reference point ``p`` is
``M_k @ A[k-1] @ p - p``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def build_base_scene(w: int, h: int, seed: int) -> np.ndarray:
    """Build a texture-rich base scene (BGR) used as the reference frame."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 127, np.uint8)
    n_shapes = max(120, (w * h) // 6000)
    for _ in range(n_shapes):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(8, max(9, w // 20)))
        color = tuple(int(v) for v in rng.integers(30, 225, 3))
        if rng.random() < 0.7:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        else:
            rw, rh = r, max(4, int(r * rng.uniform(0.3, 1.7)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(img, (cx, cy), (rw, rh), angle, 0, 360, color, -1, cv2.LINE_AA)
    return img


def motion_matrix(n: int, cx: float, cy: float, vx: float, vy: float,
                  rot_deg: float, scale: float) -> np.ndarray:
    """2x3 affine matrix sending base points to their position in frame n+1.

    ``n`` is the 0-based motion index (frame k uses n = k-1).  The motion is
    a rotation by ``rot_deg * n`` degrees and a scale of ``scale ** n``
    composed about ``(cx, cy)``, plus a translation of ``(vx, vy) * n``.
    """
    theta = math.radians(rot_deg) * n
    s = scale ** n
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]], float) * s
    t = np.array([cx, cy]) - R @ np.array([cx, cy])
    t = t + np.array([vx, vy]) * n
    return np.hstack([R, t.reshape(2, 1)]).astype(np.float64)


def candidate_points(rng: np.random.Generator, w: int, h: int, count: int,
                     keep_out: List[Tuple[float, float]], avoid_dist: float,
                     spacing: float, margin: int) -> List[Tuple[int, int]]:
    """Random points with a minimum spacing and a clearance to keep_out."""
    pts: List[Tuple[int, int]] = []
    tries = 0
    while len(pts) < count and tries < count * 500:
        tries += 1
        x = int(rng.integers(margin, w - margin))
        y = int(rng.integers(margin, h - margin))
        too_close = any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 < spacing
                        for px, py in pts)
        too_close |= any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 < avoid_dist
                         for px, py in keep_out)
        if not too_close:
            pts.append((x, y))
    if len(pts) < count:
        raise RuntimeError(
            f'candidate_points: only placed {len(pts)}/{count} points; '
            'parameters too tight for image size')
    return pts


def generate(video_path: Path, csv_path: Path, meta_path: Path, *,
             width: int = 1280, height: int = 720, frames: int = 61,
             vx: float = 2.0, vy: float = 1.0, rot_deg: float = 0.2,
             scale: float = 1.002, seed: int = 42, tpl_size: int = 64,
             n_templates: int = 6,
             fps: float = 30.0) -> Dict[str, Any]:
    """Generate the synthetic benchmark video plus template CSV and metadata.

    Returns the ground-truth metadata (motion matrices, template rows and
    background evaluation points) that the harness stores next to its report.
    """
    rng = np.random.default_rng(seed)
    base = build_base_scene(width, height, seed)
    ccx, ccy = (width - 1) / 2.0, (height - 1) / 2.0

    A = [np.hstack([np.eye(2), np.zeros((2, 1))])]
    for n in range(1, frames):
        A.append(motion_matrix(n, ccx, ccy, vx, vy, rot_deg, scale))

    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc if hasattr(
        cv2, 'VideoWriter_fourcc') else cv2.VideoWriter.fourcc
    vw = cv2.VideoWriter(str(video_path), fourcc(*'mp4v'), fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f'cannot open video writer: {video_path}')
    for k in range(1, frames + 1):  # frame k uses A[k-1]
        frame = cv2.warpAffine(base, A[k - 1], (width, height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        vw.write(frame)
    vw.release()

    # Largest displacement over the clip, for the report only.
    grid = np.stack(np.meshgrid(np.arange(0, width, 32), np.arange(0, height, 32)),
                    -1).reshape(-1, 2).astype(np.float64)
    max_disp = max(
        float(np.max(np.linalg.norm(
            (M[:, :2] @ grid.T + M[:, 2:3]).T - grid, axis=1)))
        for M in A)

    # Per-template search margins derived from that template's own motion, so
    # every true match stays inside its ROI without inflating the global ROI.
    th = tpl_size // 2

    def margins_for(x: float, y: float) -> Tuple[int, int, int, int]:
        d = np.array([(M[0, 0] * x + M[0, 1] * y + M[0, 2] - x,
                       M[1, 0] * x + M[1, 1] * y + M[1, 2] - y) for M in A])
        l = int(math.ceil(max(0.0, -float(d[:, 0].min())) + th))
        r = int(math.ceil(max(0.0, float(d[:, 0].max())) + th))
        t = int(math.ceil(max(0.0, -float(d[:, 1].min())) + th))
        b = int(math.ceil(max(0.0, float(d[:, 1].max())) + th))
        return l, r, t, b

    centre_margin_cap = 240
    centres: List[Tuple[int, int, int, int, int, int]] = []
    tries = 0
    while len(centres) < n_templates and tries < n_templates * 2000:
        tries += 1
        x = int(rng.integers(th + 24, width - th - 24))
        y = int(rng.integers(th + 24, height - th - 24))
        if any((x - px) ** 2 + (y - py) ** 2 < (tpl_size * 2) ** 2
               for px, py, *_ in centres):
            continue
        l, r, t, b = margins_for(x, y)
        if max(l, r, t, b) > centre_margin_cap:
            continue
        centres.append((x, y, l, r, t, b))
    if len(centres) < n_templates:
        raise RuntimeError(
            f'template placement: only {len(centres)}/{n_templates} placed; '
            'reduce motion magnitude or templates count')

    rows = [dict(No=i, x=x, y=y, size=tpl_size, left=l, right=r, top=t,
                 bottom=b)
            for i, (x, y, l, r, t, b) in enumerate(centres, 1)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write('No.,x座標,y座標,テンプレートの大きさ（正方形）,探索左,探索右,探索上,探索下\n')
        for r_ in rows:
            f.write(f"{r_['No']},{r_['x']},{r_['y']},{r_['size']},"
                    f"{r_['left']},{r_['right']},{r_['top']},{r_['bottom']}\n")

    bg_pts = candidate_points(rng, width, height, 40,
                              [(x, y) for x, y, *_ in centres],
                              avoid_dist=float(tpl_size) * 0.75,
                              spacing=48.0, margin=th + 8)

    meta = dict(kind='synthetic',
                video=str(video_path), frame_count=frames,
                width=width, height=height, fps=fps, seed=seed,
                motion=dict(vx=vx, vy=vy, rot_deg=rot_deg, scale=scale),
                max_displacement_px=max_disp,
                tpl_margin_cap_px=centre_margin_cap,
                A=[m.tolist() for m in A],
                templates=rows,
                bg_points=[[int(x), int(y)] for x, y in bg_pts])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    return meta
