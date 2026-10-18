"""Research prototypes only; no application method is changed.
Run: uv run python benchmarks/search_candidates.py --output <new.json>
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '02_code'))
import stabilizer_core as core
from synth import build_base_scene


def scene(name, seed, count=61):
    width, height = 640, 360
    base = cv2.cvtColor(build_base_scene(width, height, seed), cv2.COLOR_BGR2GRAY)
    centres = [(128, 96), (512, 96), (128, 264), (512, 264), (320, 180)]
    if name == 'screen_edge':
        centres = [(4, 4), (636, 4), (4, 356), (636, 356), (320, 180)]
    if name == 'low_texture':
        base[:] = 127
    templates = [dict(id=i+1, cx=x, cy=y, th=16, l=96, r=96, t=96, b=96,
                      patch=core.crop(base, x, y, 16)) for i, (x, y) in enumerate(centres)]
    frames, motion = [], []
    for n in range(count):
        dx, dy = 12*np.sin(n/10), 8*np.sin(n/15)
        if name == 'jump' and n >= 25:
            dx += 55
            dy -= 30
        if name == 'search_edge':
            dx = 88*np.sin(n/15)
        if name == 'large':
            dx, dy = 70*np.sin(n/8), 40*np.sin(n/10)
        angle = 2*np.sin(n/12) if name in ('rotation', 'normal') else 0
        scale = 1 + .02*np.sin(n/15) if name in ('scale', 'normal') else 1
        matrix = cv2.getRotationMatrix2D((320, 180), angle, scale)
        matrix[:, 2] += [dx, dy]
        frame = cv2.warpAffine(base, matrix, (width, height))
        if name == 'occlusion' and 20 <= n <= 35:
            x,y = matrix @ np.array([*centres[0], 1])
            cv2.rectangle(frame, (int(x)-24,int(y)-24), (int(x)+24,int(y)+24), 180, -1)
        if name == 'illumination':
            frame = np.clip(frame.astype(float)*(1+.25*np.sin(n/15))+12*np.sin(n/10),0,255).astype(np.uint8)
        if name == 'foreground' and n > 0:
            x = (n*9) % (width-120)
            cv2.rectangle(frame,(x,100),(x+120,260),220,-1)
        frames.append(frame)
        motion.append(matrix)
    return frames, templates, motion


def window_match(frame, template, prediction, radius):
    half = template['th']
    cx = int(round(np.clip(prediction[0], template['cx']-template['l']+half,
                           template['cx']+template['r']-half)))
    cy = int(round(np.clip(prediction[1], template['cy']-template['t']+half,
                           template['cy']+template['b']-half)))
    shifted = dict(template, cx=cx, cy=cy)
    left = min(half+radius, cx-template['cx']+template['l'])
    right = min(half+radius, template['cx']+template['r']-cx)
    top = min(half+radius, cy-template['cy']+template['t'])
    bottom = min(half+radius, template['cy']+template['b']-cy)
    matcher = core.TemplateMatcher.__new__(core.TemplateMatcher)
    return matcher._match_window(frame, shifted, left, right, top, bottom)


def predicted_match(frame, template, prediction):
    return window_match(frame, template, prediction, 16)


def pyramid_match(frame, template, prediction):
    roi = core.crop_lrbt(frame, template['cx'], template['cy'], template['l'],
                         template['r'], template['t'], template['b'])
    coarse = cv2.matchTemplate(cv2.pyrDown(roi), cv2.pyrDown(template['patch']), cv2.TM_CCORR_NORMED)
    _, _, _, peak = cv2.minMaxLoc(coarse)
    centre = (template['cx']-template['l']+peak[0]*2+template['th'],
              template['cy']-template['t']+peak[1]*2+template['th'])
    return window_match(frame, template, centre, 4)


def track(frames, templates, method):
    matcher = core.TemplateMatcher.__new__(core.TemplateMatcher)
    previous = np.array([[t['cx'],t['cy']] for t in templates],float)
    velocity = np.zeros_like(previous)
    positions = []
    fallback_indices = []
    fallback_seconds = 0
    started = time.perf_counter()
    for n,frame in enumerate(frames[1:],1):
        points=[]
        for i,template in enumerate(templates):
            if method == 'baseline':
                mx,my,_ = matcher._match(frame,template)
            else:
                match = predicted_match if method == 'prediction' else pyramid_match
                mx,my,score,clear,edge,cc = match(frame,template,previous[i]+velocity[i])
                if score < core.MATCH_SCORE_MIN or clear < core.MATCH_CLEAR_MIN or edge or cc < core.MATCH_CCOEFF_ALARM:
                    start=time.perf_counter()
                    mx,my,_ = matcher._match(frame,template)
                    fallback_seconds += time.perf_counter()-start
                    fallback_indices.append((n-1,i))
            points.append([mx,my])
        points=np.array(points)
        velocity = points-previous
        previous=points
        positions.append(points)
    return np.array(positions),dict(track_s=time.perf_counter()-started,
                                   fallback_indices=fallback_indices, fallback_s=fallback_seconds)


def evaluate(positions, templates, motion, counters):
    reference=np.array([[t['cx'],t['cy']] for t in templates],float)
    truth=np.array([(matrix[:,:2]@reference.T+matrix[:,2:3]).T for matrix in motion[1:]])
    errors=np.linalg.norm(positions-truth,axis=2)
    grid=np.array([(x,y) for x in [64,192,448,576] for y in [64,180,296]
                   if min(np.linalg.norm(np.array([x,y])-p) for p in reference)>40],float)
    residuals=[]
    failed=0
    started=time.perf_counter()
    for points,matrix,frame_index in zip(positions,motion[1:],range(1,len(motion))):
        cv2.setRNGSeed(42)
        estimated,_=cv2.estimateAffinePartial2D(points.astype(np.float32),reference.astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=2)
        if estimated is None:
            failed+=1
            continue
        moved=(matrix[:,:2]@grid.T+matrix[:,2:3]).T
        residuals.extend(np.linalg.norm((estimated[:,:2]@moved.T+estimated[:,2:3]).T-grid,axis=1))
    estimate_seconds=time.perf_counter()-started
    fallbacks=counters.pop('fallback_indices')
    fallback_correct=sum(errors[n,i]<=2 for n,i in fallbacks)
    return dict(**counters, estimate_s=estimate_seconds,
                track_and_estimate_s=counters['track_s']+estimate_seconds,
                mean_error_px=float(errors.mean()),p95_error_px=float(np.percentile(errors,95)),
                max_error_px=float(errors.max()),gross_errors_gt2=int((errors>2).sum()),
                fallback_attempts=len(fallbacks),fallback_correct=int(fallback_correct),
                background_mean_px=float(np.mean(residuals)) if residuals else None,
                background_p95_px=float(np.percentile(residuals,95)) if residuals else None,
                estimation_failures=failed,
                recovery_failures_frames36_45=int((errors[35:45]>2).sum()))


def run(output):
    report=dict(platform=platform.platform(),python=sys.version,opencv=cv2.__version__,
                seeds=[42,43,44],frames=61,templates=5,results=[])
    for name in ['normal','jump','search_edge','screen_edge','occlusion']:
        for seed in report['seeds']:
            frames,templates,motion=scene(name,seed)
            for method in ['baseline','prediction','pyramid']:
                positions,counters=track(frames,templates,method)
                report['results'].append(dict(scenario=name,seed=seed,method=method,
                    **evaluate(positions,templates,motion,counters)))
    with open(output,'x') as destination:
        json.dump(report,destination,indent=2)
        destination.write('\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
