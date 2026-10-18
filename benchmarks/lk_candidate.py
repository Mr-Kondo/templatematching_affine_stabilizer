"""LK with initial-template reanchoring; research only."""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import tempfile
import time

import cv2
import numpy as np

from search_candidates import core, scene, track, evaluate


def initial_features(templates):
    offsets=[]
    valid=[]
    for template in templates:
        corner=cv2.goodFeaturesToTrack(template['patch'],maxCorners=1,qualityLevel=.01,minDistance=3)
        offsets.append(corner.reshape(2)-template['th'] if corner is not None else [0,0])
        valid.append(corner is not None)
    return np.asarray(offsets,float),np.asarray(valid)


def lk_track(frames, templates):
    started=time.perf_counter()
    offsets,valid_features=initial_features(templates)
    centres=np.array([[t['cx'],t['cy']] for t in templates],float)
    previous=frames[0]
    matcher=core.TemplateMatcher.__new__(core.TemplateMatcher)
    positions=[]
    rejected=shortages=0
    fallback_indices=[]
    fallback_seconds=0
    for n,frame in enumerate(frames[1:],1):
        source=(centres+offsets).astype(np.float32).reshape(-1,1,2)
        forward,status,error=cv2.calcOpticalFlowPyrLK(previous,frame,source,None,winSize=(31,31),maxLevel=3)
        good=np.zeros(len(templates),bool)
        if forward is not None:
            backward,back_status,_=cv2.calcOpticalFlowPyrLK(frame,previous,forward,None,winSize=(31,31),maxLevel=3)
            if backward is not None:
                fb=np.linalg.norm(backward-source,axis=2).ravel()
                good=(status.ravel()==1)&(back_status.ravel()==1)&valid_features&(fb<=1)&(error.ravel()<=20)
                good &= np.isfinite(forward).all(axis=(1,2))
        rejected+=int((~good).sum())
        shortages+=int(good.sum()<2)
        points=centres.copy() if forward is None else forward.reshape(-1,2)-offsets
        for i,template in enumerate(templates):
            if not good[i] or n%10==0:
                start=time.perf_counter()
                mx,my,_=matcher._match(frame,template)
                fallback_seconds+=time.perf_counter()-start
                points[i]=[mx,my]
                fallback_indices.append((n-1,i))
        positions.append(points.copy())
        centres=points
        previous=frame
    return np.array(positions),dict(track_s=time.perf_counter()-started,
        fallback_indices=fallback_indices,fallback_s=fallback_seconds,
        lk_rejected_points=rejected,lk_point_shortage_frames=shortages)


def video_trial(method):
    started=time.perf_counter()
    with core.open_video('01_input/input.mp4') as cap:
        frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    definitions=core.load_templates('01_input/Input.csv',frames[0].shape)
    templates=[dict(id=t['id'],cx=t['cx'],cy=t['cy'],th=t['tpl']//2,l=t['l'],r=t['r'],t=t['t'],b=t['b'],
                    patch=core.crop(frames[0],t['cx'],t['cy'],t['tpl']//2)) for t in definitions]
    positions,counters=track(frames,templates,'baseline') if method=='baseline' else lk_track(frames,templates)
    with tempfile.TemporaryDirectory(prefix='affine-lk-trial-') as temporary:
        folder=Path(temporary)
        with open(folder/'Matched.csv','w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['Frame_No','Template_No','x','y','mx','my'])
            for n,points in enumerate(positions,2):
                for t,point in zip(templates,points):writer.writerow([n,t['id'],t['cx'],t['cy'],*point])
        corrector=core.AffineCorrector('01_input/input.mp4',str(folder/'Matched.csv'),str(folder/'out.mp4'))
        corrector.precompute();corrector.correct()
        total=time.perf_counter()-started
        from run_benchmark import eval_video, bg_residuals_lk
        background=bg_residuals_lk(Path('01_input/input.mp4'),definitions,corrector.estimate_stats)
        quality=eval_video(folder,corrector.estimate_stats,background)
    counters.pop('fallback_indices')
    return dict(method=method,decoded_frames=len(frames),whole_pipeline_s=total,**counters,accuracy=quality),positions


def run(output):
    report=dict(platform=platform.platform(),python=sys.version,opencv=cv2.__version__,results=[])
    for scenario,count in [('normal',601),('large',61),('occlusion',61),('low_texture',61)]:
        for seed in [42,43,44]:
            frames,templates,motion=scene(scenario,seed,count)
            for method in ['baseline','lk']:
                positions,counters=track(frames,templates,'baseline') if method=='baseline' else lk_track(frames,templates)
                report['results'].append(dict(scenario=scenario,frames=count,seed=seed,method=method,
                    **evaluate(positions,templates,motion,counters)))
    a,pa=video_trial('baseline');b,pb=video_trial('lk')
    delta=np.linalg.norm(pa-pb,axis=2)
    report['real_video']=[a,b]
    report['real_coordinate_difference']={'mean_px':float(delta.mean()),'max_px':float(delta.max())}
    with open(output,'x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
