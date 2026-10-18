"""Phase-guided search and ECC refinement experiments; not application modes."""
from pathlib import Path
import argparse
import json
import platform
import sys
import time

import cv2
import numpy as np
from search_candidates import core, scene, track, evaluate, window_match


def phase_shift(reference,current):
    first=cv2.pyrDown(reference[16:96,32:608]).astype(np.float32)
    second=cv2.pyrDown(current[16:96,32:608]).astype(np.float32)
    window=cv2.createHanningWindow((first.shape[1],first.shape[0]),cv2.CV_32F)
    displacement,response=cv2.phaseCorrelate(first,second,window)
    return np.array(displacement)*2,float(response)


def phase_track(frames,templates):
    started=time.perf_counter();positions=[];fallback_indices=[];fallback_seconds=0
    low_response=0
    matcher=core.TemplateMatcher.__new__(core.TemplateMatcher)
    for n,frame in enumerate(frames[1:],1):
        displacement,response=phase_shift(frames[0],frame)
        low_response+=int(response<.2)
        points=[]
        for i,t in enumerate(templates):
            prediction=np.array([t['cx'],t['cy']])+displacement
            mx,my,score,clear,edge,cc=window_match(frame,t,prediction,16)
            if response<.2 or score<core.MATCH_SCORE_MIN or clear<core.MATCH_CLEAR_MIN or edge or cc<core.MATCH_CCOEFF_ALARM:
                start=time.perf_counter();mx,my,_=matcher._match(frame,t)
                fallback_seconds+=time.perf_counter()-start;fallback_indices.append((n-1,i))
            points.append([mx,my])
        positions.append(points)
    return np.array(positions),dict(track_s=time.perf_counter()-started,
        fallback_indices=fallback_indices,fallback_s=fallback_seconds,phase_low_response_frames=low_response)


def refine_ecc(reference,current,initial,mask):
    warp=cv2.invertAffineTransform(initial).astype(np.float32)
    valid=cv2.warpAffine(np.full(reference.shape,255,np.uint8),warp,(current.shape[1],current.shape[0]))
    valid=np.where(valid==255,255,0).astype(np.uint8)
    valid=cv2.erode(valid,np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    mask=cv2.bitwise_and(mask,valid)
    warp[:,2]/=2
    try:
        correlation,warp=cv2.findTransformECC(cv2.pyrDown(reference),cv2.pyrDown(current),warp,
            cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-5),
            cv2.resize(mask,None,fx=.5,fy=.5,interpolation=cv2.INTER_NEAREST),5)
    except cv2.error:
        return initial.copy(),False
    warp[:,2]*=2
    refined=cv2.invertAffineTransform(warp)
    if not np.isfinite(refined).all() or abs(np.linalg.det(refined[:,:2]))<1e-10:
        return initial.copy(),False
    return refined,True


def ecc_trial(frames,templates,motion,baseline_positions,scenario):
    centres=np.array([[t['cx'],t['cy']] for t in templates],np.float32)
    grid=np.array([(x,y) for x in [64,192,448,576] for y in [64,180,296]],float)
    residuals=[];failures=0;started=time.perf_counter()
    for n,(frame,points,truth) in enumerate(zip(frames[1:],baseline_positions,motion[1:]),1):
        cv2.setRNGSeed(42)
        initial,_=cv2.estimateAffinePartial2D(points.astype(np.float32),centres,method=cv2.RANSAC,ransacReprojThreshold=2)
        if initial is None:
            failures+=1
            continue
        mask=np.full(frame.shape,255,np.uint8)
        # Oracle foreground mask gives this candidate an optimistic upper bound.
        if scenario == 'foreground':
            x=(n*9)%(frame.shape[1]-120)
            mask[100:261,x:x+121]=0
        refined,ok=refine_ecc(frames[0],frame,initial,mask)
        failures+=int(not ok)
        moved=(truth[:,:2]@grid.T+truth[:,2:3]).T
        residuals.extend(np.linalg.norm((refined[:,:2]@moved.T+refined[:,2:3]).T-grid,axis=1))
    return dict(refine_and_estimate_s=time.perf_counter()-started,
        fallback_or_initial_failures=failures,background_mean_px=float(np.mean(residuals)),
        background_p95_px=float(np.percentile(residuals,95)))


def refine_real_video(frames,corrector):
    started=time.perf_counter();failures=0
    entries={entry['frame']:entry for entry in corrector.estimate_stats}
    for n,frame in enumerate(frames[1:],2):
        initial=corrector._M(n)
        refined,ok=refine_ecc(frames[0],frame,initial,np.full(frame.shape,255,np.uint8))
        failures+=int(not ok)
        corrector._mcache[n]=refined
        entries[n]['M']=refined.tolist()
    return dict(ecc_fallbacks=failures,ecc_refinement_s=time.perf_counter()-started)


def real_trials():
    from lk_candidate import video_trial
    baseline=lambda f,t:track(f,t,'baseline')
    a,_=video_trial('baseline',baseline)
    b,_=video_trial('phase',phase_track)
    c,_=video_trial('ecc',baseline,refine_real_video)
    return [a,b,c]


def run(output):
    report=dict(platform=platform.platform(),python=sys.version,opencv=cv2.__version__,results=[])
    for scenario in ['translation','rotation','scale','illumination','foreground']:
        for seed in [42,43,44]:
            frames,templates,motion=scene(scenario,seed)
            baseline_positions,counters=track(frames,templates,'baseline')
            baseline=evaluate(baseline_positions,templates,motion,counters)
            positions,counters=phase_track(frames,templates)
            phase=evaluate(positions,templates,motion,counters)
            ecc=ecc_trial(frames,templates,motion,baseline_positions,scenario)
            ecc['track_and_refine_s']=baseline['track_s']+ecc['refine_and_estimate_s']
            report['results'].append(dict(scenario=scenario,seed=seed,baseline=baseline,phase=phase,ecc=ecc))
    report['real_video']=real_trials()
    with open(output,'x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
