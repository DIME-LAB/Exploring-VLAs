#!/usr/bin/env python3
"""Replay retained clean/legacy pilot commands for offline smooth-DART metrics."""
import argparse,json,math
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from dart import DEFAULT_NOISE,SmoothDart
from storage import confined
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control'))
from so_arm101_control.grasp_geometry import JOINT_LIMITS
from contract import JOINTS,corrective_action


def metrics(values):
    v=np.diff(values[:,:5],axis=0)*30;a=np.diff(v,axis=0)*30;j=np.diff(a,axis=0)*30
    out={}
    for name,data in [('velocity',v),('acceleration',a),('jerk',j)]:
        out[name]=dict(rms=float(np.sqrt(np.mean(data**2))),p95=float(np.quantile(np.abs(data),.95)),maximum=float(np.max(np.abs(data))))
    return out


def replay(rows,seed):
    out=[];baseline=[];offsets=[];index=0;operation=0;specs=[]
    while index<len(rows):
        phase=rows[index]['phase'];end=index+1
        while end<len(rows) and rows[end]['phase']==phase:end+=1
        block=rows[index:end];duration=len(block)/30
        program=SmoothDart(duration,seed+104729*operation,DEFAULT_NOISE,
            enabled=phase not in ('gripper','home','grasp_home','grasp_home_data','hold'))
        specs.append(dict(phase=phase,**program.spec))
        for tick,row in enumerate(block):
            reference=list(row['expert_reference']);noise=program.sample(tick/30)
            out.append([q+n for q,n in zip(reference,noise)]);baseline.append(reference);offsets.append(noise)
        index=end;operation+=1
    return np.array(out),np.array(baseline),np.array(offsets),specs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    source=confined(args.input);output=confined(args.output);output.mkdir(parents=True,exist_ok=True)
    receipts=[json.loads(line) for line in (source/'episodes.jsonl').read_text().splitlines()]
    combined={name:[] for name in ('expert_labels','legacy','planner','smooth')};results=[];example=None
    for r in receipts:
        if not r['accepted']:continue
        trial=r['params']['trial'];rows=[json.loads(line) for line in (source/f'attempt-{trial:04d}'/'frames.jsonl').read_text().splitlines()]
        smooth,planner,noise,spec=replay(rows,r['params']['noise_seed'])
        arrays=dict(expert_labels=np.array([row['action'] for row in rows]),legacy=np.array([row['executed_action'] for row in rows]),planner=planner,smooth=smooth)
        results.append(dict(trial=trial,seed=r['params']['seed'],color=r['params']['color'],frames=len(rows),
            metrics={k:metrics(v) for k,v in arrays.items()},noise_peak_rad=float(np.max(np.abs(noise))),
            replay_frames=len(smooth),planner_offset_max_rad=float(np.max(np.abs(smooth-planner))),
            planner_offset_p95_rad=float(np.quantile(np.abs(smooth-planner),.95)),noise_on_fraction=float(np.mean(np.max(np.abs(noise),axis=1)>1e-10)),
            noise_metrics=metrics(noise),noise_operations=spec))
        for name,array in arrays.items():combined[name].append(array)
        if example is None:example=arrays
    summary={name:{quantity:{stat:float(np.mean([r['metrics'][name][quantity][stat] for r in results])) for stat in ('rms','p95','maximum')} for quantity in ('velocity','acceleration','jerk')} for name in combined}
    report=dict(scope='Offline planner-reference plus smooth offset replay at original timestamps; phase-block seeds, not exact historical operation identities; not physical yield or collision proof',
        episodes=len(results),fps=30,per_episode=results,mean_per_episode_metrics=summary,
        maximum_noise_acceleration=max(r['noise_metrics']['acceleration']['maximum'] for r in results),
        maximum_noise_velocity=max(r['noise_metrics']['velocity']['maximum'] for r in results),
        maximum_noise_offset_rad=max(r['planner_offset_max_rad'] for r in results))
    assert report['maximum_noise_acceleration']<=.30001 and report['maximum_noise_velocity']<=.050001
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    colors=dict(expert_labels='#3b82f6',legacy='#ef4444',planner='#9333ea',smooth='#16a34a')
    for name,array in example.items():axes[0].plot(np.arange(len(array))/30,array[:,1],label=name,color=colors[name],lw=1.)
    axes[0].set(xlabel='sim seconds',ylabel='shoulder target (rad)',title='Recorded pilot commands and offline smooth replay');axes[0].legend()
    for axis,quantity in zip(axes[1:],('acceleration','jerk')):
        axis.bar(list(summary),[summary[name][quantity]['rms'] for name in summary],color=[colors[name] for name in summary])
        axis.set(ylabel=f'{quantity} RMS (rad/s^'+('2)' if quantity=='acceleration' else '3)'),title=f'Mean per-episode {quantity} RMS, arm joints only')
    fig.savefig(output/'command-metrics.png',dpi=160)
    print(json.dumps({k:v for k,v in report.items() if k!='per_episode'},indent=2))
if __name__=='__main__':main()
