#!/usr/bin/env python3
"""Replay retained clean/legacy pilot commands for offline smooth-DART metrics."""
import argparse,json,math
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from dart import DEFAULT_NOISE,SmoothDart,TargetLimiter,ExpertSchedule
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
    limiter=TargetLimiter([JOINT_LIMITS[n] for n in JOINTS[:5]])
    out=[];clean_out=[];offsets=[];deltas=[];index=0;specs=[]
    while index<len(rows):
        phase=rows[index]['phase'];end=index+1
        while end<len(rows) and rows[end]['phase']==phase:end+=1
        block=rows[index:end];duration=len(block)/30
        points=[]
        for i,row in enumerate(block):
            stamp=round(i/30*1e9)
            points.append(NS(positions=row['expert_reference'][:5],time_from_start=NS(sec=stamp//10**9,nanosec=stamp%10**9)))
        stamp=round(duration*1e9)
        points.append(NS(positions=block[-1]['expert_reference'][:5],time_from_start=NS(sec=stamp//10**9,nanosec=stamp%10**9)))
        schedule=ExpertSchedule(NS(points=points),DEFAULT_NOISE)
        program=SmoothDart(schedule.duration,seed+index,DEFAULT_NOISE,
            enabled=phase not in ('gripper','home','grasp_home','grasp_home_data','hold'))
        specs.append(dict(phase=phase,expert_schedule=schedule.spec,**program.spec))
        for tick in range(math.ceil(schedule.duration*30)):
            t=min(schedule.duration,(tick+1)/30)
            u=schedule.source_time(t)/schedule.source_duration
            location=min(len(block)-1,u*len(block));left=int(location);right=min(len(block)-1,left+1);alpha=location-left
            state=[a+(b-a)*alpha for a,b in zip(block[left]['state'],block[right]['state'])]
            reference=schedule.sample(t)+[block[left]['expert_reference'][5]]
            clean=corrective_action(state,reference);noise=program.sample(t)
            stamp=round(len(out)/30*1e9)
            candidate=limiter.propose(clean,noise,rows[0]['physics_drive_targets'],1,stamp)
            limiter.commit(candidate,1,stamp);out.append(candidate.positions);clean_out.append(clean)
            offsets.append(noise);deltas.append(candidate.filter_delta)
        index=end
    return np.array(out),np.array(clean_out),np.array(offsets),np.array(deltas),specs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    source=confined(args.input);output=confined(args.output);output.mkdir(parents=True,exist_ok=True)
    receipts=[json.loads(line) for line in (source/'episodes.jsonl').read_text().splitlines()]
    combined={name:[] for name in ('original_none','legacy','retimed_none','smooth')};results=[];example=None
    for r in receipts:
        if not r['accepted']:continue
        trial=r['params']['trial'];rows=[json.loads(line) for line in (source/f'attempt-{trial:04d}'/'frames.jsonl').read_text().splitlines()]
        smooth,retimed_clean,noise,delta,spec=replay(rows,r['params']['noise_seed'])
        arrays=dict(original_none=np.array([row['action'] for row in rows]),legacy=np.array([row['executed_action'] for row in rows]),retimed_none=retimed_clean,smooth=smooth)
        results.append(dict(trial=trial,seed=r['params']['seed'],color=r['params']['color'],frames=len(rows),
            metrics={k:metrics(v) for k,v in arrays.items()},noise_peak_rad=float(np.max(np.abs(noise))),
            retimed_frames=len(smooth),executed_expert_deviation_max_rad=float(np.max(np.abs(smooth-retimed_clean))),
            executed_expert_deviation_p95_rad=float(np.quantile(np.abs(smooth-retimed_clean),.95)),noise_on_fraction=float(np.mean(np.max(np.abs(noise),axis=1)>1e-10)),
            filter_delta_peak_rad=float(np.max(np.abs(delta))),
            filter_active_fraction=float(np.mean(np.max(np.abs(delta),axis=1)>1e-8)),noise_operations=spec))
        for name,array in arrays.items():combined[name].append(array)
        if example is None:example=arrays
    summary={name:{quantity:{stat:float(np.mean([r['metrics'][name][quantity][stat] for r in results])) for stat in ('rms','p95','maximum')} for quantity in ('velocity','acceleration','jerk')} for name in combined}
    report=dict(scope='Offline phase-level minimum-jerk retiming and interpolated observation replay; not a physical yield/recovery test',
        episodes=len(results),fps=30,per_episode=results,mean_per_episode_metrics=summary,
        maximum_smooth_acceleration=max(r['metrics']['smooth']['acceleration']['maximum'] for r in results),
        maximum_filter_delta_rad=max(r['filter_delta_peak_rad'] for r in results))
    assert report['maximum_smooth_acceleration']<=2.00001
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    colors=dict(original_none='#3b82f6',legacy='#ef4444',retimed_none='#9333ea',smooth='#16a34a')
    for name,array in example.items():axes[0].plot(np.arange(len(array))/30,array[:,1],label=name,color=colors[name],lw=1.)
    axes[0].set(xlabel='sim seconds',ylabel='shoulder target (rad)',title='Recorded pilot commands and offline smooth replay');axes[0].legend()
    for axis,quantity in zip(axes[1:],('acceleration','jerk')):
        axis.bar(list(summary),[summary[name][quantity]['rms'] for name in summary],color=[colors[name] for name in summary])
        axis.set(ylabel=f'{quantity} RMS (rad/s^'+('2)' if quantity=='acceleration' else '3)'),title=f'Mean per-episode {quantity} RMS, arm joints only')
    fig.savefig(output/'command-metrics.png',dpi=160)
    print(json.dumps({k:v for k,v in report.items() if k!='per_episode'},indent=2))
if __name__=='__main__':main()
