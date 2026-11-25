"""Identical measured-trajectory diagnostics for policy and expert episodes."""
from pathlib import Path
import argparse,csv,importlib.util,json,math,sys
import numpy as np
REPO=Path.home()/'Downloads/Exploring-VLAs'
sys.path.insert(0,str(REPO/'linux-env/scripts/collect'))
from pose_math import rotate
fk_path=REPO/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py'
if not fk_path.exists():fk_path=Path(__file__).with_name('compute_workspace.py')
spec=importlib.util.spec_from_file_location('measured_fk',fk_path);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
COLORS=('red','green','blue')

def diagnostics(snaps,color):
    stamps=np.asarray([s['sim_ns']/1e9 for s in snaps]);q=np.asarray([s['state'] for s in snaps]);t=stamps-stamps[0]
    assert len(t)>2 and np.all(np.diff(t)>0)
    local=fk.forward_kinematics_batch(q[:,:5]);tcp=np.asarray([np.asarray(rotate(s['robot_base_world_pose']['orientation'],p.tolist()))+s['robot_base_world_pose']['position'] for p,s in zip(local,snaps)])
    names=sorted(n for n in snaps[0]['bodies'] if any(n.startswith(c+'_') for c in COLORS));positions=np.asarray([[s['bodies'][n]['position'] for n in names] for s in snaps]);delta=tcp[:,None,:]-positions;dist=np.linalg.norm(delta,axis=2);xy=np.linalg.norm(delta[:,:,:2],axis=2)
    target=np.asarray([n.startswith(color+'_') for n in names]);other=~target;assert target.any() and other.any()
    dtarget=dist[:,target].min(axis=1);dother=dist[:,other].min(axis=1);nearest=dist.argmin(axis=1);grip=np.degrees(q[:,5]);closing=np.flatnonzero(np.maximum.accumulate(grip)-grip>=2)
    close=int(closing[0]) if len(closing) else None;before=close+1 if close is not None else len(t);closest=int(dtarget.argmin());xy_target=xy[:,target].min(axis=1);entries=np.flatnonzero(xy_target[:before]<=.05);entry=int(entries[0]) if len(entries) else None
    dq=np.degrees(np.gradient(q,t,axis=0));speed=np.linalg.norm(np.gradient(tcp,t,axis=0),axis=1);dt=np.diff(t,prepend=t[0]);idle=(np.abs(dq[:,:5]).max(axis=1)<1)&(np.abs(dq[:,5])<.2)
    intervals=[];start=None
    for i,yes in enumerate(list(idle)+[False]):
        if yes and start is None:start=i
        if not yes and start is not None:
            end=i-1
            if t[end]-t[start]>=.5:intervals.append([float(t[start]),float(t[end])])
            start=None
    idle_s=sum(b-a for a,b in intervals);result=dict(duration_sim_s=float(t[-1]),min_tcp_target_m=float(dtarget.min()),min_tcp_other_m=float(dother.min()),initial_target_distance_m=float(dtarget[0]),target_distance_reduction_m=float(dtarget[0]-dtarget.min()),min_preclosure_target_m=float(dtarget[:before].min()),min_preclosure_other_m=float(dother[:before].min()),closest_target_time_s=float(t[closest]),closest_target_tcp_world=tcp[closest].tolist(),nearest_colour_at_closest_target=names[nearest[closest]].split('_')[0],target_nearer_than_other_at_closest_target=bool(dtarget[closest]<=dother[closest]),
        ever_closed_ge2deg=close is not None,gripper_min_deg=float(grip.min()),gripper_max_deg=float(grip.max()),idle_s=idle_s,idle_pct=100*idle_s/t[-1],longest_idle_s=max((b-a for a,b in intervals),default=0),idle_intervals_sim_s=intervals,tcp_path_length_m=float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum()),tcp_speed_p95_m_s=float(np.quantile(speed,.95)),tcp_z_start_m=float(tcp[0,2]),tcp_z_min_m=float(tcp[:,2].min()),tcp_z_max_m=float(tcp[:,2].max()),approach_xy50mm_time_s=float(t[entry]) if entry is not None else None)
    for c in COLORS:result['min_tcp_'+c+'_m']=float(dist[:,[n.startswith(c+'_') for n in names]].min())
    if close is not None:
        target_ids=np.flatnonzero(target);obj=int(target_ids[dist[close,target].argmin()]);offset=delta[close,obj]
        result.update(closing_time_sim_s=float(t[close]),closing_tcp_world=tcp[close].tolist(),closing_target_object=names[obj],closing_target_world=positions[close,obj].tolist(),closing_offset_world_m=offset.tolist(),closing_distance_m=float(dtarget[close]),closing_other_distance_m=float(dother[close]),closing_xy_target_m=float(np.linalg.norm(offset[:2])),closing_height_above_target_center_m=float(offset[2]),nearest_colour_at_closing=names[nearest[close]].split('_')[0],nearest_object_at_closing=names[nearest[close]],target_nearer_at_closing=bool(dtarget[close]<=dother[close]))
    else:result.update(closing_time_sim_s=None,closing_distance_m=None,target_nearer_at_closing=None)
    if entry is not None:
        target_ids=np.flatnonzero(target);obj=int(target_ids[xy[entry,target].argmin()]);result['approach_height_above_target_center_m']=float(delta[entry,obj,2]);result['descent_from_xy50_to_closure_m']=float(tcp[entry:close+1,2].max()-tcp[close,2]) if close is not None else None
    else:result.update(approach_height_above_target_center_m=None,descent_from_xy50_to_closure_m=None)
    # Dwell ignores instantaneous PhysX servo jitter: half-second windows must
    # remain within a 5mm TCP box diagonal and 0.5deg gripper range.
    dwell=np.zeros(len(t),bool)
    for i in range(len(t)):
        j=int(np.searchsorted(t,t[i]+.5))
        if j<len(t) and np.linalg.norm(np.ptp(tcp[i:j+1],axis=0))<=.005 and np.ptp(grip[i:j+1])<=.5:dwell[i:j+1]=True
    result['tcp_dwell_s']=float(dt[dwell].sum());result['tcp_dwell_pct']=100*result['tcp_dwell_s']/t[-1]
    fine=np.flatnonzero(xy_target[:before]<=.015);result['aligned_xy15mm_before_closing']=bool(len(fine));result['min_preclosure_target_xy_m']=float(xy_target[:before].min())
    if len(fine):
        fi=int(fine[0]);ids=np.flatnonzero(target);obj=int(ids[xy[fi,target].argmin()]);result['fine_alignment_time_sim_s']=float(t[fi]);result['fine_alignment_height_above_target_center_m']=float(delta[fi,obj,2]);result['descent_after_fine_alignment_m']=float(tcp[fi:before,2].max()-tcp[before-1,2])
    else:result.update(fine_alignment_time_sim_s=None,fine_alignment_height_above_target_center_m=None,descent_after_fine_alignment_m=None)
    if close is not None:
        one_second_before=max(0,int(np.searchsorted(t,max(0,t[close]-1))))
        target_ids=np.flatnonzero(target);obj=int(target_ids[dist[one_second_before,target].argmin()])
        result['height_above_target_center_1s_before_closing_m']=float(delta[one_second_before,obj,2])
        result['descent_in_last_second_before_closing_m']=float(tcp[one_second_before,2]-tcp[close,2])
    arrays=dict(t=t,q=q,tcp=tcp,target_distance=dtarget,other_distance=dother,grip_deg=grip,target_xy=xy_target,block_positions=positions,block_names=np.asarray(names),closest_block_indices=nearest,idle=idle)
    return result,arrays

def convert(frame):
    return dict(sim_ns=frame['sim_ns'],state=frame['physics_state'],bodies=frame['physical_bodies'],robot_base_world_pose=frame['robot_base_world_pose'])

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'behaviour';out.mkdir(exist_ok=True);rows=[];experts=[];holdouts=[];caseframes=[]
    for file in sorted(root.glob('attempt-*/ep*/physical.jsonl')):
        r=json.loads(file.with_name('result.json').read_text());snaps=[json.loads(l) for l in file.open()];d,arr=diagnostics(snaps,r['color']);d.update(seed=r['seed'],color=r['color'],source='scored',raw_outcome=r['outcome'],timing_valid=r['client']['missed_action_ticks_after_startup']==0);rows.append(d);np.savez_compressed(out/f"eval-{r['seed']}.npz",**arr)
        if r.get('closing_distance_m') is None:assert d.get('closing_distance_m') is None
        else:assert d.get('closing_distance_m') is not None and abs(d['closing_distance_m']-r['closing_distance_m'])<1e-6,(r['seed'],d.get('closing_distance_m'),r['closing_distance_m'])
    base=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/datasets')
    for batch,indices,kind,dest in [('rtx6000-normal-i3-b00',range(5),'seen-training',experts),('pilot-3.1-v4',range(3),'held-out-pilot',holdouts)]:
        normal=base/batch/'normal';receipts=[json.loads(l) for l in (normal/'episodes.jsonl').open()]
        for idx in indices:
            frames=[json.loads(l) for l in (normal/f'attempt-{idx:04d}/frames.jsonl').open()];receipt=receipts[idx];assert receipt['accepted'];color=receipt['params']['color'];seed=receipt['params']['seed'];snaps=[convert(f) for f in frames];d,arr=diagnostics(snaps,color);d.update(seed=seed,color=color,source=kind,batch=batch,episode_index=idx,frame_count=len(frames));dest.append(d);np.savez_compressed(out/f'{kind}-{seed}.npz',**arr)
            close=np.flatnonzero(np.maximum.accumulate(arr['grip_deg'])-arr['grip_deg']>=2);closure=int(close[0]) if len(close) else len(frames)//2
            moving=[]
            for candidate in range(1,max(2,min(closure,len(frames)-50))):
                state=np.degrees(np.asarray(frames[candidate]['state']))
                future=np.degrees(np.asarray([f['action'] for f in frames[candidate:candidate+50]]))
                if np.abs(future[:10,:5]-state[:5]).max()>1 and np.abs(future[:,:5]-state[:5]).max()>5:moving.append(candidate)
            if not moving:raise ValueError('No informative moving approach window')
            anchor=moving[-1]
            if idx<3:
                case=dict(kind=kind,batch=batch,dataset=str(normal/'dataset'),episode_index=idx,seed=seed,color=color,task=receipt['params']['task'],anchor_frame=anchor,closure_frame=int(close[0]) if len(close) else None,source_frames=str(normal/f'attempt-{idx:04d}/frames.jsonl'),source_top_video=str(normal/f'attempt-{idx:04d}/top.mp4'),frame=frames[anchor],receipt=receipt)
                caseframes.append(case)
    for name,rs in [('episodes',rows),('training-experts',experts),('held-out-experts',holdouts)]:
        (out/(name+'.json')).write_text(json.dumps(rs,indent=2)+'\n');write_csv(out/(name+'.csv'),rs)
    (out/'open-loop-cases.json').write_text(json.dumps(caseframes,indent=2)+'\n')
    print(json.dumps(dict(eval_episodes=len(rows),training_experts=len(experts),held_out_experts=len(holdouts),closure_nearest_target=sum(r['target_nearer_at_closing'] is True for r in rows),mean_idle_pct=float(np.mean([r['idle_pct'] for r in rows])),output=str(out)),indent=2))
if __name__=='__main__':main()
