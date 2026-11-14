"""Transport-independent re-grasp geometry, quotas and measured miss checks."""
import math,random
from tick_contract import digest,row_errors,STEP_NS

REGRASP_POLICY=dict(version=2,share=.18,offset_min_m=.02,offset_max_m=.04,
    verification_lift_m=.03,object_motion_max_m=.003,jaw_close_tolerance_rad=.008)


def regrasp_slot(accepted_index):
    return ((accepted_index+1)*18)//100>(accepted_index*18)//100


def recovery_parameters(seed,enabled):
    if not enabled:return dict(kind='none')
    rng=random.Random(seed ^ 0xA3B5C7)
    return dict(kind='regrasp',signed_offset_m=rng.uniform(.02,.04)*rng.choice((-1,1)),
                lift_m=.03,seed=seed ^ 0xA3B5C7)


def recovery_geometry(pose,bbox,params,**settings):
    from so_arm101_control.grasp_geometry import make_grasp_plan, find_reachable_grasp_yaw
    plan=make_grasp_plan(pose,bbox,**settings)
    distance=params['signed_offset_m']
    if not .02<=abs(distance)<=.04:raise ValueError('Re-grasp offset must be 2–4 cm')
    dx,dy=-math.sin(plan['yaw'])*distance,math.cos(plan['yaw'])*distance
    poses=[]
    for name,x,y,z in plan['poses']:
        poses.extend([('true_'+name,x,y,z),('miss_'+name,x+dx,y+dy,z)])
    final=next(p for p in poses if p[0]=='miss_final')
    poses.append(('miss_lift',final[1],final[2],final[3]+params['lift_m']))
    yaw,solutions,debug=find_reachable_grasp_yaw(poses,plan['yaw'])
    return dict(offset_m=[dx,dy,0.],yaw=yaw,poses=poses,solutions=solutions,debug=debug,
                true_geometry=plan)


def detect_miss(window,object_name,close_target):
    if len(window)<2:raise ValueError('Whole close/lift window is required')
    before,after=window[0],window[-1]
    names=[n for n in before['bodies'] if not n.startswith('cup_')]
    if object_name not in names:raise ValueError('Target missing from miss window')
    for i,snapshot in enumerate(window):
        if snapshot['epoch']!=before['epoch'] or snapshot.get('state_source')!='isaac_articulation':raise ValueError('Miss window has a stale epoch or unmeasured state')
        if len(snapshot['state'])!=6 or set(snapshot['bodies'])!=set(before['bodies']):raise ValueError('Incomplete miss window')
        if i and abs(snapshot['sim_ns']-window[i-1]['sim_ns']-STEP_NS)>10:raise ValueError('Miss window has a clock gap')
        if any(len(body['position'])!=3 or not all(math.isfinite(v) for v in body['position']) for body in snapshot['bodies'].values()):raise ValueError('Invalid miss-window poses')
        if not all(math.isfinite(v) for v in snapshot['state']):raise ValueError('Invalid miss-window state')
    motion={name:max(math.dist(before['bodies'][name]['position'],s['bodies'][name]['position']) for s in window) for name in names}
    jaw_reached=math.isfinite(close_target) and abs(after['state'][5]-close_target)<=REGRASP_POLICY['jaw_close_tolerance_rad']
    return dict(confirmed=jaw_reached and all(d<=REGRASP_POLICY['object_motion_max_m'] for d in motion.values()),
        target_motion_m=motion[object_name],all_object_motion_m=motion,jaw_position_rad=after['state'][5],
        close_target_rad=close_target,jaw_reached_close_command=jaw_reached,
        primary_signal='peak object displacement over every close/lift tick',
        before_sim_ns=before['sim_ns'],after_sim_ns=after['sim_ns'],epoch=after['epoch'],ticks=len(window))


def receipt_error(receipt,rows=None):
    """Bind a measured miss to this episode's actual raw rows, epochs and poses."""
    if receipt.get('params',{}).get('recovery')!='regrasp':return None
    try:
        report=receipt['recovery_report'];episode_id=receipt['episode_id']
        if not rows:return 'recovery rows are required'
        if not episode_id or report['episode_id']!=episode_id or any(r['episode_id']!=episode_id for r in rows):return 'recovery episode identity differs'
        if any(r['epoch']!=report['epoch'] for r in rows):return 'recovery epoch differs from recorded episode'
        if not report['miss_confirmed']:return 'first grasp did not miss'
        before,after=report['measured_before'],report['measured_after_lift']
        window=[r for r in rows if before['sim_ns']<=r['sim_ns']<=after['sim_ns']]
        if not window or window[0]['sim_ns']!=before['sim_ns'] or window[-1]['sim_ns']!=after['sim_ns']:return 'miss window is outside recorded episode'
        if [digest(r) for r in window]!=report['window_frame_sha256s']:return 'miss window row hashes differ'
        if any(row_errors(r,rows[i-1] if i else None) for i,r in enumerate(rows)):return 'recovery rows fail pipeline contract'
        snapshots=[dict(epoch=r['epoch'],sim_ns=r['sim_ns'],state_source=r['state_source'],state=r['state'],bodies=r['physical_bodies']) for r in window]
        for endpoint,snapshot in ((before,snapshots[0]),(after,snapshots[-1])):
            if any(endpoint[k]!=snapshot[k] for k in snapshot):return 'recovery endpoint differs from raw frame'
        detection=detect_miss(snapshots,receipt['params']['object'],report['detection']['close_target_rad'])
        if not detection['confirmed'] or detection!=report['detection']:return 'whole-window miss evidence differs'
        stages=[s['stage'] for s in report['steps']]
        if stages!=['regrasp_miss_open','regrasp_miss_approach','regrasp_miss_descent','regrasp_miss_close','regrasp_miss_lift','regrasp_reopen']:return 'recovery sequence is incomplete'
        for step in report['steps'][1:]:
            end=step.get('end_sim_ns',step.get('sim_ns'))
            if not any(r['sim_ns']==end and r['phase']==step['stage'] for r in rows):return 'recovery stage is not in recorded frames'
        distance=math.sqrt(sum(x*x for x in report['geometry']['offset_m']))
        if not .02-1e-12<=distance<=.04+1e-12:return 'recovery offset outside 2–4 cm'
        if abs(distance-abs(receipt['params']['regrasp']['signed_offset_m']))>1e-12:return 'recovery offset differs from seed'
        if not report['scene_after_reopen']['passed']:return 'recovery scene was not verified'
    except (KeyError,TypeError,ValueError,IndexError):return 'recovery evidence is incomplete'
    return None


def recovery_summary(manifest):
    counts={'none':0,'regrasp':0};offsets=[]
    for receipt in manifest:
        kind=receipt['params'].get('recovery','none');counts[kind]+=1
        if kind=='regrasp':offsets.append(abs(receipt['params']['regrasp']['signed_offset_m']))
    return dict(counts=counts,share=counts['regrasp']/max(1,len(manifest)),offsets_m=offsets,
                metadata_path='meta/collection_episodes.json')
