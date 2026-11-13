"""Transport-independent re-grasp geometry, quotas and measured miss checks."""
import math,random

REGRASP_POLICY=dict(version=1,share=.18,offset_min_m=.02,offset_max_m=.04,
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


def detect_miss(before,after,object_name,close_target):
    if before['epoch']!=after['epoch'] or after['sim_ns']<=before['sim_ns']:
        raise ValueError('Re-grasp miss check crosses an epoch or has no physical progress')
    if before.get('state_source')!='isaac_articulation' or after.get('state_source')!='isaac_articulation':
        raise ValueError('Re-grasp requires measured physical feedback')
    if any(len(s['state'])!=6 or any(len(body['position'])!=3 for body in s['bodies'].values()) for s in (before,after)):
        raise ValueError('Incomplete measured recovery state or body position')
    if set(before['bodies'])!=set(after['bodies']):raise ValueError('Rigid-body inventory changed during recovery')
    names=[n for n in before['bodies'] if not n.startswith('cup_')]
    motion={n:math.sqrt(sum((a-b)**2 for a,b in zip(before['bodies'][n]['position'],after['bodies'][n]['position']))) for n in names}
    if object_name not in motion:raise ValueError('Target object missing from re-grasp evidence')
    if not all(math.isfinite(v) for v in [*motion.values(),after['state'][5],close_target]):
        raise ValueError('Non-finite re-grasp evidence')
    jaw_reached=abs(after['state'][5]-close_target)<=REGRASP_POLICY['jaw_close_tolerance_rad']
    unchanged=all(d<=REGRASP_POLICY['object_motion_max_m'] for d in motion.values())
    return dict(confirmed=jaw_reached and unchanged,target_motion_m=motion[object_name],
        all_object_motion_m=motion,jaw_position_rad=after['state'][5],close_target_rad=close_target,
        jaw_reached_close_command=jaw_reached,primary_signal='no object followed the verification lift',
        before_sim_ns=before['sim_ns'],after_sim_ns=after['sim_ns'],epoch=after['epoch'])


def receipt_error(receipt):
    """Require measured miss evidence before a recovery can enter the dataset."""
    if receipt.get('params',{}).get('recovery')!='regrasp':return None
    try:
        report=receipt['recovery_report']
        if not report['miss_confirmed']:return 'first grasp did not miss'
        before,after=report['measured_before'],report['measured_after_lift']
        detection=detect_miss(before,after,receipt['params']['object'],report['detection']['close_target_rad'])
        if not detection['confirmed'] or detection!=report['detection']:return 'measured miss evidence differs'
        if report['epoch']!=before['epoch']:return 'recovery epoch differs'
        stages=[step['stage'] for step in report['steps']]
        if stages!=['regrasp_miss_open','regrasp_miss_approach','regrasp_miss_descent','regrasp_miss_close','regrasp_miss_lift','regrasp_reopen']:
            return 'recovery sequence is incomplete'
        distance=math.sqrt(sum(x*x for x in report['geometry']['offset_m']))
        if not .02-1e-12<=distance<=.04+1e-12:return 'recovery offset outside 2–4 cm'
        if abs(distance-abs(receipt['params']['regrasp']['signed_offset_m']))>1e-12:return 'recovery offset differs from seed parameters'
        if not report['scene_after_reopen']['passed']:return 'recovery scene was not verified'
    except (KeyError,TypeError,ValueError,IndexError):return 'recovery evidence is incomplete'
    return None


def recovery_summary(manifest):
    counts={'none':0,'regrasp':0};offsets=[]
    for receipt in manifest:
        error=receipt_error(receipt)
        if error:raise ValueError('Cannot export recovery: '+error)
        kind=receipt['params'].get('recovery','none');counts[kind]+=1
        if kind=='regrasp':offsets.append(abs(receipt['params']['regrasp']['signed_offset_m']))
    return dict(counts=counts,share=counts['regrasp']/max(1,len(manifest)),offsets_m=offsets,
                metadata_path='meta/collection_episodes.json')
