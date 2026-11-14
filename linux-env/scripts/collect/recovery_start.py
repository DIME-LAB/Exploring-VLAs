"""Seeded synthetic post-miss starts, static geometry and measured reset evidence."""
import copy,math,random
import numpy as np
from comparisons import matches,native_frame
from contract import JOINTS
from finger_geometry import FingerModel,body_box,separation,rotation
from tick_contract import digest,require_snapshots,row_errors,TickContractError
from miss_clearance import proof_differences

STRATEGY='synthetic_post_miss_reset'
POLICY=dict(version=1,strategy=STRATEGY,phenotypes=dict(closed=.5,partly_closed=.3,open_hover=.2),
    offset_min_m=.026,offset_max_m=.038,yaw_jitter_max_rad=math.radians(12),
    near_hover_m=[.003,.015],open_hover_m=[.018,.035],max_candidates=24,settle_ticks=6,
    finger_margin_m=.001,arithmetic_margin_m=.00002,target_corner_motion_max_m=.0002,
    stationary_linear_m_s=.0001,stationary_angular_rad_s=.005,
    origin='synthetic reset; no physical failed grasp is claimed',recorded_phase='corrective',
    training='all recorded frames train on normal-speed corrective expert; no miss prefix is recorded')


def candidates(request):
    rng=random.Random(request['seed'] ^ 0x71A53F)
    u=rng.random();kind='closed' if u<.5 else 'partly_closed' if u<.8 else 'open_hover'
    jaw_fraction=0. if kind=='closed' else rng.uniform(.2,.65) if kind=='partly_closed' else 1.
    distance=abs(request['signed_offset_m'])
    if not POLICY['offset_min_m']<=distance<=POLICY['offset_max_m']:raise ValueError('Synthetic offset must be 26–38 mm')
    angle=rng.uniform(-math.pi,math.pi);yaw=rng.uniform(-POLICY['yaw_jitter_max_rad'],POLICY['yaw_jitter_max_rad'])
    hover=rng.uniform(*(POLICY['open_hover_m'] if kind=='open_hover' else POLICY['near_hover_m']))
    # Keep the selected phenotype while searching alternative reachable clear poses.
    for i in range(POLICY['max_candidates']):
        yield dict(index=i,phenotype=kind,jaw_fraction=jaw_fraction,
            distance_m=distance if i<8 else rng.uniform(POLICY['offset_min_m'],POLICY['offset_max_m']),
            direction_rad=angle+(i%8)*math.pi/4,hover_m=hover,
            yaw_jitter_rad=yaw if i<16 else rng.uniform(-POLICY['yaw_jitter_max_rad'],POLICY['yaw_jitter_max_rad']))


def geometry(pose,bbox,spec,settings):
    from so_arm101_control.grasp_geometry import make_grasp_plan,gripper_angles,object_grip_width
    from so_arm101_control.compute_workspace import geometric_ik
    base=make_grasp_plan(pose,bbox,**settings)
    yaw=base['yaw']+spec['yaw_jitter_rad']
    target=[base['tcp'][0]+spec['distance_m']*math.cos(spec['direction_rad']),
            base['tcp'][1]+spec['distance_m']*math.sin(spec['direction_rad']),base['tcp'][2]+spec['hover_m']]
    opening,closing=gripper_angles(object_grip_width(bbox,settings.get('cross',False)),settings.get('tcp_clearance',.001))
    jaw=closing+(opening-closing)*spec['jaw_fraction']
    # The miss is defined in TCP coordinates; it does not pretend to be a centred grasp.
    solutions=[]
    for equivalent in (0.,math.pi,-math.pi):
        for q in geometric_ik(*target,grasp_yaw=yaw+equivalent) or []:
            solutions.append([q[n] for n in JOINTS[:5]]+[jaw])
    return dict(tcp_m=target,reference_grasp_tcp_m=list(base['tcp']),requested_yaw_rad=yaw,
        projected_offset_m=[target[i]-base['tcp'][i] for i in range(3)],jaw_rad=jaw,opening_rad=opening,closing_rad=closing,solutions=solutions)


class StaticClearance:
    def __init__(self,model,snapshot,target):
        self.model=copy.deepcopy(model);self.oracle=FingerModel(model);self.all_links=FingerModel(model,include_all=True);self.baseline=copy.deepcopy(snapshot);self.target=target
        finger=model.get('effective_finger_offsets');block=model.get('effective_block_offsets')
        if not finger or not block:raise ValueError('Measured finger and block offsets are required')
        offsets=[*finger.values(),block]
        for x in offsets:
            if not x['contact_m'] or len(x['contact_m'])!=len(x['rest_m']) or not np.isfinite(x['contact_m']+x['rest_m']).all() or min(x['contact_m'])<0 or max(abs(v) for v in x['rest_m'])>1e-9:
                raise ValueError('Unverified static contact/rest offsets')
        if max(max(x['contact_m']) for x in finger.values())+max(block['contact_m'])+POLICY['arithmetic_margin_m']>POLICY['finger_margin_m']:
            raise ValueError('Static margin does not cover measured contact offsets')
    def state(self,q,snapshot=None):
        s=self.baseline if snapshot is None else snapshot
        target=body_box(s['bodies'][self.target],s['local_corners'][self.target]);base=s['robot_base_world_pose']
        gaps=[dict(finger=box['path'],separation_m=separation(box,target),floor_clearance_m=box['min_z_m']) for box in self.oracle.boxes(q,base)]
        required=POLICY['finger_margin_m']+POLICY['arithmetic_margin_m']
        from cup_geometry import PRIMITIVES
        world=[]
        for name,body in s['bodies'].items():
            if name.startswith('cup_'):
                R=rotation(body['orientation']);center=np.asarray(body['position'])
                for shape in PRIMITIVES:
                    world.append((name+'/'+shape['name'],dict(rotation=R@rotation(shape['quaternion_wxyz']),
                        center_world=center+R@np.asarray(shape['center_m']),half_extents_m=np.asarray(shape['size_m'])/2)))
            else:world.append((name,body_box(body,s['local_corners'][name])))
        scene=[]
        for link in self.all_links.boxes(q,base):
            name,gap=min(((name,separation(link,box)) for name,box in world),key=lambda x:x[1])
            scene.append(dict(link=link['path'],body=name,separation_m=gap))
        return dict(passed=all(g['separation_m']>=required and g['floor_clearance_m']>=required for g in gaps) and all(g['separation_m']>=required for g in scene),
            joints=list(q),target=self.target,required_margin_m=required,boxes=gaps,scene=scene,model_sha256=digest(self.model))
    def target_state(self,snapshot):
        body=snapshot['bodies'][self.target];before=self.baseline['bodies'][self.target]
        velocity=np.asarray(body['velocity'],dtype=float)
        if velocity.shape!=(6,) or not np.isfinite(velocity).all():raise ValueError('Invalid target velocity')
        corners=np.asarray(self.baseline['local_corners'][self.target])
        a=corners@rotation(before['orientation']).T+np.asarray(before['position'])
        b=corners@rotation(body['orientation']).T+np.asarray(body['position'])
        displacement=float(np.linalg.norm(b-a,axis=1).max());linear=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]))
        return dict(passed=displacement<=POLICY['target_corner_motion_max_m'] and linear<=POLICY['stationary_linear_m_s'] and angular<=POLICY['stationary_angular_rad_s'],
            max_corner_motion_m=displacement,linear_m_s=linear,angular_rad_s=angular,target=self.target)


def choose(pose,bbox,request,settings,oracle,check,require_epoch,limits):
    attempts=[]
    for spec in candidates(request):
        require_epoch();g=geometry(pose,bbox,spec,settings)
        attempt=dict(spec=spec,geometry=g,verdicts=[],passed=False);attempts.append(attempt)
        for q in g['solutions']:
            require_epoch()
            if any(not lo<=x<=hi for x,(lo,hi) in zip(q,limits)):attempt['verdicts'].append(dict(reason='joint_limits',joints=q));continue
            static=oracle.state(q)
            if not static['passed']:attempt['verdicts'].append(dict(reason='native_static_clearance',proof=static));continue
            valid,contacts=check(q);require_epoch()
            verdict=dict(passed=valid,joints=q,static=static,contacts=[dict(body_1=c.contact_body_1,body_2=c.contact_body_2,depth=c.depth) for c in contacts])
            attempt['verdicts'].append(verdict)
            if valid:
                attempt['passed']=True
                return dict(spec=spec,geometry=g,joints=q,proof=verdict,attempts=attempts,policy=POLICY)
    return dict(attempts=attempts,policy=POLICY,passed=False)


def receipt_error(receipt,rows):
    try:
        report=receipt['recovery_report'];episode_id=receipt['episode_id']
        if report['strategy']!=STRATEGY or report['policy']!=POLICY or report['episode_id']!=episode_id:return 'synthetic recovery identity/policy differs'
        if report['origin']!=POLICY['origin'] or not report['passed']:return 'synthetic start is uncertified'
        selection=report['selection'];attempts=selection['attempts'];specs=list(candidates(receipt['params']['regrasp']))
        index=selection['spec']['index']
        if len(attempts)!=index+1 or selection['spec']!=specs[index] or any(a['spec']!=specs[i] for i,a in enumerate(attempts)):return 'synthetic candidate order differs'
        if any(a['passed'] for a in attempts[:-1]) or not attempts[-1]['passed'] or not selection['proof']['passed']:return 'synthetic selection was not first passing'
        expected_geometry=geometry(report['request']['pose'],report['request']['bbox'],selection['spec'],report['request']['settings'])
        if proof_differences(selection['geometry'],expected_geometry,'synthetic.geometry') or not any(matches(selection['joints'],q,'arithmetic') for q in expected_geometry['solutions']):return 'synthetic IK geometry differs'
        oracle=StaticClearance(report['model'],report['baseline'],receipt['params']['object'])
        if proof_differences(selection['proof']['static'],oracle.state(selection['joints']),'synthetic.static'):return 'synthetic static proof differs'
        restore=report['restore'];snapshot=restore['snapshot']
        if restore['epoch']!=report['epoch'] or restore['epoch']<=report['baseline']['epoch'] or not matches(snapshot['sim_ns'],report['baseline']['sim_ns'],'stamp_ns'):return 'synthetic reset boundary identity differs'
        if not restore['arm_restored'] or not matches(snapshot['state'],selection['joints'],'target_rad') or not matches(snapshot['drive_targets'],selection['joints'],'target_rad') or not matches(snapshot['velocity'],[0.]*6,'float32_readback'):return 'synthetic restore differs'
        ticks=report['settle_snapshots'];require_snapshots(ticks)
        if len(ticks)!=POLICY['settle_ticks'] or any(s['epoch']!=report['epoch'] for s in ticks):return 'synthetic settle identity differs'
        for s,p in zip(ticks,report['settle_checks']):
            if not oracle.state(s['state'],s)['passed'] or not oracle.target_state(s)['passed'] or not p['scene_valid']:return 'synthetic settle is unsafe'
            if proof_differences(p['static'],oracle.state(s['state'],s),'synthetic.settle.static') or proof_differences(p['target'],oracle.target_state(s),'synthetic.settle.target'):return 'synthetic settle proof differs'
        if len(report['settle_checks'])!=len(ticks):return 'synthetic settle evidence missing'
        if not matches(report['record_after_sim_ns'],ticks[-1]['sim_ns'],'native_ns'):return 'synthetic recording boundary differs'
        if not rows or any(r['episode_id']!=episode_id or r['epoch']!=report['epoch'] or r['collection_phase']!='corrective' for r in rows):return 'synthetic corrective row identity differs'
        if row_errors(rows[0],ticks[-1]['tick_contract']['row']):return 'synthetic handover tick contract failed'
        if any(row_errors(r,rows[i-1] if i else None) for i,r in enumerate(rows)):return 'synthetic corrective tick contract failed'
        if any(native_frame(r)['sim_ns']<=report['record_after_sim_ns'] for r in rows):return 'synthetic setup frames entered dataset'
        if not report['scene_after_reset']['passed']:return 'synthetic planning scene not synchronized'
    except (KeyError,TypeError,ValueError,IndexError,TickContractError):return 'synthetic recovery evidence incomplete'
    return None
