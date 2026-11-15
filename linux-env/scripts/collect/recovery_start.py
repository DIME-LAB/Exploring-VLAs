"""Seeded synthetic post-miss starts, static geometry and measured reset evidence."""
import copy,math,random
import numpy as np
from comparisons import matches,native_frame,comparison,bound_comparison
from contract import JOINTS
from finger_geometry import FingerModel,body_box,separation,rotation
from tick_contract import digest,require_snapshots,row_errors,TickContractError,PHYSICS_DT,PHYSICS_STEPS,STEP_NS,world_step_errors
from miss_clearance import proof_differences

STRATEGY='synthetic_post_miss_reset'
POLICY=dict(version=3,strategy=STRATEGY,phenotypes=dict(closed=.5,partly_closed=.3,open_hover=.2),
    offset_min_m=.026,offset_max_m=.038,yaw_jitter_max_rad=math.radians(12),
    near_hover_m=[.003,.015],open_hover_m=[.018,.035],max_candidates=24,settle_ticks=6,
    finger_margin_m=.001,arithmetic_margin_m=.00002,target_corner_motion_max_m=.0002,
    # Empirical contact/impulse screens, NOT solver displacement bounds. See
    # fixtures/rest-velocity-floor.json: clean maxima .34691 mm/s/.0056004 rad/s.
    stationary_linear_m_s=.0007,stationary_angular_rad_s=.012,
    velocity_calibration=dict(source='linux-env/scripts/collect/fixtures/rest-velocity-floor.json',
        sha256='c1404aae15733be6f9c08d02987983aed6c9e5b672674ada96bc94a1b8efbe46',
        interpretation='empirical contact/impulse screens; report any resting peak above half a bound before production'),
    protected_bodies='every cup and block in the pre-restore world anchor',
    protected_corner_motion_max_m=.0002,protected_base_rule='float32_readback',
    settle_sampling='four pre-physics samples plus final post-solve sample per 30 Hz frame',
    physics_dt_s=PHYSICS_DT,physics_steps_per_tick=PHYSICS_STEPS,
    native_substep_clock='nondecreasing within the held frame; native_ns comparison rule; World clock supplies substep cadence',
    replay_robot_model_rule='all geometry/kinematics equal under arithmetic readback rule; derived digest recorded separately',
    first_corrective_base_rule='float32_readback',
    origin='synthetic reset; no physical failed grasp is claimed',recorded_phase='corrective',
    training='all recorded frames train on normal-speed corrective expert; no miss prefix is recorded')


def native_bbox(snapshot,target):
    """Use the held physical shape bounds, in metres, for synthetic geometry."""
    corners=np.asarray(snapshot['local_corners'][target],dtype=float)
    if corners.shape!=(8,3) or not np.isfinite(corners).all():
        raise ValueError('Synthetic target native corners are incomplete or non-finite')
    extents=np.ptp(corners,axis=0)
    if np.any(extents<=0):raise ValueError('Synthetic target native dimensions are not positive')
    return dict(zip(('sx','sy','sz'),map(float,extents)))


def candidates(request):
    rng=random.Random(request['seed'] ^ 0x71A53F)
    u=rng.random();kind='closed' if u<.5 else 'partly_closed' if u<.8 else 'open_hover'
    jaw_fraction=0. if kind=='closed' else rng.uniform(.2,.65) if kind=='partly_closed' else 1.
    distance=abs(request['signed_offset_m'])
    if not POLICY['offset_min_m']<=distance<=POLICY['offset_max_m']:raise ValueError('Synthetic lateral offset must be 26–38 mm')
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


WORLD_FIELDS=('bodies','local_corners','base_world_pose','robot_base_world_pose','robot_base_pose_source')
SAMPLE_FIELDS=('epoch','sim_ns','physics_time_s','state','velocity','drive_targets','bodies',
    'base_world_pose','robot_base_world_pose','robot_base_pose_source')


def world_anchor(snapshot):
    return copy.deepcopy({key:snapshot[key] for key in WORLD_FIELDS})


def sample_snapshot(snapshot):
    return copy.deepcopy({key:snapshot[key] for key in SAMPLE_FIELDS})


def pose_checks(expected,actual,field):
    q=actual.get('orientation');p=expected.get('orientation')
    sign=-1 if isinstance(q,(list,tuple)) and isinstance(p,(list,tuple)) and sum(a*b for a,b in zip(q,p))<0 else 1
    return [comparison(field+'.position',actual.get('position'),expected.get('position'),'float32_readback'),
            comparison(field+'.orientation',[sign*x for x in q] if isinstance(q,(list,tuple)) else q,p,'float32_readback')]


def base_checks(expected,actual):
    checks=[]
    for key in ('base_world_pose','robot_base_world_pose'):
        checks.extend(pose_checks(expected[key],actual[key],key))
    checks.append(comparison('robot_base_pose_source',actual.get('robot_base_pose_source'),expected.get('robot_base_pose_source')))
    return checks


def restore_world_checks(expected,actual):
    checks=base_checks(expected,actual)
    checks.append(comparison('restored body identities',sorted(actual['bodies']),sorted(expected['bodies'])))
    checks.append(comparison('restored local geometry',actual['local_corners'],expected['local_corners']))
    for name,body in expected['bodies'].items():
        restored=actual['bodies'].get(name,{})
        checks.extend(pose_checks(body,restored,name))
        checks.append(comparison(name+'.zero_velocity',restored.get('velocity'),[0.]*6,'float32_readback'))
    return checks


def replay_baseline(observed,saved_world):
    """Restore saved body poses; fixed world/robot bases must already match."""
    if any(not p['passed'] for p in base_checks(saved_world,observed)):
        raise ValueError('Replay world/robot base differs from its saved anchor; refusing a base teleport')
    if set(observed['bodies'])!=set(saved_world['bodies']) or observed['local_corners']!=saved_world['local_corners']:
        raise ValueError('Replay protected-body identities or geometry differ')
    result=copy.deepcopy(observed)
    for key in WORLD_FIELDS:result[key]=copy.deepcopy(saved_world[key])
    return result


def replay_model_binding(current,saved):
    a=copy.deepcopy(current);b=copy.deepcopy(saved)
    a.pop('sha256',None);b.pop('sha256',None)
    differences=proof_differences(a,b,'synthetic.replay_robot_model')
    return dict(passed=not differences,current_sha256=digest(current),saved_sha256=digest(saved),
        numeric_rule='arithmetic',differences=differences)


def protected_state(anchor,snapshot):
    """Measure every protected body's corner excursion and velocity, never re-anchor."""
    from cup_geometry import PRIMITIVES
    checks=base_checks(anchor,snapshot);bodies={}
    if set(snapshot['bodies'])!=set(anchor['bodies']):raise ValueError('Protected body identities changed during settle')
    for name,before in anchor['bodies'].items():
        body=snapshot['bodies'][name];velocity=np.asarray(body['velocity'],dtype=float)
        if velocity.shape!=(6,) or not np.isfinite(velocity).all():raise ValueError('Invalid protected-body velocity: '+name)
        if name.startswith('cup_'):
            corners=np.concatenate([np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*
                (np.asarray(shape['size_m'])/2)@rotation(shape['quaternion_wxyz']).T+np.asarray(shape['center_m']) for shape in PRIMITIVES])
        else:corners=np.asarray(anchor['local_corners'][name])
        a=corners@rotation(before['orientation']).T+np.asarray(before['position'])
        b=corners@rotation(body['orientation']).T+np.asarray(body['position'])
        displacement=float(np.linalg.norm(b-a,axis=1).max());linear=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]))
        passed=bool(np.isfinite(displacement) and displacement<=POLICY['protected_corner_motion_max_m'] and
            linear<=POLICY['stationary_linear_m_s'] and angular<=POLICY['stationary_angular_rad_s'])
        bodies[name]=dict(passed=passed,max_corner_motion_m=displacement,linear_m_s=linear,angular_rad_s=angular)
    return dict(passed=all(p['passed'] for p in checks) and all(p['passed'] for p in bodies.values()),bodies=bodies,base_checks=checks)


class ProtectedSettle:
    """One native pre-step callback per solve plus the final solve's readback."""
    def __init__(self,anchor,restored):
        if any(not c['passed'] for c in restore_world_checks(anchor,restored)):
            raise ValueError('Protected settle restore does not match its world anchor')
        self.anchor=world_anchor(anchor);self.anchor_sha256=digest(self.anchor)
        self.boundary=sample_snapshot(restored);self.epoch=restored['epoch'];self.frames=[];self.pending=[];self.failure=None
    def _sample(self,snapshot):
        s=sample_snapshot(snapshot);checks=base_checks(self.anchor,s)
        checks.extend([comparison('settle epoch',s['epoch'],self.epoch),
            comparison('settle held drives',s['drive_targets'],self.boundary['drive_targets'],'target_rad')])
        motion=protected_state(self.anchor,s)
        return dict(snapshot=s,checks=checks,motion=motion,passed=motion['passed'] and all(c['passed'] for c in checks))
    def before_substep(self,step,snapshot):
        sample=self._sample(snapshot);index=len(self.pending)
        sample.update(index=index,source='pre_physics_callback',step_s=float(step))
        sample['checks'].extend([comparison('physics dt',step,PHYSICS_DT,'world_s'),
            comparison('pre-step World progression',snapshot['physics_time_s']-self.boundary['physics_time_s'],index*PHYSICS_DT,'world_s'),
            bound_comparison('native substep clock in held frame',snapshot['sim_ns'],self.boundary['sim_ns'],self.boundary['sim_ns']+STEP_NS,'native_ns'),
            bound_comparison('native substep clock monotonic',snapshot['sim_ns'],self.pending[-1]['snapshot']['sim_ns'] if self.pending else self.boundary['sim_ns'],self.boundary['sim_ns']+STEP_NS,'native_ns')])
        sample['passed']=sample['passed'] and index<PHYSICS_STEPS and all(c['passed'] for c in sample['checks'])
        self.pending.append(sample)
        if not sample['passed']:
            self.failure=dict(stage='pre_physics_callback',sample=sample)
            raise RuntimeError('Protected scene disturbed during settle: '+str(self.failure))
    def finish_frame(self,snapshot):
        final=self._sample(snapshot);errors=world_step_errors(snapshot['physics_time_s'],self.boundary['physics_time_s'])
        if len(self.pending)!=PHYSICS_STEPS:errors.append('Missing/extra protected physics substep samples')
        if not matches(snapshot['sim_ns']-self.boundary['sim_ns'],STEP_NS,'native_ns'):errors.append('Protected frame skipped a native tick')
        frame=dict(version=1,epoch=self.epoch,frame_index=len(self.frames),anchor_sha256=self.anchor_sha256,
            start=copy.deepcopy(self.boundary),substeps=copy.deepcopy(self.pending),final=final,errors=errors,
            passed=not errors and final['passed'] and all(p['passed'] for p in self.pending))
        self.frames.append(frame)
        if not frame['passed']:
            self.failure=dict(stage='final_post_solve',frame=frame)
            raise RuntimeError('Protected settle coverage or motion failed: '+str(self.failure))
        self.boundary=sample_snapshot(snapshot);self.pending=[]
        return copy.deepcopy(frame)
    def certificate(self):
        passed=(not self.failure and not self.pending and len(self.frames)==POLICY['settle_ticks'] and all(f['passed'] for f in self.frames))
        samples=[p for f in self.frames for p in [*f['substeps'],f['final']]]+self.pending
        peaks={}
        for name in self.anchor['bodies']:
            body=[p['motion']['bodies'][name] for p in samples]
            peaks[name]=dict(samples=len(body),max_linear_m_s=max((p['linear_m_s'] for p in body),default=0.),
                max_angular_rad_s=max((p['angular_rad_s'] for p in body),default=0.))
        half=[name for name,p in peaks.items() if p['max_linear_m_s']>POLICY['stationary_linear_m_s']/2 or p['max_angular_rad_s']>POLICY['stationary_angular_rad_s']/2]
        return dict(version=1,epoch=self.epoch,anchor_sha256=self.anchor_sha256,passed=passed,
            frames=len(self.frames),pre_physics_samples=sum(len(f['substeps']) for f in self.frames),
            final_post_solve_samples=len(self.frames),frame_sha256=[digest(f) for f in self.frames],
            velocity_peaks=peaks,half_bound_exceeded=half)
    def evidence(self):
        return dict(certificate=self.certificate(),pending=copy.deepcopy(self.pending),frames=copy.deepcopy(self.frames),failure=copy.deepcopy(self.failure))


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


def setup_error(receipt):
    try:
        report=receipt['recovery_report'];episode_id=receipt['episode_id']
        if report['strategy']!=STRATEGY or report['policy']!=POLICY or report['episode_id']!=episode_id:return 'synthetic recovery identity/policy differs'
        if report['origin']!=POLICY['origin'] or not report['passed']:return 'synthetic start is uncertified'
        observed=report['pre_restore_observed'];require_snapshots([observed])
        baseline=report['baseline']
        expected=(replay_baseline(observed,report['replay_world_anchor']) if 'replay_world_anchor' in report else observed)
        if baseline!=expected or report['world_anchor_sha256']!=digest(world_anchor(baseline)):return 'synthetic world anchor differs'
        if 'replay_world_anchor' in report:
            binding=replay_model_binding(report['model'],report['replay_robot_model'])
            if not binding['passed'] or report['replay_model_binding']!=binding:return 'synthetic replay robot model differs'
        selection=report['selection'];attempts=selection['attempts'];specs=list(candidates(receipt['params']['regrasp']))
        index=selection['spec']['index']
        if len(attempts)!=index+1 or selection['spec']!=specs[index] or any(a['spec']!=specs[i] for i,a in enumerate(attempts)):return 'synthetic candidate order differs'
        if any(a['passed'] for a in attempts[:-1]) or not attempts[-1]['passed'] or not selection['proof']['passed']:return 'synthetic selection was not first passing'
        expected_geometry=geometry(report['request']['pose'],report['request']['bbox'],selection['spec'],report['request']['settings'])
        if proof_differences(selection['geometry'],expected_geometry,'synthetic.geometry') or not any(matches(selection['joints'],q,'arithmetic') for q in expected_geometry['solutions']):return 'synthetic IK geometry differs'
        oracle=StaticClearance(report['model'],baseline,receipt['params']['object'])
        if proof_differences(selection['proof']['static'],oracle.state(selection['joints']),'synthetic.static'):return 'synthetic static proof differs'
        restore=report['restore'];snapshot=restore['snapshot'];command=restore['command_baseline']
        if (restore['epoch']!=report['epoch'] or snapshot['epoch']!=report['epoch'] or command['epoch']!=report['epoch'] or
                restore['epoch']<=baseline['epoch'] or not matches(snapshot['sim_ns'],baseline['sim_ns'],'native_ns') or
                not matches(restore['invalid_through_ns'],snapshot['sim_ns']) or not matches(command['sim_ns'],snapshot['sim_ns']) or
                not matches(snapshot['physics_time_s'],baseline['physics_time_s'],'world_s') or
                not matches(command['physics_time_s'],snapshot['physics_time_s'])):return 'synthetic reset boundary identity differs'
        if (not restore['arm_restored'] or not matches(snapshot['state'],selection['joints'],'target_rad') or
                not matches(snapshot['drive_targets'],selection['joints'],'target_rad') or not matches(snapshot['velocity'],[0.]*6,'float32_readback') or
                not matches(command['drive_targets'],snapshot['drive_targets'])):return 'synthetic restore differs'
        if any(not c['passed'] for c in restore_world_checks(baseline,snapshot)):return 'synthetic restored world differs'
        monitor=ProtectedSettle(baseline,snapshot)
        if report['protected_start']!=dict(epoch=report['epoch'],sim_ns=snapshot['sim_ns'],anchor_sha256=monitor.anchor_sha256):return 'synthetic protected-start identity differs'
        ticks=report['settle_snapshots'];require_snapshots(ticks)
        if len(ticks)!=POLICY['settle_ticks'] or len(report['settle_checks'])!=len(ticks) or any(s['epoch']!=report['epoch'] for s in ticks):return 'synthetic settle identity differs'
        first=ticks[0]['tick_contract']['row'];drive_baseline=first['drive_baseline']
        if (drive_baseline.get('kind')!='restore' or any(drive_baseline.get(k)!=v for k,v in command.items()) or
                not matches(first['physics_drive_targets'],command['drive_targets'],'target_rad') or
                not matches(ticks[0]['sim_ns']-snapshot['sim_ns'],STEP_NS,'native_ns') or
                world_step_errors(ticks[0]['physics_time_s'],snapshot['physics_time_s'])):return 'synthetic first settle clock/drive boundary differs'
        for s,p in zip(ticks,report['settle_checks']):
            frame=s['protected_settle']
            if frame['start']!=monitor.boundary or frame['final']['snapshot']!=sample_snapshot(s):return 'synthetic substep boundary copies differ'
            for sample in frame['substeps']:monitor.before_substep(sample['step_s'],sample['snapshot'])
            expected_frame=monitor.finish_frame(s)
            if proof_differences(frame,expected_frame,'synthetic.substeps'):return 'synthetic protected substep evidence differs'
            # Hash the verified native receipt, not an independently rounded
            # recomputation of its geometric norms on the recorder's NumPy.
            monitor.frames[-1]=copy.deepcopy(frame)
            if not matches(s['drive_targets'],s['tick_contract']['row']['physics_drive_targets']):return 'synthetic settle drive readback differs'
            if not oracle.state(s['state'],s)['passed'] or not protected_state(baseline,s)['passed'] or not p['scene_valid']:return 'synthetic settle is unsafe'
            for key,expected in (('static',oracle.state(s['state'],s)),('target',oracle.target_state(s)),('protected',protected_state(baseline,s))):
                if proof_differences(p[key],expected,'synthetic.settle.'+key):return 'synthetic settle proof differs'
        if not monitor.certificate()['passed'] or report['protected_finish']!=monitor.certificate():return 'synthetic protected settle certificate differs'
        if not matches(report['record_after_sim_ns'],ticks[-1]['sim_ns'],'native_ns'):return 'synthetic recording boundary differs'
        if not report['scene_after_reset']['passed']:return 'synthetic planning scene not synchronized'
    except (KeyError,TypeError,ValueError,IndexError,TickContractError,RuntimeError):return 'synthetic recovery evidence incomplete or protected scene disturbed'
    return None


def receipt_error(receipt,rows):
    error=setup_error(receipt)
    if error:return error
    try:
        report=receipt['recovery_report'];episode_id=receipt['episode_id'];ticks=report['settle_snapshots']
        if not rows or any(r['episode_id']!=episode_id or r['epoch']!=report['epoch'] or r['collection_phase']!='corrective' for r in rows):return 'synthetic corrective row identity differs'
        if row_errors(rows[0],ticks[-1]['tick_contract']['row']):return 'synthetic handover tick contract failed'
        if any(not c['passed'] for c in pose_checks(ticks[-1]['robot_base_world_pose'],rows[0]['robot_base_world_pose'],'first corrective fixed robot base')) or rows[0].get('robot_base_pose_source')!=ticks[-1]['robot_base_pose_source']:
            return 'synthetic first corrective fixed base differs'
        if any(row_errors(r,rows[i-1] if i else None) for i,r in enumerate(rows)):return 'synthetic corrective tick contract failed'
        if any(native_frame(r)['sim_ns']<=report['record_after_sim_ns'] for r in rows):return 'synthetic setup frames entered dataset'
    except (KeyError,TypeError,ValueError,IndexError,TickContractError):return 'synthetic corrective evidence incomplete'
    return None
