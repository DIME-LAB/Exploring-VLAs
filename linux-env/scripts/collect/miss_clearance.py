"""Explicit no-contact corridor for an executed deliberate miss.

The target is omitted from MoveIt because the clean corrective expert aims to
re-grasp it. This additional oracle checks physical miss motion against that
same target, including the moving jaw and the complete close/open sweeps.
"""
import copy,json,math
import numpy as np
from finger_geometry import FingerModel,body_box,separation,rotation
from tick_contract import digest,frame_digest,snapshot_contracts,STEP_NS
from comparisons import comparison
from contract import JOINTS

POLICY=dict(version=3,executed_margin_m=.001,clean_open_margin_m=.0002,
    geometric_error_m=.00002,refinement_floor_motion_m=.00025,max_segment_checks=50000,
    target_corner_motion_max_m=.0002,stationary_linear_m_s=.0001,stationary_angular_rad_s=.005,
    step_s=1/30+1e-6,joint_numeric_allowance_rad=1e-6,reachable_numeric_allowance_rad=5e-6,settled_velocity_rad_s=.0125,
    velocity_caps_rad_s=dict(regrasp_miss_open=.5,regrasp_miss_approach=.1,
        regrasp_miss_descent=.05,regrasp_miss_close=.05,regrasp_miss_lift=.05,regrasp_reopen=.05),
    joint_point_radius_bounds_m=[.55,.55,.40,.25,.15,.10],
    source='held USD enabled finger boxes, fixed-base named-joint kinematics',
    clean_contact='clean close/lift can intentionally aim at the target; never executed on the miss')

def bind_target_offsets(model,offsets):
    FingerModel(model)  # Verify the transport's geometry/finger-offset identity first.
    value=copy.deepcopy(model);value.pop('sha256',None);value['effective_block_offsets']=copy.deepcopy(offsets)
    value['sha256']=digest(value)
    return value


def planning_margin(stage):
    phase=dict(open='regrasp_miss_open',approach_executed='regrasp_miss_approach',descent_executed='regrasp_miss_descent',
        close='regrasp_miss_close',lift_executed='regrasp_miss_lift',reopen='regrasp_reopen')[stage]
    cap=velocity_cap(phase)
    return POLICY['executed_margin_m']+sum(r*(v*POLICY['step_s']+POLICY['reachable_numeric_allowance_rad']) for r,v in zip(POLICY['joint_point_radius_bounds_m'],cap['values_rad_s']))


def velocity_cap(phase,stationary_target=None):
    if phase not in POLICY['velocity_caps_rad_s']:raise ValueError('Unknown deliberate-miss phase: '+str(phase))
    result=dict(on=True,phase=phase,values_rad_s=[POLICY['velocity_caps_rad_s'][phase]]*6,policy_version=POLICY['version'])
    if stationary_target is not None:result['stationary_target']=stationary_target
    return result


class MissClearance:
    def __init__(self,model,snapshot,name):
        if not isinstance(model,dict) or model.get('version')!=1:raise ValueError('Versioned native finger model is required')
        self.model=FingerModel(model)
        radii=self.model.radius_bounds()
        if any(a>b for a,b in zip(radii,POLICY['joint_point_radius_bounds_m'])):raise ValueError('Finger model exceeds declared motion radius bounds')
        self.spec=dict(model=model,baseline=snapshot,target=name,policy=POLICY,model_radius_bounds_m=radii)
        self.name=name;self.baseline=snapshot;self.body=snapshot['bodies'][name];self.corners=snapshot['local_corners'][name]
        self.box=body_box(self.body,self.corners);self.base=snapshot['robot_base_world_pose']
        self.model_sha256=digest(model)
        self.offsets=model.get('effective_finger_offsets')
        if self.offsets is None:raise ValueError('Effective native finger offsets are required')
        contact=[];rest=[]
        for value in self.offsets.values():
            contact+=value['contact_m'];rest+=value['rest_m']
            if len(value['contact_m'])!=4 or len(value['rest_m'])!=4:raise ValueError('Four shapes per finger are required')
        if len(contact)!=8 or not np.isfinite(contact+rest).all() or min(contact)<0 or max(abs(v) for v in rest)>1e-9:raise ValueError('Unverified finger contact/rest offsets')
        block=model.get('effective_block_offsets')
        if not block or not block['contact_m'] or len(block['contact_m'])!=len(block['rest_m']) or not np.isfinite(block['contact_m']+block['rest_m']).all():raise ValueError('Effective target offsets are required')
        if min(block['contact_m'])<0 or max(abs(v) for v in block['rest_m'])>1e-9:raise ValueError('Unverified target contact/rest offsets')
        skin=max(contact)+max(block['contact_m'])+POLICY['geometric_error_m']
        if POLICY['executed_margin_m']<skin or POLICY['clean_open_margin_m']+POLICY['geometric_error_m']<skin:raise ValueError('Miss margin does not cover effective pair offsets')

    def state(self,q,margin=None,body=None,base=None):
        box=self.box if body is None else body_box(body,self.corners)
        shapes=self.model.boxes(q,self.base if base is None else base)
        path,gap=min(((s['path'],separation(s,box)) for s in shapes),key=lambda p:p[1])
        required=POLICY['executed_margin_m'] if margin is None else margin
        return dict(passed=gap>=required+POLICY['geometric_error_m'],finger=path,target=self.name,
            separation_m=gap,required_margin_m=required,geometric_error_m=POLICY['geometric_error_m'],joints=list(q))

    def joint_box(self,low,high,margin=None,require_epoch=lambda:None,body=None,base=None):
        """Certify every combination of joint positions, not one shared fraction.

        A fixed separating axis at each cell centre loses at most the summed
        per-joint point-displacement bound anywhere in that cell. Recursively
        split the largest weighted coordinate; finite unresolved cells refuse.
        """
        low=np.asarray(low,dtype=float);high=np.asarray(high,dtype=float)
        if low.shape!=(6,) or high.shape!=(6,) or not np.isfinite([low,high]).all() or np.any(high<low):raise ValueError('Invalid independent-joint box')
        required=POLICY['executed_margin_m'] if margin is None else margin
        radii=np.asarray(POLICY['joint_point_radius_bounds_m'])
        checks=0;minimum=math.inf;certified=math.inf;max_radius=0.;failure=None
        def certify(lo,hi):
            nonlocal checks,minimum,certified,max_radius,failure
            require_epoch()
            if checks>=POLICY['max_segment_checks']:raise RuntimeError('Miss clearance geometric capacity exceeded')
            middle=(lo+hi)/2;proof=self.state(middle,required,body,base);checks+=1
            minimum=min(minimum,proof['separation_m'])
            weighted=(hi-lo)*radii/2;radius=float(sum(weighted))
            if not proof['passed']:
                failure=dict(reason='independent-joint posture violates clearance',proof=proof);return False
            gap=proof['separation_m']-radius
            if gap>=required+POLICY['geometric_error_m']:
                certified=min(certified,gap);max_radius=max(max_radius,radius);return True
            if radius<=POLICY['refinement_floor_motion_m']:
                failure=dict(reason='independent-joint box unresolved at finite resolution',
                    center_separation_m=proof['separation_m'],motion_bound_m=radius);return False
            index=int(np.argmax(weighted));left=hi.copy();left[index]=middle[index];right=lo.copy();right[index]=middle[index]
            return certify(lo,left) and certify(right,hi)
        passed=certify(low,high)
        return dict(passed=passed,kind='independent_joint_box',low_rad=low.tolist(),high_rad=high.tolist(),
            checks=checks,minimum_separation_m=minimum,required_margin_m=required,
            minimum_certified_separation_m=certified if passed else None,
            inter_sample_motion_bound_m=max_radius,rejection=failure)

    def segment(self,start,end,margin=None,require_epoch=lambda:None,body=None,base=None):
        # Planning checks include every independent progress combination, too.
        return self.joint_box(np.minimum(start,end),np.maximum(start,end),margin,require_epoch,body,base)

    def nominal_path(self,start,end,margin=None,require_epoch=lambda:None,body=None,base=None):
        # Nominal planning preflight only; actual stepping additionally requires
        # the native-cap reachable-box certificate in command().
        motion=sum(abs(a-b)*r for a,b,r in zip(start,end,POLICY['joint_point_radius_bounds_m']))
        required=POLICY['executed_margin_m'] if margin is None else margin
        checks=0;minimum=math.inf;certified=math.inf;max_radius=0.;failure=None
        def sample(alpha):
            nonlocal checks,minimum
            require_epoch()
            if checks>=POLICY['max_segment_checks']:raise RuntimeError('Miss clearance geometric capacity exceeded')
            q=[a+(b-a)*alpha for a,b in zip(start,end)]
            proof=self.state(q,required,body,base);checks+=1;minimum=min(minimum,proof['separation_m'])
            return proof
        def certify(lo,hi,left,right):
            nonlocal certified,max_radius,failure
            for alpha,proof in ((lo,left),(hi,right)):
                if not proof['passed']:
                    failure=dict(alpha=alpha,proof=proof,reason='finger inside required clearance');return False
            radius=motion*(hi-lo)/2
            gap=min(left['separation_m'],right['separation_m'])-radius
            if gap>=required+POLICY['geometric_error_m']:
                certified=min(certified,gap);max_radius=max(max_radius,radius);return True
            if radius<=POLICY['refinement_floor_motion_m']:
                failure=dict(alpha=(lo+hi)/2,reason='margin not certified at finite geometric resolution',
                    nearest_endpoint_separation_m=min(left['separation_m'],right['separation_m']),motion_bound_m=radius)
                return False
            middle=(lo+hi)/2;proof=sample(middle)
            return certify(lo,middle,left,proof) and certify(middle,hi,proof,right)
        left=sample(0.);right=sample(1.);passed=certify(0.,1.,left,right)
        return dict(passed=passed,checks=checks,minimum_separation_m=minimum,required_margin_m=required,
            minimum_certified_separation_m=certified if passed else None,
            inter_sample_motion_bound_m=max_radius,rejection=failure)

    def displacement(self,snapshot):
        body=snapshot['bodies'][self.name];corners=np.asarray(self.corners)
        initial=corners@rotation(self.body['orientation']).T+np.asarray(self.body['position'])
        current=corners@rotation(body['orientation']).T+np.asarray(body['position'])
        maximum=float(np.linalg.norm(current-initial,axis=1).max())
        return dict(passed=maximum<=POLICY['target_corner_motion_max_m'],max_corner_motion_m=maximum,
            tolerance_m=POLICY['target_corner_motion_max_m'],baseline_sim_ns=self.baseline['sim_ns'],sim_ns=snapshot['sim_ns'])

    def command(self,state,previous,target,snapshot,require_epoch=lambda:None):
        if snapshot['epoch']!=self.baseline['epoch'] or snapshot['sim_ns']<self.baseline['sim_ns']:
            raise ValueError('Miss observation belongs to another epoch or predates its anchor')
        body_velocity=snapshot['bodies'][self.name].get('velocity')
        if body_velocity is None or len(body_velocity)!=6 or not np.isfinite(body_velocity).all():
            return dict(passed=False,reason='native measured target velocity is required',target=self.name)
        stationary=dict(linear_m_s=float(np.linalg.norm(body_velocity[:3])),angular_rad_s=float(np.linalg.norm(body_velocity[3:])),
            linear_limit_m_s=POLICY['stationary_linear_m_s'],angular_limit_rad_s=POLICY['stationary_angular_rad_s'])
        if stationary['linear_m_s']>POLICY['stationary_linear_m_s'] or stationary['angular_rad_s']>POLICY['stationary_angular_rad_s']:
            return dict(passed=False,reason='guarded target is not stationary',target=self.name,stationary_target=stationary)
        movement=self.displacement(snapshot)
        body=snapshot['bodies'][self.name];base=snapshot['robot_base_world_pose'];paths=[]
        identity=dict(epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],target=list(target),model_sha256=self.model_sha256,
            measured_target_body=copy.deepcopy(body),measured_robot_base_pose=copy.deepcopy(base))
        if not movement['passed']:return dict(identity,passed=False,reason='target moved during deliberate miss',motion=movement)
        velocity=snapshot.get('velocity')
        phase=snapshot.get('miss_phase','regrasp_miss_descent')
        cap=velocity_cap(phase,self.name)
        identity['velocity_cap']=cap;identity['stationary_target']=stationary
        if velocity is None or len(velocity)!=6 or not np.isfinite(velocity).all():
            return dict(identity,passed=False,reason='native measured velocity is required',motion=movement)
        if any(abs(v)>lim+POLICY['joint_numeric_allowance_rad'] for v,lim in zip(velocity,cap['values_rad_s'])):
            return dict(identity,passed=False,reason='measured velocity exceeds recovery cap before application',measured_velocity_rad_s=list(velocity),motion=movement)
        # This geometric box is conditional on a native displacement bound.
        # The live smoke must measure solver residuals; parameter readback and
        # CPU geometry alone do not establish the physical bound. Native
        # substep monitoring rejects any observed constraint/box violation.
        half=np.asarray(cap['values_rad_s'])*POLICY['step_s']+POLICY['reachable_numeric_allowance_rad']
        reach=self.joint_box(np.asarray(state)-half,np.asarray(state)+half,require_epoch=require_epoch,body=body,base=base)
        identity['reachable_box']=reach;identity['measured_velocity_rad_s']=list(velocity)
        if not reach['passed']:return dict(identity,passed=False,reason='next-tick independent-joint motion is not clear',motion=movement)
        endpoint=self.state(target,body=body,base=base)
        identity['command_endpoint']=endpoint
        if not endpoint['passed']:return dict(identity,passed=False,reason='command endpoint violates target clearance',motion=movement)
        return dict(identity,passed=True,motion=movement)



def proof_differences(recorded,actual,path='miss_target_clearance'):
    """Derived geometry crosses NumPy/Python environments during export."""
    if isinstance(recorded,dict) and isinstance(actual,dict) and set(recorded)==set(actual):
        return [e for key in actual for e in proof_differences(recorded[key],actual[key],path+'.'+key)]
    if isinstance(recorded,list) and isinstance(actual,list) and len(recorded)==len(actual):
        return [e for i,(a,b) in enumerate(zip(recorded,actual)) for e in proof_differences(a,b,path+'['+str(i)+']')]
    copied=any(k in path for k in ('measured_target_body','measured_robot_base_pose'))
    rule='arithmetic' if not copied and (isinstance(recorded,float) or isinstance(actual,float)) else 'copy'
    result=comparison(path,recorded,actual,rule)
    return [] if result['passed'] else [result]

def receipt_error(receipt,rows):
    """Recheck the executed miss against its native model and pre-miss anchor."""
    try:
        report=receipt['recovery_report'];spec=report['clearance'];baseline=spec['baseline'];end=report['clearance_end_sim_ns']
        if report.get('frame_hash_policy')!='tick row excluding separately audited wall_profile':return 'Miss frame hash policy differs'
        contract=snapshot_contracts([baseline])
        if not contract['passed']:return 'Miss anchor contract differs: '+json.dumps(contract['comparisons'])
        if spec['policy']!=POLICY or spec['target']!=receipt['params']['object']:return 'miss target-clearance policy differs'
        # The reset-established episode anchor cannot be replaced after descent.
        from comparisons import matches,native_frame
        initial=receipt['episode_start_physical']
        if any(not matches(baseline[k],initial[k]) for k in ('epoch','sim_ns','state','bodies','robot_base_world_pose')):return 'miss anchor differs from initial episode snapshot'
        fresh=report['fresh_offsets']
        if fresh['epoch']!=baseline['epoch'] or not matches(fresh['sim_ns'],baseline['sim_ns'],'native_ns') or fresh['finger_offsets']!=spec['model']['effective_finger_offsets'] or fresh['block_offsets']!=spec['model']['effective_block_offsets']:
            return 'guarded episode fresh native offsets differ'
        guard=MissClearance(spec['model'],baseline,spec['target'])
        if report.get('velocity_cap_restore')!=dict(on=False,values_rad_s=None):return 'miss velocity cap was not restored before corrective recovery'
        kickoffs=report['kickoffs']
        phases=[item['proof']['velocity_cap']['phase'] for item in kickoffs]
        if phases!=list(POLICY['velocity_caps_rad_s']):return 'miss kickoff stages differ'
        for item in kickoffs:
            snapshot=item['snapshot'];proof=item['proof'];ack=item['ack'];cap=proof['velocity_cap']
            if item.get('advance',True) is False and (ack.get('held') is not True or ack.get('advance') is not None):
                return 'no-op miss kickoff advanced physics or lacks a held acknowledgement'
            actual=guard.command(snapshot['state'],snapshot['drive_targets'],snapshot['drive_targets'],dict(snapshot,miss_phase=cap['phase']))
            if not actual['passed'] or proof_differences(proof,actual):return 'miss kickoff clearance differs'
            if ack.get('command_miss_velocity_cap')!=cap or ack.get('epoch')!=baseline['epoch'] or not matches(ack.get('observation_stamp_ns'),snapshot['sim_ns'],'stamp_ns'):return 'miss kickoff cap was not acknowledged'
        window=[r for r in rows if baseline['sim_ns']<=native_frame(r)['sim_ns']<=end]
        if not window or [frame_digest(r) for r in window]!=report['all_miss_frame_sha256s']:return 'whole miss row hashes differ'
        first_delta=native_frame(window[0])['sim_ns']-baseline['sim_ns']
        if not (matches(first_delta,0) or matches(first_delta,STEP_NS,'native_ns')):return 'First miss row skipped its held anchor'
        if not matches(native_frame(window[-1])['sim_ns'],end):return 'miss final tick is missing'
        for row in window:
            if 'miss_target_clearance' not in row.get('execution_geometry',{}):
                if matches(native_frame(row)['sim_ns'],baseline['sim_ns']) and not row['phase'].startswith('regrasp_'):continue
                return 'miss tick lacks physical clearance proof'
            proof=row['execution_geometry']['miss_target_clearance']
            actual=guard.command(row['state'],row['physics_drive_targets'],row['executed_action'],
                dict(native_frame(row),robot_base_world_pose=row['robot_base_world_pose'],velocity=row['physics_velocity'],miss_phase=row['command_miss_velocity_cap']['phase']))
            if not proof['passed'] or not actual['passed']:return 'recorded miss target-clearance did not pass'
            if row.get('command_miss_velocity_cap')!=actual['velocity_cap']:return 'miss velocity cap was not acknowledged'
            differences=proof_differences(proof,actual)
            if differences:return 'recorded miss target-clearance proof differs: '+json.dumps(differences[:3])
    except (KeyError,TypeError,ValueError,IndexError):return 'miss target-clearance evidence is incomplete'
    return None


def residual_summary(rows,faults=()):
    """Measured excess, not an offline proof of the coupled native solver."""
    samples=[s for row in rows for s in row.get('measured_miss_velocity_cap',{}).get('substep_samples',[])]
    samples.extend(s for fault in faults for s in fault.get('samples',[]))
    result=dict(sample_count=len(samples),joint_names=list(JOINTS),certificate_pad_rad=POLICY['reachable_numeric_allowance_rad'],
        interpretation='per-substep samples; pre-step callbacks observe the preceding integration, final_snapshot closes the frame',
        physical_qualification='requires owned live smoke; any violation is terminal')
    for field in ('local_excess_rad','frame_excess_over_cap_rad','certificate_excess_rad'):
        values=[s[field] for s in samples if len(s.get(field,[]))==6 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in s[field])]
        result['max_'+field]=[max(v[i] for v in values) for i in range(6)] if values else None
    result['local_budget_rad']=POLICY['joint_numeric_allowance_rad']
    result['observed_local_budget_violation']=any(v>POLICY['joint_numeric_allowance_rad'] for v in result['max_local_excess_rad'] or [])
    result['observed_certificate_violation']=any(v>0 for v in result['max_certificate_excess_rad'] or [])
    return result
