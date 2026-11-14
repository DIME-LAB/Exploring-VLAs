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

POLICY=dict(version=1,executed_margin_m=.001,clean_open_margin_m=.0002,
    geometric_error_m=.00002,refinement_floor_motion_m=.00025,max_segment_checks=50000,
    target_corner_motion_max_m=.0002,
    joint_point_radius_bounds_m=[.55,.55,.40,.25,.15,.10],
    source='held USD enabled finger boxes, fixed-base named-joint kinematics',
    clean_contact='clean close/lift can intentionally aim at the target; never executed on the miss')


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

    def state(self,q,margin=None,body=None,base=None):
        box=self.box if body is None else body_box(body,self.corners)
        shapes=self.model.boxes(q,self.base if base is None else base)
        path,gap=min(((s['path'],separation(s,box)) for s in shapes),key=lambda p:p[1])
        required=POLICY['executed_margin_m'] if margin is None else margin
        return dict(passed=gap>=required+POLICY['geometric_error_m'],finger=path,target=self.name,
            separation_m=gap,required_margin_m=required,geometric_error_m=POLICY['geometric_error_m'],joints=list(q))

    def segment(self,start,end,margin=None,require_epoch=lambda:None,body=None,base=None):
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
        movement=self.displacement(snapshot)
        body=snapshot['bodies'][self.name];base=snapshot['robot_base_world_pose'];paths=[]
        identity=dict(epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],target=list(target),model_sha256=self.model_sha256,
            measured_target_body=copy.deepcopy(body),measured_robot_base_pose=copy.deepcopy(base))
        if not movement['passed']:return dict(identity,passed=False,reason='target moved during deliberate miss',motion=movement)
        for source in (state,previous if previous is not None else state):
            p=self.segment(source,target,require_epoch=require_epoch,body=body,base=base);paths.append(p)
            if not p['passed']:return dict(identity,passed=False,reason='finger violates deliberate-miss target clearance',paths=paths,motion=movement)
        return dict(identity,passed=True,paths=paths,motion=movement)


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
        guard=MissClearance(spec['model'],baseline,spec['target'])
        window=[r for r in rows if baseline['sim_ns']<=native_frame(r)['sim_ns']<=end]
        if not window or [frame_digest(r) for r in window]!=report['all_miss_frame_sha256s']:return 'whole miss row hashes differ'
        first_delta=native_frame(window[0])['sim_ns']-baseline['sim_ns']
        if not (matches(first_delta,0) or matches(first_delta,STEP_NS,'native_ns')):return 'First miss row skipped its held anchor'
        if not matches(native_frame(window[-1])['sim_ns'],end):return 'miss final tick is missing'
        for row in window:
            if not row['phase'].startswith('regrasp_'):continue
            proof=row['execution_geometry']['miss_target_clearance']
            actual=guard.command(row['state'],row['physics_drive_targets'],row['executed_action'],
                dict(native_frame(row),robot_base_world_pose=row['robot_base_world_pose']))
            if not proof['passed'] or not actual['passed']:return 'recorded miss target-clearance did not pass'
            differences=proof_differences(proof,actual)
            if differences:return 'recorded miss target-clearance proof differs: '+json.dumps(differences[:3])
    except (KeyError,TypeError,ValueError,IndexError):return 'miss target-clearance evidence is incomplete'
    return None
