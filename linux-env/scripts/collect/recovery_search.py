"""Bounded recovery candidates, complete motion preflight and planning allowance."""
import math,threading,time
from contextlib import contextmanager
from command_geometry import QueryBudget,QUERY_POLICY,checked_segment
from contract import JOINTS
from recovery import recovery_geometry
from tick_contract import digest

SEARCH_POLICY=dict(version=2,max_candidates=16,max_queries=16000,
    directions_rad=[0.,math.pi,math.pi/2,-math.pi/2],magnitudes_m=[.02,.03,.04],
    joint_spacing_rad=.01)


def candidate_specs(requested):
    distance=requested['signed_offset_m']
    if not .02<=abs(distance)<=.04:raise ValueError('Recovery request outside 2–4 cm')
    magnitudes=list(dict.fromkeys([abs(distance),*SEARCH_POLICY['magnitudes_m']]))
    return [dict(requested,signed_offset_m=math.copysign(m,distance),direction_offset_rad=a)
            for m in magnitudes for a in SEARCH_POLICY['directions_rad']]


class PlanningWallBudget:
    """Wall computation has its own bound; it never adds simulated motion time."""
    def __init__(self,wall=time.monotonic):
        self.wall=wall;self.lock=threading.Lock();self.completed=0.;self.active=None
    @contextmanager
    def reserve(self,seconds):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid planning wall capacity')
        start=self.wall()
        with self.lock:
            if self.active is not None:raise RuntimeError('Overlapping planning reservations')
            self.active=seconds
        try:yield
        finally:
            elapsed=self.wall()-start
            with self.lock:self.completed+=elapsed;self.active=None
            if elapsed>seconds:raise TimeoutError('Recovery planning exceeded declared query capacity')
    def allowance(self):
        with self.lock:return self.completed+(self.active or 0.)


def preflight(geometry,state,opening,closing,check,limits,require_epoch,budget,clearance):
    selected={};endpoint_checks=[]
    for stage,solutions in geometry['solutions'].items():
        for target in solutions:
            require_epoch()
            jaw=closing if stage=='miss_lift' else opening
            q=[target[n] for n in JOINTS[:5]]+[jaw]
            proof=checked_segment(q,q,check,limits,budget)
            if proof['passed']:
                target_proof=clearance.state(q,margin=None if stage.startswith('miss_') else .0002)
                proof['target_clearance']=target_proof
                proof['passed']=target_proof['passed']
            endpoint_checks.append(dict(stage=stage,target=q,proof=proof))
            if proof['passed']:selected[stage]=dict(target);break
        if stage not in selected:return None,dict(passed=False,stage=stage,endpoints=endpoint_checks)
    def q(key,jaw):return [selected[key][n] for n in JOINTS[:5]]+[jaw]
    start=list(state[:5])+[opening]
    paths=[('open',list(state),start),('approach_clean',start,q('true_approach',opening)),
           ('approach_executed',start,q('miss_approach',opening)),
           ('descent_clean',q('miss_approach',opening),q('true_final',opening)),
           ('descent_executed',q('miss_approach',opening),q('miss_final',opening)),
           ('close',q('miss_final',opening),q('miss_final',closing)),
           ('lift_clean',q('miss_final',closing),q('true_final',closing)),
           ('lift_executed',q('miss_final',closing),q('miss_lift',closing)),
           ('reopen',q('miss_lift',closing),q('miss_lift',opening))]
    checked=[]
    for stage,start,target in paths:
        require_epoch()
        proof=checked_segment(start,target,check,limits,budget)
        # A clean closed-jaw correction intentionally aims to grasp the block;
        # it is a label, never the physical trajectory of the miss.
        if proof['passed'] and stage!='lift_clean':
            target_proof=clearance.segment(start,target,
                margin=.0002 if stage in ('approach_clean','descent_clean') else None,require_epoch=require_epoch)
            proof['target_clearance']=target_proof;proof['passed']=target_proof['passed']
        checked.append(dict(stage=stage,proof=proof))
        if not proof['passed']:return None,dict(passed=False,stage=stage,endpoints=endpoint_checks,paths=checked)
    return selected,dict(passed=True,endpoints=endpoint_checks,paths=checked)


def choose(pose,bbox,requested,state,check,limits,require_epoch,settings,attempts,clearance):
    if clearance is None:raise ValueError('Explicit target clearance is required for recovery')
    budget=QueryBudget(SEARCH_POLICY['max_queries'])
    for index,spec in enumerate(candidate_specs(requested)):
        require_epoch()
        attempt=dict(index=index,spec=spec,passed=False);attempts.append(attempt)
        def validator(candidate):
            opening,closing=candidate['opening_rad'],candidate['closing_rad']
            def guarded(q):require_epoch();answer=check(q);require_epoch();return answer
            selected,proof=preflight(candidate,state,opening,closing,guarded,limits,require_epoch,budget,clearance)
            return dict(passed=selected is not None,targets=selected,proof=proof)
        geometry=recovery_geometry(pose,bbox,spec,candidate_validator=validator,**settings)
        attempt['geometry_sha256']=digest(geometry)
        attempt['yaw_attempts']=geometry['grasp_spec']['attempts']
        validated_yaws=[a['validation'] for a in attempt['yaw_attempts'] if 'validation' in a]
        if validated_yaws:attempt['proof']=validated_yaws[-1]['proof']
        attempt['query_timing']=budget.summary()
        if geometry['yaw'] is None:
            attempt['rejection']=dict(reason='no IK/collision/target-clearance-valid equivalent',debug=geometry['debug']);continue
        selection=geometry['grasp_spec']['validation'];proof=selection['proof'];selected=selection['targets']
        attempt.update(passed=True,proof=proof,query_timing=budget.summary())
        require_epoch()
        return dict(geometry=geometry,targets=selected,index=index,spec=spec,
                    opening_rad=geometry['grasp_spec']['opening_rad'],closing_rad=geometry['grasp_spec']['closing_rad'],
                    geometry_sha256=digest(geometry),query_timing=budget.summary(),policy=SEARCH_POLICY)
    return None


def selection_error(report,requested):
    try:
        selected=report['selection'];attempts=report['candidate_attempts'];specs=candidate_specs(requested)
        index=selected['index']
        if len(attempts)!=index+1 or [a['index'] for a in attempts]!=list(range(index+1)):return 'recovery search order differs'
        if any(a['passed'] for a in attempts[:-1]) or not attempts[-1]['passed']:return 'recovery did not use first passing candidate'
        if selected['spec']!=specs[index] or any(a['spec']!=specs[a['index']] for a in attempts):return 'recovery selection differs from seeded candidates'
        geometry=report['geometry']
        if digest(geometry)!=selected['geometry_sha256'] or attempts[-1]['geometry_sha256']!=selected['geometry_sha256']:return 'recovery selected geometry differs'
        angle=geometry['true_geometry']['yaw']+selected['spec']['direction_offset_rad'];distance=selected['spec']['signed_offset_m']
        expected=[-math.sin(angle)*distance,math.cos(angle)*distance,0.]
        if any(abs(a-b)>1e-12 for a,b in zip(expected,geometry['offset_m'])):return 'recovery selected offset differs'
        if not attempts[-1]['proof']['passed']:return 'recovery selection lacks complete preflight'
    except (KeyError,TypeError,ValueError,IndexError):return 'recovery selection evidence is incomplete'
    return None
