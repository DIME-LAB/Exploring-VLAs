"""Target-free MoveIt answers cannot approve an obstructed deliberate miss."""
import copy,itertools,json,math,unittest
from pathlib import Path
import numpy as np
from contract import JOINTS,parameters
from finger_geometry import FingerModel,finger_boxes,separation
from miss_clearance import MissClearance,POLICY,proof_differences
from recovery import recovery_geometry
from recovery_search import choose,candidate_specs
from so_arm101_control.grasp_geometry import JOINT_LIMITS

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('descent-obstructions.json').read_text())
MODEL=dict(effective_finger_offsets={name:dict(contact_m=[.0001]*4,rest_m=[0.]*4) for name in ('gripper_link','moving_jaw_so101_v1_link')},effective_block_offsets=dict(contact_m=[.0001],rest_m=[0.]),version=1,root_body='/so101_new_calib/base_link',joints=FIXTURE['joints'],robot_boxes=FIXTURE['robot_boxes'])


def context(seed):
    p=seed['pose'];b=seed['bbox'];name=seed['object']
    body=dict(position=[p[k] for k in ('x','y','z')],orientation=[p[k] for k in ('qw','qx','qy','qz')])
    corners=list(map(list,itertools.product(*[(-b[k]/2,b[k]/2) for k in ('sx','sy','sz')])))
    return dict(epoch=1,sim_ns=10**9,velocity=[0.]*6,bodies={name:body},local_corners={name:corners},
        robot_base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))


class MissClearanceTests(unittest.TestCase):
    def test_reviewed_original_and_small_fallbacks_cannot_pass_target_free_oracle(self):
        seed=next(s for s in FIXTURE['seeds'] if s['seed']==350007)
        requested=parameters(340000,'red',regrasp=True)['regrasp'];p=seed['params']
        settings=dict(current_wrist_roll=-1.5220963,approach_height=p['approach_height_m'],approach_offset=p['approach_offset_m'])
        guard=MissClearance(MODEL,context(seed),seed['object'])
        original=recovery_geometry(seed['pose'],seed['bbox'],requested,**settings)
        q=[original['solutions']['miss_final'][0][n] for n in JOINTS[:5]]+[original['grasp_spec']['opening_rad']]
        self.assertLess(guard.state(q)['separation_m'],-.0024)
        self.assertFalse(guard.state(q)['passed'])
        for spec in candidate_specs(requested):
            if abs(spec['signed_offset_m'])!=.02:continue
            g=recovery_geometry(seed['pose'],seed['bbox'],spec,**settings)
            if g['yaw'] is None:continue
            q=[g['solutions']['miss_final'][0][n] for n in JOINTS[:5]]+[g['grasp_spec']['opening_rad']]
            if guard.state(q)['separation_m']<POLICY['executed_margin_m']:self.assertFalse(guard.state(q)['passed'])
        attempts=[];home=[.015,-.259,.259,math.pi/2,-1.522,.1132]
        chosen=choose(seed['pose'],seed['bbox'],requested,home,lambda q:(True,[]),
            [JOINT_LIMITS[n] for n in JOINTS],lambda:None,settings,attempts,guard)
        self.assertIsNotNone(chosen)
        self.assertFalse(attempts[0]['passed'])
        self.assertTrue(chosen['geometry']['grasp_spec']['validation']['passed'])
        for path in chosen['geometry']['grasp_spec']['validation']['proof']['paths']:
            if path['stage']!='lift_clean':self.assertTrue(path['proof']['target_clearance']['passed'])
        self.assertEqual(len(chosen['geometry']['grasp_spec']['validation']['proof']['paths']),9)

    def test_actual_smoke_miss_is_rejected_before_dispatch_by_the_real_command_entrance(self):
        from dart import TargetCandidate
        from command_geometry import CommandCollision
        from test_review_regressions import ReviewRegressions
        live=json.loads(Path(__file__).with_name('fixtures').joinpath('regrasp-obstruction-340000.json').read_text())
        name=next(iter(live['baseline']['bodies']))
        # Retain physical geometry while using the fake runtime's own epoch/tick.
        baseline=dict(live['baseline'],epoch=1,sim_ns=10**9);guard=MissClearance(MODEL,baseline,name)
        q=live['target_endpoint'];self.assertLess(guard.state(q)['separation_m'],-.004)
        c,_,_=ReviewRegressions().collector();c.active_miss_clearance=(1,guard)
        # MoveIt says valid because the target was deliberately removed.
        c.node._check_state_valid_with_contacts=lambda target:(True,[])
        candidate=TargetCandidate(1,10**9,tuple(q),(0.,)*5,(0.,)*6,(0.,)*6,tuple(q))
        snapshot=dict(copy.deepcopy(baseline),velocity=[0.]*6)
        with self.assertRaises(CommandCollision):c.validate_command(candidate,q,q,snapshot)
        self.assertEqual(c.events,[])  # Neither command publication nor native stepping.
        c.timeline(False);self.assertIsNone(c.active_miss_clearance)

    def test_computed_proof_roundoff_uses_declared_numeric_rule_but_identity_is_exact(self):
        a=dict(passed=True,epoch=1,separation_m=.0023456789,target=[.1]*6,model_sha256='a'*64)
        b=copy.deepcopy(a);b['separation_m']+=1e-12
        self.assertEqual(proof_differences(a,b),[])
        b['separation_m']+=1e-5;self.assertTrue(proof_differences(a,b))
        b=copy.deepcopy(a);b['epoch']=2
        self.assertEqual(proof_differences(a,b)[0]['rule'],'copy')
        b=copy.deepcopy(a);b['passed']=False;self.assertTrue(proof_differences(a,b))

    def test_motion_radius_bounds_cover_the_authored_joint_tree(self):
        actual=FingerModel(MODEL).radius_bounds()
        self.assertTrue(all(a<=b for a,b in zip(actual,POLICY['joint_point_radius_bounds_m'])))
        # Unrecognized longer fingers cannot silently reuse a shorter sweep bound.
        bad=copy.deepcopy(MODEL);shape=next(s for s in bad['robot_boxes'] if '/gripper_link/' in s['path']);shape['low'][2]-=2.
        with self.assertRaisesRegex(ValueError,'radius bounds'):MissClearance(bad,context(FIXTURE['seeds'][0]),FIXTURE['seeds'][0]['object'])

    def test_target_motion_is_measured_from_before_approach_not_after_descent(self):
        seed=FIXTURE['seeds'][0];s=context(seed);guard=MissClearance(MODEL,s,seed['object'])
        moved=copy.deepcopy(s);moved['bodies'][seed['object']]['position'][0]+=.002
        self.assertFalse(guard.displacement(moved)['passed'])
        moved['sim_ns']+=33333333  # Staying still after the shove never resets the anchor.
        self.assertFalse(guard.displacement(moved)['passed'])
        self.assertFalse(guard.command([0.]*6,[0.]*6,[0.]*6,moved)['passed'])
        rotated=copy.deepcopy(s);rotated['bodies'][seed['object']]['orientation']=[1.,0.,0.,0.]
        self.assertFalse(guard.displacement(rotated)['passed'])

    def test_close_sweep_checks_every_intermediate_jaw_pose(self):
        seed=FIXTURE['seeds'][0];s=context(seed);model=FingerModel(MODEL);q=[.1,.1,.1,1.2,-1.2,.7]
        middle=next(b for b in model.boxes(q) if b['path'].endswith('/moving_jaw_so101_v1_link/collisions/primitives/pad_tip'))
        s['bodies'][seed['object']]=dict(position=list(middle['center_world']),orientation=[1.,0.,0.,0.])
        s['local_corners'][seed['object']]=list(map(list,itertools.product(*[(-.0005,.0005)]*3)))
        guard=MissClearance(MODEL,s,seed['object'])
        lo=q[:5]+[.1];hi=q[:5]+[1.3]
        self.assertTrue(guard.state(lo)['passed']);self.assertTrue(guard.state(hi)['passed'])
        proof=guard.segment(lo,hi)
        self.assertFalse(proof['passed']);self.assertEqual(proof['kind'],'independent_joint_box')

    def test_missing_model_or_missing_baseline_is_not_a_valid_miss(self):
        with self.assertRaises(ValueError):MissClearance(None,{},'block')
        with self.assertRaises(KeyError):MissClearance(MODEL,{},'block')


class IndependentProgressTests(unittest.TestCase):
    def guard(self):
        live=json.loads(Path(__file__).with_name('fixtures').joinpath('regrasp-obstruction-340000.json').read_text())
        baseline=dict(live['baseline'],epoch=1,sim_ns=10**9,velocity=[0.]*6)
        return MissClearance(MODEL,baseline,next(iter(baseline['bodies']))),baseline

    def test_review_mixed_posture_in_reopen_is_in_the_box_and_refused(self):
        guard,baseline=self.guard()
        arm=[-.5470024369513172,.24181999068270854,.33139967015764416,1.1475766659545439,-2.15291677513403]
        closed=arm+[.04649072078121238];target=arm+[.11324639501352214]
        state=[-.5549024369513172,.23391999068270855,.32349967015764414,1.155476665954544,-2.16081677513403,.07824639501352214]
        mixed=[target[0],state[1],state[2],target[3],state[4],state[5]]
        self.assertTrue(guard.nominal_path(closed,target)['passed'])
        self.assertTrue(guard.nominal_path(state,target)['passed'])
        self.assertLess(guard.state(mixed)['separation_m'],-.00027)
        self.assertFalse(guard.segment(state,target)['passed'])
        from dart import TargetCandidate
        from command_geometry import CommandCollision
        from test_review_regressions import ReviewRegressions
        c,_,_=ReviewRegressions().collector();c.active_miss_clearance=(1,guard,'regrasp_reopen')
        c.node._check_state_valid_with_contacts=lambda target:(True,[])
        previous=arm+[.07824639501352214]
        candidate=TargetCandidate(1,10**9,tuple(target),(0.,)*5,(0.,)*6,(0.,)*6,tuple(target))
        with self.assertRaises(CommandCollision):c.validate_command(candidate,state,previous,dict(baseline,state=state))
        self.assertEqual(c.events,[])

    def test_high_inward_velocity_is_refused_even_for_a_hold_or_reopen(self):
        guard,snapshot=self.guard();arm=[-.5470024369513172,.24181999068270854,.33139967015764416,1.1475766659545439,-2.15291677513403]
        q=arm+[.07824639501352214];snapshot=dict(snapshot,state=q,velocity=[2.28,0.,0.,0.,0.,0.],miss_phase='regrasp_reopen')
        result=guard.command(q,q,q,snapshot)
        self.assertFalse(result['passed']);self.assertIn('velocity exceeds',result['reason'])
        self.assertEqual(result['measured_velocity_rad_s'][0],2.28)

    def test_effective_pair_offsets_are_required_bounded_and_hashed(self):
        from miss_clearance import bind_target_offsets
        from tick_contract import digest
        guard,snapshot=self.guard();name=guard.name
        model=copy.deepcopy(MODEL);model.pop('effective_finger_offsets')
        with self.assertRaisesRegex(ValueError,'finger offsets'):MissClearance(model,snapshot,name)
        model=copy.deepcopy(MODEL);model['effective_finger_offsets']['gripper_link']['contact_m'][0]=.001
        with self.assertRaisesRegex(ValueError,'pair offsets'):MissClearance(model,snapshot,name)
        a=bind_target_offsets(MODEL,MODEL['effective_block_offsets'])
        model=copy.deepcopy(MODEL);model['effective_finger_offsets']['gripper_link']['contact_m'][0]=.00009
        b=bind_target_offsets(model,MODEL['effective_block_offsets'])
        self.assertNotEqual(a['sha256'],b['sha256']);MissClearance(b,snapshot,name)
        b['effective_finger_offsets']['gripper_link']['rest_m'][0]=.001
        with self.assertRaisesRegex(ValueError,'identity differs'):MissClearance(b,snapshot,name)
