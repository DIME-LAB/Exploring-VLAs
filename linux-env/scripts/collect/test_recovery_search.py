"""Recovery alternatives must pass full paths, retain contacts, and stay epoch-owned."""
import copy,math,threading,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from contract import JOINTS,parameters
from protocol import CancelledGeneration
from recovery_search import candidate_specs,choose,PlanningWallBudget,selection_error
from recovery import recovery_geometry,receipt_error
from query_service import StrictQueryClient
import test_recovery as probes
import test_review_regressions as collector_probes

POSE=dict(x=.21525059640407562,y=.07952475547790527,z=.006390673574060202,qw=.9991219639778137,qx=-8.450955419903039e-7,qy=7.380347710750357e-8,qz=-.04189673811197281)
BBOX=dict(sx=.031,sy=.0206,sz=.0128)
LIMITS=[(-3.,3.)]*6


class RecoverySearchTests(unittest.TestCase):
    def test_seeded_directions_and_magnitudes_are_bounded_and_original_first(self):
        p=parameters(340000,'red',regrasp=True)['regrasp'];specs=candidate_specs(p)
        self.assertEqual(len(specs),16);self.assertEqual(specs[0]['signed_offset_m'],p['signed_offset_m']);self.assertEqual(specs[0]['direction_offset_rad'],0.)
        self.assertEqual(specs,candidate_specs(p));self.assertEqual(len({s['direction_offset_rad'] for s in specs}),4)
        self.assertTrue(all(.02<=abs(s['signed_offset_m'])<=.04 for s in specs))

    def test_rejected_endpoint_keeps_mock_contacts_then_opposite_direction_passes(self):
        p=parameters(340000,'red',regrasp=True)['regrasp'];settings=dict(current_wrist_roll=-1.5221,approach_height=.02315197,approach_offset=[-.00392181,-.00394137])
        first=recovery_geometry(POSE,BBOX,candidate_specs(p)[0],**settings);bad=first['solutions']['miss_final'][0]
        contact=NS(contact_body_1='test_link',contact_body_2='test_obstacle',depth=.0002)
        def check(q):return (False,[contact]) if max(abs(q[i]-bad[n]) for i,n in enumerate(JOINTS[:5]))<1e-9 else (True,[])
        attempts=[];selected=choose(POSE,BBOX,p,[0.]*5+[.2],.2,.1,check,LIMITS,lambda:None,settings,attempts)
        self.assertEqual(selected['index'],1);self.assertFalse(attempts[0]['passed']);self.assertTrue(attempts[1]['passed'])
        self.assertEqual(attempts[0]['proof']['endpoints'][-1]['proof']['contacts'][0]['body_2'],'test_obstacle')
        self.assertEqual(len(attempts[1]['proof']['paths']),8)
        report=dict(selection={k:v for k,v in selected.items() if k not in ('geometry','targets')},geometry=selected['geometry'],candidate_attempts=attempts)
        self.assertIsNone(selection_error(report,p));report['selection']['index']=0
        self.assertIsNotNone(selection_error(report,p))

    def test_clear_endpoints_do_not_hide_a_collision_inside_a_path(self):
        p=parameters(1,'red',regrasp=True)['regrasp'];attempts=[];calls=[]
        # Endpoints pass. Refuse every nonconstant checked segment internally.
        contact=NS(contact_body_1='test_arm',contact_body_2='test_cup',depth=.001)
        def check(q):
            calls.append(q)
            return (False,[contact]) if len(calls)%12==0 else (True,[])
        result=choose(dict(x=.2,y=0.,z=.0064,qw=1.,qz=0.),BBOX,p,[0.]*6,.2,.1,check,LIMITS,lambda:None,dict(current_wrist_roll=0.),attempts)
        self.assertIsNone(result);self.assertTrue(any(a.get('proof',{}).get('paths') for a in attempts))
        self.assertTrue(any(any(not path['proof']['passed'] for path in a.get('proof',{}).get('paths',[])) for a in attempts))

    def test_transport_failure_and_cancellation_do_not_try_another_candidate(self):
        p=parameters(1,'red',regrasp=True)['regrasp']
        for error in (TimeoutError('no MoveIt answer'),CancelledGeneration('old epoch')):
            attempts=[]
            def check(q):raise error
            with self.assertRaises(type(error)):choose(dict(x=.2,y=0.,z=.0064,qw=1.,qz=0.),BBOX,p,[0.]*6,.2,.1,check,LIMITS,lambda:None,dict(current_wrist_roll=0.),attempts)
            self.assertEqual(len(attempts),1)

    def test_actual_prefix_exhausted_candidates_never_execute(self):
        c,events=probes.RecoveryTests().driver();contact=NS(contact_body_1='test_arm',contact_body_2='test_box',depth=.001)
        c.node._check_state_valid_with_contacts=lambda target:(False,[contact]);result={}
        with self.assertRaisesRegex(Exception,'all recovery candidates refused'):c.regrasp_prefix(parameters(1,'red',regrasp=True),1,result)
        self.assertFalse(any(e[0] in ('arm','grip') for e in events));self.assertTrue(result['recovery_report']['candidate_attempts'])

    def test_planning_reservation_is_bounded_and_then_accounts_only_elapsed_wall_time(self):
        now=[0.];budget=PlanningWallBudget(lambda:now[0])
        with budget.reserve(700.):
            self.assertEqual(budget.allowance(),700.);now[0]=600.
        self.assertEqual(budget.allowance(),600.)
        with self.assertRaises(TimeoutError):
            with budget.reserve(1.):now[0]+=2.
        self.assertEqual(budget.allowance(),602.)

    def test_refusal_observer_binds_request_context_and_records_once_after_late_completion(self):
        c,_,_=collector_probes.ReviewRegressions().collector();c.planner_refusal_lock=threading.Lock();c.planner_refusals=[];c.current_episode_id='old';c.node._motion_context=NS(epoch=1)
        req=NS(robot_state=NS(joint_state=NS(name=list(JOINTS),position=[.1]*6)))
        contact=NS(contact_body_1='test_wrist',contact_body_2='test_ground',depth=.002)
        future=NS(done=lambda:True,result=lambda:NS(valid=False,contacts=[contact]));client=StrictQueryClient(NS(call_async=lambda r:future),'validity',observer=c.observe_validity_request)
        pending=client.call_async(req);req.robot_state.joint_state.position[0]=9.;c.current_episode_id='new';c.generation.install(dict(epoch=2,invalid_through_ns=10**9));pending.result();pending.result()
        self.assertEqual(len(c.planner_refusals),1);e=c.planner_evidence(dict(episode_id='old'))[0]
        self.assertEqual(e['active_epoch_at_request'],1);self.assertEqual(e['worker_epoch'],1);self.assertEqual(e['joint_positions'][0],.1);self.assertEqual(e['contacts'][0]['body_2'],'test_ground')
        self.assertEqual(c.planner_evidence(dict(episode_id='new')),[])
