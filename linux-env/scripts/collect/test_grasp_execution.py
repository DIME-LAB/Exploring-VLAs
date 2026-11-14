"""Actual shared execution entrances bind jaws, TCP, yaws and verdicts."""
import json,math,threading,unittest
from types import MethodType,SimpleNamespace as NS
from so_arm101_control import grasp_geometry as gg
import test_motion_operations as motions
from test_safety import method
from finger_geometry import FingerModel,separation
from test_miss_clearance import MODEL

SHARED=motions.SHARED


class GraspExecutionTests(unittest.TestCase):
    def entrance(self,bbox,cross=False,reject_requested=False):
        c,r,entry,trace,threads=motions.MotionOperationTests().setup_motion();n=c.node;sent=[];dispatched=[];poses=dict(x=.20,y=0.,z=.0064,qw=1.,qz=0.)
        n.objects_lock=threading.Lock();n.objects_data={'block':poses};n._selected_object='block';n._get_selected_object_name=lambda:'block'
        n._lookup_bbox=lambda name:dict(bbox);n._ensure_lego_state=lambda:None;n._remove_single_lego_from_world=lambda name:None;n._add_lego_collision_objects=lambda:None
        def var(v):return NS(get=lambda:v)
        for key,value in dict(_grasp_cross_var=cross,_grasp_topic_var='/objects_poses',_tcp_clearance_var=1.,_bbox_enabled_var=True,
            _grasp_obj_z_var=0.,_grasp_approach_height_var=.02,_grasp_arm_duration_var=2.,_grasp_grip_duration_var=1.,_jaw_open_clearance_var=5.,_jaw_close_clearance_var=0.,_ground_z_var=0.).items():setattr(n,key,var(value))
        n._gripper_command=lambda *a,**k:None;n._select_planning_group=lambda value:None;n._trace=lambda *a,**k:None;n._append_log=lambda *a,**k:None
        for name in ('_cmd_grasp_move','_gripper_angle_for_object','_cmd_gripper_open_for_object','_cmd_gripper_close_for_object','_get_grip_width'):
            fn=method(SHARED,'PickPlaceNode',name)
            fn.__globals__.update(vars(gg));setattr(n,name,MethodType(fn,n))
        n._grasp_orientation=lambda *args:(0.,0.,0.,1.)
        n._send_gripper_goal=lambda angle,**kwargs:sent.append(angle) or True
        def attach(name):n._attached_lego_name=name;return True
        n._attach_lego_to_gripper=attach
        initial=gg.make_reachable_grasp_plan(poses,bbox,cross=cross,current_wrist_roll=0.,approach_height=.02)
        refused={tuple(round(q[k],8) for k in gg.ARM_JOINT_NAMES) for values in initial['solutions'].values() for q in values}
        queries=[]
        def valid(q):
            queries.append(dict(q))
            return not reject_requested or tuple(round(q[k],8) for k in gg.ARM_JOINT_NAMES) not in refused
        n._actual_positions['wrist_roll']=0.
        n._check_state_valid_with_contacts=lambda q:(valid(q),[])
        def dispatch(target,*args,**kwargs):
            op=kwargs['operation'];n._require_operation(op);dispatched.append((op,kwargs.get('descent')))
            event=(kwargs.get('descent') or op).event;event.verdict=dict(ok=True);event.set()
        n._ik_apply_and_act=dispatch
        return n,sent,dispatched,queries,initial,threads

    def test_jaw_commands_match_selected_plan_for_both_dimension_orders_and_cross_settings(self):
        for sx,sy in ((.032,.016),(.016,.032)):
            for cross in (False,True):
                with self.subTest(sx=sx,cross=cross):
                    n,sent,plans,queries,_,threads=self.entrance(dict(sx=sx,sy=sy,sz=.0128),cross)
                    n._cmd_gripper_open_for_object();self.assertTrue(n._current_motion_event().wait(3))
                    n._cmd_grasp_move();self.assertTrue(n._current_motion_event().wait(3));self.assertTrue(plans)
                    spec=json.loads(plans[0][0].grasp_spec_json)
                    n._cmd_gripper_close_for_object();self.assertTrue(n._current_motion_event().wait(3))
                    for t in threads:t.join(1);self.assertFalse(t.is_alive())
                    self.assertEqual(sent[0],spec['opening_rad'])
                    self.assertEqual(sent[-1],spec['closing_rad'])
                    self.assertTrue(all(abs(q['gripper_joint']-spec['opening_rad'])<1e-12 for q in queries))
                    yaw=spec['chosen_yaw'];width=spec['projected_width_m'];dx,dy=gg.jaw_offset(width,yaw)
                    self.assertAlmostEqual(spec['geometry']['tcp'][0],.2+dx,places=12)
                    self.assertAlmostEqual(spec['geometry']['tcp'][1],dy,places=12)
                    self.assertEqual(plans[0][1].grasp_spec_json,plans[0][0].grasp_spec_json)
                    joints=plans[0][1].joints();q=[joints[name] for name in gg.ARM_JOINT_NAMES]+[sent[0]]
                    target=dict(rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],center_world=[.2,0.,.0064],half_extents_m=[sx/2,sy/2,.0064])
                    boxes=FingerModel(MODEL).boxes(q)
                    self.assertGreater(min(separation(b,target) for b in boxes),0.)
                    self.assertGreater(min(b['min_z_m'] for b in boxes),.0012)

    def test_collision_rejection_tries_equivalent_yaw_through_actual_grasp_entrance(self):
        n,sent,plans,queries,initial,threads=self.entrance(dict(sx=.031,sy=.0206,sz=.0128),reject_requested=True)
        n._cmd_grasp_move();self.assertTrue(n._current_motion_event().wait(3))
        for t in threads:t.join(1);self.assertFalse(t.is_alive())
        self.assertTrue(plans)
        spec=json.loads(plans[0][0].grasp_spec_json)
        self.assertAlmostEqual(abs(spec['chosen_yaw']-spec['requested_yaw']),math.pi)
        self.assertFalse(spec['attempts'][0]['validation']['passed'])
        self.assertTrue(spec['attempts'][-1]['validation']['passed'])
        self.assertNotEqual(spec['geometry']['tcp'],initial['geometry']['tcp'])
        self.assertTrue(n._current_motion_event().verdict['ok'])
