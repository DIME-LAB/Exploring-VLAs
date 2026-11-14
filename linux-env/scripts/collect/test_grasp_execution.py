"""Actual shared execution entrances bind jaws, TCP, yaws and verdicts."""
import json,math,threading,unittest
from types import MethodType,SimpleNamespace as NS
from so_arm101_control import grasp_geometry as gg
import test_motion_operations as motions
from test_safety import method
from finger_geometry import FingerModel,separation
from test_miss_clearance import MODEL

SHARED=motions.SHARED
BASELINE_AST={'_get_grip_width': '02d87339d2b2ca44c84b8af3bfb58bd4914a8828456abb6752d83219600433e1', '_gripper_angle_for_object': '32f10e40f153ce4332b6146a2458b3811b112d17d34f9118cfe19b644faba0ad', '_cmd_gripper_open_for_object': '31a4d6f0d4477a83ceac7e9d29cb5e3fe56c55d5e9be3b3ef8b6b2a823337385', '_cmd_gripper_close_for_object': '5fc1d26fa373a15221e8f2e42d6f3c71747ad6690557fa64b38e19b4f5f45110', '_cmd_grasp_move': '348a47409f318d966f5ffa9733a43ceb87cbdf563b9e1feb6c280ff431b913be'}


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

    def test_normal_and_gui_jaw_commands_keep_bbox_min_max_policy_without_pose(self):
        for sx,sy in ((.032,.016),(.016,.032)):
            for cross in (False,True):
                n,sent,plans,queries,_,threads=self.entrance(dict(sx=sx,sy=sy,sz=.0128),cross)
                expected=gg.gripper_angles(max(sx,sy) if cross else min(sx,sy),.001,.005,0.)
                n.objects_data.clear()  # Legacy jaw commands need bbox, not pose.
                self.assertEqual(n._gripper_angle_for_object('block'),expected)
                n._cmd_gripper_open_for_object();self.assertTrue(n._current_motion_event().wait(3))
                n._cmd_gripper_close_for_object();self.assertTrue(n._current_motion_event().wait(3))
                for thread in threads:thread.join(1);self.assertFalse(thread.is_alive())
                self.assertEqual(sent,list(expected))

    def test_gui_gate_c_uses_actual_jaw_and_original_dispatch_route(self):
        from so_arm101_control.scene_transactions import SceneExecution
        for only_closed_is_valid in (False,True):
            n,sent,plans,queries,initial,threads=self.entrance(dict(sx=.031,sy=.0206,sz=.0128))
            n.execution=SceneExecution();n._actual_positions['gripper_joint']=.9
            actual_queries=[]
            def valid(arm):
                # Real _check_state_valid fills omitted joints from the measured state.
                q=dict(n._actual_positions,**arm);actual_queries.append(q)
                return (q['gripper_joint']<.5)==only_closed_is_valid
            n._check_state_valid=valid
            n._cmd_grasp_move();self.assertTrue(n._current_motion_event().wait(3))
            for thread in threads:thread.join(1);self.assertFalse(thread.is_alive())
            self.assertTrue(actual_queries);self.assertEqual(sent,[])
            self.assertTrue(all(q['gripper_joint']==.9 for q in actual_queries))
            self.assertEqual(bool(plans),not only_closed_is_valid)
            self.assertEqual(n._current_motion_event().verdict['ok'],not only_closed_is_valid)

    def test_shared_execution_methods_match_verified_baseline_ast(self):
        import ast,hashlib
        # The normalized AST hashes pin the established actual-jaw call route.
        expected=BASELINE_AST
        tree=ast.parse(SHARED.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PickPlaceNode')
        for method in cls.body:
            if isinstance(method,ast.FunctionDef) and method.name in expected:
                actual=hashlib.sha256(ast.get_source_segment(SHARED.read_text(),method).encode()).hexdigest()
                self.assertEqual(actual,expected[method.name],method.name)
