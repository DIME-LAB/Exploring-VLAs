"""Synthetic reset candidates must be seeded, clear and stationary before recording."""
import copy,json,math,collections,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from contract import JOINTS
from recovery import recovery_parameters
from recovery_start import POLICY,StaticClearance,candidates,choose
from miss_clearance import bind_target_offsets
from so_arm101_control.grasp_geometry import JOINT_LIMITS

class SyntheticStartTests(unittest.TestCase):
    def fixture(self):
        f=json.loads((Path(__file__).with_name('fixtures')/'regrasp-boot.json').read_text());s=f['initial_physical'];name=f['params']['object'];body=s['bodies'][name]
        model=bind_target_offsets(f['finger_model'],f['shape_offsets'][name]);oracle=StaticClearance(model,s,name)
        pose=dict(zip(('x','y','z'),body['position']));pose.update(dict(zip(('qw','qx','qy','qz'),body['orientation'])))
        cs=s['local_corners'][name];bbox={k:max(c[i] for c in cs)-min(c[i] for c in cs) for i,k in enumerate(('sx','sy','sz'))}
        return s,name,model,oracle,pose,bbox
    def selection(self,check=lambda q:(True,[])):
        s,name,model,o,pose,bbox=self.fixture()
        return choose(pose,bbox,recovery_parameters(340000,True),{},o,check,lambda:None,[JOINT_LIMITS[n] for n in JOINTS]),o,s
    def test_seeded_sampling_mixture_covers_closed_partial_and_open_hover(self):
        counts=collections.Counter()
        for seed in range(1000):
            request=recovery_parameters(seed,True);spec=next(candidates(request));counts[spec['phenotype']]+=1
            self.assertEqual(spec,next(candidates(request)));self.assertTrue(.026<=spec['distance_m']<=.038)
            self.assertLessEqual(abs(spec['yaw_jitter_rad']),math.radians(12))
            if spec['phenotype']=='open_hover':self.assertGreaterEqual(spec['hover_m'],.018);self.assertEqual(spec['jaw_fraction'],1.)
        for kind,p in POLICY['phenotypes'].items():self.assertLess(abs(counts[kind]-p*1000),60)
    def test_sampled_jaw_uses_shared_width_convention_for_both_bbox_orders_and_cross_modes(self):
        from recovery_start import geometry
        from so_arm101_control.grasp_geometry import gripper_angles,object_grip_width
        _,_,_,_,pose,_=self.fixture();spec=next(candidates(recovery_parameters(340000,True)))
        for sx,sy in ((.031,.0206),(.0206,.031)):
            for cross in (False,True):
                bbox=dict(sx=sx,sy=sy,sz=.0128);g=geometry(pose,bbox,spec,dict(cross=cross))
                opening,closing=gripper_angles(object_grip_width(bbox,cross))
                self.assertEqual((g['opening_rad'],g['closing_rad']),(opening,closing))

    def test_all_links_and_scene_are_checked_before_a_pose_is_selected(self):
        answer,o,s=self.selection();self.assertTrue(answer['proof']['passed']);self.assertGreater(len(answer['proof']['static']['scene']),8)
        q=answer['joints'];other=next(n for n in s['bodies'] if n!=o.target and not n.startswith('cup_'))
        link=next(x for x in o.all_links.boxes(q,s['robot_base_world_pose']) if '/gripper_link/' not in x['path'] and '/moving_jaw_' not in x['path'])
        bad=copy.deepcopy(s);bad['bodies'][other]['position']=link['center_world'].tolist()
        self.assertFalse(o.state(q,bad)['passed'])
    def test_target_and_floor_margin_reject_finger_overlap(self):
        answer,o,s=self.selection();q=answer['joints'];finger=o.oracle.boxes(q,s['robot_base_world_pose'])[0]
        bad=copy.deepcopy(s);bad['bodies'][o.target]['position']=finger['center_world'].tolist()
        self.assertFalse(o.state(q,bad)['passed'])
        low=copy.deepcopy(s);low['robot_base_world_pose']['position'][2]-=1.
        self.assertFalse(o.state(q,low)['passed'])
    def test_stationary_target_and_corner_motion_are_independent_requirements(self):
        _,_,_,o,_,_=self.fixture()
        self.assertTrue(o.target_state(o.baseline)['passed'])
        for field,value in [('velocity',[2*POLICY['stationary_linear_m_s'],0,0,0,0,0]),('velocity',[0,0,0,0,2*POLICY['stationary_angular_rad_s'],0])]:
            s=copy.deepcopy(o.baseline);s['bodies'][o.target][field]=value;self.assertFalse(o.target_state(s)['passed'])
        s=copy.deepcopy(o.baseline);s['bodies'][o.target]['position'][0]+=.0003
        self.assertFalse(o.target_state(s)['passed'])
    def test_planner_contacts_survive_refusal_and_remaining_solutions_are_tried(self):
        calls=[]
        def check(q):
            calls.append(q)
            return (len(calls)>1,[NS(contact_body_1='jaw',contact_body_2='cup',depth=.001)] if len(calls)==1 else [])
        answer,_,_=self.selection(check)
        self.assertTrue(answer['proof']['passed']);self.assertGreater(len(calls),1)
        refused=[v for a in answer['attempts'] for v in a['verdicts'] if v.get('contacts')]
        self.assertEqual(refused[0]['contacts'][0]['body_2'],'cup')
    def test_current_synthetic_request_refuses_a_physical_prefix_receipt(self):
        from recovery import receipt_error
        from contract import parameters
        receipt=dict(params=parameters(1,'red',regrasp=True),recovery_report=dict(kind='regrasp',miss_confirmed=True))
        self.assertIn('synthetic reset certificate',receipt_error(receipt,[]))

    def test_cancel_during_scene_rpc_cannot_mutate_or_publish_a_new_epoch_scene(self):
        import threading
        from contextlib import contextmanager
        from protocol import StepGeneration,CancelledGeneration
        from test_review_regressions import load_class
        Collector,_=load_class('collect.py','Collector')
        for boundary in ('clear','verify'):
            c=object.__new__(Collector);c.generation=StepGeneration();c.generation.install(dict(epoch=1,invalid_through_ns=0));c.stop=threading.Event()
            context=threading.local();effects=[]
            @contextmanager
            def guarded():
                with c.generation.own('test scene effect'):
                    c.generation.require(context.epoch);yield
            def changed():c.generation.install(dict(epoch=2,invalid_through_ns=0))
            def clear():
                if boundary=='clear':changed()
                return dict(verified=True)
            def verify(*a):
                if boundary=='verify':changed()
                return dict(passed=True)
            def ground():
                with guarded():effects.append('ground')
            c.node=NS(_motion_context=context,_motion_guard=guarded,_cmd_toggle_ground_plane=ground,
                _add_cup_collision_objects=lambda **k:None,_drop_data={},_cup_collision_names=[],_lego_collision_names=[])
            c.update_pose_cache=lambda s:effects.append('cache')
            c.scene_reset=NS(clear=clear,rebuild_legos=lambda s:None,set_policy=lambda ids:None,verify=verify)
            with self.subTest(boundary=boundary),self.assertRaises(CancelledGeneration):
                c.sync_scene(dict(epoch=1,sim_ns=10**9),epoch=1)
            self.assertFalse(hasattr(c,'scene_pose_check'))
            if boundary=='clear':self.assertEqual(effects,['cache'])

    def test_missing_full_link_geometry_and_offsets_refuse(self):
        s,name,model,_,_,_=self.fixture();bad=copy.deepcopy(model);bad.pop('all_robot_boxes');bad.pop('sha256')
        with self.assertRaises(KeyError):StaticClearance(bad,s,name)
        bad=copy.deepcopy(model);bad['effective_block_offsets']['contact_m']=[.01];bad.pop('sha256')
        with self.assertRaises(ValueError):StaticClearance(bad,s,name)

if __name__=='__main__':unittest.main()
