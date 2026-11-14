"""The actual setup entrance refuses disturbed worlds before writer admission."""
import copy,hashlib,json,threading,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from contract import JOINTS,parameters
from protocol import StepGeneration,CommandLedger
from tick_contract import TickContract,digest,PHYSICS_DT,PHYSICS_STEPS
from test_tick_contract import checked_window
from test_review_regressions import load_class
from recovery_start import ProtectedSettle,protected_state,receipt_error,world_anchor,POLICY
from miss_clearance import bind_target_offsets
from so_arm101_control.grasp_geometry import JOINT_LIMITS


class SetupHarness:
    def __init__(self,replay=None,disturb=None,shift_neighbour=0.,shift_base=0.,missing_substep=False):
        f=json.loads((Path(__file__).with_name('fixtures')/'regrasp-boot.json').read_text())
        self.initial=copy.deepcopy(f['initial_physical']);self.initial['epoch']=10
        self.initial['sim_ns']=10**9;self.initial['physics_time_s']=1.
        self.params=copy.deepcopy(replay['params'] if replay else parameters(340002,'red',regrasp=True))
        self.neighbour=next(n for n in self.initial['bodies'] if n!=self.params['object'] and not n.startswith('cup_'))
        self.initial['bodies'][self.neighbour]['position'][0]+=shift_neighbour
        self.initial['robot_base_world_pose']['position'][0]+=shift_base
        Runtime,scope=load_class('collection_runtime.py','Runtime');self.journal=[]
        scope['atomic_json']=lambda p,v:self.journal.append(copy.deepcopy(v))
        r=object.__new__(Runtime);self.runtime=r;r.world=NS(current_time=1.);r.epoch=10;r.invalid_through_ns=0
        r.timeline=NS(is_playing=lambda:False,pause=lambda:None);r.failure=None;r.busy=False;r.mode='collect'
        r.indices=list(range(6));r.ledger=CommandLedger();r.stamp=lambda:round(r.world.current_time*1e9)
        r.native_miss_fault_path='/test-only/protected-settle.json'
        self.bodies=copy.deepcopy(self.initial['bodies']);self.q=list(self.initial['state']);self.drives=list(self.q)
        self.q_writes=0;r.require_scene=lambda:None;r.bodies={}
        for name in self.bodies:
            def pose(p,q,name=name):self.bodies[name].update(position=p[0].tolist(),orientation=q[0].tolist())
            def velocity(v,name=name):self.bodies[name]['velocity']=v[0].tolist()
            r.bodies[name]=NS(set_world_poses=pose,set_velocities=velocity)
        def set_q(value,joint_indices):self.q=value[0].tolist();self.q_writes+=1
        r.art=NS(set_joint_positions=set_q,set_joint_velocities=lambda *a,**k:None)
        def apply(values,stamp):self.drives=np.asarray(values,dtype=np.float32).tolist()
        r.apply=apply
        def snapshot():
            s=copy.deepcopy(self.initial)
            s.update(epoch=r.epoch,sim_ns=r.stamp(),physics_time_s=r.world.current_time,
                state=list(self.q),velocity=[0.]*6,drive_targets=list(self.drives),bodies=copy.deepcopy(self.bodies))
            return s
        r.snapshot=snapshot
        Collector,scope=load_class('collect.py','Collector')
        scope.update(bind_target_offsets=bind_target_offsets,hashlib=hashlib,JOINT_LIMITS=JOINT_LIMITS)
        c=object.__new__(Collector);self.collector=c
        c.writer=None;c.current_episode_id='protected-setup-test';c.generation=StepGeneration()
        c.generation.install(dict(epoch=10,invalid_through_ns=0))
        replay_text=json.dumps(replay)
        c.args=NS(replay_receipt=NS(read_text=lambda:replay_text,read_bytes=lambda:replay_text.encode()) if replay else None)
        c.native_finger_model=f['finger_model'];c.native_shape_offsets=f['shape_offsets']
        body=self.bodies[self.params['object']];corners=self.initial['local_corners'][self.params['object']]
        pose=dict(zip(('x','y','z'),body['position']));pose.update(dict(zip(('qw','qx','qy','qz'),body['orientation'])))
        bbox=dict(zip(('sx','sy','sz'),np.ptp(np.asarray(corners),axis=0)))
        c.node=NS(objects_lock=threading.RLock(),objects_data={self.params['object']:pose},_lookup_bbox=lambda name:bbox,
            _tcp_clearance_var=NS(get=lambda:1.),_check_state_valid_with_contacts=lambda q:(True,[]))
        c.servo=NS(lock=threading.RLock(),reference=None,reference_epoch=None,executed_hold=None,previous_executed=None)
        c.held_snapshot=lambda epoch:checked_window([snapshot()])[0]
        c.sync_scene=lambda **kw:setattr(c,'scene_pose_check',dict(passed=True))
        self.proxy=types.ModuleType('collection_runtime');self.proxy.runtime=lambda:r
        def native(tool,request):
            if 'guarded_episode_offsets(' in request['code']:
                return dict(epoch=r.epoch,sim_ns=r.stamp(),finger_offsets=copy.deepcopy(f['finger_model']['effective_finger_offsets']),
                    block_offsets=copy.deepcopy(f['shape_offsets'][self.params['object']]))
            scope={};exec(request['code'],scope);return scope['result']
        c.mutate_kit=native;self.audit=TickContract()
        def restore(code):
            out=native('run_python',dict(code=code));c.generation.install(out)
            self.audit.install_boundary(out,restore=True);return out
        c.restore_physical=restore
        self.disturb=disturb;self.missing_substep=missing_substep;self.frame=0
        c.advance_ticks=lambda count:[self.tick() for _ in range(count)]
        self.result={}

    def tick(self):
        r=self.runtime
        for i in range(PHYSICS_STEPS):
            if self.disturb:self.disturb(self,self.frame,i)
            if not (self.missing_substep and self.frame==0 and i==1):r.check_protected_substep(PHYSICS_DT)
            r.world.current_time+=PHYSICS_DT
        r.finish_protected_frame();s=r.snapshot()
        if r._protected_settle is not None:s['protected_settle']=copy.deepcopy(r._protected_frame)
        raw=checked_window([s])[0]['tick_contract']['row'];raw['invalid_through_ns']=r.invalid_through_ns
        raw['episode_id']=self.collector.current_episode_id
        if self.audit.previous is None:raw['drive_baseline']=copy.deepcopy(self.audit.boundary)
        s['tick_contract']=self.audit.finish(raw);self.frame+=1
        return s

    def run(self):
        with patch.dict('sys.modules',{'collection_runtime':self.proxy}):
            self.collector._prepare_regrasp_start(self.params,self.result)
        receipt=dict(accepted=True,episode_id=self.collector.current_episode_id,params=self.params,**self.result)
        row=self.tick()['tick_contract']['row'];row['collection_phase']='corrective'
        return receipt,[row]


class ProtectedSettleTests(unittest.TestCase):
    def test_complete_setup_samples_every_protected_body_and_solves_before_writer(self):
        h=SetupHarness();receipt,rows=h.run();r=receipt['recovery_report']
        self.assertIsNone(receipt_error(receipt,rows));self.assertIsNone(h.collector.writer)
        self.assertEqual(r['protected_finish']['pre_physics_samples'],24)
        self.assertEqual(r['protected_finish']['final_post_solve_samples'],6)
        self.assertEqual(set(r['settle_checks'][0]['protected']['bodies']),set(r['baseline']['bodies']))
        self.assertFalse(h.journal)

    def test_cup_and_neighbour_motion_are_refused_before_writer(self):
        for kind in ('cup','neighbour','target'):
            def disturb(h,frame,step):
                if frame==0 and step==1:
                    name='cup_red' if kind=='cup' else h.neighbour if kind=='neighbour' else h.params['object']
                    h.bodies[name]['position'][0]+=.002
            h=SetupHarness(disturb=disturb)
            with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'Protected scene disturbed'):h.run()
            self.assertIsNone(h.collector.writer);self.assertFalse(h.result['recovery_report']['passed'])
            self.assertTrue(h.journal[-1]['protected_settle']['failure'])

    def test_transient_substep_motion_then_rest_cannot_be_certified(self):
        def disturb(h,frame,step):
            if frame==0:
                if step==1:h.bodies[h.neighbour]['position'][0]+=.002
                if step==2:h.bodies[h.neighbour]['position'][0]-=.002
        h=SetupHarness(disturb=disturb)
        with self.assertRaisesRegex(RuntimeError,'Protected scene disturbed'):h.run()
        h.bodies[h.neighbour]=copy.deepcopy(h.initial['bodies'][h.neighbour])
        self.assertTrue(protected_state(h.initial,h.runtime.snapshot())['passed'])
        self.assertIsNotNone(h.runtime.failure);self.assertIsNone(h.collector.writer)
        self.assertEqual(h.journal[-1]['protected_settle']['failure']['sample']['index'],1)

    def test_missing_native_callback_is_terminal_before_writer(self):
        h=SetupHarness(missing_substep=True)
        with self.assertRaisesRegex(RuntimeError,'Protected scene disturbed|coverage'):h.run()
        self.assertIsNone(h.collector.writer)

    def test_replay_restores_saved_world_despite_shifted_current_neighbour(self):
        source,_=SetupHarness().run();h=SetupHarness(replay=source,shift_neighbour=.002)
        replay,rows=h.run();self.assertIsNone(receipt_error(replay,rows))
        saved=source['recovery_report']['baseline'];restored=replay['recovery_report']['restore']['snapshot']
        for name,body in saved['bodies'].items():np.testing.assert_allclose(restored['bodies'][name]['position'],body['position'],atol=1e-6,rtol=0.)
        self.assertEqual(world_anchor(replay['recovery_report']['baseline']),world_anchor(saved))
        self.assertNotEqual(replay['recovery_report']['pre_restore_observed']['bodies'][h.neighbour]['position'],saved['bodies'][h.neighbour]['position'])

    def test_replay_refuses_different_fixed_base_before_native_write(self):
        source,_=SetupHarness().run();h=SetupHarness(replay=source,shift_base=.1)
        with self.assertRaisesRegex(ValueError,'base differs'):h.run()
        self.assertEqual(h.q_writes,0);self.assertIsNone(h.collector.writer)

    def test_receipt_binds_restore_world_base_and_clock_drive_continuity(self):
        source,rows=SetupHarness().run();self.assertIsNone(receipt_error(source,rows))
        paths=[('restore','snapshot','bodies','cup_red','position',0),
            ('restore','snapshot','bodies',next(n for n in source['recovery_report']['baseline']['bodies'] if n.startswith('green_')),'position',0),
            ('restore','snapshot','base_world_pose','position',0),('restore','snapshot','robot_base_world_pose','position',0),
            ('baseline','bodies','cup_red','position',0),('restore','snapshot','physics_time_s'),
            ('restore','command_baseline','drive_targets',0),('settle_snapshots',0,'drive_targets',0)]
        for path in paths:
            altered=copy.deepcopy(source);x=altered['recovery_report']
            for key in path[:-1]:x=x[key]
            x[path[-1]]+=.1
            with self.subTest(path=path):self.assertIsNotNone(receipt_error(altered,rows))
        altered=copy.deepcopy(source);frame=altered['recovery_report']['settle_snapshots'][0]['protected_settle']
        frame['substeps'][1]['snapshot']['bodies']['cup_red']['position'][0]+=.1
        self.assertIsNotNone(receipt_error(altered,rows))

    def test_native_geometric_rounding_is_tolerant_while_copied_boundaries_are_exact(self):
        source,rows=SetupHarness().run();report=source['recovery_report']
        frame=report['settle_snapshots'][0]['protected_settle']
        frame['substeps'][0]['motion']['bodies']['cup_red']['max_corner_motion_m']+=1e-12
        report['protected_finish']['frame_sha256'][0]=digest(frame)
        self.assertIsNone(receipt_error(source,rows))
        frame['start']['bodies']['cup_red']['position'][0]+=1e-12
        report['protected_finish']['frame_sha256'][0]=digest(frame)
        self.assertIsNotNone(receipt_error(source,rows))

    def test_rehashed_native_substep_clock_outside_frame_refuses(self):
        source,rows=SetupHarness().run();report=source['recovery_report']
        frame=report['settle_snapshots'][0]['protected_settle'];frame['substeps'][1]['snapshot']['sim_ns']+=100000000
        report['protected_finish']['frame_sha256'][0]=digest(frame)
        self.assertIsNotNone(receipt_error(source,rows))

    def test_first_corrective_fixed_base_edit_refuses(self):
        source,rows=SetupHarness().run();rows[0]['robot_base_world_pose']['position'][0]+=.1
        self.assertIn('fixed base',receipt_error(source,rows))

    def test_replay_current_robot_geometry_change_refuses_before_write(self):
        source,_=SetupHarness().run();h=SetupHarness(replay=source)
        model=copy.deepcopy(h.collector.native_finger_model)
        model['all_robot_boxes'][0]['half_extents_m'][0]-=.001
        model.pop('sha256');model['sha256']=digest(model);h.collector.native_finger_model=model
        with self.assertRaisesRegex(ValueError,'robot model differs'):h.run()
        self.assertEqual(h.q_writes,0)

    def test_replay_model_independent_arithmetic_readback_is_tolerant(self):
        from recovery_start import replay_model_binding
        h=SetupHarness();model=copy.deepcopy(h.collector.native_finger_model);current=copy.deepcopy(model)
        current['all_robot_boxes'][0]['half_extents_m'][0]+=1e-12
        current.pop('sha256');current['sha256']=digest(current)
        self.assertTrue(replay_model_binding(current,model)['passed'])

    def test_each_body_peak_and_half_bound_flag_survive_certificate_reconstruction(self):
        def disturb(h,frame,step):
            if frame==0 and step==1:h.bodies[h.neighbour]['velocity']=[.0004,0,0,0,0,.007]
        h=SetupHarness(disturb=disturb);source,rows=h.run();cert=source['recovery_report']['protected_finish']
        self.assertIsNone(receipt_error(source,rows));self.assertIn(h.neighbour,cert['half_bound_exceeded'])
        peak=cert['velocity_peaks'][h.neighbour]
        self.assertEqual(peak['samples'],30);self.assertAlmostEqual(peak['max_linear_m_s'],.0004)
        self.assertAlmostEqual(peak['max_angular_rad_s'],.007)
        altered=copy.deepcopy(source);altered['recovery_report']['protected_finish']['velocity_peaks'][h.neighbour]['max_linear_m_s']=0.
        self.assertIsNotNone(receipt_error(altered,rows))


if __name__=='__main__':unittest.main()
