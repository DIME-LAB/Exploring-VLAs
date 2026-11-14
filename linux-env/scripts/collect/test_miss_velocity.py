"""Native velocity-bound application, retirement and measured-state evidence."""
import copy,json,unittest
from types import SimpleNamespace as NS
import numpy as np
from contract import JOINTS
from miss_clearance import velocity_cap as base_cap,POLICY
def velocity_cap(phase):return base_cap(phase,'block')
from test_review_regressions import load_class


class NativeMissVelocityTests(unittest.TestCase):
    def runtime(self):
        Runtime,scope=load_class('collection_runtime.py','Runtime');r=object.__new__(Runtime)
        # Explicit healthy durable-writer adapter; publication-failure tests replace it.
        scope['atomic_json']=lambda path,value:None
        r.native_miss_fault_path='/unit-test/native-motion-fault.json'
        r.epoch=1;r.stamp=lambda:10**9
        r.mode='collect';r.busy=False;r.timeline=NS(is_playing=lambda:False,pause=lambda:None);r.failure=None
        r.indices=[4,1,5,0,2,3]  # All reads/writes use named-joint mapping.
        r._miss_cap=None;r._miss_saved_limits=None;r._miss_last_position=None
        data=dict(limits=np.asarray([[4.,5.,6.,7.,8.,9.]],dtype=np.float32),q=np.zeros((1,6),dtype=np.float32),v=np.zeros((1,6),dtype=np.float32))
        data['body_velocity']=np.zeros((1,6))
        r.bodies={'block':NS(get_velocities=lambda:data['body_velocity'].copy())}
        original=data['limits'][:,r.indices].copy()
        writes=[]
        def set_limits(value,joint_indices):
            writes.append((value.copy(),list(joint_indices)));data['limits'][:,joint_indices]=value
        r.art=NS(set_max_joint_velocities=set_limits,
            get_joint_max_velocities=lambda joint_indices:data['limits'][:,joint_indices].copy(),
            get_joint_positions=lambda:data['q'].copy(),get_joint_velocities=lambda:data['v'].copy())
        return r,data,original,writes

    def test_cap_readback_and_restore_are_named_and_leave_original_dynamics(self):
        r,d,original,writes=self.runtime();cap=velocity_cap('regrasp_miss_descent')
        self.assertEqual(r.set_miss_velocity_cap(cap),cap)
        np.testing.assert_allclose(d['limits'][:,r.indices],[[.05]*6])
        self.assertEqual(writes[-1][1],r.indices)
        r.check_miss_substep(1/120);self.assertTrue(r.miss_cap_state()['on'])
        self.assertEqual(r.set_miss_velocity_cap(None),dict(on=False,values_rad_s=None))
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)
        self.assertFalse(r.miss_cap_state()['on'])
        count=len(writes);r.set_miss_velocity_cap(None);self.assertEqual(len(writes),count)

    def test_uncapped_normal_command_uses_original_application_route(self):
        r,d,original,writes=self.runtime();r.mode='collect';r.epoch=1;r.invalid_through_ns=-1
        from protocol import CommandLedger
        r.ledger=CommandLedger();seen=[]
        r.apply=lambda command,stamp:seen.append(('apply',command)) or dict(application_stamp_ns=stamp,applied_command=command)
        r.advance=lambda:seen.append(('advance',))
        acks=[];r.ack_pub=NS(publish=lambda msg:acks.append(json.loads(msg.data)))
        r.command_received(NS(data=json.dumps(dict(epoch=1,observation_stamp_ns=10**9,command=[.1]*6,advance=True))))
        self.assertEqual(len(seen),2);self.assertEqual(writes,[])
        self.assertNotIn('command_miss_velocity_cap',acks[0])

    def test_actual_command_entrance_applies_cap_before_targets_and_returns_cap_ack(self):
        r,d,original,writes=self.runtime();r.mode='collect';r.epoch=1;r.invalid_through_ns=-1
        from protocol import CommandLedger
        r.ledger=CommandLedger();seen=[]
        def apply(command,stamp):
            np.testing.assert_allclose(d['limits'][:,r.indices],[[.05]*6]);seen.append('apply')
            return dict(application_stamp_ns=stamp,applied_command=command)
        r.apply=apply;r.advance=lambda:seen.append('advance')
        acks=[];r.ack_pub=NS(publish=lambda msg:acks.append(json.loads(msg.data)))
        cap=velocity_cap('regrasp_reopen')
        payload=dict(epoch=1,observation_stamp_ns=10**9,command=[.1]*6,advance=True,miss_velocity_cap=cap)
        r.command_received(NS(data=json.dumps(payload)));r.command_received(NS(data=json.dumps(payload)))
        self.assertEqual(seen,['apply','advance']);self.assertEqual(len(writes),1)
        self.assertEqual(acks[0]['command_miss_velocity_cap'],cap);self.assertEqual(acks[0],acks[1])

    def test_kickoff_checks_identity_state_and_cap_before_first_advance(self):
        r,d,_,writes=self.runtime();r.epoch=1;r.invalid_through_ns=0;r.stamp=lambda:10**9
        r.art.get_applied_actions=lambda:NS(joint_positions=np.zeros((1,6)))
        seen=[]
        def advance():
            self.assertTrue(r.miss_cap_state()['on']);np.testing.assert_allclose(d['limits'][:,r.indices],[[.1]*6]);seen.append('advance');return dict(target_ns=1033333333)
        r.advance=advance;cap=velocity_cap('regrasp_miss_approach')
        self.assertEqual(writes,[]);self.assertEqual(seen,[])
        ack=r.start_miss_hold(1,10**9,cap,[0.]*6,[0.]*6)
        self.assertEqual(ack['command_miss_velocity_cap'],cap);self.assertEqual(seen,['advance'])

    def test_cancel_restores_original_limits_before_installing_new_epoch(self):
        r,d,original,_=self.runtime();r.epoch=1;r.world=NS(current_time=1.);r.stamp=lambda:10**9
        r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        result=r.cancel()
        self.assertEqual(result['epoch'],2);self.assertFalse(r.miss_cap_state()['on'])
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)

    def test_bootstrap_deploy_cancel_with_no_installed_limits_is_a_native_noop(self):
        for mode in ('collect','deploy'):
            r,d,original,writes=self.runtime();r.mode=mode;r.world=NS(current_time=1.)
            # Kit has received pause(), but is_playing() still exposes the
            # preceding app update. This is the retained clock-startup failure.
            pauses=[];playing=[True];r.timeline.pause=lambda:pauses.append(True);r.timeline.is_playing=lambda:playing[0]
            r.timeline.commit=lambda:playing.__setitem__(0,False)
            published=[];r.publish_miss_fault.__func__.__globals__['atomic_json']=lambda p,v:published.append(copy.deepcopy(v))
            before=d['limits'].copy();result=r.cancel()
            with self.subTest(mode=mode):
                self.assertEqual(pauses,[True]);self.assertEqual(result['epoch'],2)
                self.assertIsNone(result['failure']);self.assertIsNone(result['miss_motion_fault'])
                self.assertEqual(writes,[]);self.assertEqual(published,[])
                self.assertIsNone(getattr(r,'_miss_motion_fault',None))
                np.testing.assert_array_equal(d['limits'],before)
                self.assertEqual(r.set_miss_velocity_cap(None),dict(on=False,values_rad_s=None))
                self.assertEqual(writes,[]);self.assertEqual(published,[])

    def test_real_cap_retirement_still_refuses_a_pending_playing_pause(self):
        r,d,original,writes=self.runtime();r.world=NS(current_time=1.)
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        r.timeline.is_playing=lambda:True
        with self.assertRaisesRegex(RuntimeError,'requires a held simulation'):
            r.set_miss_velocity_cap(None)
        self.assertIsNotNone(r.failure)
        self.assertIn('requires a held simulation',r._miss_motion_fault['error'])
        # Fault retirement is still the checked, durable terminal path, not a
        # successful caller handover or permission to advance another frame.
        with self.assertRaises(RuntimeError):r.advance()

    def test_high_measured_velocity_cannot_be_braked_by_relabelling_cap(self):
        r,d,_,writes=self.runtime();d['v'][0,r.indices[0]]=2.28
        with self.assertRaisesRegex(RuntimeError,'before advance'):r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        self.assertEqual(writes,[])

    def test_cap_change_or_substep_tracking_fault_is_terminal(self):
        for kind in ('velocity','position','limit'):
            r,d,_,_=self.runtime();r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
            if kind=='velocity':d['v'][0,r.indices[0]]=.051
            if kind=='position':d['q'][0,r.indices[2]]=.0005
            if kind=='limit':d['limits'][0,r.indices[3]]=.5
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):r.check_miss_substep(1/120)
            self.assertIsNotNone(r.failure);self.assertFalse(r.busy)

    def test_missing_or_wrong_effective_finger_offsets_refuse(self):
        r,_,_,_=self.runtime()
        def view(n):return NS(_physics_view=NS(get_contact_offsets=lambda:np.asarray([[.0001]*n]),get_rest_offsets=lambda:np.asarray([[0.]*n])))
        r.finger_views={name:view(4) for name in ('gripper_link','moving_jaw_so101_v1_link')}
        measured=r.effective_finger_offsets();self.assertEqual(len(measured),2)
        r.finger_views['gripper_link']=view(5)
        with self.assertRaisesRegex(RuntimeError,'effective finger'):r.effective_finger_offsets()

    def test_partial_install_kickoff_cancel_reset_and_fresh_attempt_preserve_original_limits(self):
        for failure in ('first_write_before','first_write','first_readback','kickoff','rollback_write','rollback_readback','normal_restore'):
            with self.subTest(failure=failure):
                r,d,original,writes=self.runtime();r.world=NS(current_time=1.);r.invalid_through_ns=0
                real_set=r.art.set_max_joint_velocities;real_get=r.art.get_joint_max_velocities
                counts=dict(writes=0,reads=0)
                def setter(value,joint_indices):
                    counts['writes']+=1
                    if failure in ('rollback_write','normal_restore') and counts['writes']==2:raise RuntimeError('rollback write failed')
                    if failure=='first_write_before' and counts['writes']==1:raise RuntimeError('first write refused')
                    real_set(value,joint_indices)
                    if failure=='first_write' and counts['writes']==1:raise RuntimeError('first write partly applied')
                def reader(joint_indices):
                    counts['reads']+=1;value=real_get(joint_indices)
                    if counts['reads']==2 and failure in ('first_readback','rollback_write','rollback_readback'):value=value+.01
                    if counts['reads']==3 and failure=='rollback_readback':value=value+.02
                    return value
                r.art.set_max_joint_velocities=setter;r.art.get_joint_max_velocities=reader
                r.art.get_applied_actions=lambda:NS(joint_positions=np.zeros((1,6)))
                def advance():raise RuntimeError('kickoff advance failed')
                r.advance=advance
                if failure=='normal_restore':
                    r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
                    with self.assertRaises(RuntimeError):r.set_miss_velocity_cap(None)
                elif failure=='kickoff':
                    with self.assertRaises(RuntimeError):r.start_miss_hold(1,10**9,velocity_cap('regrasp_reopen'),[0.]*6,[0.]*6)
                else:
                    with self.assertRaises(RuntimeError):r.start_miss_hold(1,10**9,velocity_cap('regrasp_reopen'),[0.]*6,[0.]*6)
                self.assertIsNotNone(r.failure)
                if failure.startswith('rollback_') or failure=='normal_restore':
                    self.assertIsNone(r.miss_cap_state()['on']);self.assertTrue(r._miss_limits_dirty)
                    self.assertEqual(r._miss_saved_limits,original[0].tolist())
                    with self.assertRaisesRegex(RuntimeError,'terminal'):r.set_miss_velocity_cap(velocity_cap('regrasp_miss_open'))
                    self.assertEqual(r._miss_saved_limits,original[0].tolist())
                # Make transport healthy again, then run actual cancel/reset barriers.
                r.art.set_max_joint_velocities=real_set;r.art.get_joint_max_velocities=real_get
                if failure.startswith('rollback_'):
                    with self.assertRaisesRegex(RuntimeError,'Fresh instance required'):r.restore(dict(bodies={}),arm=True)
                cancelled=r.cancel();self.assertTrue(cancelled['failure']);self.assertFalse(r.miss_cap_state()['on'])
                np.testing.assert_array_equal(d['limits'][:,r.indices],original)
                self.assertFalse(r._miss_limits_dirty)
                with self.assertRaisesRegex(RuntimeError,'Fresh instance required'):r.restore(dict(bodies={}),arm=True)
                np.testing.assert_array_equal(d['limits'][:,r.indices],original)
                # A fresh runtime, representing the required restart, sees original
                # native limits, not the previous 0.05 cap, as its new baseline.
                next_runtime,_,_,_=self.runtime();next_runtime.art=r.art
                next_runtime.set_miss_velocity_cap(velocity_cap('regrasp_miss_descent'))
                next_runtime.set_miss_velocity_cap(None)
                np.testing.assert_array_equal(d['limits'][:,r.indices],original)

    def test_native_moving_target_refuses_before_limit_write_or_advance(self):
        r,d,_,writes=self.runtime();d['body_velocity'][0,:3]=[-.2081410487926152,.0459153582818124,.4027715096285998]
        with self.assertRaisesRegex(RuntimeError,'target is moving'):r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        self.assertEqual(writes,[])
        r,d,_,_=self.runtime();r.mode='collect';r.world=NS(current_time=1.);plays=[];r.timeline.play=lambda:plays.append(True)
        r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'));d['body_velocity'][0,0]=.2
        with self.assertRaisesRegex(RuntimeError,'not stationary'):r.advance()
        self.assertEqual(plays,[])
        sample=r._miss_motion_fault['samples'][-1]
        self.assertFalse(sample['stationary_target']['passed']);self.assertEqual(len(sample['positions_rad']),6)

    def test_substep_excess_vectors_survive_terminal_rollback_and_cancellation(self):
        r,d,original,_=self.runtime();r.world=NS(current_time=1.);r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        d['q'][0,r.indices[2]]=.05*POLICY['step_s']+6e-6
        with self.assertRaises(RuntimeError):r.check_miss_substep(1/120)
        sample=r._miss_motion_fault['samples'][-1]
        self.assertEqual(sample['joint_names'],list(JOINTS));self.assertEqual(len(sample['local_excess_rad']),6)
        self.assertEqual(sample['certificate_pad_rad'],5e-6);self.assertGreater(sample['certificate_excess_rad'][2],0.)
        self.assertGreater(sample['frame_excess_over_cap_rad'][2],5e-6)
        cancelled=r.cancel();self.assertEqual(cancelled['miss_motion_fault']['samples'][-1],sample)
        from miss_clearance import residual_summary
        summary=residual_summary([], [cancelled['miss_motion_fault']])
        self.assertTrue(summary['observed_certificate_violation']);self.assertEqual(summary['certificate_pad_rad'],5e-6)
        self.assertGreater(summary['max_frame_excess_over_cap_rad'][2],5e-6)
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)

    def test_guarded_episode_start_freshly_refuses_changed_finger_or_block_offsets(self):
        r,d,_,writes=self.runtime();finger=np.full((1,4),.0001);block=np.full((1,1),.0001)
        def shape(contact):return NS(_physics_view=NS(get_contact_offsets=lambda:contact.copy(),get_rest_offsets=lambda:np.zeros_like(contact)))
        r.finger_views={name:shape(finger) for name in ('gripper_link','moving_jaw_so101_v1_link')}
        r.bodies['block']._physics_view=shape(block)._physics_view
        expected_fingers=r.effective_finger_offsets();expected_block=dict(contact_m=[.0001],rest_m=[0.])
        read=r.guarded_episode_offsets('block',1,10**9,expected_fingers,expected_block)
        self.assertEqual(read['finger_offsets'],expected_fingers)
        for changed in ('finger','block'):
            finger[:]=.0001;block[:]=.0001
            if changed=='finger':finger[0,0]=.01
            else:block[0,0]=.01
            with self.subTest(changed=changed),self.assertRaisesRegex(RuntimeError,'offsets changed'):
                r.guarded_episode_offsets('block',1,10**9,expected_fingers,expected_block)
        self.assertEqual(writes,[])

    def test_pause_failure_cannot_leave_a_resumable_native_runtime(self):
        r,d,original,_=self.runtime();r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        count=[0]
        def pause():
            count[0]+=1
            if count[0]==1:raise RuntimeError('pause failed')
        r.timeline.pause=pause
        with self.assertRaises(RuntimeError):r.cancel()
        self.assertIsNotNone(r.failure)
        with self.assertRaises(RuntimeError):r.advance()
        r.set_miss_velocity_cap(None)
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)

    def test_failing_substep_is_durable_even_when_native_rollback_and_cancel_fail(self):
        r,d,original,_=self.runtime();r.world=NS(current_time=1.)
        published=[];scope=r.publish_miss_fault.__func__.__globals__
        scope['atomic_json']=lambda path,value:published.append(copy.deepcopy(value))
        r.native_miss_fault_path='/unit-test/native-motion-fault.json'
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        def unavailable(*a,**k):raise RuntimeError('native restore unavailable')
        r.art.set_max_joint_velocities=unavailable
        d['q'][0,r.indices[0]]=.05*POLICY['step_s']+8e-6
        with self.assertRaises(RuntimeError):r.check_miss_substep(1/120)
        self.assertTrue(published);self.assertTrue(published[0]['terminal'])
        self.assertGreater(published[0]['samples'][-1]['frame_excess_over_cap_rad'][0],5e-6)
        self.assertIn('rollback_error',published[-1]);self.assertIsNone(r.miss_cap_state()['on'])
        with self.assertRaises(RuntimeError):r.cancel()
        self.assertGreater(published[-1]['samples'][-1]['frame_excess_over_cap_rad'][0],5e-6)
        self.assertEqual(r._miss_saved_limits,original[0].tolist())

    def test_fault_publication_first_failure_recovers_before_native_rollback(self):
        r,d,original,_=self.runtime();r.world=NS(current_time=1.)
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        events=[];published=[];writer_scope=r.publish_miss_fault.__func__.__globals__
        def publish(path,fault):
            events.append('publish')
            if events.count('publish')==1:raise OSError('first publication unavailable')
            published.append(copy.deepcopy(fault));events.append('durable')
        writer_scope['atomic_json']=publish
        setter=r.art.set_max_joint_velocities
        def restore(*args,**kwargs):
            self.assertTrue(r._miss_fault_durable);self.assertTrue(published)
            events.append('rollback');return setter(*args,**kwargs)
        r.art.set_max_joint_velocities=restore
        r.timeline.pause=lambda:events.append('pause')
        d['q'][0,r.indices[0]]=.05*POLICY['step_s']+8e-6
        with self.assertRaises(RuntimeError):r.check_miss_substep(1/120)
        self.assertEqual(events[:5],['pause','publish','publish','durable','rollback'])
        self.assertTrue(published[-1]['rollback_verified']);self.assertEqual(published[0]['publication_attempts'],2)
        self.assertIn('first publication unavailable',published[0]['publication_errors'][0])
        self.assertGreater(published[0]['samples'][-1]['frame_excess_over_cap_rad'][0],5e-6)
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)
        self.assertFalse(r.miss_cap_state()['on']);self.assertIsNotNone(r.failure)
        self.assertIsNotNone(r.cancel()['failure'])
        with self.assertRaises(RuntimeError):r.advance()

    def test_fault_publication_persistent_failure_keeps_limits_unresolved_on_cancel_reset(self):
        r,d,original,writes=self.runtime();r.world=NS(current_time=1.)
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        installed=d['limits'].copy();initial_writes=len(writes);attempts=[];events=[]
        writer_scope=r.publish_miss_fault.__func__.__globals__
        def unavailable(path,fault):attempts.append(copy.deepcopy(fault));raise OSError('journal unavailable')
        writer_scope['atomic_json']=unavailable;r.timeline.pause=lambda:events.append('pause')
        d['q'][0,r.indices[0]]=.05*POLICY['step_s']+8e-6
        with self.assertRaisesRegex(RuntimeError,'not durable'):r.check_miss_substep(1/120)
        self.assertEqual(len(attempts),3);self.assertEqual(events,['pause'])
        self.assertEqual(len(writes),initial_writes);np.testing.assert_array_equal(d['limits'],installed)
        self.assertEqual(r._miss_saved_limits,original[0].tolist())
        self.assertIsNone(r.miss_cap_state()['on']);self.assertTrue(r._miss_limits_dirty)
        self.assertFalse(r._miss_fault_durable);self.assertTrue(r._miss_motion_fault['durability_unresolved'])
        self.assertNotIn('rollback_verified',r._miss_motion_fault)
        for operation in (r.cancel,lambda:r.restore({},arm=True),lambda:r.set_miss_velocity_cap(None)):
            with self.assertRaisesRegex(RuntimeError,'not durable'):operation()
            self.assertEqual(r.epoch,1);self.assertEqual(len(writes),initial_writes)
            np.testing.assert_array_equal(d['limits'],installed)
            self.assertIsNone(r.miss_cap_state()['on'])
        self.assertEqual(len(attempts),12)
        with self.assertRaises(RuntimeError):r.advance()
        with self.assertRaises(RuntimeError):r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        self.assertEqual(len(writes),initial_writes)
        # Storage repair can retire limits, but cannot revive the failed runtime.
        durable=[];writer_scope['atomic_json']=lambda path,fault:durable.append(copy.deepcopy(fault))
        self.assertIsNotNone(r.cancel()['failure']);self.assertTrue(durable[-1]['rollback_verified'])
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)
        with self.assertRaises(RuntimeError):r.advance()

    def test_invalid_kickoff_restores_prior_cap_and_cannot_resume(self):
        for fault in ('epoch','stamp','state','drive'):
            r,d,original,writes=self.runtime();r.invalid_through_ns=0
            r.art.get_applied_actions=lambda:NS(joint_positions=np.zeros((1,6)))
            r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
            args=[1,10**9,velocity_cap('regrasp_reopen'),[0.]*6,[0.]*6]
            if fault=='epoch':args[0]=0
            if fault=='stamp':args[1]=0
            if fault=='state':args[3]=[.1]*6
            if fault=='drive':args[4]=[.1]*6
            with self.subTest(fault=fault),self.assertRaises(RuntimeError):r.start_miss_hold(*args)
            np.testing.assert_array_equal(d['limits'][:,r.indices],original)
            self.assertIsNotNone(r.failure)
            with self.assertRaises(RuntimeError):r.advance()
