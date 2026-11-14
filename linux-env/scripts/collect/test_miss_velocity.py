"""Native velocity-bound application, retirement and measured-state evidence."""
import copy,json,unittest
from types import SimpleNamespace as NS
import numpy as np
from contract import JOINTS
from miss_clearance import velocity_cap,POLICY
from test_review_regressions import load_class


class NativeMissVelocityTests(unittest.TestCase):
    def runtime(self):
        Runtime,scope=load_class('collection_runtime.py','Runtime');r=object.__new__(Runtime)
        r.busy=False;r.timeline=NS(is_playing=lambda:False,pause=lambda:None);r.failure=None
        r.indices=[4,1,5,0,2,3]  # All reads/writes use named-joint mapping.
        r._miss_cap=None;r._miss_saved_limits=None;r._miss_last_position=None
        data=dict(limits=np.asarray([[4.,5.,6.,7.,8.,9.]],dtype=np.float32),q=np.zeros((1,6),dtype=np.float32),v=np.zeros((1,6),dtype=np.float32))
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
        with self.assertRaisesRegex(RuntimeError,'identity'):r.start_miss_hold(0,10**9,cap,[0.]*6,[0.]*6)
        with self.assertRaisesRegex(RuntimeError,'baseline'):r.start_miss_hold(1,10**9,cap,[.1]*6,[0.]*6)
        self.assertEqual(writes,[]);self.assertEqual(seen,[])
        ack=r.start_miss_hold(1,10**9,cap,[0.]*6,[0.]*6)
        self.assertEqual(ack['command_miss_velocity_cap'],cap);self.assertEqual(seen,['advance'])

    def test_cancel_restores_original_limits_before_installing_new_epoch(self):
        r,d,original,_=self.runtime();r.epoch=1;r.world=NS(current_time=1.);r.stamp=lambda:10**9
        r.set_miss_velocity_cap(velocity_cap('regrasp_reopen'))
        result=r.cancel()
        self.assertEqual(result['epoch'],2);self.assertFalse(r.miss_cap_state()['on'])
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)

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
