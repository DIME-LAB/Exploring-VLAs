"""Deferred Kit pause is acknowledged before native limit retirement."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
from pause_barrier import pause_committed
from protocol import StepGeneration,CancelledGeneration
from contract import JOINTS
import test_miss_velocity as native_tests
from test_miss_velocity import velocity_cap
from contract import parameters
from test_review_regressions import load_class


class DelayedTimeline:
    def __init__(self,playing=True,stuck=False):
        self.playing=playing;self.pending=False;self.updates=0;self.stuck=stuck
    def is_playing(self):return self.playing
    def pause(self):self.pending=True
    def commit(self):
        self.updates+=1
        if self.pending and not self.stuck:self.playing=False;self.pending=False


class PauseBarrierTests(unittest.TestCase):
    def test_pause_commits_one_update_without_advancing_physics(self):
        timeline=DelayedTimeline();proof=pause_committed(timeline)
        self.assertFalse(timeline.is_playing());self.assertEqual(timeline.updates,1)
        self.assertEqual(proof['commits'],1);self.assertEqual(proof['physics_steps'],0)

    def test_stuck_pause_is_bounded_and_refuses_native_restore(self):
        r,d,original,writes=native_tests.NativeMissVelocityTests().runtime()
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        r.timeline=DelayedTimeline(stuck=True);count=len(writes);r.world=NS(current_time=1.)
        with self.assertRaises(RuntimeError):r.cancel()
        self.assertEqual(len(writes),count);self.assertEqual(r.timeline.updates,6)
        self.assertTrue(r._miss_limits_dirty);self.assertIsNone(r.miss_cap_state()['on'])

    def test_cancel_of_real_cap_writes_only_after_pending_pause_commits(self):
        r,d,original,writes=native_tests.NativeMissVelocityTests().runtime()
        r.set_miss_velocity_cap(velocity_cap('regrasp_miss_close'))
        r.timeline=DelayedTimeline();r.world=NS(current_time=1.)
        setter=r.art.set_max_joint_velocities
        def held_write(*a,**k):
            self.assertFalse(r.timeline.is_playing());self.assertEqual(r.timeline.updates,1)
            setter(*a,**k)
        r.art.set_max_joint_velocities=held_write
        result=r.cancel();self.assertIsNone(result['failure'])
        np.testing.assert_array_equal(d['limits'][:,r.indices],original)

    def test_noop_kickoff_ack_keeps_stamp_and_never_advances(self):
        r,d,_,writes=native_tests.NativeMissVelocityTests().runtime();r.invalid_through_ns=0
        r.art.get_applied_actions=lambda:NS(joint_positions=d['q'].copy())
        r.advance=lambda: self.fail('No-op advanced physics')
        ack=r.start_miss_hold(1,10**9,velocity_cap('regrasp_miss_open'),[0.]*6,[0.]*6,advance=False)
        self.assertTrue(ack['held']);self.assertIsNone(ack['advance'])
        self.assertEqual(ack['observation_stamp_ns'],10**9);self.assertEqual(len(writes),1)

    def test_held_ack_cannot_install_miss_limits_in_deploy_mode(self):
        r,d,_,writes=native_tests.NativeMissVelocityTests().runtime();r.mode='deploy';r.invalid_through_ns=0
        with self.assertRaisesRegex(RuntimeError,'requires collect mode'):
            r.start_miss_hold(1,10**9,velocity_cap('regrasp_miss_open'),[0.]*6,[0.]*6,advance=False)
        self.assertEqual(writes,[])

    def test_recovery_gripper_already_open_uses_held_ack_without_a_plan_or_wait(self):
        Servo,scope=load_class('collect.py','Servo');scope.update(JointTrajectory=NS,parameters=parameters);generation=StepGeneration();generation.install(dict(epoch=1,invalid_through_ns=0))
        node=NS(create_publisher=lambda *a:NS(),_motion_token=lambda:1,_actual_positions=dict(zip(JOINTS,[0.]*5+[.1132175])))
        calls=[]
        def held(play,**kwargs):calls.append((play,kwargs));return dict(held=True,advance=None)
        servo=Servo(node,generation,held);outcome={}
        self.assertTrue(servo.gripper(.1132176,3.,True,phase='regrasp_miss_open',outcome=outcome))
        self.assertEqual(calls,[(True,dict(advance=False))]);self.assertIsNone(servo.grip_plan)
        self.assertTrue(outcome['already_at_target']);self.assertEqual(servo.execution_budget(),(0.,0))

    def test_recovery_noop_revalidates_its_worker_epoch_after_ack(self):
        Servo,scope=load_class('collect.py','Servo');scope.update(JointTrajectory=NS,parameters=parameters);generation=StepGeneration();generation.install(dict(epoch=1,invalid_through_ns=0))
        node=NS(create_publisher=lambda *a:NS(),_motion_token=lambda:1,_actual_positions=dict(zip(JOINTS,[0.]*6)))
        def changed(*a,**k):generation.install(dict(epoch=2,invalid_through_ns=0));return dict(held=True,advance=None)
        servo=Servo(node,generation,changed)
        with self.assertRaises(CancelledGeneration):servo.gripper(0.,3.,True,phase='regrasp_miss_open')



if __name__=='__main__':unittest.main()
