"""Cross-episode gripper/scene continuation and stalled-owner regressions."""
import copy
import json
import threading
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from protocol import CancelledGeneration, OwnershipStalled
import test_review_regressions as probes
from test_safety import method

ROOT=Path(__file__).resolve().parents[3]
SHARED=ROOT/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'


class WorkerRetirementTests(unittest.TestCase):
    def setup_worker(self, delay):
        c,r,entry=probes.ReviewRegressions().collector()
        c.servo.generation=c.generation
        c.servo.plan=None;c.node._actual_positions['gripper_joint']=0.
        c.node._gui_ready=False
        c.node._motion_context=threading.local()
        c.node._motion_generation_provider=lambda:c.generation.epoch
        c.node._motion_effect_guard=c.motion_effect
        for name in ('_motion_guard','_require_motion_generation','_set_motion_status','_motion_token',
                     '_new_motion_event','_motion_thread','_begin_motion','_current_motion_event','_cmd_gripper_close_for_object'):
            setattr(c.node,name,MethodType(method(SHARED,'PickPlaceNode',name),c.node))
        for name in ('_plan_motion_wait','_wait_motion_event'):
            setattr(c.node,name,MethodType(method(SHARED,'PickPlaceNode',name),c.node))
        c.node._get_selected_object_name=lambda:'red_2x3_0'
        c.node._lookup_bbox=lambda name:dict(sx=.031,sy=.0206,sz=.0128)
        c.node._gripper_angle_for_object=lambda name:(.2,.1)
        c.node._grasp_grip_duration_var=NS(get=lambda:.5)
        c.node._append_log=lambda *args:None;c.node._gripper_command=lambda *a,**k:None
        c.node._attached_lego_name=None
        def attach(name):
            c.events.append('ATTACH');c.node._attached_lego_name=name;return True
        c.node._attach_lego_to_gripper=attach
        entered=threading.Event();woke=threading.Event();release=threading.Event();workers=[]
        c.servo.step=lambda play:entered.set()
        original_new=c.node._new_motion_event
        def new_event():
            event=original_new()
            if delay=='after_wait' and threading.current_thread() is not threading.main_thread():
                original_wait=event.wait
                def wait(timeout=None):
                    result=original_wait(timeout)
                    woke.set()
                    if not release.wait(3):raise TimeoutError('Test did not release delayed gripper wait')
                    return result
                event.wait=wait
            return event
        c.node._new_motion_event=new_event
        def send(position,duration_s,blocking):
            ok=c.servo.gripper(position,duration_s,blocking)
            if delay=='after_return':
                woke.set()
                if not release.wait(3):raise TimeoutError('Test did not release delayed gripper return')
            return ok
        c.node._send_gripper_goal=send
        original_thread=c.node._motion_thread
        def thread(**kwargs):
            worker=original_thread(**kwargs);workers.append(worker);return worker
        c.node._motion_thread=thread
        c.node._cmd_gripper_close_for_object()
        self.assertTrue(entered.wait(2))
        return c,r,entry,woke,release,workers

    def capture(self,c,r,entry,epoch,stamp,closed,home=False):
        positions=[.1 if home else 0.]*5+[.1 if closed else 0.]
        entry['joint'][0].position=positions
        native=entry['tick'][0];native.update(epoch=epoch,sim_ns=stamp,state=list(positions))
        entry['joint'][0].header.stamp.sec,entry['joint'][0].header.stamp.nanosec=divmod(stamp,10**9)
        r.stamp=lambda:stamp
        c.node._actual_positions.update(zip(probes.JOINTS,positions))
        c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=stamp))
        r.ack_pub=NS(publish=lambda msg:c.sensors.acks.update({(epoch,stamp):json.loads(msg.data)}))
        c.sensors.read_ack=lambda e,t:c.sensors.acks[(e,t)]
        if home:
            done=c.node._new_motion_event();c.node._motion_event=done
            c.servo.plan=dict(trajectory=NS(points=[NS(time_from_start=NS(sec=1,nanosec=0),positions=positions[:5])],joint_names=probes.JOINTS[:5]),done=done,start=stamp-2*10**9,epoch=epoch)
        def read(**kwargs):
            c.stop.set();return stamp,entry
        c.sensors.read=read;c.stop.clear();c.capture()
        self.assertIsNone(c.failure)

    def gripper_interleaving(self,closed,delay):
        c,r,entry,woke,release,workers=self.setup_worker(delay)
        try:
            if closed:
                c.servo.grip_plan.update(start=-10**9,initial=0.)
                self.capture(c,r,entry,1,10**9,True)
                self.assertTrue(woke.wait(2))
            result=c.timeline(False)
            self.assertEqual(result['epoch'],2)
            if not closed:self.assertTrue(woke.wait(2))
            # A real next-generation home convergence writes a true status.
            self.capture(c,r,entry,2,1033333333,closed,home=True)
            home_status=copy.deepcopy(c.node._last_motion_status)
            self.assertTrue(home_status['ok']);self.assertEqual(home_status['epoch'],2)
            release.set()
            for worker in workers:
                worker.join(2);self.assertFalse(worker.is_alive())
            self.assertNotIn('ATTACH',c.events)
            self.assertIsNone(c.node._attached_lego_name)
            self.assertEqual(c.node._last_motion_status,home_status)
        finally:
            release.set()
            for worker in workers:worker.join(2)

    def test_unfinished_gripper_cannot_consume_next_home_success(self):
        self.gripper_interleaving(False,'after_wait')

    def test_closed_gripper_delayed_after_wait_cannot_attach_after_cancel(self):
        self.gripper_interleaving(True,'after_wait')

    def test_closed_gripper_delayed_after_success_return_cannot_attach_after_cancel(self):
        self.gripper_interleaving(True,'after_return')

    def test_gripper_uses_its_own_failed_verdict_despite_shared_success(self):
        c,r,_,woke,release,workers=self.setup_worker('after_wait')
        try:
            done=c.servo.grip_plan['done']
            done.verdict=dict(ok=False,outcome='operation_failed');done.set()
            self.assertTrue(woke.wait(2))
            c.node._last_motion_status=dict(ok=True,outcome='other_operation')
            release.set()
            for worker in workers:worker.join(2);self.assertFalse(worker.is_alive())
            self.assertNotIn('ATTACH',c.events)
            self.assertFalse(c.node._last_motion_status['ok'])
        finally:
            release.set()
            for worker in workers:worker.join(2)

    def test_stale_scene_and_status_boundaries_refuse_mutation(self):
        c,r,_,_,release,workers=self.setup_worker('after_wait')
        try:
            c.timeline(False)
            c.node._motion_context.epoch=1
            for name,args in (('_attach_lego_to_gripper',('red_2x3_0',)),('_detach_lego_sync',()),
                              ('_dispatch_scene_update',(lambda:c.events.append('SCENE'),)),
                              ('_set_motion_status',(dict(ok=True,outcome='stale'),))):
                boundary=method(SHARED,'PickPlaceNode',name)
                boundary.__globals__['MOVEIT_AVAILABLE']=False
                with self.subTest(name=name):
                    with self.assertRaises(CancelledGeneration):boundary(c.node,*args)
            self.assertNotIn('SCENE',c.events);self.assertNotIn('ATTACH',c.events)
        finally:
            release.set()
            for worker in workers:worker.join(2)

    def test_cancel_serializes_with_attach_transaction_and_bookkeeping(self):
        c,r,_,_,release,workers=self.setup_worker('after_return')
        # Complete the physical operation, then let the continuation enter its
        # actual shared guard; hold the fake scene boundary inside that guard.
        plan=c.servo.grip_plan;plan['done'].verdict=dict(ok=True);plan['done'].set()
        entered=threading.Event();finish=threading.Event();cancelled=threading.Event()
        def attach(name):
            entered.set();self.assertTrue(finish.wait(2))
            c.node._attached_lego_name=name;c.events.append('ATTACH');return True
        c.node._attach_lego_to_gripper=attach
        release.set();self.assertTrue(entered.wait(2))
        reset=threading.Thread(target=lambda:(c.timeline(False),cancelled.set()));reset.start()
        self.assertFalse(cancelled.wait(.05))
        finish.set()
        for worker in workers:worker.join(2)
        reset.join(2);self.assertTrue(cancelled.is_set());self.assertFalse(reset.is_alive())
        # Attach is entirely before cancellation returns, never after it.
        self.assertEqual(c.events,['ATTACH','pause'])

    def test_stalled_publication_aborts_with_emergency_native_epoch_invalidation(self):
        c,r,_=probes.ReviewRegressions().collector()
        c.generation.retirement_timeout=.05
        entered=threading.Event();release=threading.Event()
        publish=c.command_pub.publish
        def blocked(msg):
            entered.set();release.wait(2);publish(msg)
        c.command_pub.publish=blocked
        capture=threading.Thread(target=c.capture);capture.start();self.assertTrue(entered.wait(2))
        start=time.monotonic()
        try:
            with self.assertRaisesRegex(OwnershipStalled,'Stalled generation owner'):c.timeline(False)
            self.assertLess(time.monotonic()-start,.5)
            self.assertEqual(c.events,['pause']);self.assertEqual(r.epoch,2)
            self.assertIsNone(c.generation.epoch);self.assertTrue(c.stop.is_set())
            self.assertIn('Stalled generation owner',c.failure)
        finally:
            release.set();capture.join(2)
        self.assertFalse(capture.is_alive());self.assertEqual(c.events,['pause'])
        with self.assertRaises(OwnershipStalled):c.generation.install(dict(epoch=3,invalid_through_ns=10**9))

    def test_stalled_reset_never_runs_pose_setters_as_emergency_cancel(self):
        c,r,_=probes.ReviewRegressions().collector();c.generation.retirement_timeout=.02
        held=threading.Event();release=threading.Event()
        def owner():
            with c.generation.own():held.set();release.wait(2)
        worker=threading.Thread(target=owner);worker.start();self.assertTrue(held.wait(2))
        try:
            with self.assertRaises(OwnershipStalled):c.cancel_owner(lambda:c.events.append('RESTORE'))
            self.assertEqual(c.events,['pause'])
        finally:release.set();worker.join(2)


if __name__=='__main__':unittest.main()
