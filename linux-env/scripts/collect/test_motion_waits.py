"""Actual shared release path under slow /clock, watchdogs and GUI parity."""
import math
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from test_safety import method
from so_arm101_control.motion_wait import MotionBudget, MotionWaiter, MotionWaitTimeout

SHARED=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'


class Clock:
    def __init__(self,rtf):self.rtf=rtf;self.sim=0.;self.wall=0.
    def advance(self,seconds):self.wall+=seconds;self.sim+=seconds*self.rtf


class Event:
    def __init__(self,clock,finish,epoch=7):
        self.clock=clock;self.finish=finish;self.done=False;self.motion_epoch=epoch
        self.motion_tag='gripper_release';self.motion_budget=MotionBudget(epoch);self.motion_budget.plan(3.)
        self.verdict=None
    def is_set(self):return self.done
    def set(self):self.done=True
    def wait(self,timeout):
        self.clock.advance(timeout)
        if self.clock.sim>=self.finish:
            self.done=True;self.verdict=dict(ok=True)
        return self.done


class MotionWaitTests(unittest.TestCase):
    def setup_release(self,rtf,finish=3.):
        clock=Clock(rtf);event=Event(clock,finish);records=[];effects=[];active=[7]
        def validate(epoch):
            if epoch!=active[0]:raise ValueError('Old operation refused')
        node=NS(_last_grasped_object='red',_attached_lego_name='red',_qs_abort_evt=threading.Event(),
            _motion_guard=nullcontext,_motion_token=lambda:7,_trace=lambda *a,**kw:None,
            _select_grasp_object=lambda *a:None,_cmd_gripper_open_for_object=lambda:None,
            _current_motion_event=lambda:event,_require_motion_generation=lambda:validate(7),
            _motion_verdict=lambda evt:evt.verdict,_ensure_lego_state=lambda:None,
            _detach_lego_sync=lambda:effects.append('detach') or True)
        def status(value,event):event.verdict=value
        node._set_motion_status=status
        node._motion_waiter=MotionWaiter(lambda:clock.sim,validate,records.append,wall=lambda:clock.wall)
        node._wait_motion_event=MethodType(method(SHARED,'PickPlaceNode','_wait_motion_event'),node)
        return node,event,clock,records,effects,active

    def test_actual_release_at_rtf_027_and_015_completes_after_ten_wall_seconds(self):
        release=method(SHARED,'PickPlaceNode','_cmd_drop_release')
        for rtf in (.27,.15):
            with self.subTest(rtf=rtf):
                node,event,clock,records,effects,_=self.setup_release(rtf)
                release(node)
                self.assertGreater(clock.wall,10.);self.assertGreaterEqual(clock.sim,3.)
                self.assertEqual(effects,['detach']);self.assertEqual(records,[])
                self.assertTrue(event.verdict['ok'])

    def test_actual_release_true_clock_hang_trips_logged_watchdog_and_never_detaches(self):
        node,event,clock,records,effects,_=self.setup_release(0.,math.inf)
        with self.assertRaises(MotionWaitTimeout) as raised:
            method(SHARED,'PickPlaceNode','_cmd_drop_release')(node)
        self.assertEqual(raised.exception.diagnostic['kind'],'wall_watchdog')
        self.assertEqual(records[-1]['planned_sim_s'],3.)
        self.assertGreater(clock.wall,records[-1]['wall_watchdog_s'])
        self.assertEqual(clock.sim,0.);self.assertEqual(effects,[])
        self.assertTrue(node._qs_abort_evt.is_set());self.assertFalse(event.verdict['ok'])

    def test_advancing_clock_noncompletion_trips_sim_deadline(self):
        node,event,clock,records,effects,_=self.setup_release(.15,math.inf)
        with self.assertRaises(MotionWaitTimeout):node._wait_motion_event(event,10.)
        self.assertEqual(records[-1]['kind'],'sim_deadline')
        self.assertGreater(clock.sim,8.);self.assertEqual(effects,[])

    def test_cancelled_worker_cannot_finish_or_detach_in_next_epoch(self):
        node,event,clock,records,effects,active=self.setup_release(.15)
        original=event.wait
        def park(timeout):
            result=original(timeout)
            if clock.wall>.5:active[0]=8
            return result
        event.wait=park
        with self.assertRaisesRegex(ValueError,'Old operation'):
            method(SHARED,'PickPlaceNode','_cmd_drop_release')(node)
        self.assertEqual(effects,[]);self.assertEqual(records,[])
        self.assertFalse(node._qs_abort_evt.is_set())

    def test_gui_release_preserves_exact_wall_wait_without_reading_sim_clock(self):
        node,event,clock,records,effects,_=self.setup_release(.15)
        del node._motion_waiter
        called=[]
        def gui_wait(timeout):
            called.append(timeout);event.verdict=dict(ok=True);return True
        event.wait=gui_wait
        node.get_clock=lambda:(_ for _ in ()).throw(AssertionError('GUI must keep its wall wait'))
        method(SHARED,'PickPlaceNode','_cmd_drop_release')(node)
        self.assertEqual(called,[10.]);self.assertEqual(effects,['detach']);self.assertEqual(records,[])

    def test_gui_action_result_keeps_duration_plus_wall_timeout(self):
        seen=[];result=object()
        node=NS(_wait_future=lambda future,**kw:seen.append(kw) or result)
        self.assertIs(method(SHARED,'PickPlaceNode','_wait_motion_future')(node,object(),3.,13.),result)
        self.assertEqual(seen,[dict(timeout_sec=13.)])

    def test_approach_and_descent_own_both_timing_budgets(self):
        parent=MotionBudget(7);parent.plan(3.)
        child=MotionBudget(7,parent);child.plan(6.)
        self.assertEqual(parent.total(),(9.,2));self.assertEqual(child.total(),(6.,1))
        with self.assertRaises(ValueError):MotionBudget(8,parent)
        self.assertEqual(parent.total(),(9.,2))

    def test_episode_wall_watchdog_grows_with_dispatched_motion(self):
        clock=Clock(.15);records=[]
        def advance(dt):
            # Twenty seconds of planning are wall work with held simulation.
            clock.wall+=dt
            if clock.wall>20:clock.sim+=dt*.15
        thread=NS(is_alive=lambda:clock.sim<30,join=lambda timeout:advance(timeout))
        def budget():
            count=0 if clock.wall<20 else min(6,1+int(clock.sim/5))
            return count*5.,count
        waiter=MotionWaiter(lambda:clock.sim,lambda epoch:None,records.append,wall=lambda:clock.wall)
        self.assertTrue(waiter.wait_thread(thread,budget,7,180.))
        self.assertGreater(clock.wall,180.);self.assertEqual(records,[])


    def test_actual_servo_gripper_long_motion_uses_its_planned_sim_duration(self):
        from test_review_regressions import load_class
        Servo,_=load_class('collect.py','Servo')
        for rtf in (.27,.15):
            node,event,clock,records,_,_=self.setup_release(rtf,15.)
            node._actual_positions={'gripper_joint':0.}
            node._new_motion_event=lambda:event
            node._plan_motion_wait=MethodType(method(SHARED,'PickPlaceNode','_plan_motion_wait'),node)
            servo=object.__new__(Servo);servo.node=node;servo.lock=threading.RLock()
            servo.generation=NS(own=lambda *a:nullcontext(),require=lambda epoch:None)
            servo.reference=None;servo.grip_plan=None;servo.planned_sim_seconds=0.;servo.planned_operations=0
            servo.step=lambda play:None
            self.assertTrue(servo.gripper(.1,15.,True))
            self.assertGreaterEqual(clock.sim,15.);self.assertEqual(records,[])
            self.assertEqual(servo.execution_budget(),(15.,1))
            if rtf==.15:self.assertGreater(clock.wall,60.)

    def test_cancellation_between_timeout_and_status_commit_has_no_new_epoch_effect(self):
        import test_motion_operations as operations
        from protocol import CancelledGeneration
        c,_,_,_,_=operations.MotionOperationTests().setup_motion()
        node=c.node;node._motion_context.epoch=1
        clock=Clock(0.);event=Event(clock,math.inf,epoch=1)
        def emit(record):
            c.generation.install(dict(epoch=2,invalid_through_ns=0))
            node._last_motion_status=dict(ok=True,epoch=2)
        node._motion_waiter=MotionWaiter(lambda:clock.sim,c.generation.require,emit,wall=lambda:clock.wall)
        with self.assertRaises(CancelledGeneration):node._wait_motion_event(event,10.)
        self.assertEqual(node._last_motion_status,dict(ok=True,epoch=2))
        self.assertFalse(node._qs_abort_evt.is_set());self.assertFalse(event.is_set())

    def test_motion_budget_rejects_nonfinite_durations_and_wait_policy(self):
        for value in (0.,-1.,math.inf,math.nan):
            with self.assertRaises(ValueError):MotionBudget(7).plan(value)
            with self.assertRaises(ValueError):MotionWaiter(lambda:0,lambda e:None,lambda r:None,rtf_floor=value)

    def test_collector_attach_refuses_unheld_state_without_waiting_under_scene_ownership(self):
        node=NS(_simulation_held=lambda:False,_motion_waiter=object())
        with self.assertRaisesRegex(RuntimeError,'requires a held physical snapshot'):
            method(SHARED,'PickPlaceNode','_wait_arm_at_rest')(node)

    def test_effective_offsets_use_tensor_values_and_refuse_sentinels(self):
        import numpy as np
        reader=method(Path(__file__).with_name('collection_runtime.py'),'Runtime','resolved_shape_offsets')
        view=NS(_physics_view=NS(get_contact_offsets=lambda:np.array([[.0013625]]),
                                get_rest_offsets=lambda:np.array([[0.]])))
        runtime=NS(bodies={'red_2x3_0':view})
        self.assertEqual(reader(runtime)['red_2x3_0'],dict(contact_m=[.0013625],rest_m=[0.]))
        view._physics_view.get_rest_offsets=lambda:np.array([[-math.inf]])
        with self.assertRaisesRegex(RuntimeError,'Unresolved effective shape offsets'):reader(runtime)

if __name__=='__main__':unittest.main()
