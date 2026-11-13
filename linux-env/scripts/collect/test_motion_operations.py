"""Immutable planner continuations and planning/commit ownership regressions."""
import ast
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
import io
import json
import math
from pathlib import Path
import threading
import time
from types import MethodType, SimpleNamespace as NS
import unittest
import test_review_regressions as probes
import test_worker_retirement as workers
from test_safety import method, MotionOperation
from protocol import CancelledGeneration, OwnershipStalled

SHARED=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
ARM=probes.JOINTS[:5]


class Trajectory:
    def __init__(self):self.joint_names=[];self.points=[]


class Point:
    def __init__(self):self.positions=[];self.time_from_start=NS(sec=0,nanosec=0)


class MotionOperationTests(unittest.TestCase):
    def setup_motion(self):
        c,r,entry=probes.ReviewRegressions().collector();node=c.node
        node._motion_context=threading.local()
        node._motion_generation_provider=lambda:c.generation.epoch
        node._motion_effect_guard=c.motion_effect
        node.speed_scale=1.;node.velocity_scale_var=NS(get=lambda:.5)
        node._planning_attempts_var=NS(get=lambda:50)
        node._gui_ready=False;node.joint_lock=threading.Lock()
        node.joint_positions=dict(zip(probes.JOINTS,[0.]*6));node.sliders={};node.slider_labels={}
        node._select_planning_group=lambda group:None;node._publish_goal_state=lambda:None
        node._append_log=lambda *args:None
        c.servo.generation=c.generation;c.servo.plan=None;c.servo.step=lambda play:None
        trace=[];thread_list=[]
        names=('_motion_guard','_require_motion_generation','_motion_token','_new_motion_event',
               '_motion_verdict','_set_motion_status','_current_motion_event','_begin_motion',
               '_motion_thread','_operation','_require_operation','_ik_apply_and_act','_dispatch_motion',
               '_plan_motion_wait','_wait_motion_event',
               '_joint_space_collision_free_execute','_plan_linear_joint_path','_execute_full_trajectory','_trace')
        for name in names:
            fn=method(SHARED,'PickPlaceNode',name)
            fn.__globals__.update(ARM_JOINT_NAMES=ARM,JointTrajectory=Trajectory,JointTrajectoryPoint=Point,
                tracer=NS(event=lambda *a,**k:trace.append((node._motion_token(),a)),snapshot_scene=lambda *a,**k:None))
            setattr(node,name,MethodType(fn,node))
        original=node._motion_thread
        def start(**kwargs):
            thread=original(**kwargs);thread_list.append(thread);return thread
        node._motion_thread=start
        node._trajectory_executor=c.servo.execute
        return c,r,entry,trace,thread_list

    def prevalidator(self,node,operation,value):
        tree=ast.parse(SHARED.read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_prevalidate_and_execute')
        poses=(('approach',.2,0.,.04),('final',.2,0.,.02))
        joints={name:value for name in ARM}
        scope=dict(self=node,operation=operation,evt=operation.event,poses_to_check=poses,
                   ground_z=0.,obj_yaw=0.,approach_h=.02,math=math,
                   check_grasp_reachable=lambda *a,**k:(True,''),
                   find_reachable_grasp_yaw=lambda *a:(0.,{stage:[dict(joints)] for stage,*_ in poses},[]))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SHARED),'exec'),scope)
        return scope['_prevalidate_and_execute']

    def test_old_prevalidator_cannot_supply_current_descent_target_or_event(self):
        c,r,entry,trace,thread_list=self.setup_motion();node=c.node
        old_event=node._begin_motion(node._new_motion_event())
        old=node._operation({},old_event,1.)
        old_check=threading.Event();release_old=threading.Event();submitted=[];approaches=[]
        def valid(target):
            if node._motion_token()==1:
                old_check.set();self.assertTrue(release_old.wait(3))
            return True
        node._check_state_valid_with_contacts=lambda target:(valid(target),[])
        def plan(target,event,**kwargs):
            operation=kwargs['operation'];node._require_operation(operation)
            if not approaches:
                approaches.append(operation);return
            submitted.append(operation)
            trajectory=Trajectory();trajectory.joint_names=ARM
            point=Point();point.positions=[target[n] for n in ARM];point.time_from_start.sec=1
            trajectory.points=[point]
            c.servo.execute(trajectory,event,operation=operation)
        node._joint_space_collision_free_execute=plan
        # Deliberately retain old-style decoy fields. The production prevalidator
        # must never write/read them or use the UI inventory event as an input.
        sentinel=object();node._grasp_final_joints=sentinel;node._grasp_motion_event=sentinel
        old_thread=node._motion_thread(target=self.prevalidator(node,old,.7));old_thread.start()
        self.assertTrue(old_check.wait(2));c.timeline(False)
        current_event=node._begin_motion(node._new_motion_event())
        current=node._operation({},current_event,1.)
        self.prevalidator(node,current,-.2)()
        self.assertEqual(len(approaches),1)
        release_old.set();old_thread.join(2);self.assertFalse(old_thread.is_alive())
        self.assertIs(node._grasp_final_joints,sentinel);self.assertIs(node._grasp_motion_event,sentinel)
        # Even a poisoned obsolete UI slot cannot replace the closure's input.
        node._motion_event=old_event
        approaches[0].event.verdict=dict(ok=True);approaches[0].event.set()
        for thread in thread_list:thread.join(2);self.assertFalse(thread.is_alive())
        self.assertEqual(len(submitted),1)
        accepted=submitted[0]
        self.assertEqual(accepted.joints(),{name:-.2 for name in ARM})
        self.assertIs(accepted.event,current_event);self.assertEqual(accepted.epoch,2)
        workers.WorkerRetirementTests().capture(c,r,entry,2,1033333333,False)
        self.assertIsNone(c.failure)
        self.assertTrue(all(value<0 for value in c.servo.reference[:5]))
        self.assertEqual(c.events,['pause','target_write','advance'])

    def test_target_is_frozen_and_consumption_checks_origin_and_event_identity(self):
        c,r,_,_,_=self.setup_motion();node=c.node
        source={name:-.2 for name in ARM};event=node._new_motion_event()
        operation=node._operation(source,event,1.)
        source[ARM[0]]=.7
        self.assertEqual(operation.joints()[ARM[0]],-.2)
        with self.assertRaises(FrozenInstanceError):operation.epoch=2
        c.timeline(False)
        with self.assertRaises(CancelledGeneration):
            node._ik_apply_and_act(operation.joints(),'grasp_execute',operation=operation)
        self.assertIsNone(c.servo.plan)
        bad=node._new_motion_event();bad.motion_epoch=1
        with self.assertRaisesRegex(ValueError,'different origins'):node._operation(source,bad,1.)

    def valid_plan_during_capture(self,delay=None):
        c,r,entry,_,_=self.setup_motion();node=c.node
        c.generation.retirement_timeout=.05 if delay is None else 15.
        entered=threading.Event();release=threading.Event();checks=[];failures=[]
        def valid(target):
            checks.append(target);entered.set()
            if delay is None:self.assertTrue(release.wait(3))
            else:threading.Event().wait(delay)
            return True,[]
        node._check_state_valid_with_contacts=valid
        event=node._begin_motion(node._new_motion_event())
        operation=node._operation({name:.1 for name in ARM},event,1.)
        def planning():
            try:node._joint_space_collision_free_execute(operation.joints(),event,operation=operation)
            except Exception as exc:failures.append(repr(exc))
        planner=node._motion_thread(target=planning)
        started=time.monotonic();planner.start();self.assertTrue(entered.wait(2))
        capture=threading.Thread(target=c.capture);capture.start();capture.join(2)
        self.assertFalse(capture.is_alive());self.assertIsNone(c.failure)
        self.assertIsNone(c.generation.stalled);self.assertEqual(r.epoch,1)
        self.assertTrue(planner.is_alive())
        capture_wall=time.monotonic()-started
        release.set();planner.join(25)
        self.assertFalse(planner.is_alive());self.assertFalse(failures)
        self.assertEqual(len(checks),51)
        self.assertIs(c.servo.plan['done'],event)
        return dict(passed=True,retirement_timeout=c.generation.retirement_timeout,
                    validity_calls=len(checks),capture_wall_seconds=capture_wall,
                    plan_wall_seconds=time.monotonic()-started,native_epoch=r.epoch,
                    poisoned=bool(c.generation.stalled))

    def test_all_51_validity_checks_run_without_generation_ownership(self):
        self.valid_plan_during_capture()

    def test_delayed_cycle_request_cannot_bind_to_a_later_epoch(self):
        c,r,_,_,_=self.setup_motion();selected=[]
        epoch=c.generation.epoch
        c.timeline(False)
        with self.assertRaises(CancelledGeneration):
            c.motion_call(lambda:selected.append('old object'),epoch=epoch)
        self.assertFalse(selected)
        original=c.node._new_motion_event();c.node._begin_motion(original)
        fresh=c.node._new_motion_event()
        self.assertIs(c.motion_call(lambda:c.node._begin_motion(fresh),epoch=2),fresh)
        self.assertIs(c.node._current_motion_event(),original)

    def test_capture_timeout_uses_common_emergency_abort_and_diagnostic(self):
        c,r,_,_,_=self.setup_motion();c.generation.retirement_timeout=.03
        held=threading.Event();release=threading.Event();output=io.StringIO()
        def owner():
            with c.generation.own():held.set();release.wait(2)
        thread=threading.Thread(target=owner);thread.start();self.assertTrue(held.wait(2))
        try:
            with redirect_stdout(output):c.capture()
            self.assertTrue(c.node._qs_abort_evt.is_set());self.assertTrue(c.stop.is_set())
            self.assertEqual(c.events,['pause']);self.assertEqual(r.epoch,2)
            message=json.loads(output.getvalue())
            self.assertEqual(message['requester'],'capture/dispatch')
            self.assertEqual(message['event'],'stalled_generation_owner')
            self.assertIsNone(c.generation.epoch)
        finally:release.set();thread.join(2)
    def test_capture_emergency_rpc_failure_keeps_abort_and_poison(self):
        c,r,_,_,_=self.setup_motion();c.generation.retirement_timeout=.02
        def unavailable(play):raise TimeoutError('injected native unreachability')
        c.abort_stalled_owner.__func__.__globals__['timeline']=unavailable
        held=threading.Event();release=threading.Event();output=io.StringIO()
        def owner():
            with c.generation.own():held.set();release.wait(2)
        thread=threading.Thread(target=owner);thread.start();self.assertTrue(held.wait(2))
        try:
            with redirect_stdout(output):c.capture()
            self.assertTrue(c.node._qs_abort_evt.is_set());self.assertTrue(c.stop.is_set())
            self.assertIn('emergency pause failed',c.failure)
            self.assertEqual(json.loads(output.getvalue())['requester'],'capture/dispatch')
            self.assertIsNone(c.generation.epoch)
        finally:release.set();thread.join(2)
        with self.assertRaises(OwnershipStalled):c.generation.install(dict(epoch=3,invalid_through_ns=0))


if __name__=='__main__':unittest.main()
