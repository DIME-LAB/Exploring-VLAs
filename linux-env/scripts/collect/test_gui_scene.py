"""Complete GUI scene transactions, late responses, callback and epoch regressions."""
import asyncio,copy,io,threading,time,unittest
from concurrent.futures import Future
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MethodType,SimpleNamespace as NS
from test_safety import method
import test_scene_lifetime as lifetime
from test_scene_lifetime import Scene, MESSAGES
import test_motion_operations as motions
from so_arm101_control.scene_transactions import SceneMutationClient,SceneMutationIncomplete,SceneExecution,GuiSceneNeedsResync
from so_arm101_control.ros_callbacks import safe_callback
from protocol import CancelledGeneration,OwnershipStalled
SHARED=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'


def until(condition,seconds=2):
    end=time.monotonic()+seconds
    while not condition():
        if time.monotonic()>end:raise TimeoutError('regression condition not reached')
        threading.Event().wait(.002)


class GuiSceneTests(unittest.TestCase):
    def fixture(self):
        c,r,entry,server,markers=lifetime.SceneLifetimeTests().setup_scene();n=c.node
        n.execution=SceneExecution();n._motion_effect_guard=None;n._scene_failure_handler=None
        n._gui_scene_warning=lambda message:None
        raw=NS(call_async=server.apply,wait_for_service=lambda **kw:True)
        repair=method(SHARED,'PickPlaceNode','_repair_gui_scene')
        repair.__globals__['GetPlanningSceneSrv']=MESSAGES.Get
        n._repair_gui_scene=MethodType(repair,n)
        n._apply_scene_client=SceneMutationClient(raw,n._motion_guard,n._scene_unresolved,
                                                collector=False,repair=n._repair_gui_scene)
        n._wait_future=MethodType(method(SHARED,'PickPlaceNode','_wait_future'),n)
        n._apply_scene_client.gui.wait_scale=.002
        n._apply_scene_client.gui.grace_seconds=.2
        n._attached_lego_tcp_offset=None
        return c,r,server,markers

    def pair(self,server,a,b):
        acm=server.scene.allowed_collision_matrix
        return acm.entry_values[acm.entry_names.index(a)].enabled[acm.entry_names.index(b)]

    def test_late_ground_add_completes_remaining_acm_and_repairs_before_next_job(self):
        c,r,s,markers=self.fixture();n=c.node;q=n._apply_scene_client.gui;s.pending_kind='world_add'
        n._cmd_toggle_ground_plane();until(lambda:bool(s.pending))
        until(lambda:q.needs_resync)  # Exceeds the actual operation's scaled 5-second wait.
        later=[];next_job=n._dispatch_scene_update(lambda:later.append(self.pair(s,'base','ground_plane')))
        self.assertFalse(later)
        s.pending.pop()();next_job.result(timeout=2)
        self.assertEqual(later,[True]);self.assertTrue(self.pair(s,'base','ground_plane'))
        self.assertFalse(q.needs_resync);self.assertTrue(n._gui_scene_repair_receipt)
        self.assertIsNone(n._apply_scene_client.failure);self.assertFalse(markers)

    def test_response_after_grace_is_reconciled_from_moveit_before_another_transaction(self):
        c,r,s,_=self.fixture();n=c.node;q=n._apply_scene_client.gui;s.pending_kind='world_add';q.grace_seconds=.015
        first=n._dispatch_scene_update(lambda:n._cmd_toggle_ground_plane())
        # Nested async dispatch adds the operation to this owner's FIFO.
        first.result(timeout=1);until(lambda:bool(s.pending));until(lambda:q.worker is None)
        self.assertTrue(q.needs_resync);s.pending.pop()()
        later=[];n._dispatch_scene_update(lambda:later.append(self.pair(s,'base','ground_plane'))).result(timeout=1)
        self.assertEqual(later,[True]);self.assertFalse(q.needs_resync)

    def test_get_service_unavailable_after_ground_add_requires_repair_before_next_job(self):
        c,r,s,_=self.fixture();n=c.node;q=n._apply_scene_client.gui
        n._get_scene_client.wait_for_service=lambda **kwargs:False
        n._cmd_toggle_ground_plane();until(lambda:q.worker is None)
        self.assertTrue(q.needs_resync)
        self.assertIn('ground_plane',[o.id for o in s.scene.world.collision_objects])
        n._get_scene_client.wait_for_service=lambda **kwargs:True
        next_job=n._dispatch_scene_update(lambda:self.pair(s,'base','ground_plane'))
        self.assertTrue(next_job.result(timeout=1));self.assertFalse(q.needs_resync)
        self.assertIsNone(n._apply_scene_client.failure)

    def test_permanent_silent_response_refuses_jobs_bounded_without_gui_poison(self):
        c,r,s,markers=self.fixture();n=c.node;q=n._apply_scene_client.gui;q.grace_seconds=.01;s.pending_kind='world_add'
        n._cmd_toggle_ground_plane();until(lambda:q.worker is None)
        later=[];job=n._dispatch_scene_update(lambda:later.append(True))
        with self.assertRaises(GuiSceneNeedsResync):job.result(timeout=1)
        self.assertFalse(later);self.assertIsNone(n._apply_scene_client.failure);self.assertFalse(markers)
        s.pending.pop()()

    def test_queued_cup_and_direct_attach_style_helper_preserve_both_acm_updates(self):
        c,r,s,_=self.fixture();n=c.node;n._cmd_toggle_ground_plane();until(lambda:not n._apply_scene_client.gui.has_work())
        gate=threading.Event();release=threading.Event();original=s.query;reads=[]
        def query(request):
            reads.append(threading.current_thread().ident)
            value=original(request)
            if not gate.is_set():gate.set();release.wait(2)
            return value
        n._get_scene_client.call_async=query
        cup=n._dispatch_scene_update(lambda:n._acm_allow_ground_rest(['cup_drop_1']))
        self.assertTrue(gate.wait(1));done=threading.Event()
        thread=threading.Thread(target=lambda:(n._acm_allow_ground_rest(['lego_red_2x3_0']),done.set()));thread.start()
        self.assertFalse(done.wait(.02));self.assertEqual(len(reads),1)
        release.set();cup.result(timeout=2);thread.join(2);self.assertTrue(done.is_set())
        self.assertTrue(self.pair(s,'cup_drop_1','ground_plane'));self.assertTrue(self.pair(s,'lego_red_2x3_0','ground_plane'))
        self.assertEqual(len(set(reads)),1)

    def test_actual_attach_and_detach_use_owner_and_finish_synchronously(self):
        c,r,s,_=self.fixture();n=c.node;n._cmd_toggle_ground_plane();until(lambda:not n._apply_scene_client.gui.has_work())
        owners=[];raw=n._apply_scene_client.client.call_async
        def call(request):owners.append(n._apply_scene_client.gui.owns_current());return raw(request)
        n._apply_scene_client.client.call_async=call
        self.assertTrue(n._attach_lego_to_gripper('red_2x3_0'))
        self.assertEqual([a.object.id for a in s.scene.robot_state.attached_collision_objects],['lego_red_2x3_0'])
        self.assertTrue(n._detach_lego_sync());self.assertFalse(s.scene.robot_state.attached_collision_objects)
        self.assertTrue(owners);self.assertTrue(all(owners))

    def test_manual_detach_queries_actual_indexed_ids_and_verifies_before_bookkeeping(self):
        c,r,s,_=self.fixture();n=c.node
        self.assertTrue(n._attach_lego_to_gripper('red_2x3_0'))
        fn=method(SHARED.with_name('control_gui.py'),'SOArm101ControlGUI','_cmd_detach_lego')
        fn.__globals__.update(MOVEIT_AVAILABLE=True,PlanningSceneMsg=Scene,
            GetPlanningSceneSrv=MESSAGES.Get,ApplyPlanningScene=MESSAGES.Apply,
            CollisionObject=lifetime.Object,AttachedCollisionObject=lifetime.Attached)
        n.root=NS(after=lambda ms,callback:callback())
        fn(n);until(lambda:not n._apply_scene_client.gui.has_work())
        self.assertFalse(s.scene.robot_state.attached_collision_objects)
        self.assertIsNone(n._attached_lego_name)
        self.assertIn('lego_red_2x3_0',[o.id for o in s.scene.world.collision_objects])

    def test_target_removal_is_barrier_before_validity_may_read_scene(self):
        c,r,s,_=self.fixture();n=c.node;s.pending_kind='world_remove'
        target=__import__('test_scene_lifetime').Object();target.id='lego_red_2x3_0';s.scene.world.collision_objects=[target]
        done=threading.Event();thread=threading.Thread(target=lambda:(n._remove_single_lego_from_world('red_2x3_0'),done.set()))
        thread.start();until(lambda:bool(s.pending));self.assertFalse(done.is_set())
        s.pending.pop()();thread.join(2);self.assertTrue(done.is_set());self.assertFalse(s.scene.world.collision_objects)

    def test_conclusive_remote_error_is_drained_then_authoritative_repair_allows_next_job(self):
        c,r,s,_=self.fixture();n=c.node;q=n._apply_scene_client.gui
        raw=q.client.call_async;failed=Future();failed.set_exception(RuntimeError('service restart'))
        q.client.call_async=lambda request:failed
        first=n._dispatch_scene_update(lambda:n._wait_future(n._apply_scene_client.call_async(NS(scene=Scene())),1.))
        with self.assertRaises(RuntimeError):first.result(timeout=1)
        self.assertTrue(q.needs_resync)
        q.client.call_async=raw
        later=n._dispatch_scene_update(lambda:'executor still alive')
        self.assertEqual(later.result(timeout=1),'executor still alive');self.assertFalse(q.needs_resync)

    def test_gate_c_keeps_collision_pair_and_depth_in_the_actual_trace(self):
        m=motions.MotionOperationTests();c,r,_,_,_=m.setup_motion();n=c.node;trace=[]
        event=n._begin_motion(n._new_motion_event());operation=n._operation({},event,1.)
        n._trace=lambda method,*args,**kwargs:trace.append((method,args,kwargs))
        n._add_lego_collision_objects=lambda:None
        n._check_state_valid_with_contacts=lambda target:(False,[NS(contact_body_1='shoulder',contact_body_2='usb_camera',depth=.00285764)])
        m.prevalidator(n,operation,.7)()
        row=next(kwargs['attempts'][0] for method,args,kwargs in trace if args==('gate_c_done',))
        self.assertFalse(row['valid']);self.assertEqual(row['contacts'],[dict(a='shoulder',b='usb_camera',depth_mm=2.85764)])
        self.assertFalse(event.verdict['ok']);self.assertTrue(event.is_set())

    def test_construction_policy_is_explicit_frozen_and_validates_entire_hook_bundle(self):
        with self.assertRaises(ValueError):SceneExecution('collector',effect=lambda:None)
        with self.assertRaises(ValueError):SceneExecution('gui',scene_failure=lambda:None)
        policy=SceneExecution('collector',*[lambda:None]*3)
        with self.assertRaises(FrozenInstanceError):policy.mode='gui'
        c,_,_,_=self.fixture()
        with self.assertRaises(AttributeError):c.node._apply_scene_client.collector=True

    def test_callback_abort_invalidates_existing_worker_before_native_pause(self):
        c,r,_,_,threads=motions.MotionOperationTests().setup_motion();n=c.node;gate=threading.Event();release=threading.Event();effects=[]
        fn=method(SHARED,'PickPlaceNode','_callback_error');n.get_logger=lambda:NS(error=lambda *a:None)
        n._callback_failure_handler=c.callback_failure
        raw=NS(call_async=lambda request:effects.append('scene mutation'))
        client=SceneMutationClient(raw,n._motion_guard,lambda exc:None)
        def worker():
            gate.set();release.wait(2)
            with n._motion_guard():
                client.call_async(NS(scene=Scene()))
                n._attached_lego_name='stale';n._set_motion_status(dict(ok=True))
        t=n._motion_thread(target=worker);t.start();self.assertTrue(gate.wait(1))
        with redirect_stdout(io.StringIO()):
            safe_callback(lambda:(_ for _ in ()).throw(RuntimeError('subscription fault')),
                          lambda name,exc:fn(n,name,exc))()
        self.assertIsNone(c.generation.epoch);self.assertTrue(c.generation.stalled)
        self.assertTrue(c.stop.is_set());release.set();t.join(2)
        self.assertFalse(effects);self.assertNotEqual(getattr(n,'_attached_lego_name',None),'stale')
        self.assertEqual(c.events,['pause'])

    def test_subscription_timer_service_and_future_callback_failures_are_contained(self):
        errors=[]
        def bad(*args):raise SceneMutationIncomplete('guard refused')
        for _ in range(3):self.assertIsNone(safe_callback(bad,lambda name,exc:errors.append(str(exc)))())
        response=NS(success=True,message='');result=safe_callback(bad,lambda name,exc:errors.append(str(exc)),service=True)(object(),response)
        self.assertIs(result,response);self.assertFalse(result.success);self.assertEqual(len(errors),4)

    def test_async_callback_and_failing_diagnostic_hook_cannot_kill_executor(self):
        async def bad(*args):raise SceneMutationIncomplete('async guard')
        def diagnostic(*args):raise RuntimeError('logger unavailable')
        self.assertIsNone(asyncio.run(safe_callback(bad,diagnostic)()))

    def test_gui_scene_error_remains_nonterminal_even_if_incidental_hook_is_assigned(self):
        callback=method(SHARED,'PickPlaceNode','_scene_unresolved');warnings=[];terminal=[]
        n=NS(execution=SceneExecution(),_gui_scene_warning=warnings.append,_scene_failure_handler=terminal.append)
        callback(n,RuntimeError('GUI overlap'));self.assertTrue(warnings);self.assertFalse(terminal)
        self.assertFalse(hasattr(n,'_scene_terminal_error'))
if __name__=='__main__':unittest.main()
