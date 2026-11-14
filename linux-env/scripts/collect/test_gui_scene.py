"""Pin GUI scene behavior separately from the strict collector protocol."""
import ast,asyncio,copy,hashlib,io,json,threading,time,unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MethodType,SimpleNamespace as NS
from test_safety import method
import test_scene_lifetime as lifetime
import test_motion_operations as motions
from so_arm101_control.scene_transactions import SceneMutationClient,SceneMutationIncomplete,SceneExecution
from so_arm101_control.ros_callbacks import safe_callback
SHARED=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('gui_scene_6e439ed.json').read_text())
GUI_METHODS={'_add_cup_collision_objects','_remove_single_lego_from_world','_attach_lego_to_gripper','_detach_lego_sync','_dispatch_scene_update'}


def source_method(path,owner,name):
    s=path.read_text();c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==owner)
    n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    return ''.join(s.splitlines(True)[start:n.end_lineno])


def plain(value):
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if isinstance(value,(tuple,list)):return [plain(x) for x in value]
    return {k:plain(v) for k,v in vars(value).items()}


class GuiSceneTests(unittest.TestCase):
    def fixture(self,baseline=False):
        c,r,entry,s,markers=lifetime.SceneLifetimeTests().setup_scene();n=c.node
        n.execution=SceneExecution();n._motion_effect_guard=None;n._scene_failure_handler=None
        n._gui_scene_warning=lambda message:None;n._gui_ready=False
        # Raw GUI clients; there is deliberately no worker, resync state or strict tracker.
        calls=[]
        def apply(request):calls.append(('apply',plain(copy.deepcopy(request.scene))));return s.apply(request)
        def query(request):calls.append(('get',request.components.components));return s.query(request)
        n._apply_scene_client=NS(call_async=apply,wait_for_service=lambda **kwargs:True)
        n._get_scene_client=NS(call_async=query,wait_for_service=lambda **kwargs:True)
        scope=dict(n._attach_lego_to_gripper.__func__.__globals__)
        # The GUI behavior class is actual production code, with only ROS messages substituted.
        tree=ast.parse(SHARED.read_text());gui=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GuiSceneBehavior')
        exec(compile(ast.Module(body=[gui],type_ignores=[]),str(SHARED),'exec'),scope)
        for fn in list(vars(n).values()):
            if hasattr(fn,'__func__'):fn.__func__.__globals__['GuiSceneBehavior']=scope['GuiSceneBehavior']
        scope.update(tracer=NS(event=lambda *a,**k:None),rclpy=NS(time=NS(Time=lambda:None)))
        if baseline:
            for item in FIXTURE['methods']:
                text=item['source'];exec(compile(ast.parse(__import__('textwrap').dedent(text)),item['file'],'exec'),scope)
                fn=scope[item['name']];fn=getattr(fn,'__func__',fn)
                setattr(n,item['name'],fn if item['name']=='_wait_future' else MethodType(fn,n))
        n._attached_lego_tcp_offset=None
        return c,r,s,markers,calls

    def test_gui_scene_methods_keep_the_verified_statements(self):
        for item in FIXTURE['methods']:
            owner=('GuiSceneBehavior' if item['name'] in GUI_METHODS else item['owner'])
            path=SHARED.with_name(item['file'])
            text=source_method(path,owner,item['name'])
            # 3.1j updates the TCP comment; executable statements stay pinned.
            import textwrap
            self.assertEqual(ast.dump(ast.parse(textwrap.dedent(text))),ast.dump(ast.parse(textwrap.dedent(item['source']))),(owner,item['name']))
            self.assertEqual(hashlib.sha256(item['source'].encode()).hexdigest(),item['sha256'])

    def test_collector_scene_client_matches_the_pilot_ready_reference(self):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('collector_scene_a8fc78b.json').read_text())
        source=SHARED.with_name('scene_transactions.py').read_text()
        classes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
        for name,sha in fixture['classes'].items():
            self.assertEqual(hashlib.sha256(classes[name].encode()).hexdigest(),sha,name)

    def exercise(self,baseline,case):
        c,r,s,markers,calls=self.fixture(baseline);n=c.node
        n._drop_data={'drop_1':dict(x=.2,y=-.2,z=.04825,qw=1.)}
        if case=='startup':
            n._cmd_toggle_ground_plane();n._add_cup_collision_objects();n._add_lego_collision_objects()
        elif case=='remove':
            n._cmd_toggle_ground_plane();n._add_cup_collision_objects();n._add_lego_collision_objects()
            n._remove_single_lego_from_world('red_2x3_0');n._remove_lego_collision_objects();n._remove_cup_collision_objects()
        elif case=='attach_detach':
            n._cmd_toggle_ground_plane();n._add_lego_collision_objects()
            self.assertTrue(n._attach_lego_to_gripper('red_2x3_0'));self.assertTrue(n._detach_lego_sync())
        elif case=='late_ground':
            s.pending_kind='world_add';n._cmd_toggle_ground_plane()
            self.assertTrue(s.pending);s.pending.pop()()
        elif case=='late_attach_remove':
            s.pending_kind='world_remove';n._attach_lego_to_gripper('red_2x3_0')
            # This intentionally preserves GUI's pre-existing nontransactional behavior.
            self.assertEqual(s.calls[-2:],['world_remove','attach']);s.pending.pop()()
        else:raise AssertionError(case)
        return calls,plain(s.scene),n._attached_lego_name,markers

    def test_gui_call_sequences_match_reference_for_startup_removal_attach_and_late_responses(self):
        for case in ('startup','remove','attach_detach','late_ground','late_attach_remove'):
            with self.subTest(case=case):self.assertEqual(self.exercise(False,case),self.exercise(True,case))

    def test_gui_dispatch_returns_without_waiting_for_its_background_callback(self):
        c,r,s,markers,calls=self.fixture();n=c.node;n._gui_ready=True
        entered=threading.Event();release=threading.Event();done=threading.Event()
        def callback():entered.set();release.wait(2);done.set()
        n._dispatch_scene_update(callback,wait=True) # wait is a collector-only contract.
        self.assertTrue(entered.wait(1));self.assertFalse(done.is_set())
        release.set();self.assertTrue(done.wait(1));self.assertFalse(hasattr(n._apply_scene_client,'gui'))

    def test_collector_target_removal_still_waits_for_completion(self):
        c,r,entry,s,markers=lifetime.SceneLifetimeTests().setup_scene();n=c.node;s.pending_kind='world_remove'
        errors=[]
        try:n._remove_single_lego_from_world('red_2x3_0')
        except SceneMutationIncomplete as exc:errors.append(exc)
        self.assertTrue(errors);self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)
        self.assertEqual(s.calls[-1:],['world_remove'])

    def test_collector_attach_phase_one_cannot_overtake_unresolved_remove(self):
        c,r,entry,s,markers=lifetime.SceneLifetimeTests().setup_scene();s.pending_kind='world_remove'
        with self.assertRaises(SceneMutationIncomplete):c.node._attach_lego_to_gripper('red_2x3_0')
        self.assertEqual(s.calls[-1:],['world_remove'])
        self.assertNotIn('attach',s.calls);self.assertFalse(s.scene.robot_state.attached_collision_objects)
        self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)

    def test_construction_policy_is_frozen_and_validates_entire_hook_bundle(self):
        with self.assertRaises(ValueError):SceneExecution('collector',effect=lambda:None)
        with self.assertRaises(ValueError):SceneExecution('gui',scene_failure=lambda:None)
        policy=SceneExecution('collector',*[lambda:None]*3)
        with self.assertRaises(FrozenInstanceError):policy.mode='gui'

    def test_callback_abort_invalidates_existing_worker_before_native_pause(self):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();n=c.node;gate=threading.Event();release=threading.Event();effects=[]
        fn=method(SHARED,'PickPlaceNode','_callback_error');n.get_logger=lambda:NS(error=lambda *a:None)
        n._callback_failure_handler=c.callback_failure
        raw=NS(call_async=lambda request:effects.append('scene mutation'))
        client=SceneMutationClient(raw,n._motion_guard,lambda exc:None)
        def worker():
            gate.set();release.wait(2)
            with n._motion_guard():
                client.call_async(NS(scene=lifetime.Scene()))
                n._attached_lego_name='stale';n._set_motion_status(dict(ok=True))
        t=n._motion_thread(target=worker);t.start();self.assertTrue(gate.wait(1))
        with redirect_stdout(io.StringIO()):
            safe_callback(lambda:(_ for _ in ()).throw(RuntimeError('subscription fault')),
                          lambda name,exc:fn(n,name,exc))()
        self.assertIsNone(c.generation.epoch);self.assertTrue(c.generation.stalled)
        self.assertTrue(c.stop.is_set());release.set();t.join(2)
        self.assertFalse(effects);self.assertNotEqual(getattr(n,'_attached_lego_name',None),'stale')
        self.assertEqual(c.events,['pause'])

    def test_gate_c_keeps_collision_pair_and_depth_in_collector_trace(self):
        m=motions.MotionOperationTests();c,r,_,_,_=m.setup_motion();n=c.node;trace=[]
        event=n._begin_motion(n._new_motion_event());operation=n._operation({},event,1.)
        n._trace=lambda method,*args,**kwargs:trace.append((method,args,kwargs));n._add_lego_collision_objects=lambda:None
        n._check_state_valid_with_contacts=lambda target:(False,[NS(contact_body_1='shoulder',contact_body_2='usb_camera',depth=.00285764)])
        m.prevalidator(n,operation,.7)()
        row=next(kwargs['attempts'][0] for method,args,kwargs in trace if args==('gate_c_done',))
        self.assertEqual(row['contacts'],[dict(a='shoulder',b='usb_camera',depth_mm=2.85764)])
        self.assertFalse(event.verdict['ok']);self.assertTrue(event.is_set())

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

    def test_gui_scene_error_is_nonterminal(self):
        callback=method(SHARED,'PickPlaceNode','_scene_unresolved');warnings=[]
        n=NS(execution=SceneExecution(),_gui_scene_warning=warnings.append,_scene_failure_handler=None)
        callback(n,RuntimeError('GUI overlap'));self.assertTrue(warnings)
        self.assertFalse(hasattr(n,'_scene_terminal_error'))
if __name__=='__main__':unittest.main()
