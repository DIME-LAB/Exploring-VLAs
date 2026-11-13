"""GUI async scene ordering and ROS callback containment without ROS imports."""
import asyncio
from concurrent.futures import Future
from pathlib import Path
import sys,threading,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control'))
from so_arm101_control.scene_transactions import SceneMutationClient,SceneMutationIncomplete
from so_arm101_control.ros_callbacks import safe_callback
from contextlib import nullcontext

class DeferredClient:
    def __init__(self):self.sent=[]
    def call_async(self,request):
        f=Future();self.sent.append((request,f));return f

class GuiSceneTests(unittest.TestCase):
    def client(self):
        raw=DeferredClient();terminal=[];warnings=[];mode=[False]
        client=SceneMutationClient(raw,nullcontext,terminal.append,collector=lambda:mode[0],warn=warnings.append)
        return client,raw,terminal,warnings,mode
    def test_two_overlapping_gui_requests_are_fifo_without_terminal_state(self):
        client,raw,terminal,_,_=self.client();a=client.call_async('ground');b=client.call_async('cups')
        self.assertEqual([v[0] for v in raw.sent],['ground']);self.assertIsNone(a.result())
        raw.sent[0][1].set_result('ground done');self.assertEqual(a.result(),'ground done')
        self.assertEqual([v[0] for v in raw.sent],['ground','cups']);self.assertFalse(b.done())
        raw.sent[1][1].set_result('cups done');self.assertEqual(b.result(),'cups done')
        self.assertIsNone(client.failure);self.assertEqual(terminal,[])
    def test_gui_operation_waits_for_late_completion_before_next_read_modify_write(self):
        client,raw,terminal,_,_=self.client();first_started=threading.Event();second_started=threading.Event()
        def first():client.call_async('first');first_started.set() # Caller has stopped waiting; remote work remains.
        def second():second_started.set();return client.call_async('second')
        a=client.submit_gui_update(first);b=client.submit_gui_update(second)
        self.assertTrue(first_started.wait(1));self.assertFalse(second_started.is_set());self.assertIsNone(client.failure)
        raw.sent[0][1].set_result(True);self.assertTrue(second_started.wait(1))
        raw.sent[1][1].set_result(True);b.result(timeout=1);self.assertTrue(a.done());self.assertEqual(terminal,[])
    def test_collector_mode_still_refuses_unresolved_gui_requests(self):
        client,raw,terminal,_,mode=self.client();client.call_async('old GUI request');mode[0]=True
        with self.assertRaises(SceneMutationIncomplete):client.require_idle()
        self.assertTrue(client.failure);self.assertEqual(len(terminal),1)
        raw.sent[0][1].set_result(True)
    def test_gui_remote_error_does_not_poison_following_request(self):
        client,raw,terminal,_,_=self.client();a=client.call_async('first');b=client.call_async('second')
        raw.sent[0][1].set_exception(RuntimeError('remote unavailable'))
        with self.assertRaises(RuntimeError):a.result()
        self.assertEqual(raw.sent[1][0],'second');raw.sent[1][1].set_result(True)
        self.assertTrue(b.result());self.assertIsNone(client.failure);self.assertEqual(terminal,[])
    def test_subscription_timer_service_and_future_callback_failures_are_contained(self):
        errors=[]
        def bad(*args):raise SceneMutationIncomplete('guard refused')
        for _ in range(3):self.assertIsNone(safe_callback(bad,lambda name,exc:errors.append(str(exc)))())
        response=NS(success=True,message='');result=safe_callback(bad,lambda name,exc:errors.append(str(exc)),service=True)(object(),response)
        self.assertIs(result,response);self.assertFalse(result.success);self.assertIn('guard refused',result.message)
        self.assertEqual(len(errors),4);self.assertEqual(safe_callback(lambda:42,lambda *args:None)(),42)
    def test_async_callback_and_failing_diagnostic_hook_cannot_kill_executor(self):
        async def bad(*args):raise SceneMutationIncomplete('async guard')
        def diagnostic(*args):raise RuntimeError('logger unavailable')
        self.assertIsNone(asyncio.run(safe_callback(bad,diagnostic)()))
    def test_collector_hook_is_called_and_callback_exception_is_contained(self):
        stops=[]
        def bad():raise SceneMutationIncomplete('terminal collector scene failure')
        safe_callback(bad,lambda name,exc:stops.append((name,str(exc))))()
        self.assertIn('terminal collector',stops[0][1])
    def test_shared_node_scene_failure_only_latches_under_collector_hook(self):
        from test_safety import method
        path=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        callback=method(path,'PickPlaceNode','_scene_unresolved');warnings=[];terminal=[]
        node=NS(_gui_scene_warning=warnings.append)
        callback(node,RuntimeError('GUI overlap'))
        self.assertTrue(warnings);self.assertFalse(hasattr(node,'_scene_terminal_error'))
        node._scene_failure_handler=terminal.append
        with self.assertRaises(SceneMutationIncomplete):callback(node,RuntimeError('collector unknown completion'))
        self.assertEqual(node._scene_terminal_error,'collector unknown completion');self.assertEqual(len(terminal),1)
    def test_shared_guard_does_not_poison_gui_from_a_legacy_scene_error(self):
        from test_safety import method
        path=Path(__file__).resolve().parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        guard=method(path,'PickPlaceNode','_motion_guard');node=NS(_scene_terminal_error='old GUI overlap',_motion_token=lambda:None)
        with guard(node):pass
        node._scene_failure_handler=lambda exc:None
        with self.assertRaises(SceneMutationIncomplete):
            with guard(node):pass
if __name__=='__main__':unittest.main()
