"""Resume preserves uncertified predecessors; timeout cannot certify callbacks."""
import hashlib,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import collection_status,storage
import test_review_regressions as probes
from test_sensor_join import sensors


def collector(root,certificate,shutdown):
    c,r,entry=probes.ReviewRegressions().collector()
    c.args=NS(output=root,resume=True);c.status_lock=threading.RLock();c.finalization_certificate=certificate
    c.startup_status_output=root.parent/'startup';c.current_attempt_path=None;c.capture_failures=[]
    c.sensors=sensors();c.executor=NS(shutdown=shutdown,_executor=NS(shutdown=lambda **kw:None,_threads=[]));c.ros_thread=NS(join=lambda **kw:None,is_alive=lambda:False)
    c.ros_context=NS(destroy=lambda:None)
    c.capture_thread=NS(join=lambda **kw:None,is_alive=lambda:False);c.node.destroy_node=lambda:None
    c.timeline=lambda play:dict(held=True)
    scope=c.close.__globals__;scope.update(confined=Path,Path=Path,mcp=Mock(return_value=True),rclpy=NS(shutdown=lambda **kw:None,get_default_context=lambda:None))
    c.persist_terminal_fault=type(c).persist_terminal_fault.__get__(c,type(c))
    return c


class ResumeRetirementTests(unittest.TestCase):
    def test_actual_resume_refuses_incomplete_predecessor_before_begin_and_preserves_it_on_close(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory)/'run';root.mkdir();certificate=collection_status.begin(root)
            attempt=root/'attempt-0000';attempt.mkdir();storage.atomic_json(attempt/'receipt.json',dict(accepted=True))
            storage.atomic_json(root/'summary.json',dict(complete=True,episodes=1,producer_invocation=certificate['invocation']))
            original={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            c=collector(root,None,lambda **kw:True);c.mutate_kit=Mock(side_effect=AssertionError('No Kit request permitted'))
            with patch('collection_status.begin',wraps=collection_status.begin) as begin:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):c.run()
                begin.assert_not_called()
            c.close()
            self.assertEqual(original,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            c.mutate_kit.assert_not_called()
            self.assertFalse(json.loads((c.startup_status_output/'collector-finalization.json').read_text())['passed'])

    def test_actual_constructor_refuses_uncertified_resume_before_importing_ros_node(self):
        import sys
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);collection_status.begin(root)
            cls,scope=probes.load_class('collect.py','Collector');scope['confined']=Path
            instance=object.__new__(cls)
            with patch.dict(sys.modules,{'so_arm101_control.pick_place':None}):
                with self.assertRaisesRegex(RuntimeError,'incomplete'):cls.__init__(instance,NS(output=root,resume=True))

    def test_begin_also_refuses_incomplete_predecessor_and_archives_a_healthy_one(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);first=collection_status.begin(root);raw=(root/'collector-finalization.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'incomplete'):collection_status.begin(root)
            self.assertEqual((root/'collector-finalization.json').read_bytes(),raw)
            collection_status.finish(root,first,[]);old_summary=(root/'summary.json').read_text()
            second=collection_status.begin(root)
            self.assertEqual(second['predecessor_invocation'],first['invocation'])
            archived=json.loads((root/'finalizations'/(first['invocation']+'.json')).read_text())
            self.assertEqual(archived['summary_raw_json'],old_summary);self.assertTrue(archived['certificate']['passed'])
            self.assertNotEqual(first['invocation'],second['invocation'])

    def test_executor_false_and_unretired_spin_thread_are_failed_finalization(self):
        for result,alive in ((False,False),(True,True),(True,False)):
            with self.subTest(result=result,alive=alive),tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
                root=Path(directory);certificate=collection_status.begin(root)
                c=collector(root,certificate,lambda **kw:result);c.ros_thread.is_alive=lambda:alive
                if result and not alive:
                    c.close();self.assertTrue(collection_status.require_finalized(root)['passed'])
                else:
                    with self.assertRaisesRegex(RuntimeError,'finalization failed'):c.close()
                    self.assertTrue(c.generation.stalled)
                    record=json.loads((root/'collector-finalization.json').read_text())
                    self.assertFalse(record['passed']);self.assertEqual(record['cleanup_errors'][0]['stage'],'executor shutdown')
                    with self.assertRaises(RuntimeError):collection_status.require_finalized(root)

    def test_unretired_capture_refuses_native_node_and_context_destruction(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);certificate=collection_status.begin(root);c=collector(root,certificate,lambda **kw:True)
            c.capture_thread.is_alive=lambda:True;c.node.destroy_node=Mock();c.ros_context.destroy=Mock()
            with self.assertRaisesRegex(RuntimeError,'finalization failed'):c.close()
            c.node.destroy_node.assert_not_called();c.ros_context.destroy.assert_not_called()
            final=json.loads((root/'collector-finalization.json').read_text());self.assertFalse(final['passed'])
            self.assertTrue(any(e['stage']=='capture retirement' for e in final['cleanup_errors']))

    def test_incomplete_schema_and_mixed_invocation_fail_consistency_checks(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);cert=collection_status.begin(root);good=collection_status.finish(root,cert,[])
            for field in ('version','invocation','completed_wall_ns'):
                bad=dict(good);bad.pop(field);storage.atomic_json(root/'collector-finalization.json',bad)
                with self.assertRaisesRegex(RuntimeError,'Malformed'):collection_status.require_finalized(root)
            bad=dict(good,invocation='a'*32);storage.atomic_json(root/'collector-finalization.json',bad)
            with self.assertRaisesRegex(RuntimeError,'invocation differs'):collection_status.require_finalized(root)
