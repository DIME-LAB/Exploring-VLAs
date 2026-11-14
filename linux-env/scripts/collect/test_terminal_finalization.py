"""A late conflicting frame plus failed cancellation cannot reuse success receipts."""
import ast,json,os,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from unittest.mock import Mock,patch
import collection_status,storage,run_smoke
import test_review_regressions as probes
from test_frame_join import image
from test_sensor_join import sensors,feed
from frame_join import SensorAmbiguity


class TerminalFinalizationTests(unittest.TestCase):
    def test_actual_late_camera_fault_failed_close_and_runner_refuse_stale_success(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);run=root/'normal';run.mkdir();attempt=run/'attempt-0000';attempt.mkdir()
            storage.atomic_json(run/'summary.json',dict(complete=True,episodes=5,end_marker='COLLECTION_FINISHED'))
            storage.atomic_json(attempt/'receipt.json',dict(accepted=True))
            certificate=collection_status.begin(run);collection_status.finish(run,certificate,[])
            c,r,entry=probes.ReviewRegressions().collector();c.args=NS(output=run);c.status_lock=threading.RLock()
            c.current_attempt_path=attempt;c.finalization_certificate=certificate;c.capture_failures=[]
            c.persist_terminal_fault=MethodType(type(c).persist_terminal_fault,c)
            s=sensors();s.generation=c.generation;feed(s,10**9,0);s.read(timeout=0)
            with self.assertRaises(SensorAmbiguity):s.put('wrist',image(10**9+500,2))
            c.sensors=s;c.callback_failure('wrist',s.failure)
            first=json.loads((run/'terminal-fault.json').read_text())['events'][0]
            self.assertEqual(len(first['sensor_evidence']['candidates']),2)
            self.assertFalse(json.loads((attempt/'receipt.json').read_text())['accepted'])
            c.executor=NS(shutdown=lambda **kw:None);c.node.destroy_node=lambda:None
            c.capture_thread=NS(join=lambda **kw:None,is_alive=lambda:False)
            scope=c.close.__globals__;scope['timeline']=Mock(side_effect=ConnectionError('native cancel unavailable'))
            scope['mcp']=Mock(return_value=True);scope['rclpy']=NS(shutdown=lambda:None)
            with self.assertRaisesRegex(RuntimeError,'finalization failed'):c.close()
            final=json.loads((run/'collector-finalization.json').read_text())
            self.assertFalse(final['passed']);self.assertTrue(any('native cancel unavailable' in e['error'] for e in final['cleanup_errors']))
            # Even stale successful documents cannot override the root marker.
            storage.atomic_json(run/'summary.json',dict(complete=True,episodes=5,end_marker='COLLECTION_FINISHED'))
            storage.atomic_json(run/'collector-finalization.json',dict(certificate,completed=True,passed=True,cleanup_errors=[]))
            with self.assertRaisesRegex(RuntimeError,'Terminal collection marker'):collection_status.require_finalized(run)
            tree=ast.parse(Path(run_smoke.__file__).read_text())
            loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='item' and isinstance(n.iter,ast.Subscript))
            restore=Mock();env=dict(recipe=dict(stages=[dict(name='normal-resume')]),stage=lambda item:1,output=root,
                stages=[dict(pid=os.getpid(),started_wall_ns=0)],require_finalized=collection_status.require_finalized,
                restore_for_next_process=restore,json=json)
            with self.assertRaisesRegex(RuntimeError,'Terminal collection marker'):
                exec(compile(ast.Module(body=[loop],type_ignores=[]),'actual runner stage validation','exec'),env)
            restore.assert_not_called()

    def test_affirmative_finalization_is_required_for_the_same_producing_stage(self):
        with tempfile.TemporaryDirectory() as directory,patch('storage.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path):
            root=Path(directory);certificate=collection_status.begin(root)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):collection_status.require_finalized(root)
            collection_status.finish(root,certificate,[])
            self.assertTrue(collection_status.require_finalized(root,os.getpid(),certificate['started_wall_ns'])['passed'])
            with self.assertRaisesRegex(RuntimeError,'another producer'):collection_status.require_finalized(root,os.getpid()+1)
            with self.assertRaisesRegex(RuntimeError,'predates'):collection_status.require_finalized(root,os.getpid(),certificate['started_wall_ns']+1)
