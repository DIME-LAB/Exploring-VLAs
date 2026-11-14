"""Camera/native clock drift never hides the primary failure or loses its receipt."""
import ast,copy,io,json,queue,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np
from wall_profile import joined_profiles,ProfileContractError
from frame_join import SensorAmbiguity
from episode_policy import FailureStreak
from contract import timing_audit
import test_review_regressions as probes
from test_sensor_join import sensors
from test_native_camera_tick import FIXTURE


def profile(pair,index=0):return dict(profile_id=f"8:{pair['native_sim_ns']}:{index}",epoch=8,sim_ns=pair['native_sim_ns'],native_sim_ns=pair['native_sim_ns'],camera_stamp_ns=pair['camera_stamp_ns'],recorded=True,tick_total_ms=150.)


class ProfileReceiptTests(unittest.TestCase):
    def test_actual_async_writer_copies_native_id_and_keeps_raw_camera_time(self):
        Writer,scope=probes.load_class('collect.py','EpisodeWriter')
        with tempfile.TemporaryDirectory() as d:
            writer=object.__new__(Writer);writer.path=Path(d);writer.rows=[];writer.write_profiles=[];writer.failure=None
            writer.queue=queue.Queue();writer.encoders={k:NS(stdin=io.BytesIO()) for k in ('wrist','top')}
            captures=[]
            for i,pair in enumerate(FIXTURE['async_clock_pairs']):
                capture=profile(pair,i);captures.append(capture)
                writer.queue.put((dict(epoch=8,sim_ns=pair['camera_stamp_ns'],wall_profile=capture),{k:np.zeros((2,2,3),dtype=np.uint8) for k in ('wrist','top')},0))
            writer.queue.put(None);writer.write_loop();self.assertIsNone(writer.failure)
            joined=joined_profiles(captures,writer.write_profiles);self.assertEqual(len(joined),14)
            for p,w in zip(captures,writer.write_profiles):
                self.assertEqual(w['native_sim_ns'],p['native_sim_ns']);self.assertEqual(w['camera_stamp_ns'],p['camera_stamp_ns'])
            persisted=[json.loads(l) for l in (writer.path/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['sim_ns'] for r in persisted],[p['camera_stamp_ns'] for p in FIXTURE['async_clock_pairs']])

    def test_copied_native_ids_remain_exact_and_independent_producer_bounds_remain_strict(self):
        p=profile(FIXTURE['async_clock_pairs'][-1]);w=dict(profile_id=p['profile_id'],epoch=8,sim_ns=p['sim_ns'],native_sim_ns=p['native_sim_ns'],camera_stamp_ns=p['camera_stamp_ns'],row_queue_ms=1.,row_video_write_ms=2.)
        for rule,change in [('copy',dict(native_sim_ns=w['native_sim_ns']+1)),('stamp_ns',dict(camera_stamp_ns=w['native_sim_ns']+1001))]:
            capture=dict(p)
            if rule=='stamp_ns':capture['camera_stamp_ns']=change['camera_stamp_ns']
            with self.subTest(rule=rule),self.assertRaises(ProfileContractError) as error:joined_profiles([capture],[dict(w,**change)])
            self.assertTrue(any(proof['rule']==rule and not proof['passed'] for proof in error.exception.proofs))

    def test_actual_attempt_finally_publishes_receipt_and_ledger_when_profile_join_fails(self):
        c,r,entry=probes.ReviewRegressions().collector();c.sensors=sensors();original=SensorAmbiguity(FIXTURE['sensor_evidence']);c.sensors.failure=original
        c.sensors.rerender_messages.update(wrist=7,top=9);c.failure=repr(original);c.generation.poison(original)
        c.writer_lock=threading.RLock();c.workers=NS(retire=lambda:dict(passed=True));c.abort_stalled_owner=Mock();c.persist_terminal_fault=Mock()
        c.servo.noise_specs=[];c.capture_failures=[];c.wait_diagnostics=[];p=profile(FIXTURE['async_clock_pairs'][-1]);c.wall_profiles=[p]
        bad=dict(profile_id=p['profile_id'],epoch=8,sim_ns=p['sim_ns'],native_sim_ns=p['native_sim_ns']+1,camera_stamp_ns=p['camera_stamp_ns'])
        writer=NS(close=Mock(),rows=[],write_profiles=[bad])
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text());run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run')
        loop=next(n for n in run.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='trial')
        attempt=next(n for n in loop.body if isinstance(n,ast.Try));body=attempt.finalbody
        with tempfile.TemporaryDirectory() as d,patch('storage.confined',side_effect=Path):
            root=Path(d);path=root/'attempt-0000';receipt=dict(accepted=False,error='wrapper cancellation',params=dict(seed=330000,trial=0),initial_physical=dict(saved='exact reset receipt'))
            scope=dict(c.failed_finalization.__globals__,timing_audit=timing_audit,self=c,receipt=receipt,writer=writer,receipt_cycle=None,args=NS(diagnostic=False),path=path,root=root,params=dict(object='red'),snapshots=[],motion_epochs={8},failure_log_start=0,cycle_result={},duplicate_start={},rerender_start={},stale_camera_start={},root_growth=NS(finish=lambda:{}),wall_profile_start=0,wait_log_start=0,streak=FailureStreak([]))
            exec(compile(ast.Module(body=body,type_ignores=[]),'actual attempt finalization','exec'),scope)
            saved=json.loads((path/'receipt.json').read_text());ledger=json.loads((root/'episodes.jsonl').read_text())
            self.assertEqual(saved,ledger);self.assertFalse(saved['accepted']);self.assertEqual(saved['sensor_join_failure'],original.evidence)
            self.assertTrue(saved['original_sensor_error_preserved']);self.assertIn('SensorAmbiguity',saved['error'])
            self.assertEqual(saved['params']['seed'],330000);self.assertEqual(saved['initial_physical'],receipt['initial_physical'])
            self.assertEqual(saved['camera_rerenders'],dict(wrist=7,top=9));self.assertTrue(saved['failure_streak']['stop'])
            self.assertTrue(saved['finalization_errors'][0]['proofs']);writer.close.assert_called_once()
