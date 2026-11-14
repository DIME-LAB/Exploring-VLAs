"""Stage timings preserve raw clock boundaries and report overlapping shares."""
import unittest
from wall_profile import profile_tick,summary,milliseconds,joined_profiles

class WallProfileTests(unittest.TestCase):
    def test_actual_native_and_camera_boundaries_and_native_state_has_no_bridge_rpc(self):
        snapshot=dict(epoch=8,sim_ns=10**9,wall_profile_native=dict(step_start_ns=10**9,step_end_ns=1100000000,state_read_ms=2.))
        entry={k:(None,1110000000) for k in ('wrist','top')}
        value=profile_tick(snapshot,entry,1090000000,1120000000)
        self.assertEqual(value['kit_step_render_ms'],100.)
        self.assertEqual(value['step_to_both_cameras_ms'],110.)
        self.assertEqual(value['image_transport_join_ms'],20.)
        self.assertEqual(value['native_state_read_ms'],2.)
        self.assertEqual(value['native_state_bridge_calls'],0)
        value.update(tick_total_ms=150.,validity_query_ms=20.,validity_queries=4)
        report=summary([value]);self.assertEqual(report['validity_query_count'],4)
        self.assertAlmostEqual(report['stages']['kit_step_render_ms']['mean_share_of_tick'],2/3)

    def test_missing_boundaries_remain_unknown_and_p95_is_measured(self):
        value=profile_tick(dict(epoch=1,sim_ns=1),{'wrist':(None,5),'top':(None,6)},1,10)
        self.assertIsNone(value['kit_step_render_ms']);self.assertIsNone(value['native_state_read_ms'])
        result=summary([dict(tick_total_ms=x,validity_queries=1) for x in range(1,21)])
        self.assertEqual(result['stages']['tick_total_ms']['p95'],19.)
        self.assertEqual(milliseconds(4,2),0.)

    def test_encoder_worker_keeps_per_tick_write_cost_for_the_final_receipt(self):
        import io,json,queue,tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS
        import numpy as np
        import test_review_regressions as probes
        Writer,scope=probes.load_class('collect.py','EpisodeWriter')
        with tempfile.TemporaryDirectory() as directory:
            writer=object.__new__(Writer);writer.path=Path(directory);writer.rows=[];writer.write_profiles=[];writer.failure=None
            writer.queue=queue.Queue();writer.encoders={k:NS(stdin=io.BytesIO()) for k in ('wrist','top')}
            profile=dict(profile_id='tick-1',epoch=1,sim_ns=1,row_video_write_ms=None,capture_processing_ms=2.,tick_total_ms=150.);row=dict(epoch=1,sim_ns=1,wall_profile=profile)
            images={k:np.zeros((2,2,3),dtype=np.uint8) for k in ('wrist','top')}
            writer.queue.put((row,images,0));writer.queue.put(None);writer.write_loop()
            self.assertIsNone(writer.failure);self.assertIsNone(profile['row_video_write_ms']);self.assertGreater(writer.write_profiles[0]['row_video_write_ms'],0.)
            self.assertEqual(len(writer.rows),1);self.assertEqual(len(writer.encoders['top'].stdin.getvalue()),12)
            persisted=json.loads((writer.path/'frames.jsonl').read_text())
            self.assertEqual(persisted['wall_profile'],profile)
            async_row=json.loads((writer.path/'write-profiles.jsonl').read_text())
            self.assertEqual(async_row,writer.write_profiles[0])
            self.assertGreater(joined_profiles([profile],[async_row])[0]['row_video_write_ms'],0.)

    def test_geometry_failure_keeps_partial_tick_and_failing_stage(self):
        import test_review_regressions as probes
        c,r,entry=probes.ReviewRegressions().collector()
        c.node._check_state_valid_with_contacts=lambda q:(False,[])
        c.capture()
        self.assertTrue(c.failure);self.assertEqual(len(c.wall_profiles),1)
        profile=c.wall_profiles[0]
        self.assertEqual(profile['stage'],'execution_geometry');self.assertEqual(profile['outcome'],'failed')
        self.assertGreater(profile['validity_queries'],0);self.assertGreater(profile['validity_query_ms'],0.)
        self.assertIn('error',profile);self.assertIn('tick_total_ms',profile)

    def test_profile_is_complete_before_immediate_encoder_consumption(self):
        from types import SimpleNamespace as NS
        import copy
        import test_review_regressions as probes
        c,r,entry=probes.ReviewRegressions().collector();persisted=[]
        c.capture.__globals__['Sensors']=probes.load_class('collect.py','Sensors')[0]
        c.writer=NS(add=lambda row,images:persisted.append(copy.deepcopy(row)))
        c.capture()
        self.assertEqual(len(persisted),1)
        profile=persisted[0]['wall_profile']
        self.assertIn('capture_processing_ms',profile);self.assertIn('tick_total_ms',profile)
        self.assertIsNone(profile['row_video_write_ms']);self.assertEqual(profile,c.wall_profiles[0])
