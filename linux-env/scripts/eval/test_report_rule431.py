import unittest,json,tempfile
from pathlib import Path
from report_rule431 import summarize,wilson,audit
class Tests(unittest.TestCase):
 def test_invalid_preserved_and_geometry_phase_explicit(self):
  row=dict(valid=True,color='red',requested_colour_success=True,requested_colour_grasp=True,rule_target_success=False,rule_target_grasp=False,wrong_color_lift=False,wrong_same_colour_target_grasp=True,rule_target_selection=dict(ambiguous=True),rule_closure_confirmed_geometry=dict(status='available',xy_mm=20,yaw_deg=-30))
  result=summarize([row,dict(valid=False)])
  self.assertEqual(result['invalid'],1);self.assertEqual(result['requested_colour_success']['count'],1);self.assertEqual(result['rule_target_success']['count'],0)
  self.assertEqual(result['yaw_deg']['mean'],30);self.assertIn('CONFIRMED',result['closure_phase'])
  self.assertEqual(result['ambiguous_rule_targets'],1);self.assertEqual(result['rule_target_success_excluding_ambiguous']['denominator'],0)
  self.assertGreater(wilson(0,15)[1],0);self.assertIsNone(wilson(0,0))
 def test_unscored_smoke_cannot_enter_report(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'window-ledger.json').write_text(json.dumps(dict(scored=False,stopped_utc='now',cleanup_exit_code=0)))
   with self.assertRaisesRegex(ValueError,'Unscored'):audit(root,root/'report')
   self.assertFalse((root/'report').exists())
 def test_held_crosscheck_is_diagnostic_and_keeps_live_target(self):
  import csv
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ep=root/'default/ep00';ep.mkdir(parents=True)
   sealed=root/'sealed.json';sealed.write_text(json.dumps(dict(initial_physical=dict(base_world_pose=dict(position=[0,0,0],orientation=[1,0,0,0]),bodies={'red_2x3_0':dict(position=[.1,0,0]),'red_2x3_1':dict(position=[.2,0,0])}))))
   (root/'window-ledger.json').write_text(json.dumps(dict(status='policy-finished',stopped_utc='now',cleanup_exit_code=0,suite='heldout-bank-v2',bank='fixture',bank_sha256='fixture')))
   row=dict(seed=900000,color='red',valid=True,reset_protocol='collector-sealed-postsample-v1',readiness_budget_s=60,requested_colour_success=False,requested_colour_grasp=False,rule_target_success=False,rule_target_grasp=False,wrong_color_lift=False,wrong_same_colour_target_grasp=False,rule_target_selection=dict(object='red_2x3_1',candidates=[],margin_m=.01,ambiguous=False))
   (ep/'result.json').write_text(json.dumps(row))
   with patch('native_bank.load_bank',return_value=([dict(seed=900000,source_receipt=str(sealed))],'heldout-bank-v2')):audit(root,root/'report')
   summary=json.loads((root/'report/summary.json').read_text());self.assertEqual(summary['rule_target_success_excluding_ambiguous']['denominator'],1)
   with (root/'report/episodes.csv').open() as stream:record=next(csv.DictReader(stream))
   diagnostic=json.loads(record['rule_target_selection']);self.assertFalse(diagnostic['live_matches_bank']);self.assertEqual(diagnostic['live_choice'],'red_2x3_1');self.assertEqual(diagnostic['bank_choice'],'red_2x3_0')
 def test_native_reset_metadata_survives_csv(self):
  import csv
  for metadata in (dict(reset_protocol='collector-recorded-postsample-v1'),dict(matched_reset=dict(protocol='collector-recorded-postsample-v1'))):
   with self.subTest(metadata=metadata),tempfile.TemporaryDirectory() as d:
    root=Path(d);ep=root/'default/ep00';ep.mkdir(parents=True)
    (root/'window-ledger.json').write_text(json.dumps(dict(scored=True,status='policy-finished',stopped_utc='now',cleanup_exit_code=0,readiness_budget_s=180,suite='rule-seen-training-home-15')))
    row=dict(seed=530002,color='blue',valid=True,readiness_budget_s=180,requested_colour_success=False,requested_colour_grasp=False,rule_target_success=False,rule_target_grasp=False,wrong_color_lift=False,wrong_same_colour_target_grasp=False,**metadata)
    (ep/'result.json').write_text(json.dumps(row));audit(root,root/'report')
    with (root/'report/episodes.csv').open() as f:record=next(csv.DictReader(f))
    self.assertEqual(record['reset_protocol'],'collector-recorded-postsample-v1');self.assertEqual(record['readiness_budget_s'],'180')
 def test_invalid_reset_metadata_refused_before_output(self):
  for metadata in (dict(),dict(reset_protocol='collector-matched-reset-v1',readiness_budget_s=180),dict(reset_protocol='collector-recorded-postsample-v1',matched_reset=dict(protocol='collector-matched-reset-v1'),readiness_budget_s=180),dict(reset_protocol='collector-recorded-postsample-v1',readiness_budget_s=60),dict(reset_protocol='collector-recorded-postsample-v1',readiness_budget_s=180,matched_reset=dict(readiness_budget_s=60))):
   with self.subTest(metadata=metadata),tempfile.TemporaryDirectory() as d:
    root=Path(d);ep=root/'default/ep00';ep.mkdir(parents=True)
    (root/'window-ledger.json').write_text(json.dumps(dict(scored=True,status='policy-finished',stopped_utc='now',cleanup_exit_code=0,suite='rule-seen-training-home-15',readiness_budget_s=180)))
    (ep/'result.json').write_text(json.dumps(dict(valid=True,**metadata)))
    with self.assertRaisesRegex(ValueError,'Missing or conflicting'):audit(root,root/'report')
    self.assertFalse((root/'report').exists())
 def test_missing_rule_fields_refused(self):
  with self.assertRaises(ValueError):summarize([dict(valid=True)])
 def test_episode_outcomes_and_evidence_columns(self):
  import csv
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'window-ledger.json').write_text(json.dumps(dict(scored=True,status='policy-finished',stopped_utc='now',cleanup_exit_code=0,suite='rule-seen-training-home-15',readiness_budget_s=180)))
   cases=[(True,True,False,True,'success','grasped-not-placed'),(False,True,False,False,'grasped-not-placed','no-grasp'),(False,False,False,False,'no-grasp','no-grasp')]
   for i,(s,g,rs,rg,_,__) in enumerate(cases):
    ep=root/'default'/f'ep{i:02d}';ep.mkdir(parents=True)
    row=dict(seed=530002+i,color='blue',valid=True,reset_protocol='collector-recorded-postsample-v1',readiness_budget_s=180,requested_colour_success=s,requested_colour_grasp=g,rule_target_success=rs,rule_target_grasp=rg,wrong_color_lift=False,wrong_same_colour_target_grasp=False,time_to_grasp_s=1.25,rtc_enabled=True,host_load_start=[1,2,3],video=dict(output_path='viewport.mp4'),wrist_preclosure=dict(manifest='wrist.json'),client=dict(flow_noise=dict(episode_seed=42),requests=[dict(wall_s=.1),dict(wall_s=.3)],observe_every_actions=10,actions_per_chunk=50,chunk_size_threshold=.8,queue_aggregation='weighted'))
    (ep/'result.json').write_text(json.dumps(row))
   ep=root/'default/ep03';ep.mkdir();(ep/'result.json').write_text(json.dumps(dict(seed=530005,color='red',valid=False,error='interrupted')))
   audit(root,root/'report')
   with (root/'report/episodes.csv').open() as f:records=list(csv.DictReader(f))
   self.assertEqual([(x['outcome'],x['rule_outcome']) for x in records],[x[-2:] for x in cases]+[('invalid','invalid')])
   r=records[0];self.assertEqual(r['time_to_grasp_s'],'1.25');self.assertEqual(r['rtc_enabled'],'True');self.assertEqual(r['video_path'],'viewport.mp4');self.assertEqual(r['wrist_manifest'],'wrist.json');self.assertEqual(json.loads(r['flow_noise'])['episode_seed'],42);self.assertEqual(float(r['inference_wall_mean_ms']),200);self.assertEqual(r['observe_every_actions'],'10')
if __name__=='__main__':unittest.main()
