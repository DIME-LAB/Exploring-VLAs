import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report_scored_window import collect,summarize,select_examples
from phase3 import manifest,BANK_SHA256
class ReportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.attempt=self.root/'attempt-01';self.ep=self.attempt/'ep00';self.ep.mkdir(parents=True)
  e=manifest()['per_checkpoint'][0]
  self.row=dict(e,rtc_enabled=True,outcome='success',clock_basis='simulation',sim_cap_hz=30,success=True,grasp_success=True,correct_cup=True,target_any_placement=True,client=dict(clock_basis='simulation',send_action_costs_ms=[.4],missed_action_ticks_after_startup=0,nonmonotonic_action_steps=0,observe_every_actions=10,clock_dispatch_trace=[dict(jitter_ms=0)],observation_cadence_events=[{}],chunk_events=[]))
  self.config=dict(pretrained='/model/010000/pretrained_model',scene_bank_sha256=BANK_SHA256,inference_mode='eager',source_sha256={},fork_commit='fixed',rtc_enabled=True)
  (self.attempt/'run_config.json').write_text(json.dumps(self.config));(self.attempt/'window-health.json').write_text('{"passed":true}');(self.attempt/'episode-manifest.json').write_text(json.dumps(manifest()['per_checkpoint']))
 def tearDown(self):self.tmp.cleanup()
 def write(self): (self.ep/'result.json').write_text(json.dumps(self.row))
 def test_missing_scenes_remain_in_denominator(self):
  self.write();rows,_=collect(self.root);s=summarize(rows);self.assertEqual((s['n'],s['valid'],s['not_run'],s['success']['k']),(30,1,29,1))
 def test_invalid_observed_success_is_not_confirmed_success(self):
  self.row['client']['missed_action_ticks_after_startup']=1;self.write();rows,_=collect(self.root);self.assertTrue(rows[0]['observed_success']);self.assertFalse(rows[0]['success']);self.assertEqual(rows[0]['reported_outcome'],'invalid');self.assertEqual(summarize(rows)['invalid'],1)
 def test_starvation_is_outcome_not_exclusion(self):
  self.row.update(empty_ticks_after_startup=12,empty_ticks_pct=1.0);self.write();rows,_=collect(self.root);self.assertTrue(rows[0]['valid']);self.assertEqual(summarize(rows)['starved'],1)
 def test_missing_result_is_invalid_attempt(self):
  rows,_=collect(self.root);self.assertEqual(rows[0]['reported_outcome'],'invalid');self.assertEqual(summarize(rows)['not_run'],29)
 def test_duplicates_refused(self):
  self.write();other=self.attempt/'ep01';other.mkdir();(other/'result.json').write_text(json.dumps(self.row))
  with self.assertRaisesRegex(ValueError,'manifest entry'):collect(self.root)
 def test_bad_window_health_invalidates_scene(self):
  self.write();(self.attempt/'window-health.json').write_text('{"passed":false}');rows,_=collect(self.root);self.assertIn('window-health',rows[0]['validity_reasons'])
 def test_early_error_nullable_faults(self):
  self.row.update(outcome='error',isaac_major_faults_start=None,isaac_major_faults_end=None,host_load_start=None);self.row['client']={};self.write();rows,_=collect(self.root);self.assertEqual(rows[0]['reported_outcome'],'invalid');self.assertIsNone(rows[0]['isaac_major_faults_delta'])
 def test_crashed_attempt_cannot_be_retried(self):
  other=self.root/'attempt-02';other.mkdir()
  for name in ('run_config.json','episode-manifest.json','window-health.json'):(other/name).write_text((self.attempt/name).read_text())
  (other/'ep00').mkdir();(other/'ep00/result.json').write_text(json.dumps(self.row))
  with self.assertRaisesRegex(ValueError,'duplicate seed attempt'):collect(self.root)
 def test_wrong_checkpoint_without_result_refused(self):
  self.config['pretrained']='/model/060000/pretrained_model';(self.attempt/'run_config.json').write_text(json.dumps(self.config))
  with self.assertRaisesRegex(ValueError,'checkpoint step'):collect(self.root)
 def test_wrong_root_refused(self):
  with self.assertRaisesRegex(ValueError,'window root'):collect(self.ep)
 def test_missing_manifest_refused(self):
  (self.attempt/'episode-manifest.json').unlink()
  with self.assertRaises(FileNotFoundError):collect(self.root)
 def test_rtc_off_arm_keeps_false_flag_and_planned_cohort(self):
  self.config['rtc_enabled']=False;(self.attempt/'run_config.json').write_text(json.dumps(self.config))
  entries=[r for r in manifest()['best_checkpoint_only'] if r['suite']=='main-rtc-off'];(self.attempt/'episode-manifest.json').write_text(json.dumps(entries));self.row.update(suite='main-rtc-off',rtc_enabled=False);self.write()
  rows,_=collect(self.root,suite='main-rtc-off');self.assertFalse(rows[0]['rtc_enabled']);self.assertTrue(rows[0]['valid']);self.assertEqual(summarize(rows)['n'],30)
 def test_rtc_off_cannot_be_mislabeled_as_rtc_on(self):
  entries=[r for r in manifest()['best_checkpoint_only'] if r['suite']=='main-rtc-off'];(self.attempt/'episode-manifest.json').write_text(json.dumps(entries));self.row.update(suite='main-rtc-off',rtc_enabled=False);self.write()
  with self.assertRaisesRegex(ValueError,'planned MAIN'):collect(self.root)
 def test_config_rtc_mismatch_refused(self):
  with self.assertRaisesRegex(ValueError,'RTC configuration'):collect(self.root,suite='main-rtc-off')
 def test_guided_prefix_in_off_invalidates_scene(self):
  self.config['rtc_enabled']=False;(self.attempt/'run_config.json').write_text(json.dumps(self.config))
  entries=[r for r in manifest()['best_checkpoint_only'] if r['suite']=='main-rtc-off'];(self.attempt/'episode-manifest.json').write_text(json.dumps(entries))
  self.row.update(suite='main-rtc-off',rtc_enabled=False);self.row['client']['chunk_events']=[{'eval_rtc':{'prefix_steps':3}}];self.write()
  rows,_=collect(self.root,suite='main-rtc-off');self.assertFalse(rows[0]['valid']);self.assertIn('guided-prefix-in-rtc-off',rows[0]['validity_reasons'])
 def test_same_step_different_checkpoint_family_refused(self):
  (self.root/'preflight.json').write_text(json.dumps(dict(source_sha256={},checkpoint=dict(path='/other-family/010000/pretrained_model'))))
  with self.assertRaisesRegex(ValueError,'Checkpoint path'):collect(self.root)
 def test_no_close_video_is_not_labeled_as_a_closing_distance(self):
  rows=[dict(seed=900000,color='red',valid=True,result_file='result.json',success=False,closing_distance_m=None)]
  chosen=select_examples(rows);self.assertEqual(len(chosen),1);self.assertIn('no ≥2° closure',chosen[0][1])
if __name__=='__main__':unittest.main()
