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
   row=dict(seed=900000,color='red',valid=True,requested_colour_success=False,requested_colour_grasp=False,rule_target_success=False,rule_target_grasp=False,wrong_color_lift=False,wrong_same_colour_target_grasp=False,rule_target_selection=dict(object='red_2x3_1',candidates=[],margin_m=.01,ambiguous=False))
   (ep/'result.json').write_text(json.dumps(row))
   with patch('native_bank.load_bank',return_value=([dict(seed=900000,source_receipt=str(sealed))],'heldout-bank-v2')):audit(root,root/'report')
   summary=json.loads((root/'report/summary.json').read_text());self.assertEqual(summary['rule_target_success_excluding_ambiguous']['denominator'],1)
   with (root/'report/episodes.csv').open() as stream:record=next(csv.DictReader(stream))
   diagnostic=json.loads(record['rule_target_selection']);self.assertFalse(diagnostic['live_matches_bank']);self.assertEqual(diagnostic['live_choice'],'red_2x3_1');self.assertEqual(diagnostic['bank_choice'],'red_2x3_0')
 def test_missing_rule_fields_refused(self):
  with self.assertRaises(ValueError):summarize([dict(valid=True)])
if __name__=='__main__':unittest.main()
