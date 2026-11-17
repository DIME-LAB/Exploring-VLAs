import unittest
from report_rule431 import summarize,wilson
class Tests(unittest.TestCase):
 def test_invalid_preserved_and_geometry_phase_explicit(self):
  row=dict(valid=True,color='red',requested_colour_success=True,requested_colour_grasp=True,rule_target_success=False,rule_target_grasp=False,wrong_color_lift=False,wrong_same_colour_target_grasp=True,rule_closure_confirmed_geometry=dict(status='available',xy_mm=20,yaw_deg=-30))
  result=summarize([row,dict(valid=False)])
  self.assertEqual(result['invalid'],1);self.assertEqual(result['requested_colour_success']['count'],1);self.assertEqual(result['rule_target_success']['count'],0)
  self.assertEqual(result['yaw_deg']['mean'],30);self.assertIn('CONFIRMED',result['closure_phase'])
  self.assertGreater(wilson(0,15)[1],0);self.assertIsNone(wilson(0,0))
 def test_missing_rule_fields_refused(self):
  with self.assertRaises(ValueError):summarize([dict(valid=True)])
if __name__=='__main__':unittest.main()
