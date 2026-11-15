import unittest
import numpy as np
from direction_metrics import angle_degrees,motion_directions,seed_consistency

class DirectionTests(unittest.TestCase):
 def test_angles_and_stationary_refusal(self):
  self.assertAlmostEqual(angle_degrees([1,0],[1,0]),0)
  self.assertAlmostEqual(angle_degrees([1,0],[0,1]),90)
  self.assertAlmostEqual(angle_degrees([1,0],[-1,0]),180)
  self.assertIsNone(angle_degrees([0,0],[1,0]))
 def test_requested_direction_and_every_block(self):
  blocks={'red_a':[1,0,0],'red_b':[-1,0,0],'green_a':[0,1,0],'blue_a':[0,-1,0]}
  m=motion_directions([0,0,0],[[.1,0,0]],blocks,'red')
  self.assertEqual(len(m['blocks']),4);self.assertTrue(m['xy']['requested_winner']);self.assertEqual(m['xy']['winner_blocks'],['red_a'])
 def test_vertical_motion_does_not_claim_horizontal_grounding(self):
  m=motion_directions([0,0,1],[[0,0,.5]],{'red_a':[1,0,0]},'red')
  self.assertTrue(m['low_horizontal_motion']);self.assertIsNone(m['xy']['requested_winner']);self.assertIsNotNone(m['blocks'][0]['angle_3d_deg'])
 def test_valid_multimodal_same_colour_can_cancel_mean(self):
  blocks={'red_a':[1,0,0],'red_b':[-1,0,0],'blue_a':[0,1,0]}
  ms=[motion_directions([0,0,0],[[x,0,0]],blocks,'red') for x in (.1,-.1)]
  s=seed_consistency(ms);self.assertEqual(s['requested_direction_wins'],2);self.assertAlmostEqual(s['direction_resultant'],0);self.assertAlmostEqual(s['max_seed_direction_angle_deg'],180)
 def test_negligible_motion_is_not_a_wrong_direction(self):
  m=motion_directions([0,0,0],[[1e-5,0,0]],{'red_a':[1,0,0]},'red')
  self.assertTrue(m['stationary']);self.assertIsNone(m['xy']['requested_winner'])
 def test_different_colour_same_bearing_is_ambiguous(self):
  m=motion_directions([0,0,0],[[.1,0,0]],{'red_a':[1,0,0],'blue_a':[2,0,0]},'red')
  self.assertIsNone(m['xy']['requested_winner']);self.assertEqual(m['xy']['winner_colors'],['blue','red'])
  s=seed_consistency([m]);self.assertEqual(s['ambiguous_colour_n'],1);self.assertEqual(s['wrong_colour_n'],0)

if __name__=='__main__':unittest.main()
