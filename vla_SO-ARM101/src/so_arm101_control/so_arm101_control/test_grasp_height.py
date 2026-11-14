"""Regress the physical jaw/table bound at the shared grasp entrance."""
import unittest
from so_arm101_control.grasp_geometry import (
    make_grasp_plan,check_grasp_reachable,FINGER_BELOW_TCP_M,FINGER_TABLE_CLEARANCE_M)


class GraspHeightTests(unittest.TestCase):
    def test_tabletop_grasp_tracks_block_bottom_and_clears_jaw_nose(self):
        for bottom in (0.,.025):
            for height in (.0096,.0128,.016):
                pose=dict(x=.2,y=.02,z=bottom+height/2,qz=0.,qw=1.)
                plan=make_grasp_plan(pose,dict(sx=.031,sy=.0206,sz=height),approach_height=.02)
                tcp_z=plan['tcp'][2]
                self.assertAlmostEqual(tcp_z-bottom-FINGER_BELOW_TCP_M,FINGER_TABLE_CLEARANCE_M)
                self.assertAlmostEqual(plan['poses'][0][3]-tcp_z,.02)
                self.assertTrue(check_grasp_reachable(*plan['tcp'],ground_z=bottom)[0])

    def test_height_override_and_drop_target_preserve_their_meaning(self):
        pose=dict(x=.2,y=0.,z=.0064,qz=0.,qw=1.)
        bbox=dict(sx=.031,sy=.0206,sz=.0128)
        self.assertAlmostEqual(make_grasp_plan(pose,bbox,object_z_override=.017)['tcp'][2],.017)
        self.assertAlmostEqual(make_grasp_plan(pose,bbox,drop=True)['tcp'][2],.06445)


if __name__=='__main__':
    unittest.main()
