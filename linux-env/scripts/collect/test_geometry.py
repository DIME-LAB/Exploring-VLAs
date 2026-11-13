"""Offline geometry boundary and inherited Quickstart regression checks."""
import ast
from pathlib import Path
import unittest
import sys

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / 'vla_SO-ARM101/src/so_arm101_control/so_arm101_control'


sys.path.insert(0, str(PACKAGE.parent))
from so_arm101_control.grasp_geometry import make_grasp_plan, gripper_angles, jaw_offset


class GeometryBoundaryTests(unittest.TestCase):
    def test_grasp_plan_preserves_final_target_and_offsets_approach(self):
        pose = dict(x=.2, y=0., z=.0064, qz=0., qw=1.)
        bbox = dict(sx=.03, sy=.02, sz=.0128)
        plan = make_grasp_plan(pose, bbox, approach_offset=(.003, -.002))
        self.assertAlmostEqual(plan['tcp'][0], .2)
        self.assertAlmostEqual(plan['tcp'][1], .011)
        self.assertAlmostEqual(plan['tcp'][2], .01445)
        self.assertAlmostEqual(plan['poses'][0][1], .203)
        self.assertAlmostEqual(plan['poses'][0][2], .009)
        self.assertEqual(plan['poses'][-1][1:], plan['tcp'])
        self.assertEqual(pose['y'], 0.)

    def test_jaw_clearance_and_limits(self):
        opening, closing = gripper_angles(.02)
        self.assertGreater(opening, closing)
        self.assertEqual(jaw_offset(.02, 1., enabled=False), (0., 0.))

    def test_geometry_has_no_transport_imports(self):
        tree = ast.parse((PACKAGE / 'grasp_geometry.py').read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(name and name.startswith(('rclpy', 'moveit', 'tkinter')) for name in imports))

    def test_gui_uses_the_shared_motion_class(self):
        tree = ast.parse((PACKAGE / 'control_gui.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertIn('PickPlaceNode', [n.id for n in cls.bases if isinstance(n, ast.Name)])
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name in ('_cmd_grasp_move', '_execute_full_trajectory', '_attach_lego_to_gripper', '_qs_run'):
            self.assertNotIn(name, methods)


if __name__ == '__main__':
    unittest.main()
