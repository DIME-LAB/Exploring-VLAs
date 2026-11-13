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


class LegacyParityTests(unittest.TestCase):
    def test_1000_random_cases_match_verified_gui_geometry(self):
        import contextlib
        import math
        import random
        from types import SimpleNamespace
        from legacy_geometry_reference import LegacyGeometry
        from so_arm101_control.grasp_geometry import object_grip_width
        rng = random.Random(3101)
        def setting(value):
            return SimpleNamespace(get=lambda:value)
        for _ in range(1000):
            yaw = rng.uniform(-math.pi,math.pi)
            pose = dict(x=rng.uniform(-.30,.30), y=rng.uniform(-.20,.20), z=rng.uniform(.004,.02), qz=math.sin(yaw/2),qw=math.cos(yaw/2))
            bbox = dict(sx=rng.uniform(.01,.06),sy=rng.uniform(.008,.04),sz=rng.uniform(.008,.02))
            cross=rng.choice((True,False));enabled=rng.choice((True,False));wr=rng.uniform(-2.7,2.8)
            override=rng.choice((0.,rng.uniform(.01,.03)));height=rng.uniform(0,.04);drop=rng.choice((True,False))
            tcp=rng.uniform(0,.003);opening=rng.uniform(0,.01);closing=rng.uniform(0,.002)
            old=LegacyGeometry();old.joint_lock=contextlib.nullcontext();old._lookup_bbox=lambda name:bbox
            old._actual_positions={'wrist_roll':wr}
            for name,value in {'_grasp_cross_var':cross,'_bbox_enabled_var':enabled,'_tcp_clearance_var':tcp*1000,
                '_jaw_open_clearance_var':opening*1000,'_jaw_close_clearance_var':closing*1000,
                '_grasp_obj_z_var':override,'_grasp_topic_var':'/drop_poses' if drop else '/objects_poses_sim',
                '_grasp_approach_height_var':height}.items():setattr(old,name,setting(value))
            expected=old.plan(pose)
            actual=make_grasp_plan(pose,bbox,cross=cross,current_wrist_roll=wr,tcp_clearance=tcp,
                bbox_enabled=enabled,object_z_override=override,approach_height=height,drop=drop)
            for key in ('tcp','jaw_offset'):
                for a,b in zip(actual[key],expected[key]):self.assertAlmostEqual(a,b,delta=1e-12)
            self.assertAlmostEqual(actual['yaw'],expected['yaw'],delta=1e-12)
            self.assertEqual(actual['poses'],expected['poses'])
            angles=gripper_angles(object_grip_width(bbox,cross),tcp,opening,closing)
            for a,b in zip(angles,old._gripper_angle_for_object('block')):self.assertAlmostEqual(a,b,delta=1e-12)


if __name__ == '__main__':
    unittest.main()
