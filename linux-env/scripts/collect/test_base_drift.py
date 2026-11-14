"""Native base drift reporting has no scene setters and does not assume DOF order."""
import math,unittest
from base_drift import summary


class BaseDriftTests(unittest.TestCase):
    def test_measured_wobble_is_reported_without_relabelling_the_ros_reference_frame(self):
        initial=dict(robot_base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))
        q=[math.cos(.0063/2),0.,0.,math.sin(.0063/2)]
        rows=[dict(robot_base_world_pose=dict(position=[.00011,0.,0.],orientation=q),robot_base_pose_source='physx_rigid_body')]
        d=summary(initial,rows);self.assertAlmostEqual(d['max_translation_m'],.00011);self.assertAlmostEqual(d['max_rotation_rad'],.0063,places=10)

    def test_missing_or_unmeasured_native_pose_refuses_a_partial_sequence(self):
        initial=dict(robot_base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))
        row=dict(robot_base_world_pose=initial['robot_base_world_pose'],robot_base_pose_source='physx_rigid_body')
        with self.assertRaises(ValueError):summary(initial,[row,{}])
        with self.assertRaises(ValueError):summary(initial,[dict(row,robot_base_pose_source='USD')])
        self.assertFalse(summary(None,[])['available'])

    def test_quaternion_sign_and_norm_do_not_fake_rotation(self):
        initial=dict(robot_base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))
        row=dict(robot_base_world_pose=dict(position=[0.,0.,0.],orientation=[-1.00000001,0.,0.,0.]),robot_base_pose_source='physx_rigid_body')
        self.assertEqual(summary(initial,[row])['max_rotation_rad'],0.)
