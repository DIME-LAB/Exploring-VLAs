#!/usr/bin/env python3
"""
IK Solver Test Script for SO-ARM101

Calls /compute_ik with multiple test poses in the `base` frame,
measures solve time and success rate. Tests both position+orientation
and position-only goals.

Usage:
  ros2 run so_arm101_control test_ik_solvers
"""

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState


ARM_JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']

# Test poses: (x, y, z, qx, qy, qz, qw, description)
#
# SO-ARM101 joint axes at home config:
#   Rotation: -Z (base yaw), Pitch/Elbow/Wrist_Pitch: +X (pitch),
#   Wrist_Roll: +Y (tool roll)
# Controllable EE orientation: pitch (X-rot) + tool-roll (Y-rot)
# CANNOT independently control: side-tilt (Y-rot decoupled from tool-roll)
#
# Orientation (0,0,0,1) = identity quaternion (EE pointing up at home)
TEST_POSES = [
    # --- Reachable positions, identity orientation ---
    (0.12, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0, 'front_center_up'),
    (0.10, 0.05, 0.12, 0.0, 0.0, 0.0, 1.0, 'front_right'),
    (0.10, -0.05, 0.12, 0.0, 0.0, 0.0, 1.0, 'front_left'),
    (0.08, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0, 'high_center'),
    (0.15, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0, 'low_front'),
    # --- Achievable orientations (pitch about X + tool roll about Y) ---
    # 45-deg pitch about X (uses Pitch/Elbow/Wrist_Pitch)
    (0.12, 0.0, 0.12, 0.383, 0.0, 0.0, 0.924, 'pitch_45'),
    # 90-deg tool roll about Y (uses Wrist_Roll, within +-2.79 rad limit)
    (0.12, 0.0, 0.15, 0.0, 0.707, 0.0, 0.707, 'tool_roll_90'),
    # --- Impossible orientation (side-tilt about Z = missing DOF) ---
    (0.12, 0.0, 0.15, 0.0, 0.0, 0.383, 0.924, 'side_tilt_45_IMPOSSIBLE'),
    # --- Edge-of-workspace positions ---
    (0.20, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0, 'far_front'),
    (0.05, 0.10, 0.10, 0.0, 0.0, 0.0, 1.0, 'far_right'),
]


class IKSolverTester(Node):
    def __init__(self):
        super().__init__('ik_solver_tester')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/compute_ik service not available!')
            return
        self.get_logger().info('/compute_ik service ready')

    def test_pose(self, x, y, z, qx, qy, qz, qw, description,
                  position_only=False):
        """Test a single IK query. Returns (success, solve_time_ms)."""
        request = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = 'arm'
        ik_req.avoid_collisions = False

        # Seed state at home (all zeros)
        robot_state = RobotState()
        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        robot_state.joint_state.position = [0.0] * len(ARM_JOINT_NAMES)
        ik_req.robot_state = robot_state

        pose = PoseStamped()
        pose.header.frame_id = 'base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        if position_only:
            # Identity orientation — solver should treat as "don't care"
            # with approximate IK / pick_ik's low rotation_scale
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

        ik_req.pose_stamped = pose
        request.ik_request = ik_req

        t0 = time.monotonic()
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if future.result() is None:
            return False, elapsed_ms, 'timeout'

        response = future.result()
        error_code = response.error_code.val
        success = (error_code == 1)  # MoveItErrorCodes.SUCCESS = 1

        if success:
            sol = response.solution.joint_state
            joints_str = ', '.join(
                f'{n}={sol.position[i]:.3f}'
                for i, n in enumerate(sol.name)
                if n in ARM_JOINT_NAMES
            )
            return True, elapsed_ms, joints_str
        else:
            return False, elapsed_ms, f'error_code={error_code}'

    def run_all_tests(self):
        """Run all test poses and print results."""
        self.get_logger().info('=' * 70)
        self.get_logger().info('IK SOLVER TEST — SO-ARM101 (5-DOF arm)')
        self.get_logger().info('=' * 70)

        for mode_label, position_only in [
            ('FULL 6-DOF (position + orientation)', False),
            ('POSITION-ONLY (orientation = identity)', True),
        ]:
            self.get_logger().info('')
            self.get_logger().info(f'--- {mode_label} ---')
            successes = 0
            total = 0
            total_time = 0.0

            for pose in TEST_POSES:
                x, y, z, qx, qy, qz, qw, desc = pose
                total += 1
                ok, ms, info = self.test_pose(
                    x, y, z, qx, qy, qz, qw, desc,
                    position_only=position_only)
                total_time += ms
                status = 'OK' if ok else 'FAIL'
                if ok:
                    successes += 1
                self.get_logger().info(
                    f'  [{status}] {desc:20s}  '
                    f'({x:.2f},{y:.2f},{z:.2f})  '
                    f'{ms:7.1f}ms  {info}')

            rate = successes / total * 100 if total > 0 else 0
            avg_ms = total_time / total if total > 0 else 0
            self.get_logger().info(
                f'  RESULT: {successes}/{total} = {rate:.0f}%  '
                f'avg {avg_ms:.1f}ms')

        self.get_logger().info('')
        self.get_logger().info('=' * 70)
        self.get_logger().info('Test complete')


def main(args=None):
    rclpy.init(args=args)
    node = IKSolverTester()
    if node.ik_client.service_is_ready():
        node.run_all_tests()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
