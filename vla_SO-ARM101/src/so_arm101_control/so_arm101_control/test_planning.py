#!/usr/bin/env python3
"""
Planning Diagnostic for SO-ARM101

Tests:
1. IK solve (position-only) for several target poses
2. Joint-space motion planning from home to each IK solution
3. Reports collision state of both start and goal

Usage:
  ros2 run so_arm101_control test_planning
"""

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetStateValidity
from moveit_msgs.msg import (
    PositionIKRequest,
    RobotState,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
)
from sensor_msgs.msg import JointState


ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

# Target positions (x, y, z, description) — will use position-only IK
TARGETS = [
    (0.12, 0.0, 0.15, 'front_center_up'),
    (0.10, 0.05, 0.12, 'front_right'),
    (0.10, -0.05, 0.12, 'front_left'),
    (0.08, 0.0, 0.20, 'high_center'),
    (0.15, 0.0, 0.05, 'low_front'),
    (0.12, 0.0, 0.12, 'mid_center'),
    (0.14, 0.03, 0.10, 'low_right'),
]


def make_robot_state(positions):
    rs = RobotState()
    rs.joint_state.name = list(ARM_JOINT_NAMES)
    rs.joint_state.position = list(positions)
    return rs


class PlanningTester(Node):
    def __init__(self):
        super().__init__('planning_tester')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.validity_client = self.create_client(
            GetStateValidity, '/check_state_validity'
        )

        self.get_logger().info('Waiting for services...')
        for name, client in [
            ('/compute_ik', self.ik_client),
            ('/plan_kinematic_path', self.plan_client),
            ('/check_state_validity', self.validity_client),
        ]:
            if not client.wait_for_service(timeout_sec=15.0):
                self.get_logger().error(f'{name} not available!')
                return
        self.get_logger().info('All services ready')

    def solve_ik(self, x, y, z, avoid_collisions=True, seed=None):
        """Position-only IK. Returns joint positions or None."""
        request = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = 'arm'
        ik_req.avoid_collisions = avoid_collisions
        ik_req.robot_state = make_robot_state(seed if seed else [0.0] * 5)

        pose = PoseStamped()
        pose.header.frame_id = 'base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        ik_req.pose_stamped = pose
        request.ik_request = ik_req

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            return None
        resp = future.result()
        if resp.error_code.val != 1:
            return None
        sol = resp.solution.joint_state
        # Extract arm joints in order
        positions = []
        for jn in ARM_JOINT_NAMES:
            if jn in sol.name:
                positions.append(sol.position[sol.name.index(jn)])
            else:
                positions.append(0.0)
        return positions

    def check_validity(self, positions):
        """Check if a joint configuration is collision-free. Returns (valid, contacts)."""
        request = GetStateValidity.Request()
        request.group_name = 'arm'
        request.robot_state = make_robot_state(positions)
        future = self.validity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return None, 'timeout'
        resp = future.result()
        contacts = []
        for c in resp.contacts:
            contacts.append(f'{c.contact_body_1}<->{c.contact_body_2}')
        return resp.valid, contacts

    def plan_joint_goal(self, start_positions, goal_positions, planning_time=5.0):
        """Plan from start to goal in joint space. Returns (success, error_code, time_ms)."""
        request = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.num_planning_attempts = 10
        mpr.allowed_planning_time = planning_time
        mpr.max_velocity_scaling_factor = 0.5
        mpr.max_acceleration_scaling_factor = 0.5

        # Start state
        mpr.start_state = make_robot_state(start_positions)

        # Goal: joint constraints
        constraints = Constraints()
        for i, jn in enumerate(ARM_JOINT_NAMES):
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = goal_positions[i]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        mpr.goal_constraints.append(constraints)

        request.motion_plan_request = mpr

        t0 = time.monotonic()
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if future.result() is None:
            return False, 'timeout', elapsed_ms
        resp = future.result()
        ec = resp.motion_plan_response.error_code.val
        return ec == 1, ec, elapsed_ms

    def run_tests(self):
        self.get_logger().info('=' * 70)
        self.get_logger().info('PLANNING DIAGNOSTIC — SO-ARM101')
        self.get_logger().info('=' * 70)

        home = [0.0] * 5

        # Check home validity
        valid, contacts = self.check_validity(home)
        self.get_logger().info(f'Home state valid: {valid}  contacts: {contacts}')

        for label, avoid_col in [
            ('IK avoid_collisions=True (RViz mode)', True),
            ('IK avoid_collisions=False (raw)', False),
        ]:
            self.get_logger().info('')
            self.get_logger().info(f'--- {label} ---')

            ok_count = 0
            total = 0

            for x, y, z, desc in TARGETS:
                total += 1
                # Step 1: IK
                joints = self.solve_ik(x, y, z, avoid_collisions=avoid_col)
                if joints is None:
                    self.get_logger().info(
                        f'  [{desc:20s}] IK FAIL  ({x:.2f},{y:.2f},{z:.2f})'
                    )
                    continue

                joints_str = ', '.join(
                    f'{ARM_JOINT_NAMES[i]}={joints[i]:.3f}' for i in range(5)
                )

                # Step 2: Check goal validity
                valid, contacts = self.check_validity(joints)
                collision_str = 'VALID' if valid else f'COLLISION {contacts}'

                # Step 3: Plan
                ok, ec, ms = self.plan_joint_goal(home, joints)
                status = 'OK' if ok else 'FAIL'
                if ok:
                    ok_count += 1

                self.get_logger().info(
                    f'  [{status}] {desc:20s}  '
                    f'({x:.2f},{y:.2f},{z:.2f})  '
                    f'plan_ec={ec}  {ms:7.1f}ms  '
                    f'goal={collision_str}'
                )
                self.get_logger().info(f'         joints: {joints_str}')

            self.get_logger().info(
                f'  RESULT: {ok_count}/{total} plans succeeded'
            )

        self.get_logger().info('')
        self.get_logger().info('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningTester()
    if node.ik_client.service_is_ready():
        node.run_tests()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
