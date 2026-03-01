#!/usr/bin/env python3
"""
Test script for SO-ARM101 debug services and IK solver.

Exercises the GUI's debug services and tests IK with multiple positions/orientations.
Run: ros2 run so_arm101_control test_debug_services
"""

import math
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import PositionIKRequest, RobotState
    from geometry_msgs.msg import PoseStamped
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False

ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
GUI_NODE = '/so_arm101_control_gui'

# Test positions: (label, x, y, z)
# Note: arm home points FORWARD (+X at zero), new_calib convention
IK_POSITIONS = [
    ('front_center', 0.12, 0.0, 0.15),
    ('right_near',   0.10, -0.10, 0.12),
    ('front_left',   0.10, -0.08, 0.12),
    ('low_front',    0.15, 0.0, 0.05),
    ('high_mid',     0.10, 0.0, 0.18),
]

# Test orientations: (label, qx, qy, qz, qw)
IK_ORIENTATIONS = [
    ('identity',     0.0, 0.0, 0.0, 1.0),
    ('pitch_45',     0.0, 0.383, 0.0, 0.924),
    ('yaw_90',       0.0, 0.0, 0.707, 0.707),
]


class TestDebugServices(Node):
    def __init__(self):
        super().__init__('test_debug_services')
        self.passed = 0
        self.failed = 0
        self.skipped = 0

        # IK client
        if MOVEIT_AVAILABLE:
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        else:
            self.ik_client = None

    def call_trigger(self, service_name, timeout=5.0):
        """Call a Trigger service on the GUI node. Returns (success, message)."""
        full_name = f'{GUI_NODE}/{service_name}'
        client = self.create_client(Trigger, full_name)
        if not client.wait_for_service(timeout_sec=timeout):
            return None, f'Service {full_name} not available'

        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        if future.result() is None:
            return None, 'Service call timed out'

        resp = future.result()
        return resp.success, resp.message

    def call_ik(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                seed=None, avoid_collisions=True):
        """Call /compute_ik directly. Returns (error_code, joint_dict) or (None, error_msg)."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            return None, 'MoveIt not available'

        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            return None, '/compute_ik not available'

        request = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = 'arm'
        ik_req.avoid_collisions = avoid_collisions

        robot_state = RobotState()
        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        robot_state.joint_state.position = seed or [0.0] * len(ARM_JOINT_NAMES)
        ik_req.robot_state = robot_state

        pose = PoseStamped()
        pose.header.frame_id = 'base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        ik_req.pose_stamped = pose

        request.ik_request = ik_req

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            return None, 'IK call timed out'

        resp = future.result()
        code = resp.error_code.val

        if code == 1:
            joints = {}
            for i, name in enumerate(resp.solution.joint_state.name):
                if name in ARM_JOINT_NAMES and i < len(resp.solution.joint_state.position):
                    joints[name] = resp.solution.joint_state.position[i]
            return code, joints
        return code, {}

    def report(self, test_name, passed, detail=''):
        status = 'PASS' if passed else 'FAIL'
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        msg = f'  [{status}] {test_name}'
        if detail:
            msg += f' — {detail}'
        self.get_logger().info(msg)

    def skip(self, test_name, reason=''):
        self.skipped += 1
        msg = f'  [SKIP] {test_name}'
        if reason:
            msg += f' — {reason}'
        self.get_logger().info(msg)

    def run_all(self):
        self.get_logger().info('=== SO-ARM101 Debug Services Test ===')
        self.get_logger().info('')

        # --- 1. Service smoke tests ---
        self.get_logger().info('--- Service Smoke Tests ---')

        services = [
            'zero_arm', 'get_joint_positions', 'randomize_arm',
            'gripper_open', 'gripper_close',
        ]

        for svc in services:
            ok, msg = self.call_trigger(svc)
            if ok is None:
                self.report(f'service/{svc}', False, msg)
            else:
                self.report(f'service/{svc}', ok, msg[:80] if msg else '')

        # set_joints (after zero)
        self.call_trigger('zero_arm')
        time.sleep(0.5)
        ok, msg = self.call_trigger('set_joints')
        if ok is None:
            self.report('service/set_joints', False, msg)
        else:
            self.report('service/set_joints', ok, msg[:80] if msg else '')

        # plan_execute (after zero — trivial plan)
        time.sleep(1.0)
        ok, msg = self.call_trigger('plan_execute')
        if ok is None:
            self.report('service/plan_execute', False, msg)
        else:
            self.report('service/plan_execute', ok, msg[:80] if msg else '')

        self.get_logger().info('')

        # --- 2. IK Position Tests ---
        self.get_logger().info('--- IK Position Tests ---')
        if not MOVEIT_AVAILABLE:
            self.skip('ik/positions', 'MoveIt not available')
        else:
            for label, x, y, z in IK_POSITIONS:
                # Try with zero seed
                code, joints = self.call_ik(x, y, z, seed=[0.0]*5)
                if code == 1:
                    jstr = ', '.join(f'{n}:{joints[n]:.3f}' for n in ARM_JOINT_NAMES if n in joints)
                    self.report(f'ik/pos/{label}', True, jstr)
                else:
                    # Try with heuristic seed
                    rot = math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0
                    code2, joints2 = self.call_ik(x, y, z, seed=[rot, 0.0, 0.0, 0.0, 0.0])
                    if code2 == 1:
                        jstr = ', '.join(f'{n}:{joints2[n]:.3f}' for n in ARM_JOINT_NAMES if n in joints2)
                        self.report(f'ik/pos/{label}', True, f'heuristic seed — {jstr}')
                    else:
                        self.report(f'ik/pos/{label}', False, f'error codes: zero={code}, heuristic={code2}')

        self.get_logger().info('')

        # --- 3. IK Orientation Tests ---
        self.get_logger().info('--- IK Orientation Tests ---')
        if not MOVEIT_AVAILABLE:
            self.skip('ik/orientations', 'MoveIt not available')
        else:
            # Use front_center position for orientation tests
            tx, ty, tz = 0.12, 0.0, 0.15
            solutions = {}

            for label, qx, qy, qz, qw in IK_ORIENTATIONS:
                code, joints = self.call_ik(tx, ty, tz, qx, qy, qz, qw)
                if code == 1:
                    solutions[label] = joints
                    jstr = ', '.join(f'{n}:{joints[n]:.3f}' for n in ARM_JOINT_NAMES if n in joints)
                    self.report(f'ik/orient/{label}', True, jstr)
                else:
                    self.report(f'ik/orient/{label}', False, f'error code: {code}')

            # Check if different orientations produce different solutions
            if len(solutions) >= 2:
                labels = list(solutions.keys())
                all_same = True
                for i in range(1, len(labels)):
                    for n in ARM_JOINT_NAMES:
                        if n in solutions[labels[0]] and n in solutions[labels[i]]:
                            if abs(solutions[labels[0]][n] - solutions[labels[i]][n]) > 0.01:
                                all_same = False
                                break
                if all_same:
                    self.get_logger().info(
                        '  [INFO] All orientations gave same solution — solver ignoring orientation '
                        '(rotation_scale=0.0)')
                else:
                    self.get_logger().info(
                        '  [INFO] Different orientations gave different solutions — orientation active')

        self.get_logger().info('')

        # --- 4. EE Orientation Readback ---
        self.get_logger().info('--- EE Orientation Readback ---')
        ok, msg = self.call_trigger('randomize_arm')
        if ok:
            time.sleep(1.0)
            ok2, msg2 = self.call_trigger('get_joint_positions')
            self.report('ee/readback_after_randomize', ok2,
                       msg2[:80] if msg2 else '')
        else:
            self.report('ee/readback_after_randomize', False, 'randomize failed')

        # --- Summary ---
        self.get_logger().info('')
        total = self.passed + self.failed + self.skipped
        self.get_logger().info(
            f'=== Results: {self.passed} passed, {self.failed} failed, '
            f'{self.skipped} skipped / {total} total ===')

        return self.failed == 0


def main(args=None):
    rclpy.init(args=args)
    node = TestDebugServices()
    try:
        success = node.run_all()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
