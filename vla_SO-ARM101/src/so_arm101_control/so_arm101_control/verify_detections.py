#!/usr/bin/env python3
"""
Verify YOLOE detected poses against Gazebo ground truth for SO-ARM101.

Subscribes to /objects_poses (TFMessage) and compares detected positions
against known ground truth from lego_world.sdf.

Ported from RoboSort/JETANK_description/verify_detections.py.
Stripped depth diagnostics (SO-ARM101 uses RGB-only YOLOE pipeline).

Usage:
    ros2 run so_arm101_control verify_detections
"""

import re
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
import tf2_ros

# Lego model names in Gazebo -> color keys for detection matching
LEGO_MODELS = {
    'red_lego_2x4':   'red',
    'green_lego_2x3': 'green',
    'blue_lego_2x2':  'blue',
}


def query_lego_world_positions():
    """Read actual lego positions from Gazebo's dynamic_pose topic."""
    try:
        result = subprocess.run(
            ["ign", "topic", "-e", "-t",
             "/world/so_arm101_lego_world/pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    positions = {}
    blocks = re.split(r'(?=^pose\s*\{)', result.stdout, flags=re.MULTILINE)
    for block in blocks:
        for model_name, color in LEGO_MODELS.items():
            if f'"{model_name}"' in block:
                def _val(txt, key):
                    m = re.search(rf'{key}:\s*([-\d.e+]+)', txt)
                    return float(m.group(1)) if m else 0.0

                pos_m = re.search(r'position\s*\{([^}]*)\}', block)
                pb = pos_m.group(1) if pos_m else ""
                positions[color] = np.array([
                    _val(pb, 'x'), _val(pb, 'y'), _val(pb, 'z')])
                break
    return positions


class DetectionVerifier(Node):
    def __init__(self):
        super().__init__('detection_verifier')
        self.subscription = self.create_subscription(
            TFMessage, '/objects_poses', self.objects_callback, 10
        )
        self.cam_pose_sub = self.create_subscription(
            PoseStamped, '/camera_pose', self.camera_pose_callback, 10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera_info', self.camera_info_callback, 10
        )

        self.detections = {}
        self.detection_count = 0
        self.start_time = time.time()

        # Camera state
        self.cam_pos = None
        self.cam_quat = None
        self.cam_info = None

        # TF2 for world -> base
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ground_truth = {}

        self.get_logger().info('Detection Verifier started (SO-ARM101)')
        self.get_logger().info('Waiting for world->base TF and detections...')

        # Timer to print comparison every 5 seconds
        self.timer = self.create_timer(5.0, self.print_results)

    def camera_pose_callback(self, msg):
        self.cam_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self.cam_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

    def camera_info_callback(self, msg):
        if self.cam_info is None:
            self.cam_info = msg
            self.get_logger().info(
                f'Camera info: {msg.width}x{msg.height}, '
                f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}')

    def objects_callback(self, msg):
        self.detection_count += 1
        for transform in msg.transforms:
            name = transform.child_frame_id
            pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])
            quat = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ])
            self.detections[name] = {
                'pos': pos,
                'quat': quat,
                'time': time.time(),
            }

    def match_detection_to_gt(self, det_name):
        """Match a detection name to a ground truth entry by color."""
        det_lower = det_name.lower()
        for gt_name in self.ground_truth:
            if gt_name in det_lower:
                return gt_name
        return None

    def _update_ground_truth_from_tf(self):
        """Refresh ground truth using live Gazebo poses + world->base TF."""
        try:
            from scipy.spatial.transform import Rotation as R

            t = self.tf_buffer.lookup_transform(
                "base", "world", rclpy.time.Time())
            p = t.transform.translation
            q = t.transform.rotation
            tf_pos = np.array([p.x, p.y, p.z])
            tf_rot = R.from_quat([q.x, q.y, q.z, q.w])

            world_positions = query_lego_world_positions()
            if not world_positions:
                return len(self.ground_truth) > 0

            for color, p_world in world_positions.items():
                p_base = tf_rot.apply(p_world) + tf_pos
                self.ground_truth[color] = {
                    'x': p_base[0], 'y': p_base[1], 'z': p_base[2]}
            return True
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return False

    def print_results(self):
        elapsed = time.time() - self.start_time

        if not self._update_ground_truth_from_tf():
            if not self.ground_truth:
                self.get_logger().info(
                    f'[{elapsed:.0f}s] Waiting for world->base TF...')
                return

        if not self.detections:
            self.get_logger().info(
                f'[{elapsed:.0f}s] No detections yet '
                f'(received {self.detection_count} messages)...'
            )
            return

        lines = []
        lines.append('=' * 70)
        lines.append(f'DETECTION vs GROUND TRUTH  [{elapsed:.0f}s elapsed, '
                      f'{self.detection_count} msgs, '
                      f'{len(self.detections)} objects]')
        lines.append('=' * 70)

        matched = 0
        total_error = 0.0

        for det_name, det_data in sorted(self.detections.items()):
            det_pos = det_data['pos']
            age = time.time() - det_data['time']

            gt_name = self.match_detection_to_gt(det_name)

            if gt_name is not None:
                gt = self.ground_truth[gt_name]
                gt_pos = np.array([gt['x'], gt['y'], gt['z']])
                error_vec = det_pos - gt_pos
                error_mm = np.linalg.norm(error_vec) * 1000

                lines.append(
                    f'  {det_name:20s} -> {gt_name:8s}  '
                    f'det=({det_pos[0]:.4f}, {det_pos[1]:.4f}, {det_pos[2]:.4f})  '
                    f'gt=({gt_pos[0]:.4f}, {gt_pos[1]:.4f}, {gt_pos[2]:.4f})  '
                    f'err={error_mm:.1f}mm  '
                    f'(dx={error_vec[0]*1000:.1f}, dy={error_vec[1]*1000:.1f}, '
                    f'dz={error_vec[2]*1000:.1f})  '
                    f'age={age:.1f}s'
                )
                matched += 1
                total_error += error_mm
            else:
                lines.append(
                    f'  {det_name:20s} -> NO MATCH  '
                    f'pos=({det_pos[0]:.4f}, {det_pos[1]:.4f}, {det_pos[2]:.4f})  '
                    f'age={age:.1f}s'
                )

        if matched > 0:
            avg_error = total_error / matched
            lines.append(f'\n  Matched: {matched}/{len(self.ground_truth)} | '
                          f'Avg error: {avg_error:.1f}mm | '
                          f'{"PASS" if avg_error < 50 else "FAIL"} (threshold: 50mm)')

        lines.append('=' * 70)

        for line in lines:
            self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
