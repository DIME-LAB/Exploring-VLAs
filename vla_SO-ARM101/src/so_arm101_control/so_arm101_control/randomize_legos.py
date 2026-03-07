#!/usr/bin/env python3
"""Randomize lego positions in Gazebo via ign service for SO-ARM101.

Reads camera_link pose from /camera_pose and intrinsics from /camera_info.
Uses the empirically-derived camera_link-to-OpenCV rotation to project
world points to pixels and ensure objects are visible in the camera frame.

Usage:
    ros2 run so_arm101_control randomize_legos              # all fully in frame
    ros2 run so_arm101_control randomize_legos --edge       # one partially outside
    ros2 run so_arm101_control randomize_legos --reset      # default SDF positions
"""
import argparse
import math
import random
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo

# Default SDF positions (world frame) — (x, y, z, yaw)
DEFAULTS = {
    "red_lego_2x4":   (0.18,  0.03, 0.0055, 0.8),
    "green_lego_2x3": (0.20,  0.06, 0.0055, -0.4),
    "blue_lego_2x2":  (0.22,  0.03, 0.0055, 0.0),
}

# Lego half-sizes in meters (length/2, width/2)
HALF_SIZES = {
    "red_lego_2x4":   (0.016, 0.008),
    "green_lego_2x3": (0.012, 0.008),
    "blue_lego_2x2":  (0.008, 0.008),
}

TABLE_Z = 0.0055
MIN_SPACING = 0.03

# Rotation from camera_link frame to OpenCV frame (X=right, Y=down, Z=forward).
# Derived empirically: cv_X = -Z_link, cv_Y = -Y_link, cv_Z = -X_link.
# This accounts for the sensor <pose> rotation in so_arm101.gazebo.xacro
# (roll=pi/2, yaw=pi/2) plus the Gazebo-to-OpenCV convention change.
LINK_TO_OPENCV = np.array([
    [ 0,  0, -1],
    [ 0, -1,  0],
    [-1,  0,  0],
], dtype=float)


def forward_project(point_world, cam_pos, R_link, K):
    """Forward-project a world point to pixel via camera_link frame."""
    v_world = point_world - cam_pos
    v_link = R_link.T @ v_world
    v_cv = LINK_TO_OPENCV @ v_link
    if v_cv[2] <= 0:
        return None
    u = K[0, 0] * v_cv[0] / v_cv[2] + K[0, 2]
    v = K[1, 1] * v_cv[1] / v_cv[2] + K[1, 2]
    return u, v


class CameraHelper:
    def __init__(self):
        self.cam_pos = None
        self.cam_rot = None  # 3x3 rotation matrix (base -> camera_link)
        self.K = None
        self.img_w = self.img_h = None

        rclpy.init()
        self.node = Node("randomize_legos")
        self.node.create_subscription(PoseStamped, "/camera_pose", self._pose_cb, 1)
        self.node.create_subscription(CameraInfo, "/camera_info", self._info_cb, 1)

    def _pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.cam_pos = np.array([p.x, p.y, p.z])
        self.cam_rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def _info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.img_w = msg.width
        self.img_h = msg.height

    def wait_for_data(self, timeout=15.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if self.cam_pos is not None and self.K is not None:
                return True
        return False

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def check_camera_looking_down(self, threshold=0.8):
        """Return True if camera's optical axis points predominantly downward.

        The camera forward direction in OpenCV convention is +Z.
        In camera_link frame, that corresponds to LINK_TO_OPENCV^-1 @ [0,0,1].
        Since LINK_TO_OPENCV is its own inverse: forward_link = LINK_TO_OPENCV @ [0,0,1] = [-1,0,0].
        In world frame: forward_world = R_link @ [-1,0,0] = -R_link[:,0].
        """
        forward_world = -self.cam_rot[:, 0]
        return np.dot(forward_world, np.array([0.0, 0.0, -1.0])) >= threshold

    def backproject_pixel_to_ground(self, u, v):
        """Back-project a pixel to the ground plane (z=TABLE_Z) in world frame."""
        K_inv = np.linalg.inv(self.K)
        ray_cv = K_inv @ np.array([u, v, 1.0])
        # OpenCV -> camera_link -> world
        ray_link = LINK_TO_OPENCV @ ray_cv  # LINK_TO_OPENCV is its own inverse
        ray_world = self.cam_rot @ ray_link

        if abs(ray_world[2]) < 1e-9:
            return None
        t = (TABLE_Z - self.cam_pos[2]) / ray_world[2]
        if t <= 0:
            return None
        hit = self.cam_pos + t * ray_world
        return hit[0], hit[1]

    def compute_search_bounds(self, margin=40):
        """Compute world-frame search bounds by back-projecting image corners
        onto the ground plane."""
        corners = [
            (margin, margin),
            (self.img_w - margin, margin),
            (margin, self.img_h - margin),
            (self.img_w - margin, self.img_h - margin),
        ]
        xs, ys = [], []
        for cu, cv in corners:
            hit = self.backproject_pixel_to_ground(cu, cv)
            if hit is not None:
                xs.append(hit[0])
                ys.append(hit[1])

        if len(xs) < 2:
            return None

        x_min = max(min(xs), -0.5)
        x_max = min(max(xs), 0.5)
        y_min = max(min(ys), -0.5)
        y_max = min(max(ys), 0.5)

        if x_max - x_min < 0.02 or y_max - y_min < 0.02:
            return None

        return {"x": (x_min, x_max), "y": (y_min, y_max)}

    def project_world(self, wx, wy, wz):
        """Project a world-frame point to pixel coordinates."""
        p = np.array([wx, wy, wz])
        return forward_project(p, self.cam_pos, self.cam_rot, self.K)

    def _rotated_corners(self, wx, wy, half_l, half_w, yaw):
        """Return 4 world-frame corners of a brick rotated by yaw."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = []
        for dx in (-half_l, half_l):
            for dy in (-half_w, half_w):
                rx = c * dx - s * dy
                ry = s * dx + c * dy
                corners.append((wx + rx, wy + ry))
        return corners

    def object_fully_in_frame(self, wx, wy, half_l, half_w, yaw=0.0, margin=20):
        for cx, cy in self._rotated_corners(wx, wy, half_l, half_w, yaw):
            pix = self.project_world(cx, cy, TABLE_Z)
            if pix is None:
                return False
            u, v = pix
            if u < margin or u > self.img_w - margin or v < margin or v > self.img_h - margin:
                return False
        return True

    def object_partially_in_frame(self, wx, wy, half_l, half_w, yaw=0.0):
        in_count = 0
        for cx, cy in self._rotated_corners(wx, wy, half_l, half_w, yaw):
            pix = self.project_world(cx, cy, TABLE_Z)
            if pix is not None:
                u, v = pix
                if 0 <= u <= self.img_w and 0 <= v <= self.img_h:
                    in_count += 1
        return 0 < in_count < 4


def set_pose(name, x, y, z, yaw=0.0):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (f"name: '{name}', position: {{x: {x}, y: {y}, z: {z}}}, "
           f"orientation: {{z: {qz}, w: {qw}}}")
    result = subprocess.run(
        ["ign", "service", "-s", "/world/so_arm101_lego_world/set_pose",
         "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
         "--timeout", "3000", "-r", req],
        capture_output=True, text=True
    )
    return "true" in result.stdout


def random_positions_in_view(cam, names, bounds):
    positions = []
    for name in names:
        hl, hw = HALF_SIZES[name]
        for _attempt in range(500):
            x = random.uniform(*bounds["x"])
            y = random.uniform(*bounds["y"])
            yaw = random.uniform(0, 2 * math.pi)
            if not cam.object_fully_in_frame(x, y, hl, hw, yaw):
                continue
            if all(((x - px)**2 + (y - py)**2)**0.5 > MIN_SPACING
                   for px, py, _yaw in positions):
                positions.append((x, y, yaw))
                break
        else:
            print(f"  ERROR: could not place {name} in frame after 500 attempts")
            return None
    return positions


def random_positions_edge(cam, names, bounds):
    positions = []
    need_edge = random.randint(0, len(names) - 1)
    for i, name in enumerate(names):
        hl, hw = HALF_SIZES[name]
        want_edge = (i == need_edge)
        for _attempt in range(1000):
            x = random.uniform(*bounds["x"])
            y = random.uniform(*bounds["y"])
            yaw = random.uniform(0, 2 * math.pi)
            if want_edge:
                if not cam.object_partially_in_frame(x, y, hl, hw, yaw):
                    continue
            else:
                if not cam.object_fully_in_frame(x, y, hl, hw, yaw):
                    continue
            if all(((x - px)**2 + (y - py)**2)**0.5 > MIN_SPACING
                   for px, py, _yaw in positions):
                positions.append((x, y, yaw))
                break
        else:
            print(f"  ERROR: could not place {name} in frame after 1000 attempts")
            return None
    return positions


def main():
    parser = argparse.ArgumentParser(description="Randomize lego positions in Gazebo")
    parser.add_argument("--edge", action="store_true",
                        help="At least one object partially outside camera frame")
    parser.add_argument("--reset", action="store_true",
                        help="Reset to default SDF positions")
    args = parser.parse_args()

    if args.reset:
        print("Resetting legos to default positions...")
        for name, (x, y, z, yaw) in DEFAULTS.items():
            ok = set_pose(name, x, y, z, yaw)
            print(f"  {name}: ({x:.3f}, {y:.3f}, {z:.4f}, yaw={yaw:.2f}) {'OK' if ok else 'FAIL'}")
        return

    cam = CameraHelper()
    print("Reading camera data from ROS2...")
    if not cam.wait_for_data():
        print("ERROR: Could not read camera data. Is the sim running?")
        cam.shutdown()
        sys.exit(1)

    if not cam.check_camera_looking_down():
        print("ERROR: Camera is not looking downward. "
              "Move arm to grasp home (wrist_flex=pi/2) first.")
        cam.shutdown()
        sys.exit(1)

    print(f"  Camera pos: ({cam.cam_pos[0]:.4f}, {cam.cam_pos[1]:.4f}, {cam.cam_pos[2]:.4f})")
    print(f"  K: fx={cam.K[0,0]:.1f} fy={cam.K[1,1]:.1f} [{cam.img_w}x{cam.img_h}]")

    bounds = cam.compute_search_bounds()
    if bounds is None:
        print("ERROR: Camera cannot see the ground plane. Aim the camera downward.")
        cam.shutdown()
        sys.exit(1)
    print(f"  Search bounds (world): x=({bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}) "
          f"y=({bounds['y'][0]:.3f}, {bounds['y'][1]:.3f})")

    # Sanity check with known default red lego
    test = cam.project_world(0.18, 0.03, TABLE_Z)
    if test:
        print(f"  Projection sanity (red default): px=({test[0]:.1f}, {test[1]:.1f})")

    names = list(DEFAULTS.keys())
    if args.edge:
        positions = random_positions_edge(cam, names, bounds)
    else:
        positions = random_positions_in_view(cam, names, bounds)

    if positions is None:
        print("ERROR: Could not place all objects in camera frame. "
              "Try a different camera angle.")
        cam.shutdown()
        sys.exit(1)

    mode = "with edge case" if args.edge else "all fully in frame"
    print(f"\nRandomizing legos ({mode})...")
    for name, (x, y, yaw) in zip(names, positions):
        z = DEFAULTS[name][2]
        ok = set_pose(name, x, y, z, yaw)
        hl, hw = HALF_SIZES[name]
        full = cam.object_fully_in_frame(x, y, hl, hw, yaw)
        partial = cam.object_partially_in_frame(x, y, hl, hw, yaw)
        label = "FULL" if full else ("EDGE" if partial else "OUT")
        pix = cam.project_world(x, y, TABLE_Z)
        px_str = f"px=({pix[0]:.0f},{pix[1]:.0f})" if pix else "px=N/A"
        print(f"  {name}: world=({x:.3f}, {y:.3f}, yaw={yaw:.2f}) {px_str} {'OK' if ok else 'FAIL'} [{label}]")

    cam.shutdown()


if __name__ == "__main__":
    main()
