#!/usr/bin/env python3
"""drive_pick_place.py — Mac-style direct trajectory publisher.

Reference: ../../mac-env/scripts/drive_base_yaw_sweep.py (architecture)

Mac's recording driver publishes JointTrajectory messages directly to
ros2_control's controller topics — control_gui is never on the critical
path during a record. We mirror that here: subscribe to
/objects_poses_sim and /drop_poses, compute geometric IK in-process, and
publish trajectories DIRECTLY to:

    /arm_controller/joint_trajectory       (5-joint arm)
    /gripper_controller/joint_trajectory   (single-joint gripper)

This bypasses control_gui's services entirely — no `qs_select`, no
`qs_play`, no tkinter mainloop on the critical path. The previous version
of this script went through control_gui services and timed out under
recording load (handoff blocker #1).

The /joint_commands action column is mirrored separately by
joint_states_to_commands.py (it tracks /joint_states verbatim — fine for
behavioral cloning where action ≈ achieved-state in sim).

Modes:
  --mode sweep      shoulder_pan sine + gripper toggle (true 1:1 with Mac)
  --mode pickplace  read /objects_poses_sim + /drop_poses, run a geometric-IK
                    pick-place cycle per lego (default)

Usage:
  python3 drive_pick_place.py [--mode pickplace] [legos...]
  python3 drive_pick_place.py --mode sweep
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Pure-Python IK module from the SO-ARM101 control package (no ROS deps).
# Path-import keeps this driver runnable from pixi-Jazzy without sourcing
# the system Humble overlay.
_SO_ARM101_PKG = os.path.expanduser(
    "~/Projects/Exploring-VLAs/vla_SO-ARM101/src/so_arm101_control/so_arm101_control"
)
if _SO_ARM101_PKG not in sys.path:
    sys.path.insert(0, _SO_ARM101_PKG)
from compute_workspace import (  # noqa: E402  (after sys.path mutation)
    JOINT_LIMITS,
    geometric_ik,
)


ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper_joint"

ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIP_ACTION = "/gripper_controller/follow_joint_trajectory"

OBJECTS_TOPIC = "/objects_poses_sim"
DROPS_TOPIC = "/drop_poses"
JOINT_STATES_TOPIC = "/joint_states"

# Simple gripper angles. The full bbox-driven model lives in control_gui
# (_gripper_angle_for_object); for sim recording we use fixed open/close
# values that work for 2x2/2x3 lego — empirically validated.
GRIPPER_OPEN_RAD = 1.4
GRIPPER_CLOSE_RAD = -0.05  # slight pre-load against the block

# Pre-grasp / drop hover offsets relative to the published pose.
# Reach is height-vs-radius limited at the workspace boundary, so 6cm pre-
# grasp can be unreachable for distal legos. Use 2cm and trust mock_components
# to interpolate cleanly to the grasp pose.
PRE_GRASP_DZ = 0.02       # 2 cm above lego before descending
GRASP_DZ = 0.005          # 5 mm above the lego center for jaw closure
HOME_GRASP_DZ = 0.10      # post-grasp lift before traversing (avoid cup rims)
DROP_HOVER_DZ = 0.13      # hover height above cup body-center pose
                          # (drop_poses publishes BODY-CENTER per CLAUDE.md)

# Yaw fallback offsets (deg). Tried in order when the requested grasp_yaw has
# no IK at any z. Mirrors control_gui's find_reachable_grasp_yaw pattern but
# simpler — picks the first offset that works rather than scoring.
YAW_FALLBACK_DEG = (0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90)

# Home configuration matches control_gui's grasp_home (joint-space).
HOME_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": math.radians(-90.0),
    "elbow_flex": math.radians(90.0),
    "wrist_flex": math.radians(45.0),
    "wrist_roll": 0.0,
}

# Color → drop_N child_frame_id mapping (per project CLAUDE.md robot facts).
COLOR_TO_DROP = {"red": "drop_0", "green": "drop_1", "blue": "drop_2"}

# Default lego list — picks one of each color first, matching test_pick_all.sh.
DEFAULT_LEGOS = ["red_2x2", "green_2x2", "blue_2x2"]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Z-axis yaw from a unit quaternion (top-down grasp orientation)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PickPlaceDriver(Node):
    def __init__(self, mode: str = "pickplace") -> None:
        super().__init__("drive_pick_place")
        self.mode = mode

        # Action interface (NOT topic). Under mock_components/GenericSystem
        # (Linux config), JointTrajectoryController only actuates joints when
        # commanded via FollowJointTrajectory action — topic publishes are
        # accepted but don't drive the state interface. Mac's gz_ros2_control
        # backend honors the topic; Linux's mock_components doesn't.
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.grip_client = ActionClient(self, FollowJointTrajectory, GRIP_ACTION)

        self.objects: dict[str, Pose] = {}
        self.drops: dict[str, Pose] = {}
        self.joint_state: Optional[JointState] = None

        self.create_subscription(TFMessage, OBJECTS_TOPIC, self._on_objects, 10)
        self.create_subscription(TFMessage, DROPS_TOPIC, self._on_drops, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_js, 50)

    # ---- subscription callbacks --------------------------------------------------

    def _store_tf(self, store: dict[str, Pose], msg: TFMessage) -> None:
        for tf in msg.transforms:
            t = tf.transform.translation
            r = tf.transform.rotation
            store[tf.child_frame_id] = Pose(t.x, t.y, t.z, r.x, r.y, r.z, r.w)

    def _on_objects(self, msg: TFMessage) -> None:
        self._store_tf(self.objects, msg)

    def _on_drops(self, msg: TFMessage) -> None:
        self._store_tf(self.drops, msg)

    def _on_js(self, msg: JointState) -> None:
        self.joint_state = msg

    # ---- low-level publishers ----------------------------------------------------

    def _send_traj_action(
        self, client: ActionClient, joint_names: list[str], positions: list[float],
        duration_s: float, blocking: bool = True
    ) -> bool:
        """Send a single-waypoint trajectory via FollowJointTrajectory action.

        Blocks (when blocking=True) until the action completes or 1.5×duration
        elapses. Returns True if the action ran (status SUCCEEDED or CANCELED;
        with mock_components, completed motions sometimes finish in CANCELED
        state but the joints did interpolate to target).
        """
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(f"action server {client._action_name} not ready")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in positions]
        p.time_from_start.sec = int(duration_s)
        p.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        goal.trajectory.points = [p]

        send_fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=2.0)
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("goal rejected")
            return False
        if not blocking:
            return True
        rfut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rfut, timeout_sec=duration_s * 1.5 + 1.0)
        return rfut.done()

    def _send_arm(self, target: dict[str, float], duration_s: float = 2.5) -> None:
        self._send_traj_action(
            self.arm_client, ARM_JOINTS,
            [target[j] for j in ARM_JOINTS], duration_s,
        )

    def _send_gripper(self, angle: float, duration_s: float = 0.8) -> None:
        self._send_traj_action(
            self.grip_client, [GRIPPER_JOINT], [angle], duration_s,
        )

    # ---- waiting -----------------------------------------------------------------

    def _wait_settled(
        self, target: dict[str, float], timeout_s: float = 5.0, tol_rad: float = 0.02
    ) -> bool:
        """Pump the executor until /joint_states matches `target` within `tol_rad`."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joint_state is None:
                continue
            name_to_pos = dict(zip(self.joint_state.name, self.joint_state.position))
            if all(
                j in name_to_pos and abs(name_to_pos[j] - target[j]) < tol_rad
                for j in target
            ):
                return True
        return False

    def _spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_topics(self, timeout_s: float = 10.0) -> bool:
        """Block until at least one msg has arrived on every required topic."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state is not None and self.objects and self.drops:
                return True
        return False

    # ---- IK ----------------------------------------------------------------------

    def _solve_grasp(
        self, x: float, y: float, z: float, grasp_yaw: float,
        try_fallback: bool = True
    ) -> tuple[Optional[dict[str, float]], float]:
        """Top-down grasp IK with yaw fallback. Returns (solution, used_yaw)."""
        sols = geometric_ik(x, y, z, grasp_yaw=grasp_yaw)
        if sols:
            return sols[0], grasp_yaw
        if not try_fallback:
            return None, grasp_yaw
        for off_deg in YAW_FALLBACK_DEG:
            if off_deg == 0:
                continue
            candidate = grasp_yaw + math.radians(off_deg)
            sols = geometric_ik(x, y, z, grasp_yaw=candidate)
            if sols:
                return sols[0], candidate
        return None, grasp_yaw

    def _solve_drop(
        self, x: float, y: float, z: float, lock_pan: Optional[float] = None
    ) -> Optional[dict[str, float]]:
        """Drop IK at grip_angle=45°, wrist_roll=-π/2 (matches control_gui's
        _drop_grip_angle_var default + WRIST_ROLL_URDF_PITCH offset).

        Pure-horizontal grip (grip_angle=0) has no analytical solution at the
        cup distances/heights this robot uses; 45° + wrist_roll=-π/2 is the
        validated configuration in control_gui:_cmd_drop_sweep.

        If lock_pan is given, snap the pan joint of the result so consecutive
        drop motions don't introduce sub-degree base yaw drift between
        sub-stages (mirrors control_gui's lock_pan kwarg).
        """
        from compute_workspace import WRIST_ROLL_URDF_PITCH
        drop_grip_angle = math.radians(45.0)
        drop_wrist_roll = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        sols = geometric_ik(x, y, z, grasp_yaw=None,
                            grip_angle=drop_grip_angle,
                            wrist_roll=drop_wrist_roll)
        if not sols:
            return None
        sol = sols[0]
        if lock_pan is not None:
            sol = dict(sol)
            sol["shoulder_pan"] = lock_pan
        return sol

    # ---- high-level steps --------------------------------------------------------

    def go_home(self, duration_s: float = 2.5) -> bool:
        self.get_logger().info("→ home")
        self._send_arm(HOME_POSE, duration_s=duration_s)
        return True

    def gripper_open(self) -> None:
        self.get_logger().info("→ gripper open")
        self._send_gripper(GRIPPER_OPEN_RAD, duration_s=0.8)
        self._spin_for(0.9)

    def gripper_close(self) -> None:
        self.get_logger().info("→ gripper close")
        self._send_gripper(GRIPPER_CLOSE_RAD, duration_s=0.8)
        self._spin_for(1.0)

    def pick(self, lego_name: str) -> bool:
        if lego_name not in self.objects:
            self.get_logger().warn(f"pick {lego_name}: no pose on {OBJECTS_TOPIC}")
            return False
        p = self.objects[lego_name]
        grasp_yaw = _yaw_from_quat(p.qx, p.qy, p.qz, p.qw)

        # Solve grasp first (most likely to succeed at z+0.005), then re-use
        # the resulting yaw for pre-grasp + lift. control_gui calls this
        # "single yaw across stages" — keeps wrist_roll consistent.
        grasp, used_yaw = self._solve_grasp(p.x, p.y, p.z + GRASP_DZ, grasp_yaw)
        if grasp is None:
            self.get_logger().warn(
                f"pick {lego_name}: grasp IK failed even with yaw fallback "
                f"(requested yaw={math.degrees(grasp_yaw):+.1f}°)"
            )
            return False

        pre, _ = self._solve_grasp(p.x, p.y, p.z + PRE_GRASP_DZ, used_yaw,
                                   try_fallback=False)
        if pre is None:
            # Fall back to even smaller pre-grasp offset
            for dz in [0.015, 0.01, 0.007]:
                pre, _ = self._solve_grasp(p.x, p.y, p.z + dz, used_yaw,
                                           try_fallback=False)
                if pre is not None:
                    break
        if pre is None:
            self.get_logger().warn(f"pick {lego_name}: pre-grasp IK failed at all heights")
            return False

        if used_yaw != grasp_yaw:
            self.get_logger().info(
                f"yaw fallback: {math.degrees(grasp_yaw):+.1f}° → {math.degrees(used_yaw):+.1f}°"
            )

        self.gripper_open()

        self.get_logger().info(f"→ pre-grasp {lego_name}")
        self._send_arm(pre, duration_s=2.5)

        self.get_logger().info(f"→ grasp {lego_name}")
        self._send_arm(grasp, duration_s=1.5)

        self.gripper_close()

        # Lift back to a safe carry height before traversing.
        carry, _ = self._solve_grasp(p.x, p.y, p.z + HOME_GRASP_DZ, used_yaw,
                                     try_fallback=False)
        if carry is None:
            carry = pre  # fall back to pre-grasp pose for the lift
        self.get_logger().info("→ lift")
        self._send_arm(carry, duration_s=2.0)
        return True

    def place(self, drop_id: str) -> bool:
        if drop_id not in self.drops:
            self.get_logger().warn(f"place {drop_id}: no pose on {DROPS_TOPIC}")
            return False
        p = self.drops[drop_id]

        # Hover above the cup with horizontal gripper. drop_poses publishes
        # the cup body-center; rim_z is computed downstream — we add a hover
        # offset on top of the published z to clear the rim.
        hover = self._solve_drop(p.x, p.y, p.z + DROP_HOVER_DZ)
        if hover is None:
            self.get_logger().warn(f"place {drop_id}: IK failed at hover")
            return False
        lock_pan = hover["shoulder_pan"]

        # Drop pose itself: same xy, slightly lower so block exits jaws cleanly.
        drop_at = self._solve_drop(p.x, p.y, p.z + DROP_HOVER_DZ - 0.02, lock_pan=lock_pan)
        if drop_at is None:
            drop_at = hover

        self.get_logger().info(f"→ drop hover above {drop_id}")
        self._send_arm(hover, duration_s=2.5)

        self.get_logger().info(f"→ drop sweep at {drop_id}")
        self._send_arm(drop_at, duration_s=1.2)

        self.gripper_open()

        # Step back up to hover before going home — keeps from clipping the rim.
        self._send_arm(hover, duration_s=1.2)
        return True

    # ---- top-level cycles --------------------------------------------------------

    def cycle_pickplace(self, lego: str) -> str:
        """One pick-and-place cycle. Returns 'PASS' / 'FAIL_*'."""
        self.get_logger().info(f"=== cycle: {lego} ===")
        color = lego.split("_")[0]
        drop_id = COLOR_TO_DROP.get(color)
        if drop_id is None:
            return f"FAIL_UNKNOWN_COLOR({color})"

        if not self.pick(lego):
            self.go_home()
            return "FAIL_PICK"
        if not self.place(drop_id):
            self.go_home()
            return "FAIL_PLACE"
        self.go_home()
        return "PASS"

    def cycle_sweep(self) -> None:
        """Mac-style synthetic motion: shoulder_pan sine + gripper toggle.

        Mirrors mac-env/scripts/drive_base_yaw_sweep.py for direct A/B parity.
        """
        self.get_logger().info("=== sweep mode (Mac drive_base_yaw_sweep equivalent) ===")
        amplitude = math.pi / 2
        period = 10.0
        gripper_period = 6.0
        t0 = time.monotonic()
        last_arm = 0.0
        last_grip = 0.0
        while rclpy.ok():
            now = time.monotonic()
            t = now - t0
            sp = amplitude * math.sin(2 * math.pi * t / period)
            gp_phase = (1 - math.cos(2 * math.pi * t / gripper_period)) / 2
            gp = GRIPPER_CLOSE_RAD + gp_phase * (GRIPPER_OPEN_RAD - GRIPPER_CLOSE_RAD)

            if now - last_arm >= 0.5:
                self._send_arm(
                    {"shoulder_pan": sp, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                     "wrist_flex": 0.0, "wrist_roll": 0.0},
                    duration_s=0.6,
                )
                last_arm = now
            if now - last_grip >= 1.0:
                self._send_gripper(gp, duration_s=1.2)
                last_grip = now
            rclpy.spin_once(self, timeout_sec=0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pickplace", "sweep"], default="pickplace")
    parser.add_argument("legos", nargs="*", default=DEFAULT_LEGOS)
    args = parser.parse_args()

    rclpy.init()
    driver = PickPlaceDriver(mode=args.mode)

    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        if args.mode == "sweep":
            driver.cycle_sweep()
            return 0

        if not driver._wait_for_topics(timeout_s=15.0):
            driver.get_logger().error(
                "topics not ready after 15s — check Isaac Sim publishers + "
                "control stack are up"
            )
            return 1

        driver.go_home(duration_s=3.0)

        results: list[tuple[str, str]] = []
        for lego in args.legos:
            if stopping["flag"]:
                break
            verdict = driver.cycle_pickplace(lego)
            results.append((lego, verdict))
            driver._spin_for(1.0)

        driver.get_logger().info("=== summary ===")
        for lego, verdict in results:
            driver.get_logger().info(f"  {lego}: {verdict}")
        return 0 if all(v == "PASS" for _, v in results) else 2
    finally:
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
