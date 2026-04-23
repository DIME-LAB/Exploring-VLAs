#!/usr/bin/env python
"""drive_base_yaw_sweep.py — programmatic stand-in for an interactive
control_gui session, paired with `record_sim.sh`.

Continuously sends a two-channel motion to the SO-ARM101:
  1. `/arm_controller/joint_trajectory`  (5-joint JointTrajectory)
       shoulder_pan sweeps ±π/2 as a slow sine; other arm joints 0.
  2. `/gripper_controller/joint_trajectory`  (single-joint)
       gripper_joint alternates open/closed on its own period.
  3. `/joint_commands`  (sensor_msgs/JointState, all 6 joints)
       parallel publish at 30 Hz so the so101_ros2 Teleoperator plugin
       captures these values into the dataset's `action` column.

Because (1)+(2) are what actually actuates the arm in Gazebo, the
resulting `observation.state` tracks (1)+(2) through the controllers
— the dataset row-to-row shows real camera POV changes and joint
motion.

Exits on SIGINT / SIGTERM.
"""

from __future__ import annotations

import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINTS = ["gripper_joint"]

ARM_CONTROLLER = "/arm_controller/joint_trajectory"
GRIPPER_CONTROLLER = "/gripper_controller/joint_trajectory"
JOINT_COMMANDS = "/joint_commands"

# Motion parameters
YAW_AMPLITUDE = math.pi / 2      # ±90°
YAW_PERIOD_S = 10.0              # full sweep period
GRIPPER_CLOSED = 0.0             # rest
GRIPPER_OPEN = 1.4               # near upper limit (1.745)
GRIPPER_PERIOD_S = 6.0           # open/close period
PUBLISH_HZ = 30.0                # rate for /joint_commands


def _shoulder_pan_at(t: float) -> float:
    return YAW_AMPLITUDE * math.sin(2 * math.pi * t / YAW_PERIOD_S)


def _gripper_at(t: float) -> float:
    # Smooth open/close via cosine — avoids sharp controller steps.
    phase = (1 - math.cos(2 * math.pi * t / GRIPPER_PERIOD_S)) / 2  # 0..1
    return GRIPPER_CLOSED + phase * (GRIPPER_OPEN - GRIPPER_CLOSED)


class Driver(Node):
    def __init__(self) -> None:
        super().__init__("drive_base_yaw_sweep")
        self.pub_arm = self.create_publisher(JointTrajectory, ARM_CONTROLLER, 10)
        self.pub_grip = self.create_publisher(JointTrajectory, GRIPPER_CONTROLLER, 10)
        self.pub_cmd = self.create_publisher(JointState, JOINT_COMMANDS, 10)

        # Seed the arm at home once, then let sine drive it.
        self._send_arm_goal(shoulder_pan=0.0, hold_s=1.0)
        self._send_gripper_goal(GRIPPER_CLOSED, hold_s=1.0)

        self.t0 = time.monotonic()
        self.last_arm_pub = 0.0
        self.last_grip_pub = 0.0
        # send new trajectory goals at this rate (sparser than /joint_commands
        # because each JointTrajectory carries a time-parameterised segment)
        self.arm_goal_period_s = 0.5
        self.grip_goal_period_s = 1.0

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self._count = 0

    def _send_arm_goal(self, shoulder_pan: float, hold_s: float = 0.5) -> None:
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        p = JointTrajectoryPoint()
        p.positions = [shoulder_pan, 0.0, 0.0, 0.0, 0.0]
        p.time_from_start.sec = int(hold_s)
        p.time_from_start.nanosec = int((hold_s - int(hold_s)) * 1e9)
        msg.points = [p]
        self.pub_arm.publish(msg)

    def _send_gripper_goal(self, position: float, hold_s: float = 0.5) -> None:
        msg = JointTrajectory()
        msg.joint_names = GRIPPER_JOINTS
        p = JointTrajectoryPoint()
        p.positions = [position]
        p.time_from_start.sec = int(hold_s)
        p.time_from_start.nanosec = int((hold_s - int(hold_s)) * 1e9)
        msg.points = [p]
        self.pub_grip.publish(msg)

    def _tick(self) -> None:
        now = time.monotonic()
        t = now - self.t0

        sp = _shoulder_pan_at(t)
        gp = _gripper_at(t)

        # Always publish /joint_commands at 30 Hz — this is what the teleop
        # captures into the dataset's `action` column.
        cmd = JointState()
        cmd.name = [*ARM_JOINTS, "gripper_joint"]
        cmd.position = [sp, 0.0, 0.0, 0.0, 0.0, gp]
        self.pub_cmd.publish(cmd)

        # Send new JointTrajectory goals less frequently — each carries a
        # time-parameterised segment so the controller interpolates between
        # updates.
        if now - self.last_arm_pub >= self.arm_goal_period_s:
            # 0.6s "horizon" — slightly longer than our update period so the
            # trajectory is always ahead of the current time.
            self._send_arm_goal(shoulder_pan=sp, hold_s=0.6)
            self.last_arm_pub = now

        if now - self.last_grip_pub >= self.grip_goal_period_s:
            self._send_gripper_goal(gp, hold_s=1.2)
            self.last_grip_pub = now

        self._count += 1
        if self._count % 150 == 0:  # every ~5 s
            print(
                f"t={t:5.1f}s  shoulder_pan={math.degrees(sp):+7.1f}°  "
                f"gripper={math.degrees(gp):+6.1f}°  count={self._count}",
                flush=True,
            )


def main() -> int:
    rclpy.init()
    node = Driver()

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(
        "drive_base_yaw_sweep: publishing to\n"
        f"  {ARM_CONTROLLER}\n"
        f"  {GRIPPER_CONTROLLER}\n"
        f"  {JOINT_COMMANDS}\n"
        "Ctrl-C to stop.",
        flush=True,
    )

    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
