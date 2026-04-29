#!/usr/bin/env python3
"""Publish controller-reference (commanded) joint positions to
/joint_commands_lerobot at 30 Hz, so lerobot's so101_ros2 Teleoperator
sees an `action` column that genuinely leads `observation.state`.

Background
----------
The previous version of this script mirrored /joint_states verbatim →
/joint_commands_lerobot, which made action == state in the saved
dataset. Real-world teleop datasets have action LEADING state by the
controller's tracking lag (~30-100 ms): the leader-arm command is
recorded at sample time, then the follower physically catches up.

Behaviour-cloning policies trained on action == state learn the
identity function and don't drive the arm at deploy time.

Source of truth
---------------
ros2_controllers' JointTrajectoryController publishes
control_msgs/JointTrajectoryControllerState on
  /arm_controller/controller_state    (5 arm joints)
  /gripper_controller/controller_state (1 gripper joint)

The `reference.positions` field is the per-tick interpolated setpoint
the JTC drives the hardware/sim toward. That is exactly what an action
column should contain: what the planner commanded, BEFORE physics
tracked.

Output
------
sensor_msgs/JointState on /joint_commands_lerobot at 30 Hz, with
joint_names in the lerobot-canonical order:

  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

(The trailing 'gripper' here matches the dataset feature key
`gripper.pos`; the JTC controlling it is `gripper_controller` and the
URDF joint inside it is `gripper_joint`. We remap on output.)

The 30 Hz rate matches the dataset.fps lerobot is configured to.
Below that the latest reference is held; the mirror is decoupled from
the controller's publish frequency so action sampling stays uniform
even when controller_state arrives at irregular intervals.

Why /joint_commands_lerobot and NOT /joint_commands
---------------------------------------------------
control_gui's _ext_cmd_callback subscribes to /joint_commands and treats
each message as an external teleop command (dispatches a new
FollowJointTrajectory action goal per message). Publishing at 30 Hz on
that topic would saturate control_gui's executor and starve qs_*
service callbacks (regression observed 2026-04-27). Recorder-only
topic decouples lerobot's action source from the "external teleop"
semantic and breaks the feedback loop.

Usage
-----
  python3 joint_states_to_commands.py [--rate HZ]

Defaults to 30 Hz to match dataset.fps.
"""
from __future__ import annotations

import argparse
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState


# Output joint order MUST match dataset features. lerobot v3.0 datasets
# for so_follower use these joint names in order:
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_OUT_NAME = "gripper"


class CommandedActionPublisher(Node):
    def __init__(self, rate_hz: float):
        super().__init__("joint_commanded_to_actions")

        self._lock = threading.Lock()
        # Latest reference positions, keyed by output joint name. Once
        # both subscribers have populated their joints we publish.
        self._latest_ref: dict[str, float] = {}

        self.create_subscription(
            JointTrajectoryControllerState,
            "/arm_controller/controller_state",
            self._on_arm_state,
            10,
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/gripper_controller/controller_state",
            self._on_gripper_state,
            10,
        )

        self.pub = self.create_publisher(
            JointState, "/joint_commands_lerobot", 10
        )

        period_s = 1.0 / float(rate_hz)
        self.create_timer(period_s, self._publish)

        self.get_logger().info(
            f"publishing controller reference → /joint_commands_lerobot @ "
            f"{rate_hz:.1f} Hz "
            f"(joints={ARM_JOINT_NAMES + [GRIPPER_OUT_NAME]})"
        )

    def _store_state(
        self,
        msg: JointTrajectoryControllerState,
        joint_remap: dict[str, str] | None = None,
    ) -> None:
        # `reference` carries the per-tick interpolated commanded
        # position. Both /arm_controller and /gripper_controller publish
        # this in JointTrajectoryControllerState.
        if not msg.joint_names or not msg.reference.positions:
            return
        n = min(len(msg.joint_names), len(msg.reference.positions))
        with self._lock:
            for i in range(n):
                src = msg.joint_names[i]
                dst = (joint_remap or {}).get(src, src)
                self._latest_ref[dst] = float(msg.reference.positions[i])

    def _on_arm_state(self, msg: JointTrajectoryControllerState) -> None:
        self._store_state(msg)

    def _on_gripper_state(self, msg: JointTrajectoryControllerState) -> None:
        # gripper_controller's URDF joint is `gripper_joint`; dataset
        # feature key is `gripper.pos`. Remap on store so the output
        # JointState carries the lerobot-canonical name `gripper`.
        self._store_state(msg, joint_remap={"gripper_joint": GRIPPER_OUT_NAME})

    def _publish(self) -> None:
        with self._lock:
            ref = dict(self._latest_ref)

        # Wait until at least one of each controller has reported. This
        # avoids publishing partial JointStates that confuse lerobot's
        # action stale-detection.
        out_names = ARM_JOINT_NAMES + [GRIPPER_OUT_NAME]
        if not all(n in ref for n in out_names):
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = out_names
        msg.position = [ref[n] for n in out_names]
        # velocities/effort intentionally empty — lerobot's so101_ros2
        # teleop only reads .position from this stream.
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0,
                    help="Publish rate in Hz (default 30 to match dataset.fps)")
    args = ap.parse_args()

    rclpy.init()
    node = CommandedActionPublisher(rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
