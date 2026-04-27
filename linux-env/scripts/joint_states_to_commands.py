#!/usr/bin/env python3
"""Mirror /joint_states → /joint_commands_lerobot at the source rate.

When recording pick-and-place demos in sim, the commanded trajectory is
delivered as JointTrajectory on /arm_controller/joint_trajectory and the
controller manager interpolates it; the resulting follower-side positions
appear on /joint_states. There is no separate "leader" topic in sim
because the planner *is* the leader.

For lerobot's so101_ros2 Teleoperator (which expects a sensor_msgs/JointState
as the action column source), this mirroring approximates
action ≈ achieved-state, which is what the planner intended at each tick.

CRITICAL: Publish to /joint_commands_lerobot, NOT /joint_commands.
control_gui's _ext_cmd_callback subscribes to /joint_commands and treats
each message as an external teleop command — it dispatches a
FollowJointTrajectory action goal per message. Mirroring at /joint_states
rate produces ~60 goals/sec flood that saturates control_gui's executor
and starves all qs_* service callbacks (verified 2026-04-27 by toggling the
mirror — services blocked at 30 s timeout, returned in 10 ms once mirror
killed). Recorder-only topic decouples lerobot's action source from the
"external teleop" semantic and breaks the feedback loop.

Diagnosis is in this repo's PLAN/HANDOFF artifacts; fix recommendation came
from GPT review (codex/gpt-5.4) of the diagnosis writeup.

Usage:
  python3 joint_states_to_commands.py
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Mirror(Node):
    def __init__(self):
        super().__init__("joint_states_to_commands_mirror")
        # Publish to a recorder-only topic. /joint_commands is reserved for
        # control_gui's external-teleop semantic — see module docstring.
        self.pub = self.create_publisher(JointState, "/joint_commands_lerobot", 10)
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._cb, 10
        )
        self.get_logger().info(
            "mirroring /joint_states → /joint_commands_lerobot"
        )

    def _cb(self, msg: JointState):
        # Re-emit verbatim. Header timestamps stay; downstream sees a
        # synthetic "leader" stream that perfectly tracks the follower.
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Mirror()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
