#!/usr/bin/env python
"""Drive /joint_commands at 30 Hz with a slow shoulder_pan sine sweep.

Stands in for an interactive control_gui session during a smoke record.
Exits on SIGINT. Joint names match the HF SO-101 canonical set; the
so101_ros2 teleop remaps gripper_joint->gripper at read time if present.
"""
import math
import signal
import sys
import time

import rclpy
from sensor_msgs.msg import JointState


def main():
    rclpy.init()
    node = rclpy.create_node("smoke_joint_commands_pub")
    pub = node.create_publisher(JointState, "/joint_commands", 10)

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Names the control_gui publishes (the teleop remaps gripper_joint -> gripper).
    names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper_joint",
    ]

    # Sweep shoulder_pan between -0.4 and +0.4 rad over 6s.
    t0 = time.monotonic()
    dt = 1.0 / 30.0
    period = 6.0
    print("publishing /joint_commands @ 30 Hz (sine shoulder_pan); Ctrl-C to stop", flush=True)
    count = 0
    while not stopping:
        elapsed = time.monotonic() - t0
        msg = JointState()
        msg.name = names
        msg.position = [
            0.4 * math.sin(2 * math.pi * elapsed / period),  # shoulder_pan
            0.0,  # shoulder_lift
            0.0,  # elbow_flex
            0.0,  # wrist_flex
            0.0,  # wrist_roll
            0.0,  # gripper_joint
        ]
        pub.publish(msg)
        count += 1
        if count % 60 == 0:
            print(f"  published {count} msgs, t={elapsed:.1f}s, shoulder_pan={msg.position[0]:+.3f}", flush=True)
        time.sleep(dt)

    node.destroy_node()
    rclpy.shutdown()
    print(f"stopped after {count} messages")


if __name__ == "__main__":
    main()
