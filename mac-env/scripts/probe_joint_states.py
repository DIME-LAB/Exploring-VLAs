#!/usr/bin/env python3
"""Quick /joint_states probe: capture 4 seconds, print first/middle/last
sample in degrees per arm joint. Useful for verifying jsr is publishing
all 6 joints, sanity-checking calibration immediately after stack
restart, or watching settle behavior post-motion.

Usage (stack must be running):
  pixi run --manifest-path /tmp/mac-env/pixi.toml python3 \\
      mac-env/scripts/probe_joint_states.py
"""
import rclpy, math, time
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()
n = Node("probe3")
got = []
def cb(msg):
    got.append({nm: p for nm, p in zip(msg.name, msg.position)})
n.create_subscription(JointState, "/joint_states", cb, 10)
t0 = time.time()
while time.time() - t0 < 4:
    rclpy.spin_once(n, timeout_sec=0.2)
print(f"Captured {len(got)} samples over 4s")
for label, idx in [("first", 0), ("middle", len(got)//2), ("last", -1)]:
    s = got[idx]
    e = math.degrees(s.get('elbow_flex', 0))
    wf = math.degrees(s.get('wrist_flex', 0))
    wr = math.degrees(s.get('wrist_roll', 0))
    print(f"  {label:>6}: elbow_flex={e:+8.2f}  wrist_flex={wf:+8.2f}  wrist_roll={wr:+8.2f}")
rclpy.shutdown()
