#!/usr/bin/env python3
"""Passive observer: subscribe to /joint_states for N seconds, compute
per-joint vibration statistics. Does NOT touch the Feetech bus, does NOT
kill jsr. Just reads the existing topic stream.

Per joint over the sample window:
  mean      — mean position (rad → deg)
  stdev     — vibration amplitude (encoder noise floor ~0.05°)
  peak_to_peak — max - min (visible oscillation if > ~0.3°)
  range     — peak_to_peak in degrees

Vibration verdict:
  std < 0.10°  → quiet
  std 0.10-0.30° → small ringing
  std > 0.30°  → significant vibration
"""
import argparse
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Observer(Node):
    def __init__(self):
        super().__init__('vib_observer')
        self.samples = []
        self.create_subscription(JointState, '/joint_states', self._cb, 50)

    def _cb(self, msg):
        snap = {n: msg.position[i] for i, n in enumerate(msg.name)}
        self.samples.append((time.time(), snap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    obs = Observer()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(obs, timeout_sec=0.05)
    obs.destroy_node()
    rclpy.shutdown()

    if not obs.samples:
        print("FAIL: no /joint_states received in", args.seconds, "s")
        return

    n = len(obs.samples)
    duration = obs.samples[-1][0] - obs.samples[0][0]
    rate = n / duration if duration > 0 else 0
    print(f"# {n} samples over {duration:.2f}s ({rate:.1f} Hz)")
    print(f"# Vibration verdict per joint (stdev in degrees):")
    print(f"# | Joint           | mean (°) | stdev (°) | p2p (°) | verdict |")
    print(f"# |-----------------|----------|-----------|---------|---------|")

    joints = list(obs.samples[-1][1].keys())
    for j in joints:
        vals = [s[j] for _, s in obs.samples if j in s]
        if len(vals) < 5:
            continue
        mean_rad = statistics.mean(vals)
        stdev_rad = statistics.pstdev(vals)
        p2p_rad = max(vals) - min(vals)
        mean_deg = math.degrees(mean_rad)
        stdev_deg = math.degrees(stdev_rad)
        p2p_deg = math.degrees(p2p_rad)
        if stdev_deg < 0.10:
            verdict = "QUIET"
        elif stdev_deg < 0.30:
            verdict = "small ring"
        else:
            verdict = "VIBRATING"
        print(f"# | {j:<15} | {mean_deg:+8.2f} | {stdev_deg:8.3f}  | {p2p_deg:6.2f}  | {verdict} |")


if __name__ == '__main__':
    main()
