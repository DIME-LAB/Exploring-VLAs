#!/usr/bin/env python3
"""Record /joint_targets, /joint_commands, /joint_states with monotonic
timestamps for a window, dump JSONL for offline analysis with
analyze_motion.py.

Use case: empirical motion-quality debugging when /joint_states or RViz
shows behavior you can't explain from logs alone. Captures all three
streams time-aligned so you can quantify lag, RMS error, settling time,
oscillation frequency per joint.

Run with the stack ALREADY UP. The recorder subscribes; YOU drive
motion (click GUI buttons, ros2 service call, etc.) during the window.

  bash mac-env/scripts/stack_start.sh --mode real --rviz follower_port:=...
  pixi run --manifest-path /tmp/mac-env/pixi.toml python3 \\
      mac-env/scripts/record_motion_topics.py --duration 30 --out /tmp/run.jsonl
  # ... drive motion during the 30s window ...
  pixi run --manifest-path /tmp/mac-env/pixi.toml python3 \\
      mac-env/scripts/analyze_motion.py /tmp/run.jsonl

Note: shim publishes to /joint_commands directly (no /joint_targets)
since gravity-comp went inline 2026-04-27. The /joint_targets channel
is captured but will be empty in current architecture; analyze_motion.py
detects motion segments from /joint_commands gaps instead.
"""
import argparse, json, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Recorder(Node):
    def __init__(self, out_path, duration_s):
        super().__init__('motion_recorder')
        self.out_path = out_path
        self.deadline = time.monotonic() + duration_s
        self.rows = []
        # Three independent buffers — flush in order each time we record
        self.create_subscription(
            JointState, '/joint_targets',
            lambda m: self._on('targets', m), 100)
        self.create_subscription(
            JointState, '/joint_commands',
            lambda m: self._on('commands', m), 100)
        self.create_subscription(
            JointState, '/joint_states',
            lambda m: self._on('states', m), 100)
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f'Recording to {out_path} for {duration_s:.1f}s — '
            f'TRIGGER MOTION NOW (Grasp Home, etc.)')

    def _on(self, kind, msg):
        self.rows.append({
            't': time.monotonic(),
            'kind': kind,
            'name': list(msg.name),
            'pos': list(msg.position),
        })

    def _tick(self):
        if time.monotonic() >= self.deadline:
            with open(self.out_path, 'w') as f:
                for r in self.rows:
                    f.write(json.dumps(r) + '\n')
            self.get_logger().info(
                f'Saved {len(self.rows)} samples to {self.out_path}')
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/motion_record.jsonl')
    ap.add_argument('--duration', type=float, default=25.0)
    args, _ = ap.parse_known_args()
    rclpy.init()
    rclpy.spin(Recorder(args.out, args.duration))


if __name__ == '__main__':
    main()
