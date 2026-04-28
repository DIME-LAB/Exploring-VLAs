#!/usr/bin/env python3
"""Stream-to-disk recorder. Each /joint_commands and /joint_states message
is appended to the JSONL file as it arrives, so SIGINT/SIGTERM at any
point still leaves a usable file.

Usage: python3 start_jerk_recorder_v2.py <out_path> [duration_s]
"""
import json
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class StreamRecorder(Node):
    def __init__(self, out_path: str, duration_s: float):
        super().__init__('stream_recorder')
        self.out_f = open(out_path, 'w', buffering=1)  # line-buffered
        self.deadline = time.monotonic() + duration_s
        self.count = {'c': 0, 's': 0}
        for topic, kind in [('/joint_commands', 'c'), ('/joint_states', 's')]:
            self.create_subscription(
                JointState, topic,
                lambda m, k=kind: self._on_msg(m, k),
                100,
            )
        self.create_timer(0.5, self._tick)
        self.get_logger().info(f'streaming to {out_path}')

    def _on_msg(self, msg, kind):
        self.out_f.write(json.dumps({
            't': time.monotonic(),
            'kind': kind,
            'name': list(msg.name),
            'pos': list(msg.position),
        }) + '\n')
        self.count[kind] += 1

    def _tick(self):
        if time.monotonic() >= self.deadline:
            self.shutdown()

    def shutdown(self):
        try:
            self.out_f.flush()
            self.out_f.close()
        except Exception:
            pass
        self.get_logger().info(
            f"saved {self.count['c']} cmds + {self.count['s']} states"
        )
        rclpy.shutdown()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/stream.jsonl'
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    rclpy.init()
    node = StreamRecorder(out_path, duration_s)

    def _sigint(_a, _b):
        node.shutdown()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    main()
