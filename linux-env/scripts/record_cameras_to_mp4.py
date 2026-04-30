#!/usr/bin/env python3
"""record_cameras_to_mp4.py — minimal camera-to-mp4 recorder.

Standalone helper used to capture wrist + workspace camera videos during
async inference (which doesn't go through lerobot-record's dataset path
and so doesn't write videos itself). Lives in pixi-Jazzy env to share the
same rclpy Jazzy that the async client uses.

Run alongside the async inference subprocess:
    python3 record_cameras_to_mp4.py \\
        --topic /wrist_camera_rgb_sim --output /tmp/wrist.mp4 &
    python3 record_cameras_to_mp4.py \\
        --topic /workspace_camera_sim --output /tmp/top.mp4 &

SIGINT or SIGTERM → finalizes the mp4 cleanly via cv2.VideoWriter.release().
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraRecorder(Node):
    def __init__(self, topic: str, output: str, fps: int):
        super().__init__("camera_recorder")
        self._topic = topic
        self._output = output
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._t0 = time.monotonic()
        self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data,
        )
        self.get_logger().info(f"recording {topic} → {output} @ {fps} fps")

    def _on_image(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8",):
                img = arr.reshape((msg.height, msg.width, 3))
                bgr = img[..., ::-1].copy()  # rgb → bgr for cv2
            elif msg.encoding in ("bgr8",):
                bgr = arr.reshape((msg.height, msg.width, 3))
            elif msg.encoding in ("rgba8",):
                rgba = arr.reshape((msg.height, msg.width, 4))
                bgr = rgba[..., 2::-1].copy()  # rgba → bgr (drop alpha, reverse)
            elif msg.encoding in ("bgra8",):
                bgra = arr.reshape((msg.height, msg.width, 4))
                bgr = bgra[..., :3].copy()
            else:
                self.get_logger().warning(
                    f"unsupported encoding {msg.encoding!r}; dropping frame")
                return
        except Exception as e:
            self.get_logger().warning(f"decode failed: {e}")
            return

        if self._writer is None:
            h, w = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._output, fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                self.get_logger().error(
                    f"failed to open writer for {self._output}")
                return
        self._writer.write(bgr)
        self._frame_count += 1

    def shutdown(self):
        if self._writer is not None:
            self._writer.release()
        elapsed = time.monotonic() - self._t0
        self.get_logger().info(
            f"{self._topic}: wrote {self._frame_count} frames in {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    rclpy.init()
    node = CameraRecorder(args.topic, args.output, args.fps)

    def _shutdown(*_a):
        node.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
