"""Publishes synthetic sensor_msgs/Image frames for smoke-testing ROS2Camera.

Runs forever. Each frame is a 640x480 BGR8 moving-gradient so you can tell
frames are genuinely updating (not a static buffer).

Usage (inside the pixi env):
  pixi run python scripts/smoke_publisher.py --topic /smoke/image --fps 30
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg


def ndarray_to_image_msg(img: np.ndarray, stamp, frame_id: str = "") -> ImageMsg:
    """Build a sensor_msgs/Image from a (H, W, 3) uint8 BGR ndarray.

    Pure-Python construction — avoids cv_bridge entirely so we don't pull in
    an ABI dependency on the publisher side.
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    h, w, _ = img.shape
    msg = ImageMsg()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = img.tobytes()
    return msg


class SmokePublisher(Node):
    def __init__(self, topic: str, fps: int, width: int, height: int):
        super().__init__("smoke_publisher")
        self.pub = self.create_publisher(ImageMsg, topic, 10)
        self.width = width
        self.height = height
        self.t0 = time.time()
        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f"Publishing {width}x{height} @ {fps}fps on {topic}")

    def tick(self):
        t = time.time() - self.t0
        # Moving-gradient frame so content changes over time.
        # int32 intermediates to avoid numpy>=2 uint8-overflow errors when
        # adding a Python int phase that may equal 255.
        x = np.linspace(0, 255, self.width, dtype=np.int32)
        y = np.linspace(0, 255, self.height, dtype=np.int32)
        gx, gy = np.meshgrid(x, y)
        phase = int((t * 60) % 256)
        b = ((gx + phase) % 256).astype(np.uint8)
        g = ((gy + phase) % 256).astype(np.uint8)
        r = ((gx + gy + phase) % 256).astype(np.uint8)
        img = np.stack([b, g, r], axis=-1)
        # Stamp a frame counter in the corner so the subscriber can sanity-check
        # it's genuinely getting live frames.
        cv2.putText(img, f"f={self.frame_idx}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        msg = ndarray_to_image_msg(img, self.get_clock().now().to_msg())
        self.pub.publish(msg)
        self.frame_idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/smoke/image")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    rclpy.init()
    node = SmokePublisher(args.topic, args.fps, args.width, args.height)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
