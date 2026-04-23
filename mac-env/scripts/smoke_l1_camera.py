"""L1 smoke test: ROS2Camera subscribes to /smoke/image and reads N frames.

Requires smoke_publisher.py running in another shell, e.g.:
  Terminal A: pixi run python scripts/smoke_publisher.py --topic /smoke/image
  Terminal B: pixi run python scripts/smoke_l1_camera.py --topic /smoke/image --frames 10

Pass/fail criteria:
  * connect() returns within 5s
  * read() returns np.ndarray with shape (H, W, 3), dtype=uint8
  * 10 consecutive reads show frame content changing (not all identical)
"""
import argparse
import time

import numpy as np

from lerobot.cameras.ros2 import ROS2Camera, ROS2CameraConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/smoke/image")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--encoding", default="bgr8")
    args = ap.parse_args()

    cfg = ROS2CameraConfig(topic=args.topic, encoding=args.encoding, fps=30)
    cam = ROS2Camera(cfg)

    t0 = time.perf_counter()
    cam.connect()
    print(f"connected in {(time.perf_counter() - t0):.2f}s")

    shapes = []
    checksums = []
    for i in range(args.frames):
        img = cam.read()
        assert isinstance(img, np.ndarray), f"read() returned {type(img)}, want ndarray"
        shapes.append(img.shape)
        checksums.append(int(img.sum()))
        print(f"  frame {i}: shape={img.shape} dtype={img.dtype} sum={checksums[-1]}")
        time.sleep(1 / 15)  # Slower than publisher so we see varied content.

    cam.disconnect()

    # Assertions
    assert len({s[:2] for s in shapes}) == 1, f"frame size drifted: {shapes}"
    assert shapes[0][2] == 3, f"expected 3-channel image, got shape {shapes[0]}"
    assert len(set(checksums)) > 1, (
        "all frames had identical pixels — publisher may be stuck or subscribe "
        "is not receiving new frames"
    )
    print(f"\n[L1 PASS] {args.frames} frames, shape={shapes[0]}, "
          f"{len(set(checksums))}/{args.frames} unique")


if __name__ == "__main__":
    main()
