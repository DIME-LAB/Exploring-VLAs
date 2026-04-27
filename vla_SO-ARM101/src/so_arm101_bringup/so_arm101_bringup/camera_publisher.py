#!/usr/bin/env python3
"""Camera Publisher Node.

Opens a USB camera via cv2.VideoCapture (AVFoundation on macOS, V4L2 on
Linux) and publishes sensor_msgs/Image frames to a ROS2 topic.

Ported from aruco_camera_localizer/camera_publisher.py so the lerobot
real-hardware stack stays standalone (no runtime dep on the aruco repo).
Changes vs the original:

* Resolution / frame rate / frame_id now come from CLI flags
  (``--width/--height/--fps/--frame-id``) with sensible defaults matching
  the sim pipeline (640x480 @ 30 fps, ``camera_optical_frame``).
* Exposure / white-balance hardcoding from the aruco version is removed —
  for dataset recording we want the camera's deployment-default behaviour.
* Actual capture resolution / frame rate is read back after ``cap.set``
  and a warning is logged on mismatch (AVFoundation silently downgrades
  many resolutions).
"""

import argparse
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from so_arm101_bringup.camera_selection import (
    detect_available_cameras,
    select_camera,
)


class CameraPublisher(Node):
    def __init__(
        self,
        camera_id: int,
        publish_topic: str = '/camera/image_raw',
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        frame_id: str = 'camera_optical_frame',
    ):
        super().__init__('camera_publisher')

        self.publish_topic = publish_topic
        self.frame_id = frame_id
        self.raw_image_pub = self.create_publisher(Image, publish_topic, 10)
        self.bridge = CvBridge()

        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera {self.camera_id}")
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_w != width or actual_h != height:
            self.get_logger().warn(
                f"Requested {width}x{height} but driver gave {actual_w}x{actual_h} — "
                "recording will reflect actual resolution."
            )
        if actual_fps and abs(actual_fps - fps) > 1.0:
            self.get_logger().warn(
                f"Requested {fps:.1f} fps but driver gave {actual_fps:.1f} fps."
            )

        self.get_logger().info(
            f"Camera {self.camera_id} opened: {actual_w}x{actual_h} @ "
            f"{actual_fps:.1f} fps → {self.publish_topic}"
        )

        period = 1.0 / fps if fps > 0 else 0.033
        self.timer = self.create_timer(period, self.capture_and_publish)
        self.frame_count = 0

    def capture_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.frame_id
            self.raw_image_pub.publish(img_msg)

            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f"Published {self.frame_count} frames on {self.publish_topic}"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to publish frame: {e}")

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Publish USB camera frames to a ROS2 Image topic.'
    )
    parser.add_argument(
        '--camera-id', type=int, default=None,
        help='Camera device ID for cv2.VideoCapture. If omitted, an '
             'interactive picker walks available cameras.',
    )
    parser.add_argument(
        '--publish-topic', type=str, default='/camera/image_raw',
        help='ROS2 topic to publish images on (default: /camera/image_raw).',
    )
    parser.add_argument('--width', type=int, default=640,
                        help='Requested capture width (default: 640).')
    parser.add_argument('--height', type=int, default=480,
                        help='Requested capture height (default: 480).')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Requested capture frame rate (default: 30.0).')
    parser.add_argument('--frame-id', type=str, default='camera_optical_frame',
                        help='sensor_msgs/Image header.frame_id.')
    return parser.parse_known_args()


def main(args=None):
    # argparse on sys.argv so ros2 run ... -- flags works; rclpy consumes
    # anything after --ros-args via args=.
    parsed_args, _ = parse_args()

    if parsed_args.camera_id is not None:
        cam_id = parsed_args.camera_id
        print(f"Using specified camera ID: {cam_id}")
    else:
        print("Detecting available cameras...")
        available = detect_available_cameras()
        if not available:
            print("No cameras detected!")
            return
        cam_id = select_camera(available)
        if cam_id is None:
            print("No camera selected!")
            return

    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisher(
            camera_id=cam_id,
            publish_topic=parsed_args.publish_topic,
            width=parsed_args.width,
            height=parsed_args.height,
            fps=parsed_args.fps,
            frame_id=parsed_args.frame_id,
        )
        print(
            f"\n[camera_publisher] cam {cam_id} → {parsed_args.publish_topic} "
            f"({parsed_args.width}x{parsed_args.height} @ {parsed_args.fps:g} fps). "
            "Ctrl+C to stop.\n"
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down camera publisher...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
