#!/usr/bin/env python3
# Reference: ~/ros2_ws/install/JETANK_description/lib/python3.10/site-packages/JETANK_description/camera_pose_publisher.py
"""
Camera Pose Publisher for SO-ARM101.
Uses TF2 to look up the transform from base to camera_link and publishes as PoseStamped.

The camera is mounted on the wrist via a fixed joint chain:
  gripper -> camera_mount -> camera_link -> camera_optical_frame

This publisher tracks camera_link (the physical camera origin) by default.
Use camera_optical_frame for vision pipelines that need the optical convention
(Z forward, X right, Y down).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class CameraPosePublisher(Node):
    def __init__(self):
        super().__init__('camera_pose_publisher')

        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('startup_delay', 3.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        startup_delay = self.get_parameter('startup_delay').value

        self.pose_pub = self.create_publisher(PoseStamped, '/camera_pose', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'Camera Pose Publisher: {self.base_frame} -> {self.camera_frame} '
            f'at {self.publish_rate} Hz')

        self.startup_timer = self.create_timer(startup_delay, self._start_publishing)
        self.publish_timer = None
        self._last_warning_time = 0.0

    def _start_publishing(self):
        self.startup_timer.cancel()
        self.startup_timer.destroy()
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate, self._publish_camera_pose)
        self.get_logger().info('Started publishing camera pose')

    def _publish_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame

            msg.pose.position.x = transform.transform.translation.x
            msg.pose.position.y = transform.transform.translation.y
            msg.pose.position.z = transform.transform.translation.z
            msg.pose.orientation.x = transform.transform.rotation.x
            msg.pose.orientation.y = transform.transform.rotation.y
            msg.pose.orientation.z = transform.transform.rotation.z
            msg.pose.orientation.w = transform.transform.rotation.w

            self.pose_pub.publish(msg)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_warning_time > 5.0:
                self.get_logger().warn(
                    f'TF lookup failed ({self.base_frame}->{self.camera_frame}): {e}')
                self._last_warning_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CameraPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
