#!/usr/bin/env python3
"""Broadcast the dynamic `tcp_link` TF frame.

`tcp_link` is the midpoint between `gripper_tip` (fixed-jaw tip) and
`jaw_tip` (moving-jaw tip) — the true grasp center. It depends on the
live `gripper_joint` angle, so it can't be a static URDF link.

This node:
  1. Looks up TF base->gripper_tip and base->jaw_tip continuously.
  2. Computes the midpoint position and an orientation derived from
     gripper_tip's frame (so consumers can use tcp_link as a drop-in
     replacement for the legacy fixed-jaw tcp_link orientation).
  3. Broadcasts base->tcp_link at `publish_rate` Hz.

The midpoint is in base frame (not gripper frame) because that's how
all current TF consumers query it (`lookup_transform('base', 'tcp_link')`).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer, TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class GraspCenterPublisher(Node):
    def __init__(self):
        super().__init__('grasp_center_publisher')

        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('gripper_tip_frame', 'gripper_tip')
        self.declare_parameter('jaw_tip_frame', 'jaw_tip')
        self.declare_parameter('output_frame', 'tcp_link')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('startup_delay', 3.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.gripper_tip_frame = self.get_parameter('gripper_tip_frame').value
        self.jaw_tip_frame = self.get_parameter('jaw_tip_frame').value
        self.output_frame = self.get_parameter('output_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        startup_delay = self.get_parameter('startup_delay').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f'Grasp center publisher: midpoint({self.gripper_tip_frame}, '
            f'{self.jaw_tip_frame}) -> {self.output_frame} at '
            f'{self.publish_rate} Hz')

        self._last_warning_time = 0.0
        self.startup_timer = self.create_timer(startup_delay, self._start_publishing)
        self.publish_timer = None

    def _start_publishing(self):
        self.startup_timer.cancel()
        self.startup_timer.destroy()
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate, self._publish_grasp_center)
        self.get_logger().info('Started publishing grasp center TF')

    def _publish_grasp_center(self):
        try:
            t_grip = self.tf_buffer.lookup_transform(
                self.base_frame, self.gripper_tip_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            t_jaw = self.tf_buffer.lookup_transform(
                self.base_frame, self.jaw_tip_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_warning_time > 5.0:
                self.get_logger().warn(f'TF lookup failed: {e}')
                self._last_warning_time = now
            return

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame
        out.child_frame_id = self.output_frame

        out.transform.translation.x = 0.5 * (
            t_grip.transform.translation.x + t_jaw.transform.translation.x)
        out.transform.translation.y = 0.5 * (
            t_grip.transform.translation.y + t_jaw.transform.translation.y)
        out.transform.translation.z = 0.5 * (
            t_grip.transform.translation.z + t_jaw.transform.translation.z)

        # Orientation: copy gripper_tip's. The legacy tcp_link was a child
        # of gripper with rpy=(0,π,0); after the rename gripper_tip carries
        # that same orientation, so consumers that previously read tcp_link
        # quaternion get bit-identical behavior here.
        out.transform.rotation = t_grip.transform.rotation

        self.tf_broadcaster.sendTransform(out)


def main(args=None):
    rclpy.init(args=args)
    node = GraspCenterPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
