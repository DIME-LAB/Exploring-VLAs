#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import serial
import time
import struct
import math


class SOARMTF2Broadcaster(Node):
    def __init__(self):
        super().__init__('soarm_tf2_broadcaster')

        self.declare_parameter('serial_port', '/dev/ttyACM1')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('inter_servo_delay_s', 0.003)

        self.serial_device = str(self.get_parameter('serial_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.inter_servo_delay_s = float(self.get_parameter('inter_servo_delay_s').value)

        # TF2 broadcaster for publishing transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Joint names from the actual robot (excluding gripper)
        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
        ]

        # Parent-child relationship between links and associated joint
        self.link_connections = [
            ['world', 'base_link', None, -1],
            ['base_link', 'shoulder_link', 'shoulder_pan', 0],
            ['shoulder_link', 'upper_arm_link', 'shoulder_lift', 1],
            ['upper_arm_link', 'lower_arm_link', 'elbow_flex', 2],
            ['lower_arm_link', 'wrist_link', 'wrist_flex', 3],
            ['wrist_link', 'gripper_link', 'wrist_roll', 4],
            ['gripper_link', 'ee_control_frame', None, -1],
        ]

        self.serial_port = None
        self.connect_to_robot()

        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self.read_and_publish_tf)

        self.last_positions = [0.0] * len(self.joint_names)
        self.last_raw_ticks = [2048] * len(self.joint_names)

        self.read_errors = [0] * len(self.joint_names)
        self.total_reads = 0
        self.consecutive_errors = 0

        self.get_logger().info(
            f"SO-ARM101 TF2 Broadcaster started at {self.publish_rate_hz:.1f} Hz | port={self.serial_device}"
        )
        self.get_logger().info("Direct hardware interface - reads from physical robot")
        self.get_logger().info("Publishing TF tree with exact URDF joint geometry and link names")

    def connect_to_robot(self):
        """Connect to the SO-ARM101 hardware."""
        try:
            self.serial_port = serial.Serial(self.serial_device, self.baud_rate, timeout=0.1)
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            time.sleep(0.1)
            self.get_logger().info(f"Connected to SO-ARM101 robot on {self.serial_device}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot: {e}")
            self.serial_port = None

    def read_servo_position(self, servo_id):
        """Read position from STS3215 servo using official protocol."""
        if not self.serial_port:
            return None

        try:
            length = 4
            instruction = 0x02
            address = 0x38
            read_length = 0x02

            checksum = (~(servo_id + length + instruction + address + read_length)) & 0xFF
            cmd = bytes([0xFF, 0xFF, servo_id, length, instruction, address, read_length, checksum])

            self.serial_port.reset_input_buffer()
            self.serial_port.write(cmd)

            time.sleep(0.001)
            response = self.serial_port.read(8)

            if len(response) >= 7:
                if response[0] != 0xFF or response[1] != 0xFF:
                    return None
                if response[2] != servo_id:
                    return None

                pos = struct.unpack('<H', response[5:7])[0]
                if 0 <= pos <= 4095:
                    return pos
                self.get_logger().debug(f"Servo {servo_id}: Invalid position {pos} (out of range 0-4095)")
                return None

            self.get_logger().debug(f"Servo {servo_id}: Short response ({len(response)} bytes)")
            return None

        except Exception as e:
            self.read_errors[servo_id - 1] += 1
            if self.read_errors[servo_id - 1] % 50 == 1:
                self.get_logger().warn(
                    f"Servo {servo_id}: Communication error #{self.read_errors[servo_id - 1]}: {e}"
                )

        return None

    def ticks_to_radians(self, ticks, joint_idx):
        """Convert servo ticks (0-4095) to radians (-pi to pi)."""
        if ticks is None:
            return self.last_positions[joint_idx]

        normalized = (ticks - 2048) / 2048.0
        return normalized * math.pi

    def rpy_to_quaternion(self, roll, pitch, yaw):
        """Convert roll, pitch, yaw (in radians) to quaternion (w, x, y, z)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return w, x, y, z

    def create_transform(self, parent_frame, child_frame, joint_angle=None, joint_name=None):
        """Create a transform between parent and child frames using URDF-like joint definitions."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.w = 1.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0

        if joint_angle is not None and joint_name is not None:
            if joint_name == 'shoulder_pan':
                transform.transform.translation.x = 0.0388353
                transform.transform.translation.y = -8.97657e-09
                transform.transform.translation.z = 0.0624
                static_quat_w, static_quat_x, static_quat_y, static_quat_z = self.rpy_to_quaternion(
                    3.14159, 4.18253e-17, -3.14159
                )
                joint_quat_w = math.cos(joint_angle / 2)
                joint_quat_z = math.sin(joint_angle / 2)
                transform.transform.rotation.w = static_quat_w * joint_quat_w - static_quat_z * joint_quat_z
                transform.transform.rotation.x = static_quat_x * joint_quat_w + static_quat_y * joint_quat_z
                transform.transform.rotation.y = static_quat_y * joint_quat_w - static_quat_x * joint_quat_z
                transform.transform.rotation.z = static_quat_z * joint_quat_w + static_quat_w * joint_quat_z

            elif joint_name == 'shoulder_lift':
                transform.transform.translation.x = -0.0303992
                transform.transform.translation.y = -0.0182778
                transform.transform.translation.z = -0.0542
                static_quat_w, static_quat_x, static_quat_y, static_quat_z = self.rpy_to_quaternion(
                    -1.5708, -1.5708, 0
                )
                joint_quat_w = math.cos(joint_angle / 2)
                joint_quat_z = math.sin(joint_angle / 2)
                transform.transform.rotation.w = static_quat_w * joint_quat_w - static_quat_z * joint_quat_z
                transform.transform.rotation.x = static_quat_x * joint_quat_w + static_quat_y * joint_quat_z
                transform.transform.rotation.y = static_quat_y * joint_quat_w - static_quat_x * joint_quat_z
                transform.transform.rotation.z = static_quat_z * joint_quat_w + static_quat_w * joint_quat_z

            elif joint_name == 'elbow_flex':
                transform.transform.translation.x = -0.11257
                transform.transform.translation.y = -0.028
                transform.transform.translation.z = 1.73763e-16
                static_quat_w, static_quat_x, static_quat_y, static_quat_z = self.rpy_to_quaternion(
                    -3.63608e-16, 8.74301e-16, 1.5708
                )
                joint_quat_w = math.cos(joint_angle / 2)
                joint_quat_z = math.sin(joint_angle / 2)
                transform.transform.rotation.w = static_quat_w * joint_quat_w - static_quat_z * joint_quat_z
                transform.transform.rotation.x = static_quat_x * joint_quat_w + static_quat_y * joint_quat_z
                transform.transform.rotation.y = static_quat_y * joint_quat_w - static_quat_x * joint_quat_z
                transform.transform.rotation.z = static_quat_z * joint_quat_w + static_quat_w * joint_quat_z

            elif joint_name == 'wrist_flex':
                transform.transform.translation.x = -0.1349
                transform.transform.translation.y = 0.0052
                transform.transform.translation.z = 3.62355e-17
                static_quat_w, static_quat_x, static_quat_y, static_quat_z = self.rpy_to_quaternion(
                    4.02456e-15, 8.67362e-16, -1.5708
                )
                joint_quat_w = math.cos(joint_angle / 2)
                joint_quat_z = math.sin(joint_angle / 2)
                transform.transform.rotation.w = static_quat_w * joint_quat_w - static_quat_z * joint_quat_z
                transform.transform.rotation.x = static_quat_x * joint_quat_w + static_quat_y * joint_quat_z
                transform.transform.rotation.y = static_quat_y * joint_quat_w - static_quat_x * joint_quat_z
                transform.transform.rotation.z = static_quat_z * joint_quat_w + static_quat_w * joint_quat_z

            elif joint_name == 'wrist_roll':
                transform.transform.translation.x = 5.55112e-17
                transform.transform.translation.y = -0.0611
                transform.transform.translation.z = 0.0181
                static_quat_w, static_quat_x, static_quat_y, static_quat_z = self.rpy_to_quaternion(
                    1.5708, 0.0486795, 3.14159
                )
                joint_quat_w = math.cos(joint_angle / 2)
                joint_quat_z = math.sin(joint_angle / 2)
                transform.transform.rotation.w = static_quat_w * joint_quat_w - static_quat_z * joint_quat_z
                transform.transform.rotation.x = static_quat_x * joint_quat_w + static_quat_y * joint_quat_z
                transform.transform.rotation.y = static_quat_y * joint_quat_w - static_quat_x * joint_quat_z
                transform.transform.rotation.z = static_quat_z * joint_quat_w + static_quat_w * joint_quat_z

        elif joint_name is None and parent_frame == 'gripper_link' and child_frame == 'ee_control_frame':
            transform.transform.translation.x = 0.015
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = -0.1
            transform.transform.rotation.w = 1.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0

        return transform

    def read_and_publish_tf(self):
        """Read all joint positions and publish as TF2 transforms."""
        if not self.serial_port:
            self.connect_to_robot()
            return

        self.total_reads += 1

        joint_positions = []
        new_raw_ticks = []
        successful_reads = 0

        for i in range(len(self.joint_names)):
            servo_id = i + 1
            ticks = self.read_servo_position(servo_id)
            radians = self.ticks_to_radians(ticks, i)
            joint_positions.append(radians)

            if ticks is not None:
                new_raw_ticks.append(ticks)
                successful_reads += 1
            else:
                new_raw_ticks.append(self.last_raw_ticks[i])

            time.sleep(self.inter_servo_delay_s)

        if successful_reads == 0:
            self.consecutive_errors += 1
            if self.consecutive_errors > 10:
                self.get_logger().error("Too many consecutive read failures - attempting reconnection")
                self.serial_port = None
                self.consecutive_errors = 0
                return
        else:
            self.consecutive_errors = 0

        self.last_positions = list(joint_positions)
        self.last_raw_ticks = new_raw_ticks.copy()

        for parent, child, joint_name, joint_idx in self.link_connections:
            joint_angle = None
            if joint_idx >= 0:
                joint_angle = joint_positions[joint_idx]
            transform = self.create_transform(parent, child, joint_angle, joint_name)
            self.tf_broadcaster.sendTransform(transform)

        if self.total_reads % 250 == 0:
            pos_str = [f'{p:.3f}' for p in joint_positions]
            tick_str = [f'{t}' for t in new_raw_ticks]
            error_str = [f'{e}' for e in self.read_errors]

            self.get_logger().info(f"=== Status Report (Read #{self.total_reads}) ===")
            self.get_logger().info("Publishing TF2 transforms for SO-ARM101")
            self.get_logger().info(f"Joint positions (rad): {pos_str}")
            self.get_logger().info(f"Raw servo ticks: {tick_str}")
            self.get_logger().info(f"Read errors per servo: {error_str}")
            self.get_logger().info(f"Successful reads: {successful_reads}/{len(self.joint_names)}")

    def destroy_node(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
        super().destroy_node()


def main():
    rclpy.init()

    broadcaster = None
    try:
        broadcaster = SOARMTF2Broadcaster()
        rclpy.spin(broadcaster)
    except KeyboardInterrupt:
        print("\nShutting down SO-ARM101 TF2 broadcaster...")
    finally:
        if broadcaster is not None:
            broadcaster.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
