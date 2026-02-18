#!/usr/bin/env python3

import math
import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial


class JointStateReader(Node):
    """Read leader SO-ARM joints, mirror to follower SO-ARM, publish /joint_states."""

    def __init__(self):
        super().__init__('joint_state_reader')

        self.declare_parameter('leader_port', '/dev/ttyACM1')
        self.declare_parameter('follower_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('mirror_to_follower', True)
        self.declare_parameter('inter_servo_delay_s', 0.002)
        self.declare_parameter('reconnect_interval_s', 1.0)

        self.leader_port_name = str(self.get_parameter('leader_port').value)
        self.follower_port_name = str(self.get_parameter('follower_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.mirror_to_follower = bool(self.get_parameter('mirror_to_follower').value)
        self.inter_servo_delay_s = float(self.get_parameter('inter_servo_delay_s').value)
        self.reconnect_interval_s = float(self.get_parameter('reconnect_interval_s').value)

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.joint_names = [
            'Rotation',
            'Pitch',
            'Elbow',
            'Wrist_Pitch',
            'Wrist_Roll',
            'Jaw',
        ]

        self.leader_serial = None
        self.follower_serial = None
        self.last_reconnect_attempt = 0.0

        self.last_raw_ticks = [2048] * len(self.joint_names)
        self.last_positions = [0.0] * len(self.joint_names)

        self.total_cycles = 0
        self.read_errors = [0] * len(self.joint_names)
        self.write_errors = [0] * len(self.joint_names)

        self._connect_serials(force=True)

        timer_period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.read_mirror_publish)

        self.get_logger().info(
            f"Started joint_state_reader at {self.publish_rate_hz:.1f} Hz | "
            f"leader={self.leader_port_name} follower={self.follower_port_name} "
            f"mirror={self.mirror_to_follower}"
        )

    def _connect_port(self, port_name: str):
        try:
            ser = serial.Serial(port_name, self.baud_rate, timeout=0.05)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.05)
            self.get_logger().info(f"Connected: {port_name}")
            return ser
        except Exception as exc:
            self.get_logger().warn(f"Could not connect {port_name}: {exc}")
            return None

    def _connect_serials(self, force: bool = False):
        now = time.time()
        if not force and (now - self.last_reconnect_attempt) < self.reconnect_interval_s:
            return

        self.last_reconnect_attempt = now

        if self.leader_serial is None:
            self.leader_serial = self._connect_port(self.leader_port_name)

        if self.mirror_to_follower and self.follower_serial is None:
            self.follower_serial = self._connect_port(self.follower_port_name)

    @staticmethod
    def _checksum(payload_bytes):
        return (~sum(payload_bytes)) & 0xFF

    def _read_servo_position(self, ser, servo_id: int):
        try:
            payload = [servo_id, 4, 0x02, 0x38, 0x02]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])

            ser.reset_input_buffer()
            ser.write(packet)
            time.sleep(0.0015)
            response = ser.read(8)

            if len(response) < 8:
                return None
            if response[0] != 0xFF or response[1] != 0xFF:
                return None
            if response[2] != servo_id:
                return None
            if response[4] != 0x00:
                return None

            position = struct.unpack('<H', response[5:7])[0]
            if 0 <= position <= 4095:
                return position
            return None
        except Exception:
            return None

    def _write_servo_goal(self, ser, servo_id: int, ticks: int):
        ticks = max(0, min(4095, int(ticks)))
        low = ticks & 0xFF
        high = (ticks >> 8) & 0xFF

        try:
            payload = [servo_id, 5, 0x03, 0x2A, low, high]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.write(packet)
            return True
        except Exception:
            return False

    def _ticks_to_radians(self, ticks: int):
        return ((ticks - 2048) / 2048.0) * math.pi

    def _close_serial(self, attr_name: str):
        ser = getattr(self, attr_name)
        if ser is None:
            return
        try:
            ser.close()
        except Exception:
            pass
        setattr(self, attr_name, None)

    def read_mirror_publish(self):
        self._connect_serials(force=False)
        if self.leader_serial is None:
            return

        self.total_cycles += 1

        leader_ticks = []
        successful_reads = 0

        for i in range(len(self.joint_names)):
            servo_id = i + 1
            ticks = self._read_servo_position(self.leader_serial, servo_id)
            if ticks is None:
                self.read_errors[i] += 1
                ticks = self.last_raw_ticks[i]
            else:
                successful_reads += 1
            leader_ticks.append(ticks)
            time.sleep(self.inter_servo_delay_s)

        if successful_reads == 0:
            self._close_serial('leader_serial')
            return

        self.last_raw_ticks = leader_ticks

        if self.mirror_to_follower and self.follower_serial is not None:
            writes_ok = 0
            for i, ticks in enumerate(leader_ticks):
                servo_id = i + 1
                ok = self._write_servo_goal(self.follower_serial, servo_id, ticks)
                if ok:
                    writes_ok += 1
                else:
                    self.write_errors[i] += 1
                time.sleep(self.inter_servo_delay_s)

            if writes_ok == 0:
                self._close_serial('follower_serial')

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [self._ticks_to_radians(t) for t in leader_ticks]
        self.last_positions = msg.position
        self.joint_pub.publish(msg)

        if self.total_cycles % 100 == 0:
            self.get_logger().info(
                f"cycle={self.total_cycles} read_ok={successful_reads}/6 "
                f"leader={self.leader_port_name} follower_connected={self.follower_serial is not None}"
            )

    def destroy_node(self):
        self._close_serial('leader_serial')
        self._close_serial('follower_serial')
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = JointStateReader()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
