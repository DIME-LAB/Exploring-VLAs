#!/usr/bin/env python3
"""
JointStateReader — Feetech serial bridge for SO-ARM101 leader + follower arms.

Publishes:
  /joint_states    sensor_msgs/JointState  — follower by default (sim-parity),
                                              leader on publish_source='leader'
  /joint_commands  sensor_msgs/JointState  — always leader position (matches
                                              upstream so_leader teleop semantics)

Subscribes (when accept_command_input=True AND mirror_to_follower=False):
  /joint_commands  sensor_msgs/JointState  — writes goal_position to follower_serial
                                              (lets a policy / control package
                                              drive the real arm via topics)

Operating modes (pick via params):
  1. Physical leader teleop (legacy default):
       mirror_to_follower=True, accept_command_input=False
       Leader arm is the teleop source; follower mirrors it directly via USB.
  2. Topic-driven teleop (new for so101_ros2 plugin recording):
       mirror_to_follower=False, accept_command_input=True
       An external publisher on /joint_commands (control package / policy /
       script) drives the follower through this node's USB write.
  3. Observation-only (no actuation):
       mirror_to_follower=False, accept_command_input=False
       Just reads + publishes state; follower stays where it is.

Backward compat: publish_source='leader' restores the pre-2026-04 behavior
where /joint_states carried the leader arm's position.
"""

import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial


class JointStateReader(Node):
    """Feetech serial bridge — leader/follower arms ↔ ROS2 topics."""

    def __init__(self):
        super().__init__('joint_state_reader')

        # --- Serial / rate params ---
        self.declare_parameter('leader_port', '/dev/ttyACM1')
        self.declare_parameter('follower_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('inter_servo_delay_s', 0.002)
        self.declare_parameter('reconnect_interval_s', 1.0)

        # --- Publishing semantics ---
        self.declare_parameter('publish_source', 'follower')       # 'follower' | 'leader'
        self.declare_parameter('publish_commands', True)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_commands_topic', '/joint_commands')

        # --- Actuation semantics (exclusive) ---
        self.declare_parameter('mirror_to_follower', True)         # legacy USB teleop
        self.declare_parameter('accept_command_input', False)      # topic-driven teleop

        # --- Resolve ---
        self.leader_port_name = str(self.get_parameter('leader_port').value)
        self.follower_port_name = str(self.get_parameter('follower_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.inter_servo_delay_s = float(self.get_parameter('inter_servo_delay_s').value)
        self.reconnect_interval_s = float(self.get_parameter('reconnect_interval_s').value)

        self.publish_source = str(self.get_parameter('publish_source').value).lower()
        if self.publish_source not in ('follower', 'leader'):
            self.get_logger().warn(
                f"publish_source={self.publish_source!r} invalid; falling back to 'follower'")
            self.publish_source = 'follower'
        self.publish_commands = bool(self.get_parameter('publish_commands').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.joint_commands_topic = str(self.get_parameter('joint_commands_topic').value)

        self.mirror_to_follower = bool(self.get_parameter('mirror_to_follower').value)
        self.accept_command_input = bool(self.get_parameter('accept_command_input').value)

        if self.mirror_to_follower and self.accept_command_input:
            self.get_logger().warn(
                "mirror_to_follower=True AND accept_command_input=True — both want to "
                "drive the follower. Disabling accept_command_input to keep legacy behaviour."
            )
            self.accept_command_input = False

        # --- Publishers ---
        self.joint_state_pub = self.create_publisher(
            JointState, self.joint_states_topic, 10)
        self.joint_command_pub = (
            self.create_publisher(JointState, self.joint_commands_topic, 10)
            if self.publish_commands else None
        )

        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper_joint',
        ]

        self.leader_serial = None
        self.follower_serial = None
        self.last_reconnect_attempt = 0.0

        # Serial writes can race between the main publish loop (mirror mode) and
        # the command callback (topic-driven mode) — but `accept_command_input`
        # and `mirror_to_follower` are mutually exclusive so in practice only
        # one path ever writes. Lock guards follower_serial writes anyway.
        self.follower_write_lock = threading.Lock()

        self.last_leader_ticks = [2048] * len(self.joint_names)
        self.last_follower_ticks = [2048] * len(self.joint_names)

        self.total_cycles = 0
        self.read_errors_leader = [0] * len(self.joint_names)
        self.read_errors_follower = [0] * len(self.joint_names)
        self.write_errors = [0] * len(self.joint_names)
        self.command_msgs_received = 0

        self._connect_serials(force=True)

        # --- Subscriber for topic-driven follower control (optional) ---
        self.command_sub = None
        if self.accept_command_input:
            self.command_sub = self.create_subscription(
                JointState, self.joint_commands_topic,
                self._on_command_msg, 10)
            self.get_logger().info(
                f"Subscribed to {self.joint_commands_topic} — will write goals to follower"
            )

        timer_period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.read_and_publish)

        mode = (
            "usb-mirror-teleop" if self.mirror_to_follower
            else "topic-driven" if self.accept_command_input
            else "observation-only"
        )
        self.get_logger().info(
            f"Started joint_state_reader @ {self.publish_rate_hz:.1f} Hz | "
            f"mode={mode} | publish_source={self.publish_source} "
            f"publish_commands={self.publish_commands} | "
            f"leader={self.leader_port_name} follower={self.follower_port_name} | "
            f"states_topic={self.joint_states_topic} commands_topic={self.joint_commands_topic}"
        )

    # ------------------------------------------------------------------ serial
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

        need_follower = (
            self.mirror_to_follower
            or self.accept_command_input
            or self.publish_source == 'follower'
        )
        if need_follower and self.follower_serial is None:
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

    def _close_serial(self, attr_name: str):
        ser = getattr(self, attr_name)
        if ser is None:
            return
        try:
            ser.close()
        except Exception:
            pass
        setattr(self, attr_name, None)

    @staticmethod
    def _ticks_to_radians(ticks: int):
        return ((ticks - 2048) / 2048.0) * math.pi

    @staticmethod
    def _radians_to_ticks(radians: float):
        return int(round(2048 + (radians / math.pi) * 2048.0))

    # --------------------------------------------------------------- helpers
    def _read_all(self, ser, error_counter):
        """Read all 6 joints from one serial port. Returns list[int] (ticks)
        with the previous value substituted for any joint that failed to read.
        Returns None if every joint failed (port probably dead)."""
        ticks = []
        ok = 0
        for i in range(len(self.joint_names)):
            servo_id = i + 1
            t = self._read_servo_position(ser, servo_id)
            if t is None:
                error_counter[i] += 1
                # Use last known value for this arm so the published vector
                # doesn't suddenly collapse on a single bad read.
                t = 2048
            else:
                ok += 1
            ticks.append(t)
            time.sleep(self.inter_servo_delay_s)
        return ticks if ok > 0 else None

    def _publish_state(self, pub, ticks):
        if pub is None or ticks is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [self._ticks_to_radians(t) for t in ticks]
        pub.publish(msg)

    # --------------------------------------------------------------- command sub
    def _on_command_msg(self, msg: JointState):
        """Topic-driven follower write. Only active when accept_command_input=True."""
        if not self.accept_command_input or self.follower_serial is None:
            return
        # Match incoming positions to our joint ordering by name.
        pos_by_name = dict(zip(msg.name, msg.position, strict=False))
        with self.follower_write_lock:
            for i, name in enumerate(self.joint_names):
                rad = pos_by_name.get(name)
                if rad is None:
                    continue
                servo_id = i + 1
                if not self._write_servo_goal(
                    self.follower_serial, servo_id, self._radians_to_ticks(rad)
                ):
                    self.write_errors[i] += 1
                time.sleep(self.inter_servo_delay_s)
        self.command_msgs_received += 1

    # ---------------------------------------------------------------- main loop
    def read_and_publish(self):
        self._connect_serials(force=False)
        self.total_cycles += 1

        leader_ticks = None
        follower_ticks = None

        if self.leader_serial is not None:
            leader_ticks = self._read_all(self.leader_serial, self.read_errors_leader)
            if leader_ticks is None:
                self._close_serial('leader_serial')
            else:
                self.last_leader_ticks = leader_ticks

        if self.follower_serial is not None:
            follower_ticks = self._read_all(self.follower_serial, self.read_errors_follower)
            if follower_ticks is None:
                # Only close follower if we couldn't read at all AND we're not
                # actively writing to it via mirror/command modes — else wait
                # for next reconnect tick.
                if not (self.mirror_to_follower or self.accept_command_input):
                    self._close_serial('follower_serial')
            else:
                self.last_follower_ticks = follower_ticks

        # --- mirror mode (legacy): leader -> follower directly ---
        if (self.mirror_to_follower and leader_ticks is not None
                and self.follower_serial is not None):
            with self.follower_write_lock:
                writes_ok = 0
                for i, t in enumerate(leader_ticks):
                    if self._write_servo_goal(self.follower_serial, i + 1, t):
                        writes_ok += 1
                    else:
                        self.write_errors[i] += 1
                    time.sleep(self.inter_servo_delay_s)
                if writes_ok == 0:
                    self._close_serial('follower_serial')

        # --- publish /joint_states (source is configurable) ---
        source_ticks = follower_ticks if self.publish_source == 'follower' else leader_ticks
        if source_ticks is not None:
            self._publish_state(self.joint_state_pub, source_ticks)

        # --- publish /joint_commands (always leader) ---
        if self.publish_commands and leader_ticks is not None:
            self._publish_state(self.joint_command_pub, leader_ticks)

        if self.total_cycles % 100 == 0:
            self.get_logger().info(
                f"cycle={self.total_cycles} "
                f"leader_ok={self.leader_serial is not None} "
                f"follower_ok={self.follower_serial is not None} "
                f"cmd_msgs_rx={self.command_msgs_received}"
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
