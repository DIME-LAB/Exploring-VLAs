#!/usr/bin/env python3
"""
STS3215 servo driver for SO-ARM101.
Reads joint positions from 6 servos and publishes to /joint_states.
Subscribes to /joint_commands and writes goal positions to servos.
Supports mirror-to-follower teleoperation, torque control, speed/acceleration
parameters, position verification, and diagnostic logging.

Source: adapted from Exploring-VLAs/vla_SO-101/jointstatereader/joint_state_reader.py
        (leader/follower mirroring, serial protocol, error tracking)
Write command / torque / init reference: harryzy/lododo-arm ft_arm_controller.py
        (speed, acceleration, torque enable/disable, damping, init-to-center, RegAction)
"""

import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

try:
    import serial
except ImportError:
    serial = None

# STS3215 register addresses
_ADDR_GOAL_POSITION = 0x2A
_ADDR_CURRENT_POSITION = 0x38
_ADDR_TORQUE_ENABLE = 0x28
_ADDR_GOAL_SPEED = 0x2E
_ADDR_GOAL_ACC = 0x29

# STS3215 constants
CENTER_POSITION = 2048
TICKS_MAX = 4095
DEFAULT_SPEED = 800
DEFAULT_ACC = 80


class ServoDriver(Node):
    """Read SO-ARM101 servo positions, publish joint_states, accept joint_commands.

    Features carried over from the original jointstatereader and lododo-arm:
    - Leader-to-follower mirror teleoperation (from joint_state_reader.py)
    - Per-servo error tracking and diagnostic logging (from joint_state_reader.py)
    - Speed and acceleration parameters for servo writes (from ft_arm_controller.py)
    - Torque enable/disable/damping (from ft_arm_controller.py)
    - Position verification (from ft_arm_controller.py)
    - Initialize-to-center sequence (from ft_arm_controller.py)
    """

    JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    NUM_SERVOS = 6

    def __init__(self):
        super().__init__('servo_driver')

        # -- Parameters --
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('follower_port', '')  # empty = no follower
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('mirror_to_follower', False)
        self.declare_parameter('inter_servo_delay_s', 0.002)
        self.declare_parameter('reconnect_interval_s', 1.0)
        self.declare_parameter('moving_speed', DEFAULT_SPEED)
        self.declare_parameter('moving_acc', DEFAULT_ACC)
        self.declare_parameter('init_to_center', False)
        self.declare_parameter('position_verify_tolerance', 10)  # ticks

        self.port_name = str(self.get_parameter('serial_port').value)
        self.follower_port_name = str(self.get_parameter('follower_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.mirror_to_follower = bool(self.get_parameter('mirror_to_follower').value)
        self.inter_servo_delay_s = float(self.get_parameter('inter_servo_delay_s').value)
        self.reconnect_interval_s = float(self.get_parameter('reconnect_interval_s').value)
        self.moving_speed = int(self.get_parameter('moving_speed').value)
        self.moving_acc = int(self.get_parameter('moving_acc').value)
        self.init_to_center = bool(self.get_parameter('init_to_center').value)
        self.verify_tolerance = int(self.get_parameter('position_verify_tolerance').value)

        # -- Publishers / subscribers --
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self._joint_command_cb, 10)

        # Service: enable/disable torque
        self.torque_srv = self.create_service(
            SetBool, '~/set_torque', self._set_torque_callback)

        # -- Serial state --
        self.leader_serial = None
        self.follower_serial = None
        self.last_reconnect = 0.0

        # -- Servo state --
        self.last_ticks = [CENTER_POSITION] * self.NUM_SERVOS
        self.last_positions = [0.0] * self.NUM_SERVOS
        self.pending_goals = {}  # servo_id -> ticks
        self._goals_lock = threading.Lock()  # Issue 1: protect pending_goals

        # -- Diagnostics (from joint_state_reader.py) --
        self.total_cycles = 0
        self.read_errors = [0] * self.NUM_SERVOS
        self.write_errors = [0] * self.NUM_SERVOS

        # -- Connect and optionally initialize --
        self._connect_serials(force=True)

        if self.init_to_center and self.leader_serial is not None:
            self._initialize_to_center()

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._read_mirror_publish)

        self.get_logger().info(
            f'ServoDriver started: port={self.port_name} '
            f'follower={self.follower_port_name or "none"} '
            f'mirror={self.mirror_to_follower} '
            f'rate={self.publish_rate_hz} Hz '
            f'speed={self.moving_speed} acc={self.moving_acc}')

    # ================================================================
    # Serial connection (from joint_state_reader.py)
    # ================================================================

    def _connect_port(self, port_name):
        """Connect to a single serial port."""
        if serial is None or not port_name:
            return None
        try:
            ser = serial.Serial(port_name, self.baud_rate, timeout=0.05)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.05)
            self.get_logger().info(f'Connected: {port_name}')
            return ser
        except Exception as exc:
            self.get_logger().warn(f'Could not connect {port_name}: {exc}')
            return None

    def _connect_serials(self, force=False):
        """Connect leader and optionally follower serial ports."""
        now = time.time()
        if not force and (now - self.last_reconnect) < self.reconnect_interval_s:
            return
        self.last_reconnect = now

        if self.leader_serial is None:
            self.leader_serial = self._connect_port(self.port_name)

        if self.mirror_to_follower and self.follower_serial is None:
            self.follower_serial = self._connect_port(self.follower_port_name)

    def _close_serial(self, attr_name):
        """Close a serial port by attribute name."""
        ser = getattr(self, attr_name)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            setattr(self, attr_name, None)

    # ================================================================
    # STS3215 low-level protocol
    # ================================================================

    @staticmethod
    def _checksum(payload):
        return (~sum(payload)) & 0xFF

    def _read_position(self, ser, servo_id):
        """Read current position (ticks 0-4095) from a single servo."""
        try:
            payload = [servo_id, 4, 0x02, _ADDR_CURRENT_POSITION, 0x02]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.reset_input_buffer()
            ser.write(packet)
            time.sleep(0.0015)
            resp = ser.read(8)
            if len(resp) < 8 or resp[0] != 0xFF or resp[1] != 0xFF:
                return None
            if resp[2] != servo_id or resp[4] != 0x00:
                return None
            # Issue 3: validate response checksum to reject corrupted data
            expected_cs = (~sum(resp[2:7])) & 0xFF
            if resp[7] != expected_cs:
                return None
            pos = struct.unpack('<H', resp[5:7])[0]
            return pos if 0 <= pos <= TICKS_MAX else None
        except Exception:
            return None

    def _write_goal(self, ser, servo_id, ticks):
        """Write goal position to a single servo (immediate write)."""
        ticks = max(0, min(TICKS_MAX, int(ticks)))
        low = ticks & 0xFF
        high = (ticks >> 8) & 0xFF
        try:
            payload = [servo_id, 5, 0x03, _ADDR_GOAL_POSITION, low, high]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.write(packet)
            return True
        except Exception:
            return False

    def _write_torque(self, ser, servo_id, enable_value):
        """Write torque enable register.
        Source: from lododo-arm ft_arm_controller.py setup_torque().

        enable_value:
            0 = torque off (free spin)
            1 = torque on (hold position)
            2 = damping mode (passive compliance)
            128 = calibrate current position to 2048
        """
        try:
            payload = [servo_id, 4, 0x03, _ADDR_TORQUE_ENABLE, enable_value & 0xFF]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.write(packet)
            time.sleep(0.001)
            return True
        except Exception:
            return False

    def _reg_write_goal(self, ser, servo_id, ticks, speed=None):
        """Register-buffered write (RegWrite) — servo stores command but doesn't
        execute until a RegAction packet is sent. This allows synchronized
        multi-servo movement.
        Source: from lododo-arm ft_arm_controller.py execute_movement().

        Uses instruction 0x04 (RegWrite) instead of 0x03 (Write).
        """
        ticks = max(0, min(TICKS_MAX, int(ticks)))
        speed = speed if speed is not None else self.moving_speed
        pos_lo = ticks & 0xFF
        pos_hi = (ticks >> 8) & 0xFF
        spd_lo = speed & 0xFF
        spd_hi = (speed >> 8) & 0xFF
        try:
            payload = [servo_id, 9, 0x04, _ADDR_GOAL_POSITION,
                       pos_lo, pos_hi, 0x00, 0x00, spd_lo, spd_hi]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.write(packet)
            time.sleep(0.001)
            return True
        except Exception:
            return False

    def _reg_action(self, ser):
        """Send RegAction broadcast packet to trigger all buffered RegWrite commands.
        Source: from lododo-arm ft_arm_controller.py execute_movement().
        Broadcast ID = 0xFE, instruction = 0x05 (Action).
        """
        try:
            payload = [0xFE, 2, 0x05]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.write(packet)
            return True
        except Exception:
            return False

    # ================================================================
    # Conversions
    # ================================================================

    @staticmethod
    def _ticks_to_radians(ticks):
        return ((ticks - CENTER_POSITION) / 2048.0) * math.pi

    @staticmethod
    def _radians_to_ticks(radians):
        return int((radians / math.pi) * 2048.0 + CENTER_POSITION)

    # ================================================================
    # Higher-level hardware operations (from ft_arm_controller.py)
    # ================================================================

    def _set_torque_all(self, ser, enable_value):
        """Set torque for all 6 servos."""
        for sid in range(1, self.NUM_SERVOS + 1):
            self._write_torque(ser, sid, enable_value)
            time.sleep(self.inter_servo_delay_s)

    def _verify_positions(self, ser, target_ticks, tolerance=None):
        """Verify all servos have reached their target positions.
        Source: from lododo-arm ft_arm_controller.py verify_positions().
        Returns True if all servos are within tolerance of their targets.
        """
        if tolerance is None:
            tolerance = self.verify_tolerance
        for i in range(self.NUM_SERVOS):
            current = self._read_position(ser, i + 1)
            if current is None:
                return False
            if abs(current - target_ticks[i]) > tolerance:
                return False
            time.sleep(self.inter_servo_delay_s)
        return True

    def _initialize_to_center(self):
        """Initialize all servos to center position (2048).
        Source: from lododo-arm ft_arm_controller.py initialize_to_center().

        Sequence:
        1. Enable torque on all servos
        2. Write center position with speed/acceleration
        3. Wait for movement
        4. Verify positions
        """
        ser = self.leader_serial
        if ser is None:
            self.get_logger().warn('Cannot init-to-center: not connected')
            return

        self.get_logger().info('Initializing all servos to center position...')

        # Step 1: Enable torque
        self._set_torque_all(ser, 1)
        time.sleep(0.1)

        # Step 2: Write center position using RegWrite for synchronized movement
        for sid in range(1, self.NUM_SERVOS + 1):
            self._reg_write_goal(ser, sid, CENTER_POSITION,
                                 speed=self.moving_speed)
            time.sleep(self.inter_servo_delay_s)

        # Step 3: Trigger synchronized movement
        self._reg_action(ser)
        time.sleep(2.0)  # Wait for movement to complete

        # Step 4: Verify
        target = [CENTER_POSITION] * self.NUM_SERVOS
        if self._verify_positions(ser, target):
            self.get_logger().info('Init-to-center: all servos at center')
        else:
            self.get_logger().warn('Init-to-center: some servos did not reach center')

    def _synchronized_write(self, ser, goals_dict):
        """Write goals to multiple servos and execute simultaneously.
        goals_dict: {servo_id: ticks}
        Source: from lododo-arm ft_arm_controller.py execute_movement().
        """
        for sid, ticks in goals_dict.items():
            self._reg_write_goal(ser, sid, ticks, speed=self.moving_speed)
            time.sleep(self.inter_servo_delay_s)
        self._reg_action(ser)

    # ================================================================
    # Service callback: torque enable/disable
    # ================================================================

    def _set_torque_callback(self, request, response):
        """ROS2 service to enable/disable torque on all servos."""
        if self.leader_serial is None:
            response.success = False
            response.message = 'Not connected to serial port'
            return response
        value = 1 if request.data else 0
        self._set_torque_all(self.leader_serial, value)
        response.success = True
        response.message = f'Torque {"enabled" if request.data else "disabled"} on all servos'
        self.get_logger().info(response.message)
        return response

    # ================================================================
    # Callbacks
    # ================================================================

    def _joint_command_cb(self, msg):
        """Queue goal positions received on /joint_commands."""
        with self._goals_lock:
            for i, name in enumerate(msg.name):
                if i >= len(msg.position):
                    break
                try:
                    idx = self.JOINT_NAMES.index(name)
                except ValueError:
                    continue
                servo_id = idx + 1
                self.pending_goals[servo_id] = self._radians_to_ticks(msg.position[i])

    # ================================================================
    # Main loop (from joint_state_reader.py read_mirror_publish)
    # ================================================================

    def _read_mirror_publish(self):
        """Read leader servos, mirror to follower, publish joint_states.
        Source: from joint_state_reader.py read_mirror_publish()."""
        self._connect_serials(force=False)
        if self.leader_serial is None:
            return

        self.total_cycles += 1

        # -- Write pending goals using synchronized write --
        with self._goals_lock:
            goals = dict(self.pending_goals)
            self.pending_goals.clear()
        if goals:
            self._synchronized_write(self.leader_serial, goals)

        # -- Read all servo positions from leader --
        leader_ticks = []
        successful_reads = 0
        for i in range(self.NUM_SERVOS):
            servo_id = i + 1
            ticks = self._read_position(self.leader_serial, servo_id)
            if ticks is None:
                self.read_errors[i] += 1
                ticks = self.last_ticks[i]
            else:
                successful_reads += 1
            leader_ticks.append(ticks)
            time.sleep(self.inter_servo_delay_s)

        if successful_reads == 0:
            self._close_serial('leader_serial')
            return

        self.last_ticks = leader_ticks

        # -- Mirror to follower (from joint_state_reader.py) --
        if self.mirror_to_follower and self.follower_serial is not None:
            writes_ok = 0
            for i, ticks in enumerate(leader_ticks):
                servo_id = i + 1
                ok = self._write_goal(self.follower_serial, servo_id, ticks)
                if ok:
                    writes_ok += 1
                else:
                    self.write_errors[i] += 1
                time.sleep(self.inter_servo_delay_s)

            if writes_ok == 0:
                self._close_serial('follower_serial')

        # -- Publish joint states --
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = [self._ticks_to_radians(t) for t in leader_ticks]
        self.last_positions = list(msg.position)
        self.joint_pub.publish(msg)

        # -- Periodic diagnostics (from joint_state_reader.py) --
        if self.total_cycles % 100 == 0:
            self.get_logger().info(
                f'cycle={self.total_cycles} read_ok={successful_reads}/{self.NUM_SERVOS} '
                f'leader={self.port_name} '
                f'follower_connected={self.follower_serial is not None} '
                f'read_errors={self.read_errors} write_errors={self.write_errors}')

    # ================================================================
    # Cleanup
    # ================================================================

    def destroy_node(self):
        self._close_serial('leader_serial')
        self._close_serial('follower_serial')
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = ServoDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
