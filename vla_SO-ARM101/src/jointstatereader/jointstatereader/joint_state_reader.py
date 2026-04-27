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
from std_srvs.srv import Trigger
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

        # ---- Service: ~/torque_reset (std_srvs/Trigger) ----
        # Public name resolves to /<node_name>/torque_reset, e.g.
        # /joint_state_reader/torque_reset. Replaces the
        # kill+respawn-jsr dance previously needed by mac-env scripts
        # (arm_reset.py / arm_trim.py) just to access the half-duplex
        # Feetech bus. The handler runs inside the rclpy
        # SingleThreadedExecutor, so it naturally blocks the read
        # timer + the /joint_commands subscriber while it executes.
        # /joint_states pauses for ~1 sec, then resumes seamlessly.
        self._torque_reset_service = self.create_service(
            Trigger, '~/torque_reset', self._on_torque_reset)
        # Companion services for freedrive: torque_off leaves the arm
        # back-drivable indefinitely; torque_on anti-jolts and re-engages.
        # Use these from the GUI's Freedrive button. The pair is a
        # *stateful* alternative to the single-shot torque_reset cycle.
        self._torque_off_service = self.create_service(
            Trigger, '~/torque_off', self._on_torque_off)
        self._torque_on_service = self.create_service(
            Trigger, '~/torque_on', self._on_torque_on)
        # Per-joint freedrive + recenter (single-joint zeroing).
        # Reads ROS param `target_joint_name` (string, in self.joint_names)
        # so the same Trigger srv type handles any joint.
        self.declare_parameter('target_joint_name', '')
        self._freedrive_one_service = self.create_service(
            Trigger, '~/freedrive_one', self._on_freedrive_one)
        self._recenter_one_service = self.create_service(
            Trigger, '~/recenter_one', self._on_recenter_one)
        self._cancel_freedrive_one_service = self.create_service(
            Trigger, '~/cancel_freedrive_one',
            self._on_cancel_freedrive_one)
        self.get_logger().info(
            f"Hosting services: ~/torque_reset, ~/torque_off, "
            f"~/torque_on, ~/freedrive_one, ~/recenter_one, "
            f"~/cancel_freedrive_one")

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

    def _write_servo_byte(self, ser, servo_id: int, addr: int, value: int):
        """Write a single-byte EEPROM/RAM register to a Feetech servo.

        Used by ~/torque_reset for register 40 (Torque_Enable). Returns
        True on a complete status-packet ack from the servo. Mirrors the
        `write_byte` helper in mac-env/scripts/arm_reset.py so the
        in-process and out-of-process paths agree on protocol framing.
        """
        try:
            payload = [servo_id, 4, 0x03, addr, value & 0xFF]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.reset_input_buffer()
            ser.write(packet)
            time.sleep(0.005)
            response = ser.read(6)
            return (len(response) >= 6
                    and response[0] == 0xFF and response[1] == 0xFF
                    and response[2] == servo_id and response[4] == 0)
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
    def _read_all(self, ser, error_counter, last_known):
        """Read all 6 joints from one serial port. Returns list[int] (ticks)
        with the previous value substituted for any joint that failed to read.
        Returns None if every joint failed (port probably dead).

        ``last_known`` is the per-arm tick vector from the most recent fully
        successful cycle (last_leader_ticks or last_follower_ticks). Falling
        back to it — instead of 2048 (mid-encoder) — keeps /joint_states
        stable across transient read failures. The previous behavior caused
        a visible RViz snap-to-zero per joint whenever the serial bus was
        contended (e.g. during topic-driven goal writes).
        """
        ticks = []
        ok = 0
        for i in range(len(self.joint_names)):
            servo_id = i + 1
            t = self._read_servo_position(ser, servo_id)
            if t is None:
                error_counter[i] += 1
                t = last_known[i]
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

    # --------------------------------------------------------------- torque_reset
    def _on_torque_reset(self, request, response):
        """Cycle the follower's torque without releasing the bus.

        Procedure (matches mac-env/scripts/arm_reset.py's bus protocol):
          1. anti-jolt — Goal_Position ← Present_Position per servo so
             torque-on produces zero PID error → no startup motion
          2. Torque_Enable=0 per servo → vibration stops, gravity parks
          3. sleep 700 ms → let inertia decay
          4. Torque_Enable=1 per servo → re-engage holding settled pose

        The SingleThreadedExecutor serializes callbacks, so this body
        naturally blocks the read timer + /joint_commands subscriber
        for its duration (~1 s). No explicit pause flag needed; rclpy
        drops missed timer ticks rather than queueing them.

        The follower_write_lock is taken to be defensive against any
        future MultiThreadedExecutor migration — under the default
        single-threaded executor the lock is never contended.

        Replaces the kill+respawn-jsr dance previously needed by
        mac-env/scripts/arm_reset.py (which had to release the
        half-duplex bus before any external bus access).
        """
        if self.follower_serial is None:
            response.success = False
            response.message = (
                f"follower_serial not connected (port="
                f"{self.follower_port_name}); nothing to torque-cycle")
            self.get_logger().warn(response.message)
            return response

        REG_TORQUE_ENABLE = 40
        try:
            with self.follower_write_lock:
                # Step 1: anti-jolt
                for i in range(len(self.joint_names)):
                    sid = i + 1
                    present = self._read_servo_position(self.follower_serial, sid)
                    if present is not None:
                        self._write_servo_goal(self.follower_serial, sid, present)
                    time.sleep(self.inter_servo_delay_s)
                # Step 2: torque off (per-servo)
                for i in range(len(self.joint_names)):
                    self._write_servo_byte(
                        self.follower_serial, i + 1, REG_TORQUE_ENABLE, 0)
                    time.sleep(self.inter_servo_delay_s)
                # Step 3: settle under gravity
                time.sleep(0.7)
                # Step 4: torque on (zero PID error → no jolt)
                for i in range(len(self.joint_names)):
                    self._write_servo_byte(
                        self.follower_serial, i + 1, REG_TORQUE_ENABLE, 1)
                    time.sleep(self.inter_servo_delay_s)
            response.success = True
            response.message = (
                f"torque cycled on {self.follower_port_name} "
                f"(6 servos, anti-jolt applied)")
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"torque_reset failed: {exc}"
            self.get_logger().error(response.message)
        return response

    # --------------------------------------------------------------- torque_off / torque_on (freedrive)
    def _on_torque_off(self, request, response):
        """Disengage torque on all 6 follower servos and STAY OFF.

        Use as the "freedrive enter" primitive: arm becomes back-drivable
        and you can hand-pose it. Caller is responsible for re-engaging
        via ~/torque_on (the anti-jolt path) when done. Without re-engage
        the servos draw no holding current and the arm sags under gravity.

        SAFETY: caller must ensure the arm is in a rested / supported
        pose before calling. Calling with arm extended in air → arm
        falls under gravity → potential collision with workspace.
        """
        if self.follower_serial is None:
            response.success = False
            response.message = (
                f"follower_serial not connected (port="
                f"{self.follower_port_name}); nothing to torque-off")
            self.get_logger().warn(response.message)
            return response

        REG_TORQUE_ENABLE = 40
        try:
            with self.follower_write_lock:
                for i in range(len(self.joint_names)):
                    self._write_servo_byte(
                        self.follower_serial, i + 1, REG_TORQUE_ENABLE, 0)
                    time.sleep(self.inter_servo_delay_s)
            response.success = True
            response.message = (
                f"torque OFF on {self.follower_port_name} — arm is "
                f"back-drivable. Re-engage via ~/torque_on when done.")
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"torque_off failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def _on_torque_on(self, request, response):
        """Re-engage torque on all 6 servos with anti-jolt.

        Use as the "freedrive exit" primitive after ~/torque_off. Anti-jolt
        first writes Goal_Position = Present_Position per servo so the
        servo's PID error is zero when torque comes back, preventing the
        arm from snapping to whatever Goal_Position was last commanded
        (which may have drifted during the freedrive period if any
        publisher wrote /joint_commands while torque was off).
        """
        if self.follower_serial is None:
            response.success = False
            response.message = (
                f"follower_serial not connected (port="
                f"{self.follower_port_name}); nothing to torque-on")
            self.get_logger().warn(response.message)
            return response

        REG_TORQUE_ENABLE = 40
        try:
            with self.follower_write_lock:
                # Anti-jolt: Goal_Position ← Present_Position per servo.
                for i in range(len(self.joint_names)):
                    sid = i + 1
                    present = self._read_servo_position(self.follower_serial, sid)
                    if present is not None:
                        self._write_servo_goal(self.follower_serial, sid, present)
                    time.sleep(self.inter_servo_delay_s)
                # Torque on per servo.
                for i in range(len(self.joint_names)):
                    self._write_servo_byte(
                        self.follower_serial, i + 1, REG_TORQUE_ENABLE, 1)
                    time.sleep(self.inter_servo_delay_s)
            response.success = True
            response.message = (
                f"torque ON on {self.follower_port_name} (6 servos, "
                f"anti-jolt applied — arm holds current pose)")
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"torque_on failed: {exc}"
            self.get_logger().error(response.message)
        return response

    # --------------------------------------------------------------- per-joint freedrive + recenter
    def _resolve_target_joint(self):
        """Read target_joint_name param + map to (name, servo_id).
        Returns (name, sid) on success, (None, error_message) on failure."""
        name = str(self.get_parameter('target_joint_name').value).strip()
        if not name:
            return None, ("target_joint_name param is empty — set first: "
                          "ros2 param set /joint_state_reader "
                          "target_joint_name <joint>")
        if name not in self.joint_names:
            return None, (f"target_joint_name={name!r} not in "
                          f"{self.joint_names}")
        return (name, self.joint_names.index(name) + 1), None

    def _read_servo_word_signed(self, ser, sid, addr):
        """Read 2-byte signed register (e.g. Homing_Offset reg 31).
        Convention: top-bit = sign, mirrors arm_reset.py's read_signed_word."""
        try:
            payload = [sid, 4, 0x02, addr, 2]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.reset_input_buffer()
            ser.write(packet)
            time.sleep(0.003)
            r = ser.read(8)
            if (len(r) < 8 or r[0] != 0xFF or r[1] != 0xFF
                    or r[2] != sid or r[4] != 0):
                return None
            raw = struct.unpack('<H', r[5:7])[0]
            return raw - 65536 if raw >= 32768 else raw
        except Exception:
            return None

    def _write_servo_word(self, ser, sid, addr, value):
        """Write 2-byte little-endian register. Negative values mask
        to uint16 so signed Homing_Offset round-trips correctly."""
        try:
            v = value & 0xFFFF
            lo = v & 0xFF
            hi = (v >> 8) & 0xFF
            payload = [sid, 5, 0x03, addr, lo, hi]
            packet = bytes([0xFF, 0xFF] + payload + [self._checksum(payload)])
            ser.reset_input_buffer()
            ser.write(packet)
            time.sleep(0.005)
            r = ser.read(6)
            return (len(r) >= 6 and r[0] == 0xFF and r[1] == 0xFF
                    and r[2] == sid and r[4] == 0)
        except Exception:
            return False

    def _on_freedrive_one(self, request, response):
        """Disengage torque on a single servo (the joint named in
        target_joint_name param). Other 5 servos stay torqued, holding
        the rest of the arm rigid while the user hand-positions one
        joint. Pair with ~/recenter_one (sets new zero) or
        ~/cancel_freedrive_one (re-engage without changing zero)."""
        if self.follower_serial is None:
            response.success = False
            response.message = "follower_serial not connected"
            return response
        resolved, err = self._resolve_target_joint()
        if resolved is None:
            response.success = False
            response.message = err
            return response
        joint, sid = resolved
        try:
            with self.follower_write_lock:
                ok = self._write_servo_byte(self.follower_serial, sid, 40, 0)
            if not ok:
                response.success = False
                response.message = (
                    f"{joint} (servo {sid}) torque-off bus write failed")
                return response
            response.success = True
            response.message = (
                f"{joint} (servo {sid}) torque OFF — back-drivable. "
                f"Hand-position then call ~/recenter_one to set new zero, "
                f"or ~/cancel_freedrive_one to re-engage with no change.")
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"freedrive_one failed: {e}"
            self.get_logger().error(response.message)
        return response

    def _on_recenter_one(self, request, response):
        """Snapshot the current physical pose as the new zero-reference
        for the joint named in target_joint_name. Procedure:

          1. read Present_Position (post-Homing_Offset, signed about 2048)
          2. read current Homing_Offset (reg 31, signed int16)
          3. delta_ticks = Present_Position - 2048
             new_Homing_Offset = old_Homing_Offset + delta_ticks
          4. write new Homing_Offset → firmware reinterprets, next
             Present_Position read returns ~2048
          5. anti-jolt: re-read Present, write Goal_Position = Present
          6. Torque_Enable=1
          7. update lerobot calibration JSON (homing_offset += delta_ticks)
             so the change persists across stack restarts

        Result: /joint_states reports ~0° for this joint at the user's
        current physical position. Other joints unchanged.
        """
        if self.follower_serial is None:
            response.success = False
            response.message = "follower_serial not connected"
            return response
        resolved, err = self._resolve_target_joint()
        if resolved is None:
            response.success = False
            response.message = err
            return response
        joint, sid = resolved

        REG_HOMING_OFFSET = 31
        REG_TORQUE_ENABLE = 40

        try:
            with self.follower_write_lock:
                ser = self.follower_serial
                # 1+2: snapshot present + homing
                present = self._read_servo_position(ser, sid)
                old_homing = self._read_servo_word_signed(
                    ser, sid, REG_HOMING_OFFSET)
                if present is None or old_homing is None:
                    response.success = False
                    response.message = (
                        f"{joint}: read Present/Homing failed "
                        f"(present={present}, old_homing={old_homing})")
                    return response

                # 3: compute delta + new_homing
                delta_ticks = present - 2048
                new_homing = old_homing + delta_ticks

                # Sanity: Feetech Homing_Offset signed int16 range
                # is roughly ±2047 in practical use. Reject big shifts
                # (likely a misuse or readout glitch).
                if abs(new_homing) > 2047:
                    response.success = False
                    response.message = (
                        f"{joint}: new_Homing_Offset={new_homing} out of "
                        f"safe range. delta_ticks={delta_ticks}, "
                        f"old_homing={old_homing}. Re-run lerobot-calibrate "
                        f"and recheck.")
                    return response

                # 4: write new homing offset
                ok = self._write_servo_word(
                    ser, sid, REG_HOMING_OFFSET, new_homing)
                if not ok:
                    response.success = False
                    response.message = (
                        f"{joint}: write Homing_Offset failed")
                    return response
                time.sleep(0.05)  # let firmware re-apply

                # 5: anti-jolt
                new_present = self._read_servo_position(ser, sid)
                if new_present is not None:
                    self._write_servo_goal(ser, sid, new_present)

                # 6: torque on
                self._write_servo_byte(ser, sid, REG_TORQUE_ENABLE, 1)

            # 7: persist to lerobot JSON (after releasing the bus lock —
            # JSON I/O is filesystem, not bus, so it's fine outside).
            json_status = "skipped"
            try:
                json_status = self._update_lerobot_json_homing(
                    joint, delta_ticks)
            except Exception as je:
                json_status = f"FAILED: {je}"
                self.get_logger().warn(
                    f"lerobot JSON update failed for {joint}: {je}")

            delta_deg = delta_ticks * 360.0 / 4096.0
            response.success = True
            response.message = (
                f"{joint} (servo {sid}) recentered: "
                f"Homing_Offset {old_homing}→{new_homing} "
                f"(Δ{delta_ticks:+d} ticks ≈ {delta_deg:+.2f}°). "
                f"Torque on with anti-jolt. JSON: {json_status}")
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"recenter_one failed: {e}"
            self.get_logger().error(response.message)
        return response

    def _on_cancel_freedrive_one(self, request, response):
        """Re-engage torque on the target joint with anti-jolt, WITHOUT
        changing Homing_Offset. Use to abort a freedrive_one that the
        user changed their mind about."""
        if self.follower_serial is None:
            response.success = False
            response.message = "follower_serial not connected"
            return response
        resolved, err = self._resolve_target_joint()
        if resolved is None:
            response.success = False
            response.message = err
            return response
        joint, sid = resolved
        REG_TORQUE_ENABLE = 40
        try:
            with self.follower_write_lock:
                ser = self.follower_serial
                present = self._read_servo_position(ser, sid)
                if present is not None:
                    self._write_servo_goal(ser, sid, present)
                self._write_servo_byte(ser, sid, REG_TORQUE_ENABLE, 1)
            response.success = True
            response.message = (
                f"{joint} (servo {sid}) freedrive cancelled — torque on, "
                f"Homing_Offset unchanged.")
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"cancel_freedrive_one failed: {e}"
            self.get_logger().error(response.message)
        return response

    def _update_lerobot_json_homing(self, joint_name, delta_ticks):
        """Add delta_ticks to homing_offset for the named joint in the
        most-recent lerobot calibration JSON. Returns a short status
        string for service response messages.

        Mirrors arm_trim.py's update_lerobot_json so EEPROM + JSON
        stay consistent regardless of which path made the change.
        """
        import json
        import os
        cal_dir = os.path.expanduser(
            '~/.cache/huggingface/lerobot/calibration/'
            'robots/so101_follower/')
        if not os.path.isdir(cal_dir):
            return f"calib dir missing: {cal_dir}"
        json_files = [f for f in os.listdir(cal_dir) if f.endswith('.json')]
        if not json_files:
            return f"no .json in {cal_dir}"
        path = os.path.join(cal_dir, sorted(json_files)[-1])
        with open(path, 'r') as f:
            cal = json.load(f)
        if joint_name not in cal:
            return f"{joint_name!r} not in JSON keys ({list(cal.keys())})"
        old = int(cal[joint_name].get('homing_offset', 0))
        cal[joint_name]['homing_offset'] = old + delta_ticks
        # Backup BEFORE mutation (preserve old state on disk too).
        bak = path + '.bak'
        try:
            with open(bak, 'w') as f:
                json.dump({**cal, joint_name: {
                    **cal[joint_name],
                    'homing_offset': old,
                }}, f, indent=2)
        except Exception:
            pass  # best-effort backup
        with open(path, 'w') as f:
            json.dump(cal, f, indent=2)
        return f"ok ({joint_name}.homing_offset {old}→{old + delta_ticks})"

    # ---------------------------------------------------------------- main loop
    def read_and_publish(self):
        self._connect_serials(force=False)
        self.total_cycles += 1

        leader_ticks = None
        follower_ticks = None

        if self.leader_serial is not None:
            leader_ticks = self._read_all(
                self.leader_serial, self.read_errors_leader, self.last_leader_ticks)
            if leader_ticks is None:
                self._close_serial('leader_serial')
            else:
                self.last_leader_ticks = leader_ticks

        if self.follower_serial is not None:
            follower_ticks = self._read_all(
                self.follower_serial, self.read_errors_follower, self.last_follower_ticks)
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
