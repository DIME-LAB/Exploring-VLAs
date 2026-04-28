#!/usr/bin/env python3
"""trajectory_shim — execute MoveIt trajectories on real SO-ARM101 hardware.

Action servers (matching moveit_controllers.yaml):
  /arm_controller/follow_joint_trajectory     control_msgs/FollowJointTrajectory
  /gripper_controller/follow_joint_trajectory control_msgs/FollowJointTrajectory

Plays back trajectories at ``rate_hz`` on ``joint_commands_topic``.
``jointstatereader`` (with ``accept_command_input=true``) writes those
JointState setpoints to the Feetech follower servos.

Goal completion is hybrid:
  * Time: all waypoint times must elapse (scaled by 1/velocity_scale).
  * Position: final-waypoint joints must be within ``tolerance_rad``
    of the most recent ``/joint_states``.
  * Timeout: if tolerance isn't met within ``completion_timeout_s``
    after the trajectory ends, abort with GOAL_TOLERANCE_VIOLATED.

Parameters:
  rate_hz              (20.0)   how often to publish setpoints
  velocity_scale       (0.25)   multiplier on time_from_start.
                                 <1 plays the trajectory slower.
  tolerance_rad        (0.052)  completion tolerance per joint (3°)
  completion_timeout_s (1.5)    extra wait after trajectory ends
  joint_commands_topic (/joint_commands)
  joint_states_topic   (/joint_states)

Joint-name remap (MoveIt → URDF/jointstatereader):
  ``gripper`` → ``gripper_joint``   (config split between MoveIt and URDF)

Cancel: ``send_cancel_goal`` makes the playback loop exit on its next tick.
The last published setpoint sticks (servos remain torque-on at that pose).
For an emergency torque-off, kill the shim and restart the stack with
``read_only:=true``.

Preemption: a newer goal arriving for a group signals the in-flight goal
via a ``threading.Event``; the old playback loop sees the flag, calls
``goal_handle.canceled()``, releases the per-group lock, and the new goal
takes over. Lock-acquire timeout is 2s — if the old loop fails to drain
that fast, the new goal aborts. Designed for FK-tab streaming where a
slider can fire 20+ goals/s; old "lock.acquire(blocking=False) → reject"
behavior dropped 96% of those goals on bench testing.
"""

import bisect
import json
import math
import os
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

# Inline gravity-comp: import the math from gravity_comp_node so we don't
# duplicate it. The gravity_comp_node module still ships and is invoked
# standalone for `--calibrate` mode (re-fitting K_calib); only the daemon
# path that subscribed /joint_targets and republished /joint_commands has
# been collapsed into the shim's own publish loop here.
try:
    from so_arm101_control.gravity_comp_node import (
        load_pinocchio_model,
        gravity_torque,
        map_user_pose_to_q,
        JOINT_NAMES_URDF,
    )
    _GC_AVAILABLE = True
except Exception:  # pinocchio missing or import error → run as passthrough
    _GC_AVAILABLE = False
    JOINT_NAMES_URDF = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                        'wrist_flex', 'wrist_roll', 'gripper_joint']


# MoveIt's controller config names this joint 'gripper' but the URDF /
# jointstatereader / Feetech bus all call it 'gripper_joint'. Translate
# at the publish boundary so jointstatereader's name-keyed dict picks it up.
_NAME_REMAP = {
    'gripper': 'gripper_joint',
}


def _compute_hermite_tangents(times, positions):
    """Per-joint tangent velocities for cubic Hermite interpolation.

    Uses central-difference velocities for interior waypoints and PINS the
    first and last tangents to ZERO. The zero-endpoints constraint is what
    eliminates the start/stop jerk (the prior linear interp had a velocity
    step from 0 to slope at t=0 and slope to 0 at t=t_end, both of which
    excite servo PD oscillation).

    Returns a list of [J]-tuples matching `positions`.
    """
    n = len(times)
    j_dim = len(positions[0])
    if n == 1:
        return [[0.0] * j_dim]
    tangents = [[0.0] * j_dim for _ in range(n)]
    # Interior: central diff on a uniform-or-not time grid.
    for i in range(1, n - 1):
        dt_total = times[i + 1] - times[i - 1] or 1e-9
        for k in range(j_dim):
            tangents[i][k] = (positions[i + 1][k] - positions[i - 1][k]) / dt_total
    # Endpoints stay at 0 (smooth ease-in / ease-out).
    return tangents


def _interpolate(times, positions, tangents, t):
    """Cubic Hermite interp at time t across waypoints. Returns list[J].

    Hermite basis (unit-interval s in [0,1]):
        h00(s) = 2s³ - 3s² + 1     (position weight at p0)
        h10(s) = s³  - 2s² + s     (tangent weight at v0)
        h01(s) = -2s³ + 3s²        (position weight at p1)
        h11(s) = s³  -  s²         (tangent weight at v1)
    Segment-scaled tangents are v * span_seconds (Hermite math expects
    per-unit-s tangents, but we have per-second velocities, so multiply
    by the segment time to convert).
    """
    if t <= times[0]:
        return list(positions[0])
    if t >= times[-1]:
        return list(positions[-1])
    i = bisect.bisect_right(times, t) - 1
    j = i + 1
    span = times[j] - times[i] or 1e-9
    s = (t - times[i]) / span
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    out = []
    for k in range(len(positions[i])):
        p0 = positions[i][k]
        p1 = positions[j][k]
        v0 = tangents[i][k] * span
        v1 = tangents[j][k] * span
        out.append(h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1)
    return out


def _remap_names(names):
    return [_NAME_REMAP.get(n, n) for n in names]


class TrajectoryShim(Node):
    def __init__(self):
        super().__init__('trajectory_shim')

        # 50 Hz: matches the lerobot SO-101 follower's bus rate. Each Feetech
        # bus write at 6 servos × ~1.5 ms response + 5 × 2 ms inter-servo
        # delay ≈ 19 ms, fitting the 20 ms 50 Hz window. 20 Hz (the prior
        # default) made each setpoint a 50 ms step the servo PD treated as a
        # micro-impulse → visible jitter.
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('velocity_scale', 0.25)
        # Tolerance back to 3° because empirical data showed the integrator
        # cannot reliably converge to sub-deadband (<1°) residuals — many
        # trajectories aborted with tolerance_violated. Combined with
        # `freeze_wire_at_t_end=True` below, the action server returns
        # "succeeded" once the arm is within 3° of target AND the wire
        # then freezes — so no post-trajectory drift can drive the wire
        # past the firmware deadband and produce visible ticks.
        self.declare_parameter('tolerance_rad', math.radians(3.0))
        self.declare_parameter('completion_timeout_s', 2.0)
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('joint_states_topic', '/joint_states')
        # Soft-start prefix: if measured /joint_states differs from
        # traj.points[0] by more than `soft_start_threshold_rad`, prepend a
        # synthetic waypoint at t=0 with positions=actual and shift the
        # original waypoints forward by `soft_start_s`. With Hermite
        # tangents=0 at the new endpoint, the first segment becomes a smooth
        # ease-in from physical actual → planner's first waypoint, killing
        # the impulse-step jerk at trajectory start. Threshold prevents
        # adding the prefix for trivially-small residuals where it'd just
        # slow motion for no benefit.
        self.declare_parameter('soft_start_s', 0.2)
        self.declare_parameter('soft_start_threshold_rad', math.radians(0.5))
        self.declare_parameter('k_calib_file',
                               os.path.expanduser(
                                   '~/.cache/so_arm101/gravity_comp/k_calib.json'))
        self.declare_parameter('gravity_comp_enabled', True)

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.velocity_scale = float(self.get_parameter('velocity_scale').value)
        self.tolerance_rad = float(self.get_parameter('tolerance_rad').value)
        self.completion_timeout_s = float(self.get_parameter('completion_timeout_s').value)
        self._cmd_topic = str(self.get_parameter('joint_commands_topic').value)
        self._state_topic = str(self.get_parameter('joint_states_topic').value)
        self._soft_start_s = float(self.get_parameter('soft_start_s').value)
        self._soft_start_threshold_rad = float(
            self.get_parameter('soft_start_threshold_rad').value)
        self._k_calib_file = str(self.get_parameter('k_calib_file').value)
        self._gc_enabled = bool(self.get_parameter('gravity_comp_enabled').value)

        cb_group = ReentrantCallbackGroup()

        self.cmd_pub = self.create_publisher(JointState, self._cmd_topic, 10)

        self._state_lock = threading.Lock()
        self._latest_state = None
        self.create_subscription(
            JointState, self._state_topic,
            self._on_state, 10, callback_group=cb_group,
        )

        # ---- Inline gravity compensation (was a separate node) ----
        # Why inline: the prior design had trajectory_shim publish to
        # /joint_targets, gravity_comp_node subscribe + republish to
        # /joint_commands. That IPC split caused (a) partial-target
        # bugs (per-controller publishes wiping joints absent from the
        # message), (b) phase jitter between two independently-timed
        # 50 Hz timers, and (c) integrator stalls because the hold-
        # detection ran on a different clock than the trajectory
        # completion-wait loop. Computing FF + i_term in the same
        # publish thread eliminates all three.
        self._gc_init_ok = self._init_gravity_comp()
        # Layer-3 integrator state (only used if FF is initialized)
        self._i_term = {n: 0.0 for n in JOINT_NAMES_URDF}
        self._actual_lpf = {}
        self._latest_target = {}
        self._last_motion_t = time.monotonic()
        # Tuning: data-driven from /tmp/motion_record_v2.jsonl analysis.
        # alpha=0.4 + i_max=8° hunted on fast motions (90°/s Set→Zero
        # case): arm overshoots 17°+, integrator winds up to its 8° clamp
        # but still leaves 9°+ static residual that doesn't decay. Dialed
        # back to stable-and-slow rather than aggressive-and-hunting.
        # On a 5s smooth Grasp Home, this still converges hold_e to ~1°.
        # Pre-continuous-publishing baseline was alpha=0.2 at 2 Hz tick →
        # effective learning rate 0.4*err per second. With continuous
        # 50 Hz publishing, the bus sees every i_term step live, so the 2 Hz
        # cadence shows up as visible "tick-tick" settling at end-of-motion
        # while the integrator catches up to the gravity offset.
        # Move the integrator to the publish rate (50 Hz) and scale alpha
        # by tick_ratio_old/new = 2/50 = 0.04 → alpha = 0.2 * 0.04 = 0.008.
        # Cumulative learning rate unchanged, but each individual i_term
        # update becomes 25× smaller → sub-perceptible per-tick movement
        # while the same 1-2s overall convergence is maintained.
        # _i_motion_leak similarly scaled (per-tick decay) so the total
        # decay-during-motion timeline matches what it was at 2 Hz.
        self._i_tick_hz = self.rate_hz  # 50 Hz integration
        # Old (2 Hz tick) alpha=0.2; preserve cumulative rate at new tick:
        #   new_alpha = old_alpha * (old_hz / new_hz) = 0.2 * (2 / 50)
        self._i_alpha = 0.2 * (2.0 / self._i_tick_hz)
        self._i_max = math.radians(4.0)
        # Deadband matched to STS3215 firmware reality: the servo has a
        # built-in ~0.88° insensitive zone (10 encoder counts) plus ~0.87°
        # of mechanical backlash, totalling ~1° of unrecoverable noise
        # floor. Setting the integrator deadband below that floor causes
        # it to wind up against errors the hardware cannot honor — wire
        # cmd drifts, hits the deadband threshold, servo lurches one step,
        # repeat. The user-visible "tick-tick-tick" post-motion settling
        # is exactly this dynamic. Tracking the firmware's own
        # insensitivity threshold lets the integrator stop accumulating
        # at residuals the platform can't resolve.
        # Source: Robo9 STS3215 measurement
        # (https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/).
        self._i_deadband = math.radians(1.0)
        # _i_motion_leak: at 2 Hz tick, 0.25 per tick = ~1s decay to ~10%.
        # At 50 Hz, scale: per_tick_decay = 1 - (1 - old) ** (old_hz/new_hz)
        #   = 1 - 0.75 ** (2/50) = 1 - 0.75^0.04 ≈ 0.0114
        self._i_motion_leak = 1.0 - ((1.0 - 0.25) ** (2.0 / self._i_tick_hz))
        self._lpf_alpha = 0.3
        self._hold_dwell_s = 0.5
        # Rate-limit i_term's contribution to the wire cmd so servo can't
        # step a full encoder LSB (0.088°) on a single integrator tick.
        # Decouples integrator update rate (used internally for fast
        # convergence math) from observable wire-cmd change rate (must be
        # sub-LSB to avoid the visible "tick-tick" stair-stepping the user
        # sees post-motion). With 0.04°/tick at 50 Hz = 2°/s effective
        # convergence — fast enough to settle in <2s for typical 3-4°
        # gravity offsets, slow enough that no individual tick crosses an
        # LSB so the servo sees a smooth gradient instead of stair-steps.
        self.declare_parameter('max_i_change_per_tick_deg', 0.04)
        self._max_i_change_per_tick = math.radians(
            float(self.get_parameter('max_i_change_per_tick_deg').value))
        # Per-joint accumulated wire-cmd contribution from i_term. Rate
        # limited toward self._i_term[n] in _publish; this is what actually
        # appears on the wire, NOT self._i_term itself (which can change
        # arbitrarily fast inside the integrator math).
        self._published_i_term = {n: 0.0 for n in JOINT_NAMES_URDF}

        # ---- Predictive residual model r_hat(q) = K_res[j] * tau_g[j] ----
        # The integrator alone made gravity correction reactive: the wire
        # cmd had to drift after t_end to learn each new pose's residual,
        # which — once we shipped continuous publishing — became a
        # user-visible smooth slide post-arrival. The fix: predict the
        # residual from gravity-torque features at every trajectory
        # waypoint, so the visible motion already includes the correction.
        # By t_end the wire is at goal - ff(goal) + r_hat(goal); we then
        # FREEZE wire output. No post-arrival drift, accurate end-pose
        # delivery (modulo K_res accuracy).
        # K_res starts at 0 (= behaves like FF-only). Online learner
        # observes the hold-converged i_term across trajectories and
        # updates K_res via LMS so future playbacks predict more
        # accurately. Persisted across restarts.
        self._k_res = {n: 0.0 for n in JOINT_NAMES_URDF}
        self._k_res_file = os.path.expanduser(
            '~/.cache/so_arm101/gravity_comp/k_res.json')
        # DISABLED BY DEFAULT after a GPT v3 review identified critical
        # holes in the online-learner signal path:
        #   1. i_term is no longer coupled to the plant (we apply r_hat,
        #      not i_term, to the wire), so the integrator's accumulated
        #      value is an open-loop drift, not a converged residual
        #      command. Feeding garbage labels to LMS produces garbage
        #      K_res that, in turn, gets injected into the live
        #      trajectory at every tick — visible as shake.
        #   2. _update_k_res_from_hold runs per-execute, so concurrent
        #      arm + gripper goals consume the same hold sample twice.
        #   3. NLMS regularization advertised in the docstring isn't
        #      actually implemented (no ||tau_g||²+ε term).
        #   4. Sample is time-misaligned: tau_g snapshotted at t_end,
        #      i_term read at next goal-accept (depends on user wait time).
        #   5. Single-sample per-joint updates are too brittle.
        # Right approach is offline calibration: move arm to N diverse
        # poses, hold each, measure settled residual command, fit K_res
        # via ridge regression, validate, THEN apply. Online adaptation
        # can come back later with a proper signal path.
        self.declare_parameter('r_hat_enabled', False)
        self.declare_parameter('r_hat_learning_rate', 0.3)
        # Freeze the wire cmd at the moment _play() exits (success or
        # abort). _idle_tick then republishes the literal frozen wire
        # values forever, bypassing FF/i_term re-evaluation, so the
        # integrator can never drive cmd past the firmware deadband and
        # produce visible LSB-snap ticks. Trade-off: small static residual
        # at hold (whatever FF + the integrator's work during the 2s
        # completion-wait managed to compensate). The empirical data
        # showed this is the only way to eliminate post-trajectory ticks
        # without inviting tolerance-violated aborts.
        self.declare_parameter('freeze_wire_at_t_end', True)
        self._r_hat_enabled = bool(
            self.get_parameter('r_hat_enabled').value)
        self._r_hat_eta = float(
            self.get_parameter('r_hat_learning_rate').value)
        self._freeze_wire_at_t_end = bool(
            self.get_parameter('freeze_wire_at_t_end').value)
        self._load_k_res()
        # When True, _idle_tick republishes the literal frozen wire values
        # instead of going through _publish (which would re-evaluate
        # FF + r_hat at each tick — and any K_res update by the online
        # learner would then bleed into the wire as drift, which is
        # exactly what we're trying to avoid).
        self._wire_frozen = False
        self._frozen_wire_cmd = {}  # joint -> last published value
        # For learner: snapshot of (target, tau_g) at the start of the
        # most recent hold, used to derive observed_residual at the
        # next goal-accept (= start of next trajectory).
        self._last_hold_target = {}
        self._last_hold_tau_g = {}
        if self._gc_init_ok:
            self.create_timer(1.0 / self._i_tick_hz, self._integrator_tick)
            self.get_logger().info(
                f'gravity-comp inline | K_calib={self._k_calib} | '
                f'i_tick={self._i_tick_hz:.0f}Hz '
                f'i_alpha={self._i_alpha:.4f} '
                f'i_motion_leak={self._i_motion_leak:.4f} '
                f'i_max={math.degrees(self._i_max):.1f}° '
                f'hold_dwell={self._hold_dwell_s:.2f}s')

        # One in-flight goal per group (arm, gripper). The Event is set
        # by an incoming goal to preempt the current loop; the lock then
        # serializes execute() callbacks so the new goal waits for the old
        # one to drain before taking over.
        self._arm_lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._arm_preempt = threading.Event()
        self._gripper_preempt = threading.Event()

        # ---- Continuous-hold publishing ----
        # Without this, the shim goes silent between goals: the servo enters
        # its own internal "static hold" mode (firmware-profiled at last
        # Goal_Position). When the next goal arrives, the firmware
        # transitions hold→motion, which briefly drops effective support
        # current → arm momentarily moves WITH gravity ("release"), then
        # P-loop catches and pulls to target. By streaming the last
        # corrected setpoint at 50 Hz forever, the servo never enters
        # quiescent hold; each new goal's first publish is just another
        # tick in an unbroken stream → no firmware mode transition →
        # no support-torque dip → no release.
        # Enabled only when a goal has actually run at least once
        # (otherwise we'd send arbitrary zeros at startup before _publish
        # has populated _latest_target with anything meaningful).
        self.declare_parameter('continuous_hold', True)
        self._continuous_hold = bool(
            self.get_parameter('continuous_hold').value)
        # Counter (not bool) so concurrent arm+gripper goals don't race —
        # _execute increments on entry, decrements on exit; idle publisher
        # stands down while count > 0.
        self._play_active = 0
        self._play_active_lock = threading.Lock()
        if self._continuous_hold:
            self.create_timer(1.0 / self.rate_hz, self._idle_tick)

        self.arm_server = ActionServer(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_arm,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb_group,
        )
        self.gripper_server = ActionServer(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            execute_callback=self._execute_gripper,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb_group,
        )

        self.get_logger().info(
            f"trajectory_shim ready | rate={self.rate_hz:.1f} Hz | "
            f"velocity_scale={self.velocity_scale:.3f} "
            f"({1.0 / max(self.velocity_scale, 1e-3):.2f}x slower than MoveIt) | "
            f"tolerance={math.degrees(self.tolerance_rad):.2f}° | "
            f"completion_timeout={self.completion_timeout_s:.2f}s | "
            f"soft_start={self._soft_start_s:.2f}s "
            f"(threshold={math.degrees(self._soft_start_threshold_rad):.2f}°)"
        )

    def _init_gravity_comp(self):
        """Load Pinocchio model + K_calib JSON. Returns True on success."""
        if not self._gc_enabled:
            self.get_logger().info('gravity_comp_enabled=false → passthrough mode')
            return False
        if not _GC_AVAILABLE:
            self.get_logger().warn(
                'pinocchio import failed → passthrough (no FF, no integrator)')
            return False
        # Resolve URDF path via ament_index
        try:
            from ament_index_python.packages import get_package_share_directory
            urdf_path = os.path.join(
                get_package_share_directory('so_arm101_description'),
                'urdf', 'so_arm101.urdf')
            self._gc_model, self._gc_data, self._gc_joint_idx_in_q = \
                load_pinocchio_model(urdf_path)
        except Exception as e:
            self.get_logger().warn(f'URDF load failed: {e} → passthrough')
            return False
        # Load K_calib (accept ridge-fit dict or legacy scalar)
        self._k_calib = {n: 0.0 for n in JOINT_NAMES_URDF}
        if os.path.exists(self._k_calib_file):
            try:
                with open(self._k_calib_file) as f:
                    loaded = json.load(f)
                for jn, val in loaded.get('k_calib', {}).items():
                    if isinstance(val, dict):
                        self._k_calib[jn] = float(val.get('K', 0.0))
                    else:
                        self._k_calib[jn] = float(val)
            except Exception as e:
                self.get_logger().warn(f'K_calib parse failed: {e}')
        else:
            self.get_logger().warn(
                f'K_calib file missing at {self._k_calib_file} → '
                f'FF=0, integrator-only')
        return True

    def _load_k_res(self):
        """Load K_res scalars from disk if available. Missing file → zeros
        (= behaves like the original FF-only path). Same JSON convention
        as k_calib.json so an ops engineer can edit by hand."""
        if not os.path.exists(self._k_res_file):
            self.get_logger().info(
                f'k_res file missing at {self._k_res_file} → '
                f'starting with K_res=0; learner will populate online')
            return
        try:
            with open(self._k_res_file) as f:
                loaded = json.load(f)
            for jn, val in loaded.get('k_res', {}).items():
                self._k_res[jn] = float(val)
            self.get_logger().info(f'loaded K_res: {self._k_res}')
        except Exception as e:
            self.get_logger().warn(f'K_res parse failed: {e}; using zeros')

    def _save_k_res(self):
        """Persist K_res to disk so the learner's progress survives
        restarts. Best-effort — failures are warned, not raised."""
        try:
            os.makedirs(os.path.dirname(self._k_res_file), exist_ok=True)
            with open(self._k_res_file, 'w') as f:
                json.dump({'k_res': self._k_res}, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f'K_res save failed: {e}')

    def _compute_tau_g(self, q_dict):
        """Run RNEA at pose q_dict and return per-joint gravity torque
        as a dict keyed by joint name. Returns {} if FF isn't initialised
        (passthrough mode)."""
        if not self._gc_init_ok:
            return {}
        # Build full q vector from partial q_dict (fall back to current
        # /joint_states for missing joints — same convention as _publish).
        full_q = dict(q_dict)
        with self._state_lock:
            state_snapshot = (dict(self._latest_state)
                              if self._latest_state else {})
        for n in JOINT_NAMES_URDF:
            if n not in full_q:
                full_q[n] = self._latest_target.get(
                    n, state_snapshot.get(n, 0.0))
        q = map_user_pose_to_q(full_q, self._gc_joint_idx_in_q,
                               self._gc_model)
        tau = gravity_torque(self._gc_model, self._gc_data, q)
        out = {}
        for n in JOINT_NAMES_URDF:
            idx = self._gc_joint_idx_in_q.get(n)
            out[n] = float(tau[idx]) if idx is not None else 0.0
        return out

    def _compute_r_hat(self, q_dict):
        """Predicted gravity-residual correction at pose q_dict.
        Returns dict joint -> r_hat value (radians)."""
        if not self._r_hat_enabled:
            return {n: 0.0 for n in JOINT_NAMES_URDF}
        tau_g = self._compute_tau_g(q_dict)
        return {n: self._k_res.get(n, 0.0) * tau_g.get(n, 0.0)
                for n in JOINT_NAMES_URDF}

    def _update_k_res_from_hold(self):
        """LMS update of K_res per joint. DISABLED by default — see
        r_hat_enabled docstring. Kept in the codebase as scaffolding for
        when the offline calibration path is added and online adaptation
        is reintroduced with a proper signal path."""
        if not self._r_hat_enabled:
            return
        if not self._gc_init_ok or not self._last_hold_target:
            return
        for n in JOINT_NAMES_URDF:
            tau_g_n = self._last_hold_tau_g.get(n, 0.0)
            if abs(tau_g_n) < 1e-3:
                # Joint has near-zero gravity load at this pose → no info
                # in this sample for K_res[n]. Skip.
                continue
            i_term_observed = self._i_term.get(n, 0.0)
            predicted = self._k_res.get(n, 0.0) * tau_g_n
            residual = i_term_observed - predicted
            # Normalised LMS update (per joint, scalar tau_g):
            update = self._r_hat_eta * residual / tau_g_n
            self._k_res[n] += update
        self.get_logger().info(
            f'K_res updated: {[f"{n}={self._k_res[n]:+.4f}" for n in JOINT_NAMES_URDF]}')
        self._save_k_res()

    def _on_state(self, msg: JointState):
        with self._state_lock:
            self._latest_state = dict(zip(msg.name, msg.position))
        # 1st-order LPF for the integrator (matches gravity_comp_node).
        if self._gc_init_ok:
            for n, p in zip(msg.name, msg.position):
                prev = self._actual_lpf.get(n, p)
                self._actual_lpf[n] = (1 - self._lpf_alpha) * prev + self._lpf_alpha * p
        # Seed _latest_target from the first /joint_states so idle
        # publishing starts immediately after the shim comes up — even
        # before any goal has run. Without this, the first motion still
        # exits a silent-bus quiescent hold and sees the firmware
        # transition jerk we're trying to eliminate.
        if not self._latest_target:
            for n, p in zip(msg.name, msg.position):
                self._latest_target[n] = p

    def _integrator_tick(self):
        """Layer-3 integrator update at self._i_tick_hz (default 50 Hz to
        match the publish rate). Only accumulates in hold mode (no recent
        motion); leaks during motion to avoid carrying tracking lag into
        the next hold. The value computed here is the *target* for
        self._i_term — the wire cmd separately rate-limits how fast the
        published contribution converges toward this target (see
        max_i_change_per_tick in _publish)."""
        if not self._latest_target:
            return
        in_hold = (time.monotonic() - self._last_motion_t) > self._hold_dwell_s
        for n, target in self._latest_target.items():
            actual = self._actual_lpf.get(n)
            if actual is None:
                continue
            err = target - actual
            if not in_hold:
                # Motion mode: leak so we don't bake transient tracking lag in.
                self._i_term[n] *= (1 - self._i_motion_leak)
                continue
            if abs(err) < self._i_deadband:
                continue
            self._i_term[n] += self._i_alpha * err
            self._i_term[n] = max(-self._i_max,
                                  min(self._i_max, self._i_term[n]))

    def _idle_tick(self):
        """Republish the wire setpoint at 50 Hz between goals.

        Two paths:
          (a) Wire frozen (post-trajectory): republish the EXACT wire
              values captured at t_end — bypassing FF/r_hat re-evaluation
              so K_res learner updates can't bleed into the wire as
              user-visible drift.
          (b) Wire not yet frozen (pre-first-goal): publish via _publish()
              so FF/r_hat get applied to the seeded /joint_states snapshot.

        Either way, the bus stays fed at 50 Hz so firmware never enters
        quiescent static-hold mode (which produces the original release
        jerk at trajectory start).
        """
        with self._play_active_lock:
            if self._play_active > 0:
                return
        if self._wire_frozen and self._frozen_wire_cmd:
            self._publish_frozen()
            return
        if not self._latest_target:
            # No state seeded yet; nothing meaningful to publish.
            return
        names = list(self._latest_target.keys())
        positions = [self._latest_target[n] for n in names]
        self._publish(names, positions, in_motion=False)

    def _execute_arm(self, goal_handle):
        return self._execute(goal_handle, self._arm_lock, self._arm_preempt, label='arm')

    def _execute_gripper(self, goal_handle):
        return self._execute(goal_handle, self._gripper_lock, self._gripper_preempt, label='gripper')

    def _execute(self, goal_handle, group_lock, preempt, *, label):
        traj = goal_handle.request.trajectory
        if not traj.joint_names or not traj.points:
            self.get_logger().warn(f"[{label}] empty trajectory; aborting")
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return r

        # Signal any in-flight goal to bow out, then wait for the lock.
        # The previous goal's _play loop checks `preempt` each tick and
        # releases the lock via its `finally`, so this acquire usually
        # succeeds within ~50 ms.
        preempt.set()
        if not group_lock.acquire(timeout=2.0):
            self.get_logger().warn(
                f"[{label}] previous goal didn't release within 2s; aborting new"
            )
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return r

        try:
            preempt.clear()
            # Tell the idle-publisher to stand down so it doesn't race with
            # the playback loop's publishes during this goal.
            with self._play_active_lock:
                self._play_active += 1
            # Online learner: at goal-accept, the integrator's i_term has
            # been accumulating during the prior hold. Use it to refine
            # K_res before the new trajectory starts. (We snapshot
            # last_hold_target / last_hold_tau_g at the END of each play,
            # below, to pair with the i_term observation.)
            self._update_k_res_from_hold()
            # Unfreeze the wire so playback _publish() can drive it. The
            # last frozen values are still in self._frozen_wire_cmd until
            # _publish overwrites them tick-by-tick.
            self._wire_frozen = False
            return self._play(goal_handle, traj, preempt, label=label)
        finally:
            with self._play_active_lock:
                self._play_active -= 1
            group_lock.release()

    def _play(self, goal_handle, traj, preempt, *, label):
        names_out = _remap_names(list(traj.joint_names))
        inv_scale = 1.0 / max(self.velocity_scale, 1e-3)

        times = [
            (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) * inv_scale
            for pt in traj.points
        ]
        positions = [list(pt.positions) for pt in traj.points]
        n_joints = len(names_out)

        if any(len(p) != n_joints for p in positions):
            self.get_logger().error(f"[{label}] waypoint joint count mismatch")
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return r

        # Soft-start prefix: bridge the gap between measured /joint_states
        # and traj.points[0]. MoveIt plans from the planner seed (= last
        # commanded pose), but Feetech servos with gravity-loaded joints
        # settle 3-5° below commanded at hold. First publish at t=0 would
        # otherwise jump cmd to traj.points[0] = old commanded, an impulse
        # step on every motion start. Hermite ease pinned at endpoint
        # tangents only smooths motion *between* waypoints, not the gap
        # from physical state to the first one — that gap is a position
        # discontinuity. Prepending a synthetic waypoint at t=0 with
        # positions=current_actual + shifting originals by soft_start_s
        # gives Hermite a smooth segment from actual → planner-seed-pose.
        # Skip when gap < threshold (saves a needless ease for trivial
        # residuals) or when /joint_states is missing (no worse than the
        # pre-fix behavior).
        if self._soft_start_s > 0.0:
            with self._state_lock:
                state_snap = (dict(self._latest_state)
                              if self._latest_state else {})
            if state_snap:
                actual0 = [state_snap.get(n, positions[0][k])
                           for k, n in enumerate(names_out)]
                max_gap = max(abs(a - p)
                              for a, p in zip(actual0, positions[0]))
                if max_gap > self._soft_start_threshold_rad:
                    self.get_logger().info(
                        f"[{label}] soft-start: gap "
                        f"{math.degrees(max_gap):.2f}° > "
                        f"{math.degrees(self._soft_start_threshold_rad):.2f}°"
                        f" → prepending {self._soft_start_s:.2f}s ease-in"
                    )
                    positions = [actual0] + positions
                    times = [0.0] + [t + self._soft_start_s for t in times]

        # Pre-compute Hermite tangents: per-joint velocity at each waypoint,
        # central-diff for interior, ZERO at first/last for smooth ease-in/out.
        # Tier-1's 51 linearly-spaced waypoints give central-diff velocities
        # equal to the segment slope, so interior motion stays linear (no
        # geometric distortion of the planner-validated path); only the
        # first/last segment changes shape — into a smooth easing curve —
        # which is exactly where the user-visible jerk lives.
        tangents = _compute_hermite_tangents(times, positions)

        dt = 1.0 / self.rate_hz
        t_end = times[-1]
        self.get_logger().info(
            f"[{label}] {len(times)} waypoints, scaled duration {t_end:.2f}s "
            f"(MoveIt asked {t_end * self.velocity_scale:.2f}s); "
            f"joints={names_out}"
        )

        # ---- Playback ----
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while True:
            if preempt.is_set():
                self.get_logger().info(f"[{label}] preempted by newer goal")
                # rclpy disallows EXECUTING → CANCELED unilaterally; that
                # transition is reserved for client-initiated cancel.
                # For server-initiated preemption we abort instead.
                goal_handle.abort()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            if goal_handle.is_cancel_requested:
                self.get_logger().info(f"[{label}] cancelled by client")
                goal_handle.canceled()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            t = self.get_clock().now().nanoseconds * 1e-9 - t0
            self._publish(names_out, _interpolate(times, positions, tangents, t))
            if t >= t_end:
                break
            time.sleep(dt)

        # ---- Completion check (position within tolerance) ----
        goal_pos = positions[-1]
        deadline = self.get_clock().now().nanoseconds * 1e-9 + self.completion_timeout_s
        max_err = float('inf')
        while self.get_clock().now().nanoseconds * 1e-9 < deadline:
            if preempt.is_set():
                self.get_logger().info(f"[{label}] preempted during completion-wait")
                goal_handle.abort()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            with self._state_lock:
                state = self._latest_state
            if state is not None:
                errs = [abs(state[n] - goal_pos[k])
                        for k, n in enumerate(names_out) if n in state]
                if errs:
                    max_err = max(errs)
                    if max_err <= self.tolerance_rad:
                        self.get_logger().info(
                            f"[{label}] reached goal "
                            f"(max_err={math.degrees(max_err):.2f}°)"
                        )
                        # Snapshot last-hold target + tau_g so the next
                        # _update_k_res_from_hold call can pair them with
                        # the integrator's converged i_term.
                        if self._gc_init_ok:
                            goal_q_dict = dict(zip(names_out, goal_pos))
                            self._last_hold_target = dict(goal_q_dict)
                            self._last_hold_tau_g = self._compute_tau_g(
                                goal_q_dict)
                        # Freeze the wire at the cmd value reached at t_end.
                        # _frozen_wire_cmd was populated by _publish on the
                        # final tick. The idle publisher will republish
                        # this exact value forever — no FF/r_hat
                        # re-evaluation, no learner-bleed, no drift.
                        if self._freeze_wire_at_t_end:
                            self._wire_frozen = True
                        goal_handle.succeed()
                        r = FollowJointTrajectory.Result()
                        r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        return r
            # Hold the final setpoint while waiting for the servos to catch up.
            # in_motion=False signals the integrator that we're at a steady
            # target — it can immediately start accumulating against residual
            # error instead of waiting for hold_dwell to elapse.
            self._publish(names_out, goal_pos, in_motion=False)
            time.sleep(dt)

        self.get_logger().warn(
            f"[{label}] tolerance violated: max_err={math.degrees(max_err):.2f}° "
            f"> {math.degrees(self.tolerance_rad):.2f}° after "
            f"{self.completion_timeout_s:.2f}s — aborting"
        )
        # Even on tolerance-violated abort, snapshot + freeze: the
        # integrator's i_term over the timeout window is still useful
        # signal for the learner, and freezing the wire avoids stranding
        # the user with an unwanted post-abort drift.
        if self._gc_init_ok:
            goal_q_dict = dict(zip(names_out, goal_pos))
            self._last_hold_target = dict(goal_q_dict)
            self._last_hold_tau_g = self._compute_tau_g(goal_q_dict)
        if self._freeze_wire_at_t_end:
            self._wire_frozen = True
        goal_handle.abort()
        r = FollowJointTrajectory.Result()
        r.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
        return r

    def _publish(self, names, positions, in_motion=True):
        """Publish a setpoint, optionally with gravity FF + i_term applied.

        in_motion=True  → playback loop tick (caller is interpolating).
                          Updates _last_motion_t so integrator stays in
                          motion-leak mode until hold_dwell elapses.
        in_motion=False → completion-wait or steady hold. Lets the
                          integrator immediately enter hold mode and
                          accumulate against the residual error.
        """
        # Cache the latest target for the integrator (regardless of FF status)
        for n, p in zip(names, positions):
            self._latest_target[n] = p
        if in_motion:
            self._last_motion_t = time.monotonic()

        if not self._gc_init_ok:
            # Passthrough — no FF, no integrator
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(names)
            msg.position = list(positions)
            self.cmd_pub.publish(msg)
            return

        # Build a full q vector for RNEA. Joints not in the current `names`
        # (e.g., gripper during arm motion) fall back to the latest cached
        # target, then to the latest /joint_states actual, then to 0. RNEA
        # needs the full q because gravity torque on shoulder_lift depends
        # on elbow_flex etc. — we MUST NOT default missing joints to 0.
        q_pose = dict(zip(names, positions))
        with self._state_lock:
            state_snapshot = dict(self._latest_state) if self._latest_state else {}
        for n in JOINT_NAMES_URDF:
            if n not in q_pose:
                q_pose[n] = self._latest_target.get(n,
                            state_snapshot.get(n, 0.0))
        q = map_user_pose_to_q(q_pose, self._gc_joint_idx_in_q, self._gc_model)
        tau = gravity_torque(self._gc_model, self._gc_data, q)

        # Apply FF + correction ONLY to joints actually in this publish.
        # Preserves the per-controller separation: arm-only publishes
        # don't touch gripper command, gripper-only don't touch arm.
        # Two correction modes (mutually exclusive):
        #   r_hat_enabled=True (currently disabled by default — see GPT v3
        #     review notes in __init__): predictive correction
        #     r_hat = K_res * tau_g(q) evaluated at the trajectory pose.
        #   r_hat_enabled=False: reactive rate-limited i_term (the
        #     pre-r_hat stable behavior). Per-tick wire shift bounded to
        #     ±max_i_change_per_tick so the servo doesn't stair-step.
        corrected = []
        for n, p in zip(names, positions):
            idx = self._gc_joint_idx_in_q.get(n)
            ff = 0.0
            if idx is not None and self._k_calib.get(n, 0.0) != 0:
                ff = self._k_calib[n] * float(tau[idx])
            if (self._r_hat_enabled and idx is not None
                    and self._k_res.get(n, 0.0) != 0):
                correction = self._k_res[n] * float(tau[idx])
            else:
                # Rate-limited i_term contribution — the well-tested
                # path, kept as the safe default while the predictive
                # learner is being redesigned.
                i_target = self._i_term.get(n, 0.0)
                i_prev = self._published_i_term.get(n, 0.0)
                delta = i_target - i_prev
                if delta > self._max_i_change_per_tick:
                    delta = self._max_i_change_per_tick
                elif delta < -self._max_i_change_per_tick:
                    delta = -self._max_i_change_per_tick
                correction = i_prev + delta
                self._published_i_term[n] = correction
            wire_val = p - ff + correction
            corrected.append(wire_val)
            # Cache for freeze (only consulted when freeze_wire_at_t_end
            # is True; harmless to cache always).
            self._frozen_wire_cmd[n] = wire_val

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = corrected
        self.cmd_pub.publish(msg)

    def _publish_frozen(self):
        """Republish the wire cmd snapshot taken at t_end. Bypasses
        FF/r_hat re-evaluation so K_res updates by the background
        learner don't bleed into the wire as drift. Servo holds at
        whatever pose the trajectory ended at — no post-arrival shift.
        """
        if not self._frozen_wire_cmd:
            return
        names = list(self._frozen_wire_cmd.keys())
        positions = [self._frozen_wire_cmd[n] for n in names]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = TrajectoryShim()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Persist the latest K_res snapshot so learner progress survives
        # restarts. _save_k_res() is also called on every learner update,
        # but we save here as a final guarantee in case the last update
        # raced with shutdown.
        try:
            node._save_k_res()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
