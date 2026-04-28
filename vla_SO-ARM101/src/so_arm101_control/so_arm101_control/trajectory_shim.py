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
        self.declare_parameter('tolerance_rad', math.radians(3.0))
        self.declare_parameter('completion_timeout_s', 1.5)
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('joint_states_topic', '/joint_states')
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
        self._i_alpha = 0.2
        self._i_max = math.radians(4.0)
        self._i_deadband = math.radians(0.3)
        self._i_motion_leak = 0.25
        self._lpf_alpha = 0.3
        self._hold_dwell_s = 0.5
        if self._gc_init_ok:
            # 2 Hz: matches the gentler alpha/i_max tuning. At 5 Hz the
            # integrator was effectively faster than the servo's settle
            # time, encouraging hunting on slow-tracking joints.
            self.create_timer(1.0 / 2.0, self._integrator_tick)
            self.get_logger().info(
                f'gravity-comp inline | K_calib={self._k_calib} | '
                f'i_alpha={self._i_alpha} i_max={math.degrees(self._i_max):.1f}° '
                f'hold_dwell={self._hold_dwell_s:.2f}s')

        # One in-flight goal per group (arm, gripper). The Event is set
        # by an incoming goal to preempt the current loop; the lock then
        # serializes execute() callbacks so the new goal waits for the old
        # one to drain before taking over.
        self._arm_lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._arm_preempt = threading.Event()
        self._gripper_preempt = threading.Event()

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
            f"completion_timeout={self.completion_timeout_s:.2f}s"
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

    def _on_state(self, msg: JointState):
        with self._state_lock:
            self._latest_state = dict(zip(msg.name, msg.position))
        # 1st-order LPF for the integrator (matches gravity_comp_node).
        if self._gc_init_ok:
            for n, p in zip(msg.name, msg.position):
                prev = self._actual_lpf.get(n, p)
                self._actual_lpf[n] = (1 - self._lpf_alpha) * prev + self._lpf_alpha * p

    def _integrator_tick(self):
        """Layer-3 integrator update at 5 Hz. Only accumulates in hold mode
        (no recent motion); leaks during motion to avoid carrying tracking
        lag into the next hold."""
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
            return self._play(goal_handle, traj, preempt, label=label)
        finally:
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

        # Apply FF + i_term ONLY to joints actually in this publish.
        # Preserves the per-controller separation: arm-only publishes
        # don't touch gripper command, gripper-only don't touch arm.
        # See gravity_comp_node.py:_cycle for the same partial-publish
        # discipline.
        corrected = []
        for n, p in zip(names, positions):
            idx = self._gc_joint_idx_in_q.get(n)
            ff = 0.0
            if idx is not None and self._k_calib.get(n, 0.0) != 0:
                ff = self._k_calib[n] * float(tau[idx])
            corrected.append(p - ff + self._i_term.get(n, 0.0))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = corrected
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
