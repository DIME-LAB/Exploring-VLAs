#!/usr/bin/env python3
"""Layer 2 + Layer 3 — Gravity feed-forward + adaptive integral correction.

Architecture:
  - Subscribes to /joint_targets  (sensor_msgs/JointState — desired pose)
  - Subscribes to /joint_states   (current measured pose)
  - Publishes  to /joint_commands (corrected setpoint, what jsr writes to bus)

Layer 2 (gravity feed-forward, model-based):
  Use Pinocchio's RNEA(model, data, q, v=0, a=0) → returns gravity torque
  τ_g(q) per joint. Convert to position offset via per-joint K_calib_servo:
       Δθ_ff[j] = K_calib[j] · τ_g[j](q_target)
  K_calib is the inverse of the servo's effective Kp_eff (which has units
  of [Nm/rad]). We CALIBRATE K_calib empirically on first use by stepping
  through known poses and observing steady-state error.

Layer 3 (adaptive integrator, residual correction):
  Outer loop at 1-2 Hz (configurable). Catches whatever Layer 2 misses
  (model-vs-real mismatch, friction, payload):
       err = q_target - q_actual_filtered
       i_term[j] += alpha · err[j]               (deadband-gated)
       i_term ∈ [-i_max, +i_max]                 (anti-windup)
  Reset i_term on new target (or via service).

Final command:
  q_cmd = q_target + Δθ_ff(q_target) + i_term

CLI / parameters:
  --urdf, --root-link, --tip-link
  --rate-hz, --integrator-rate-hz
  --alpha-int, --i-max-deg, --deadband-deg
  --calibrate (one-shot mode: drive to a few poses, learn K_calib, save)
  --k-calib-file (load pre-learned per-joint K_calib)

In one-shot calibrate mode the node:
  1. Drives to URDF-zero (with current K_calib=0 → no FF)
  2. Holds 5s, measures steady-state error e_zero
  3. Drives to grasp_home, holds 5s, measures e_home
  4. For each gravity-loaded joint, fits K_calib so that
     K_calib[j] · τ_g(pose) ≈ e_pose_measured
     across both poses (linear least squares).
  5. Saves /tmp/k_calib.json
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES_URDF = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                    'wrist_flex', 'wrist_roll', 'gripper_joint']

POSES = {
    'zero': {n: 0.0 for n in JOINT_NAMES_URDF},
    'home': {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': math.pi / 2,
        'wrist_roll': -math.pi / 2,
        'gripper_joint': 0.0,
    },
}


def load_pinocchio_model(urdf_path):
    """Load the SO-ARM101 URDF as a pinocchio Model.

    Returns (model, data, joint_idx_in_q) where joint_idx_in_q maps
    URDF joint name → index in the q vector.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    # Map joint name → index in q vector (skip the universe joint at index 0)
    joint_idx_in_q = {}
    for jid in range(1, model.njoints):
        name = model.names[jid]
        # pinocchio q index for revolute joint = idx_q (single-DOF joints)
        joint = model.joints[jid]
        joint_idx_in_q[name] = joint.idx_q
    return model, data, joint_idx_in_q


def gravity_torque(model, data, q_full):
    """Compute joint-space gravity torque vector via Pinocchio's RNEA
    with v=0, a=0. Returns the τ vector aligned with q_full.
    """
    nq = model.nq
    nv = model.nv
    v = np.zeros(nv)
    a = np.zeros(nv)
    tau = pin.rnea(model, data, q_full, v, a)
    return tau


def map_user_pose_to_q(pose_dict, joint_idx_in_q, model):
    """Convert {URDF_joint_name: rad} dict to a full pinocchio q vector
    of length model.nq, with zeros for unspecified joints."""
    q = np.zeros(model.nq)
    for jname, val in pose_dict.items():
        if jname in joint_idx_in_q:
            idx = joint_idx_in_q[jname]
            q[idx] = val
    return q


def map_q_to_user(q, joint_idx_in_q):
    """Inverse of above: extract our 6 known joints from the q vector."""
    return {jname: float(q[idx]) for jname, idx in joint_idx_in_q.items()
            if jname in JOINT_NAMES_URDF}


def map_user_joints_to_indices(joint_idx_in_q):
    """Return list of q-indices for our 6 known joints in canonical order."""
    return [joint_idx_in_q[n] for n in JOINT_NAMES_URDF
            if n in joint_idx_in_q]


# --- Calibration: learn K_calib per joint from observed sag ---

def calibrate_k_calib(observed_errs_per_pose, predicted_torques_per_pose,
                     joint_names, lambda_ridge=1e-3):
    """Fit per-joint K_calib via ridge regression with prior K=0.

    Solves: K = argmin (||K·τ - e||² + λ·K²)
              = (τᵀτ + λ)⁻¹ · τᵀe

    Per GPT review: the previous τ-magnitude threshold was a heuristic
    band-aid. Ridge regularization is principled: λ acts as a prior
    that K should be 0 unless the data supports otherwise. Joints with
    tiny |τ| (roll axes, no gravity moment) get K_calib≈0 naturally
    because (τᵀτ + λ) ≈ λ → K → (τᵀe)/λ which is small for small e.

    Default λ=1e-3 chosen so that for τ ≥ 0.05 N·m (gravity-loaded joints),
    τᵀτ ≫ λ → behaves like LS. For τ ≤ 0.005 N·m (roll axes),
    τᵀτ ≪ λ → K dominated by prior (≈0). Smooth transition between.

    Also returns per-joint diagnostic info (residual, condition).

    Returns: dict joint → {'K': float, 'residual': float, 'tau_norm_sq': float}
    """
    k_calib = {}
    poses = list(observed_errs_per_pose.keys())
    for jidx, jname in enumerate(joint_names):
        e_vec = np.array([observed_errs_per_pose[p][jname] for p in poses])
        t_vec = np.array([predicted_torques_per_pose[p][jidx] for p in poses])
        tau_norm_sq = float(np.dot(t_vec, t_vec))
        denom = tau_norm_sq + lambda_ridge
        K = float(np.dot(t_vec, e_vec) / denom)
        residual = float(np.sqrt(np.mean((e_vec - K * t_vec) ** 2)))
        k_calib[jname] = {
            'K': K,
            'residual_rad': residual,
            'tau_norm_sq': tau_norm_sq,
            'identifiability': tau_norm_sq / lambda_ridge,
        }
    return k_calib


# --- Core node ---

class GravityCompNode(Node):
    def __init__(self, args):
        super().__init__('gravity_comp')
        self.args = args
        self.get_logger().info(f"loading URDF: {args.urdf}")
        self.model, self.data, self.joint_idx_in_q = load_pinocchio_model(args.urdf)
        self.get_logger().info(
            f"  model: nq={self.model.nq} nv={self.model.nv} "
            f"joints_in_q={list(self.joint_idx_in_q.keys())}")

        # K_calib per joint — load if file given, else zeros.
        # Schema: {'k_calib': {joint: K_or_dict}} — accept both legacy
        # scalar values and new ridge-fit dicts {'K': ..., 'residual_rad': ...}.
        self.k_calib = {n: 0.0 for n in JOINT_NAMES_URDF}
        if args.k_calib_file and os.path.exists(args.k_calib_file):
            with open(args.k_calib_file) as f:
                loaded = json.load(f)
            for jname, val in loaded.get('k_calib', {}).items():
                if isinstance(val, dict):
                    self.k_calib[jname] = float(val.get('K', 0.0))
                else:
                    self.k_calib[jname] = float(val)
            self.get_logger().info(f"loaded K_calib: {self.k_calib}")
        elif not args.calibrate:
            self.get_logger().warn(
                "no k-calib-file and not in calibrate mode — "
                "FF will be zero (passthrough)")

        # Integrator state
        self.i_term = {n: 0.0 for n in JOINT_NAMES_URDF}
        self.alpha = args.alpha_int
        self.i_max = math.radians(args.i_max_deg)
        self.deadband = math.radians(args.deadband_deg)
        self._prev_err = {n: 0.0 for n in JOINT_NAMES_URDF}

        # Hold-detection state (for hold-only integrator gating).
        # Per GPT review v2: detect motion via finite-difference of the
        # target stream, NOT via "big jump" heuristic. A slow trajectory
        # whose per-message delta is sub-threshold would otherwise look
        # like hold after 0.5 s and the integrator would accumulate
        # motion-tracking lag.
        #
        # Approach: each time _on_target() fires, finite-difference the
        # received target against the previous one and the timestamp delta
        # to estimate cmd-velocity. Track the last instant we saw motion
        # above threshold. Hold = (now - last_motion_seen) > dwell.
        self._target_motion_threshold_rad_s = math.radians(2.0)  # 2°/s
        self.hold_dwell_s = float(getattr(args, 'hold_dwell_s', 0.5))
        # Init "last motion seen" to NOW so we start in motion mode and
        # have to earn our way into hold mode. Per GPT v2 first-target bug.
        self._last_motion_seen_t = time.monotonic()
        self._last_target_t = None
        self._last_target_snapshot = None
        # Leak rates (per integrator tick at integrator_rate_hz):
        # Per GPT v2: motion_leak=0.02 was too weak (4%/s). Bumping to
        # 0.25 per tick → 50%/s decay during motion, so old i_term has
        # mostly dissipated by the time we reach the next hold.
        self.motion_leak = 0.25
        self.signflip_leak = 0.15

        # Filtered actual position (low-pass for noise rejection)
        self.actual = {}
        self.target = {}
        self.actual_lpf = {}
        self.lpf_alpha = 0.3   # 1st-order LPF coefficient

        # ROS plumbing
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 50)
        self.create_subscription(JointState, '/joint_states', self._on_state, 50)
        self.create_subscription(JointState, '/joint_targets', self._on_target, 50)

        # Outer loop timer
        self.create_timer(1.0 / args.rate_hz, self._cycle)
        # Integrator updates at slower rate
        self.create_timer(1.0 / args.integrator_rate_hz, self._integrator_update)

        self.get_logger().info(
            f"running: outer={args.rate_hz}Hz integrator={args.integrator_rate_hz}Hz "
            f"alpha={self.alpha} i_max={args.i_max_deg}° "
            f"deadband={args.deadband_deg}°")

    def _on_state(self, msg):
        for i, n in enumerate(msg.name):
            self.actual[n] = msg.position[i]
            # 1st-order low-pass for integrator
            prev = self.actual_lpf.get(n, msg.position[i])
            self.actual_lpf[n] = (1 - self.lpf_alpha) * prev + self.lpf_alpha * msg.position[i]

    def _on_target(self, msg):
        new_target = {n: msg.position[i] for i, n in enumerate(msg.name)}
        now = time.monotonic()
        # Finite-diff to estimate commanded velocity per joint, then
        # update _last_motion_seen_t if any joint exceeds the motion
        # threshold. This catches BOTH big jumps and slow continuous
        # motion (GPT v2 fix — previous big-jump heuristic missed slow
        # trajectories and let i_term accumulate during motion).
        if self._last_target_snapshot is not None and self._last_target_t is not None:
            dt = max(now - self._last_target_t, 1e-3)
            max_velocity_rad_s = max(
                (abs(new_target.get(j, 0) - self._last_target_snapshot.get(j, 0)) / dt
                 for j in new_target), default=0.0)
            if max_velocity_rad_s > self._target_motion_threshold_rad_s:
                self._last_motion_seen_t = now
        # Reset i_term on a big jump (e.g. /clear, abrupt new pose)
        if self.target:
            max_jump = max(
                (abs(new_target.get(j, 0) - self.target.get(j, 0))
                 for j in new_target), default=0.0)
            if max_jump > math.radians(0.5):
                # Reset accumulated i_term and force motion mode for dwell_s
                for j in self.i_term:
                    self.i_term[j] = 0.0
                self._last_motion_seen_t = now
        self.target = new_target
        self._last_target_snapshot = dict(new_target)
        self._last_target_t = now

    def _is_in_hold(self):
        """True if commanded velocity has been below threshold for at
        least hold_dwell_s. GPT v2: gates on motion detection, not
        absence of big jumps. Slow continuous trajectories correctly
        keep us in motion mode."""
        return (time.monotonic() - self._last_motion_seen_t) > self.hold_dwell_s

    def _integrator_update(self):
        """Hold-only integrator with leak, per GPT-flagged improvements.

        Three operating regimes per joint:
          1. NOT in hold (target recently changed):
             apply small uniform leak — don't accumulate motion-lag,
             let i_term decay so it doesn't fight the next planner move.
          2. In hold + err sign flipped from previous tick:
             apply larger leak — kills slow oscillation around target
             (limit cycle pathology).
          3. In hold + |err| > deadband + same sign as before:
             integrate normally (alpha · err) with hard clamp.

        Why each piece:
          - Hold-gating  : prevents learning transient tracking lag.
          - Sign-flip leak: detects oscillation and bleeds energy out.
          - Hard clamp   : last-resort anti-windup if integrator runs
                            away despite the gating.
        """
        if not self.target or not self.actual_lpf:
            return

        in_hold = self._is_in_hold()

        for j in JOINT_NAMES_URDF:
            if j not in self.target or j not in self.actual_lpf:
                continue
            err = self.target[j] - self.actual_lpf[j]
            prev_err = self._prev_err.get(j, 0.0)
            self._prev_err[j] = err

            if not in_hold:
                # Motion mode: bleed i_term toward zero so we don't
                # carry the previous hold's compensation into the next move.
                self.i_term[j] *= (1.0 - self.motion_leak)
                continue

            # Hold mode: detect sign-flip oscillation, leak harder if so.
            if prev_err * err < -1e-12:
                self.i_term[j] *= (1.0 - self.signflip_leak)

            if abs(err) < self.deadband:
                continue

            # Standard integration (in hold, above deadband, same-sign or first hit)
            self.i_term[j] += self.alpha * err
            # Hard clamp (final safety net)
            self.i_term[j] = max(-self.i_max, min(self.i_max, self.i_term[j]))

    def _cycle(self):
        if not self.target:
            return
        # Build full q at target. We need the FULL q for Pinocchio's RNEA
        # (gravity torque on, say, shoulder_lift depends on elbow_flex
        # too), so missing joints fall back to the last actual reading
        # rather than 0. Falling back to 0 was the pre-fix bug: arm-only
        # targets pinned gripper to mid-range, gripper-only targets
        # slammed the arm to all-zero ("robot falls down" symptom).
        target_full = dict(self.target)
        for j in JOINT_NAMES_URDF:
            if j not in target_full:
                # Prefer last actual reading; fall back to 0 only if we've
                # never seen state for this joint.
                target_full[j] = self.actual.get(j, 0.0)
        q_target = map_user_pose_to_q(target_full, self.joint_idx_in_q, self.model)
        tau_g = gravity_torque(self.model, self.data, q_target)

        # Publish ONLY the joints actually present in the incoming target.
        # jsr's _on_command_msg uses dict(zip(msg.name, msg.position)).get()
        # so absent joints are left alone — the per-controller separation
        # in trajectory_shim (arm action server vs gripper action server)
        # publishes partial JointStates, and we must preserve that
        # partition through gravity_comp instead of broadcasting a full
        # 6-joint command on every tick.
        cmd = {}
        for j in self.target.keys():
            base = self.target[j]
            ff = 0.0
            idx = self.joint_idx_in_q.get(j)
            if idx is not None and self.k_calib.get(j, 0.0) != 0:
                # K_calib was fit to: observed_error = K · τ.
                # observed_error = actual - target (positive if sag pushes UP).
                # To CANCEL that error we command target - (predicted error),
                # i.e. SUBTRACT the FF offset from target.
                ff = self.k_calib[j] * float(tau_g[idx])
            cmd[j] = base - ff + self.i_term.get(j, 0.0)

        if not cmd:
            return

        # Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(cmd.keys())
        msg.position = list(cmd.values())
        self.cmd_pub.publish(msg)


def _default_urdf():
    """Resolve the SO-ARM101 URDF from so_arm101_description's share dir."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('so_arm101_description')
        return os.path.join(share, 'urdf', 'so_arm101.urdf')
    except Exception:
        return None


def _default_k_calib_path():
    """Standard location for gravity-comp K_calib (mirrors lerobot's
    cache pattern: user-specific data lives in ~/.cache, NOT in the repo)."""
    home = os.path.expanduser('~')
    return os.path.join(home, '.cache', 'so_arm101', 'gravity_comp',
                        'k_calib.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=None,
                    help='URDF path. Default: so_arm101_description package share.')
    ap.add_argument('--rate-hz', type=float, default=20.0)
    ap.add_argument('--integrator-rate-hz', type=float, default=2.0)
    ap.add_argument('--alpha-int', type=float, default=0.3)
    ap.add_argument('--i-max-deg', type=float, default=5.0)
    ap.add_argument('--deadband-deg', type=float, default=0.10,
                    help='Per GPT v2: 0.1° = ~1 encoder tick (0.088°/tick on '
                         'Feetech 4096-tick) + small headroom. Tighter than '
                         'this risks chasing encoder quantization noise; '
                         'looser leaves visible static error.')
    ap.add_argument('--hold-dwell-s', type=float, default=0.5,
                    help='Time after target stabilizes before integrator '
                         'starts accumulating. Prevents learning transient '
                         'motion-tracking lag.')
    ap.add_argument('--k-calib-file', default=None,
                    help='Default: ~/.cache/so_arm101/gravity_comp/k_calib.json')
    ap.add_argument('--calibrate', action='store_true')
    # When invoked under `ros2 run`, ROS prepends --ros-args to sys.argv.
    # parse_known_args lets us ignore unknown ROS-specific flags.
    args, _ = ap.parse_known_args()
    if args.urdf is None:
        args.urdf = _default_urdf()
        if args.urdf is None:
            print("FAIL: --urdf not provided and so_arm101_description not "
                  "found via ament_index", file=sys.stderr)
            return 1
    if args.k_calib_file is None:
        args.k_calib_file = _default_k_calib_path()
    # Ensure parent dir exists so --calibrate can write
    os.makedirs(os.path.dirname(args.k_calib_file), exist_ok=True)

    if args.calibrate:
        calibrate_main(args)
        return 0

    rclpy.init()
    node = GravityCompNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def calibrate_main(args):
    """One-shot: drive to N poses with FF disabled, measure steady-state
    error per joint, fit K_calib, save."""
    rclpy.init()
    node = Node('gravity_comp_calibrate')
    cmd_pub = node.create_publisher(JointState, '/joint_commands', 50)
    actual = {}
    def on_state(msg):
        for i, n in enumerate(msg.name):
            actual[n] = msg.position[i]
    node.create_subscription(JointState, '/joint_states', on_state, 50)

    model, data, jidx = load_pinocchio_model(args.urdf)
    print(f"[calibrate] pinocchio model loaded: nq={model.nq}")

    # Wait for initial state
    deadline = time.time() + 5.0
    while time.time() < deadline and not actual:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not actual:
        print("FAIL: no /joint_states received")
        rclpy.shutdown()
        return 1

    # Calibration runs in two phases per pose:
    #   1. Drive smoothly current → pose (3s quintic)
    #   2. Hold for 5s; sample last 2s to compute steady-state error
    def drive_and_measure(target_pose, label):
        print(f"[calibrate] driving to {label}: "
              f"{ {k: round(math.degrees(v),1) for k,v in target_pose.items()} }")
        start = {n: actual.get(n, 0.0) for n in JOINT_NAMES_URDF}
        T_move = 3.0
        rate = 50.0
        n_steps = int(T_move * rate)
        for i in range(n_steps + 1):
            a = i / n_steps
            s = 10*a**3 - 15*a**4 + 6*a**5  # quintic
            interp = {n: start[n] + s * (target_pose[n] - start[n])
                      for n in JOINT_NAMES_URDF}
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(interp.keys())
            msg.position = list(interp.values())
            cmd_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / rate)
        # Hold + sample for 20s — long enough for true steady-state on
        # P=16 Feetech servos. Earlier 5s under-counted real sag because
        # slow servos don't fully settle in that window. Use last 30%
        # (post-15s mark) to capture truly steady values.
        T_hold = 20.0
        n_hold = int(T_hold * rate)
        samples = []
        for i in range(n_hold):
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(target_pose.keys())
            msg.position = list(target_pose.values())
            cmd_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            samples.append(dict(actual))
            time.sleep(1.0 / rate)
        # Use last 30% as steady-state (was 50% on a shorter window)
        ss = samples[int(len(samples) * 0.7):]
        ss_means = {}
        for n in JOINT_NAMES_URDF:
            vals = [s[n] for s in ss if n in s]
            ss_means[n] = float(np.mean(vals)) if vals else 0.0
        # Error = actual_ss - target  (sign: positive means actual is HIGHER than target)
        err = {n: ss_means[n] - target_pose[n] for n in JOINT_NAMES_URDF}
        print(f"  steady-state err: { {k: round(math.degrees(v),2) for k,v in err.items()} }°")
        # Predicted gravity torque at this pose
        q = map_user_pose_to_q(target_pose, jidx, model)
        tau = gravity_torque(model, data, q)
        # Convert to dict per joint
        tau_per_joint = np.zeros(len(JOINT_NAMES_URDF))
        for ji, jn in enumerate(JOINT_NAMES_URDF):
            if jn in jidx:
                tau_per_joint[ji] = tau[jidx[jn]]
        print(f"  predicted τ_g (Pinocchio): { {n: round(tau_per_joint[i],4) for i,n in enumerate(JOINT_NAMES_URDF)} } N·m")
        return err, tau_per_joint

    err_zero, tau_zero = drive_and_measure(POSES['zero'], 'zero')
    err_home, tau_home = drive_and_measure(POSES['home'], 'home')

    observed = {'zero': err_zero, 'home': err_home}
    predicted = {'zero': tau_zero, 'home': tau_home}
    k_calib = calibrate_k_calib(observed, predicted, JOINT_NAMES_URDF)
    print(f"\n[calibrate] ridge-regularized K_calib fit (rad / N·m, λ=1e-3):")
    print(f"  {'Joint':<14} {'K':>10}  {'fit_resid':>10}  {'identifiability':>15}")
    for jn, info in k_calib.items():
        K = info['K']
        res_deg = math.degrees(info['residual_rad'])
        ident = info['identifiability']
        # Convention: identifiability >> 1 means the data strongly
        # determines K. << 1 means it's pinned to the prior (≈0).
        flag = '' if ident > 10 else '  (pinned to prior)'
        print(f"  {jn:<14} {K:+10.4f}  {res_deg:>9.3f}°  {ident:>14.1f}{flag}")

    # Save (k_calib is now a dict-of-dicts with diagnostic info per joint;
    # runtime loader handles both legacy float and new dict schema).
    out = {
        'k_calib': k_calib,
        'observed_err_rad': {p: {n: float(v) for n, v in d.items()}
                             for p, d in observed.items()},
        'predicted_tau_nm': {p: {JOINT_NAMES_URDF[i]: float(v)
                                 for i, v in enumerate(tau)}
                             for p, tau in predicted.items()},
        'comment': ('Layer 2 gravity feed-forward calibration with ridge '
                    'regularization (λ=1e-3). k_calib[j].K = ridge-fit '
                    'scalar; .residual_rad = fit RMS; .identifiability = '
                    'τᵀτ/λ (>10 = data-driven, <1 = pinned to prior). '
                    'Apply: q_cmd = q_target − K · τ_g(q_target).'),
    }
    with open(args.k_calib_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved to {args.k_calib_file}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
