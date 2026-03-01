#!/usr/bin/env python3
"""
Compute the Cartesian workspace bounding box for SO-ARM101 via Monte Carlo FK sampling.

Parses the URDF directly and computes FK with numpy (no ROS service needed).
Samples random joint configurations, computes FK for each, and extracts the
axis-aligned bounding box of reachable EE positions.

Outputs a YAML file that the GUI loads to set spinbox limits.

Usage:
    python3 compute_workspace.py                          # default 200k samples
    python3 compute_workspace.py --samples 500000
    python3 compute_workspace.py --margin 0.02
    ros2 run so_arm101_control compute_workspace          # also works as ROS2 entry point
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# URDF kinematic chain: base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper
#
# Each joint is defined by its origin (xyz, rpy) and rotates about local Z.
# ---------------------------------------------------------------------------

# Joint chain in order: base -> gripper
# (joint_name, origin_xyz, origin_rpy, lower_limit, upper_limit)
KINEMATIC_CHAIN = [
    ('shoulder_pan',    (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159),
     -1.91986, 1.91986),
    ('shoulder_lift',       (-0.0303992, -0.0182778, -0.0542),  (-1.5708, -1.5708, 0.0),
     -1.74533, 1.74533),
    ('elbow_flex',       (-0.11257, -0.028, 2.46331e-16),    (-1.22818e-15, 5.75928e-16, 1.5708),
     -1.69, 1.69),
    ('wrist_flex', (-0.1349, 0.0052, 1.65232e-16),     (3.2474e-15, 2.86219e-15, -1.5708),
     -1.65806, 1.65806),
    ('wrist_roll',  (0.0, -0.0611, 0.0181),             (1.5708, 0.0486795, 3.14159),
     -2.74385, 2.84121),
]

ARM_JOINT_NAMES = [j[0] for j in KINEMATIC_CHAIN]


def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll-pitch-yaw to a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def make_transform(xyz, rpy):
    """Create a 4x4 homogeneous transform from xyz + rpy."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def rot_z(angle):
    """4x4 rotation about Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T


# TCP transform from gripper frame to jaw-tip grasp center.
# Translation + 180° Y rotation so TCP Z points opposite to gripper Z
# (TCP Z = approach direction, points down in top-down grasp).
# Reference: https://maegantucker.com/ECE4560/assignment7-so101/ (IK frame diagram)
TCP_TRANSFORM = make_transform((-0.0079, -0.000218121, -0.0981274), (0, math.pi, 0))


def forward_kinematics(joint_angles):
    """Compute TCP position (x,y,z) for given joint angles. Returns 3-element array."""
    T = np.eye(4)
    for i, (name, xyz, rpy, lo, hi) in enumerate(KINEMATIC_CHAIN):
        T = T @ make_transform(xyz, rpy) @ rot_z(joint_angles[i])
    return (T @ TCP_TRANSFORM)[:3, 3]


def forward_kinematics_batch(all_angles):
    """Vectorized FK for N samples. all_angles: (N, 5) array. Returns (N, 3)."""
    n = all_angles.shape[0]
    positions = np.empty((n, 3))

    # Pre-compute fixed transforms for each joint
    fixed_transforms = []
    for name, xyz, rpy, lo, hi in KINEMATIC_CHAIN:
        fixed_transforms.append(make_transform(xyz, rpy))

    for i in range(n):
        T = np.eye(4)
        for j in range(len(KINEMATIC_CHAIN)):
            T = T @ fixed_transforms[j] @ rot_z(all_angles[i, j])
        positions[i] = (T @ TCP_TRANSFORM)[:3, 3]

    return positions


# ---------------------------------------------------------------------------
# Geometric IK constants — derived from FK trace and verified numerically
# Reference: https://maegantucker.com/ECE4560/assignment7-so101/
# Reference: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
# ---------------------------------------------------------------------------

# Pan joint X offset from base origin (meters)
X_PAN = 0.0388353

# Shoulder-lift pivot position in arm plane (meters, from FK trace at home)
#   r = radial distance from pan axis,  h = height from base
LIFT_R = 0.0304   # = (shoulder_lift_post.x - X_PAN) at home
LIFT_H = 0.1166   # = shoulder_lift_post.z at home

# Link lengths (meters) — measured from FK trace joint positions at home
L_UPPER = 0.1160  # shoulder_lift to elbow_flex pivot
L_LOWER = 0.1350  # elbow_flex to wrist_flex pivot

# Arm-plane angles at home position (radians, from FK trace)
UPPER_HOME = math.radians(76.03)   # upper arm angle from horizontal at θ₂=0
LOWER_HOME = math.radians(2.21)    # lower arm angle from horizontal at θ₂=0,θ₃=0
HOME_BEND = UPPER_HOME - LOWER_HOME  # relative angle between links at home (73.82°)

# Wrist_flex pivot to TCP offset at gripper-down (meters, constant for all
# gripper-down configs — verified across 15 configurations in FK trace)
WF_TCP_DR = -0.0079    # radial: TCP is 7.9mm inboard of wrist_flex pivot
WF_TCP_DH = -0.15923   # height: TCP is 159.2mm below wrist_flex pivot

# Gripper-down constraint: θ₂ + θ₃ + θ₄ = 90° (verified across all configs)
GRIPPER_DOWN_SUM = math.radians(90.0)

# Wrist-roll RPY offset from URDF (the 0.0487 rad pitch in wrist_roll joint)
WRIST_ROLL_OFFSET = math.pi / 2 - 0.0487

# Joint limits dict (for clamping)
JOINT_LIMITS = {
    'shoulder_pan':  (-1.91986, 1.91986),
    'shoulder_lift': (-1.74533, 1.74533),
    'elbow_flex':    (-1.69, 1.69),
    'wrist_flex':    (-1.65806, 1.65806),
    'wrist_roll':    (-2.74385, 2.84121),
}


def forward_kinematics_full(joint_angles):
    """Compute full 4x4 TCP transform for given joint angles.

    Unlike forward_kinematics() which returns only (x,y,z), this returns
    the complete homogeneous transform including orientation.
    """
    T = np.eye(4)
    for i, (name, xyz, rpy, lo, hi) in enumerate(KINEMATIC_CHAIN):
        T = T @ make_transform(xyz, rpy) @ rot_z(joint_angles[i])
    return T @ TCP_TRANSFORM


def _solve_2link(r_tcp, z, theta5):
    """Core 2-link IK: given radial TCP distance, height, and wrist_roll,
    solve for (theta2, theta3, theta4). Returns list of (θ₂, θ₃, θ₄) tuples."""
    # TCP-to-wrist offset: the radial component rotates with wrist_roll
    # (height component is invariant). At roll=0 the offset is -7.9mm radial;
    # at roll=±90° the offset rotates out of the arm plane.
    dr = WF_TCP_DR * math.cos(theta5)
    r_wf = r_tcp - dr
    h_wf = z - WF_TCP_DH

    r_target = r_wf - LIFT_R
    h_target = h_wf - LIFT_H

    d_sq = r_target * r_target + h_target * h_target
    d = math.sqrt(d_sq)

    if d > L_UPPER + L_LOWER or d < abs(L_UPPER - L_LOWER):
        return []

    cos_bend = (d_sq - L_UPPER * L_UPPER - L_LOWER * L_LOWER) / \
               (2.0 * L_UPPER * L_LOWER)
    cos_bend = max(-1.0, min(1.0, cos_bend))
    alpha = math.atan2(h_target, r_target)

    results = []
    for sign in [1, -1]:
        bend = sign * math.acos(cos_bend)
        delta = math.atan2(
            L_LOWER * math.sin(bend),
            L_UPPER + L_LOWER * math.cos(bend))
        phi2 = alpha + delta

        theta2 = UPPER_HOME - phi2
        theta3 = bend - HOME_BEND
        theta4 = GRIPPER_DOWN_SUM - theta2 - theta3
        results.append((theta2, theta3, theta4))
    return results


def geometric_ik(x, y, z, grasp_yaw=None):
    """Analytical IK for SO-ARM101 with gripper pointing straight down.

    Solves the 5-DOF arm using geometric decomposition:
      1. θ₁ (pan) decouples as atan2(-y, x - X_PAN)
      2. θ₅ (wrist_roll) from analytical formula with grasp_yaw
      3. Back-compute wrist_flex pivot from TCP target (roll-adjusted offset)
      4. θ₂, θ₃ (lift, elbow) solved via 2-link law of cosines
      5. θ₄ (wrist_flex) from gripper-down constraint: θ₂+θ₃+θ₄ = 90°
      6. One FK refinement step to compensate for cross-plane coupling

    Returns list of dicts (up to 2: elbow-up and elbow-down), each mapping
    joint name -> angle (radians). Returns empty list if target is unreachable.

    Args:
        x, y, z: TCP target position in base frame (meters)
        grasp_yaw: desired jaw-line direction in base frame (radians), or None
    """
    # --- Joint 1: Pan ---
    dx = x - X_PAN
    r_tcp = math.sqrt(dx * dx + y * y)
    if r_tcp < 1e-6:
        theta1 = 0.0
    else:
        theta1 = math.atan2(-y, dx)

    # --- Joint 5: Wrist roll (computed early — needed for TCP offset) ---
    if grasp_yaw is not None:
        theta5 = theta1 + grasp_yaw - WRIST_ROLL_OFFSET
    else:
        theta5 = 0.0

    # --- Solve 2-link IK ---
    arm_solutions = _solve_2link(r_tcp, z, theta5)
    if not arm_solutions:
        return []

    target = np.array([x, y, z])
    solutions = []
    for theta2, theta3, theta4 in arm_solutions:
        angles = {
            'shoulder_pan': theta1,
            'shoulder_lift': theta2,
            'elbow_flex': theta3,
            'wrist_flex': theta4,
            'wrist_roll': theta5,
        }

        # --- Check joint limits ---
        in_limits = True
        for name, val in angles.items():
            lo, hi = JOINT_LIMITS[name]
            if val < lo - 0.01 or val > hi + 0.01:
                in_limits = False
                break
            angles[name] = max(lo, min(hi, val))

        if not in_limits:
            continue

        # --- FK refinement: correct for cross-plane coupling ---
        # The arm plane has a Y offset and the TCP offset rotates with
        # wrist_roll, causing small position errors (~1-8mm). One Newton
        # step via FK typically brings error below 0.5mm.
        angle_list = [angles[n] for n in ARM_JOINT_NAMES]
        fk_pos = forward_kinematics(angle_list)
        err = target - fk_pos
        err_norm = math.sqrt(err[0]*err[0] + err[1]*err[1] + err[2]*err[2])

        if err_norm > 0.0005:  # > 0.5mm — refine
            x2, y2, z2 = x + err[0], y + err[1], z + err[2]
            dx2 = x2 - X_PAN
            r_tcp2 = math.sqrt(dx2 * dx2 + y2 * y2)
            theta1_r = math.atan2(-y2, dx2) if r_tcp2 > 1e-6 else 0.0
            theta5_r = theta1_r + grasp_yaw - WRIST_ROLL_OFFSET \
                if grasp_yaw is not None else 0.0

            refined = _solve_2link(r_tcp2, z2, theta5_r)
            if refined:
                t2r, t3r, t4r = refined[0]
                angles_r = {
                    'shoulder_pan': theta1_r,
                    'shoulder_lift': t2r,
                    'elbow_flex': t3r,
                    'wrist_flex': t4r,
                    'wrist_roll': theta5_r,
                }
                ok = True
                for name, val in angles_r.items():
                    lo, hi = JOINT_LIMITS[name]
                    if val < lo - 0.01 or val > hi + 0.01:
                        ok = False
                        break
                    angles_r[name] = max(lo, min(hi, val))
                if ok:
                    angles = angles_r

        solutions.append(angles)

    return solutions


def compute_grasp_workspace(r_step=0.005, z_step=0.005, yaw_step=0.25,
                            margin_pct=0.05):
    """Compute workspace bounds for top-down grasps using geometric_ik directly.

    Sweeps a grid of (r, z, yaw) target positions and calls geometric_ik() on
    each one. Only positions where geometric_ik returns at least one solution
    are considered reachable. This accounts for all IK constraints including
    the θ₁/θ₅ coupling to the target position and grasp yaw.

    Returns dict with keys: r_min, r_max, z_min, z_max, n_reachable, n_tested.
    """
    r_values = np.arange(0.02, 0.40, r_step)
    z_values = np.arange(-0.25, 0.15, z_step)
    yaw_values = np.arange(-math.pi, math.pi, yaw_step)

    reachable = []
    n_tested = 0

    for r in r_values:
        for z in z_values:
            for yaw in yaw_values:
                n_tested += 1
                # Place target along +X axis (pan symmetry makes angle irrelevant)
                x, y = float(r), 0.0
                sols = geometric_ik(x, y, float(z), grasp_yaw=float(yaw))
                if sols:
                    reachable.append((float(r), float(z)))
                    break  # one yaw works → this (r, z) is reachable, skip rest
            # If we already found a yaw that works, the break above exits the
            # yaw loop. If no yaw worked, this (r, z) is unreachable.

    if not reachable:
        return None

    reachable = np.array(reachable)
    r_all = reachable[:, 0]
    z_all = reachable[:, 1]

    r_min_raw = float(r_all.min())
    r_max_raw = float(r_all.max())
    z_min_raw = float(z_all.min())
    z_max_raw = float(z_all.max())

    r_extent = r_max_raw - r_min_raw
    z_extent = z_max_raw - z_min_raw
    r_margin = margin_pct * r_extent
    z_margin = margin_pct * z_extent

    return {
        'r_min': r_min_raw + r_margin,
        'r_max': r_max_raw - r_margin,
        'z_min': z_min_raw + z_margin,
        'z_max': z_max_raw - z_margin,
        'r_min_raw': r_min_raw,
        'r_max_raw': r_max_raw,
        'z_min_raw': z_min_raw,
        'z_max_raw': z_max_raw,
        'n_reachable': len(reachable),
        'n_tested': n_tested,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description='Compute SO-ARM101 workspace bounds')
    parser.add_argument('--samples', type=int, default=200000,
                        help='Number of random joint samples (default: 200000)')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Safety margin as fraction (default: 0.05 = 5%%)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output YAML path')
    parsed, _ = parser.parse_known_args(args)

    num_samples = parsed.samples
    margin_pct = parsed.margin

    print(f'SO-ARM101 Workspace Computation')
    print(f'  Samples: {num_samples}')
    print(f'  Margin:  {margin_pct*100:.0f}%')
    print()

    # Verify FK against known home position
    home_pos = forward_kinematics([0.0] * 5)
    print(f'Home EE position (all zeros): ({home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f})')
    print()

    # ---------------------------------------------------------------
    # General workspace (all orientations)
    # ---------------------------------------------------------------
    print(f'Sampling {num_samples} random joint configurations...')
    lower = np.array([j[3] for j in KINEMATIC_CHAIN])
    upper = np.array([j[4] for j in KINEMATIC_CHAIN])
    samples = np.random.uniform(lower, upper, size=(num_samples, len(KINEMATIC_CHAIN)))

    # Add home and single-joint extremes
    extras = [np.zeros(5)]
    for i in range(5):
        for val in [lower[i], upper[i]]:
            s = np.zeros(5)
            s[i] = val
            extras.append(s)
    samples = np.vstack([samples, np.array(extras)])

    print(f'Computing FK for {len(samples)} configurations...')
    t0 = time.monotonic()
    positions = forward_kinematics_batch(samples)
    elapsed = time.monotonic() - t0
    print(f'Done in {elapsed:.2f}s ({len(samples)/elapsed:.0f} FK/s)')
    print()

    # Bounding box
    raw_min = positions.min(axis=0)
    raw_max = positions.max(axis=0)

    margin = margin_pct * (raw_max - raw_min)
    safe_min = raw_min + margin
    safe_max = raw_max - margin

    # Spherical bounds
    dists = np.linalg.norm(positions, axis=1)
    r_min = dists.min()
    r_max = dists.max()

    print('=== Workspace Bounding Box (raw) ===')
    print(f'  X: [{raw_min[0]:.4f}, {raw_max[0]:.4f}]')
    print(f'  Y: [{raw_min[1]:.4f}, {raw_max[1]:.4f}]')
    print(f'  Z: [{raw_min[2]:.4f}, {raw_max[2]:.4f}]')
    print(f'  Radius: [{r_min:.4f}, {r_max:.4f}]')
    print()
    print(f'=== Workspace Bounding Box ({margin_pct*100:.0f}% margin) ===')
    print(f'  X: [{safe_min[0]:.4f}, {safe_max[0]:.4f}]')
    print(f'  Y: [{safe_min[1]:.4f}, {safe_max[1]:.4f}]')
    print(f'  Z: [{safe_min[2]:.4f}, {safe_max[2]:.4f}]')
    print()

    # ---------------------------------------------------------------
    # Grasp workspace (top-down only — computed via geometric_ik)
    # ---------------------------------------------------------------
    print(f'Computing grasp workspace via geometric IK sweep '
          f'(θ₂+θ₃+θ₄={math.degrees(GRIPPER_DOWN_SUM):.0f}° constraint)...')
    t0 = time.monotonic()
    grasp = compute_grasp_workspace(margin_pct=margin_pct)
    elapsed = time.monotonic() - t0

    if grasp is None:
        print('ERROR: No reachable grasp configurations found!')
        grasp_yaml = ""
    else:
        print(f'  Reachable: {grasp["n_reachable"]}/{grasp["n_tested"]} (r,z) points')
        print(f'  Done in {elapsed:.2f}s')
        print()
        print('=== Grasp Workspace (top-down, raw) ===')
        print(f'  Radius: [{grasp["r_min_raw"]:.4f}, {grasp["r_max_raw"]:.4f}]')
        print(f'  Z:      [{grasp["z_min_raw"]:.4f}, {grasp["z_max_raw"]:.4f}]')
        print()
        print(f'=== Grasp Workspace (top-down, {margin_pct*100:.0f}% margin) ===')
        print(f'  Radius: [{grasp["r_min"]:.4f}, {grasp["r_max"]:.4f}]')
        print(f'  Z:      [{grasp["z_min"]:.4f}, {grasp["z_max"]:.4f}]')
        print()

        grasp_yaml = (
            "\n"
            f"# Grasp workspace: top-down geometric IK (θ₂+θ₃+θ₄={math.degrees(GRIPPER_DOWN_SUM):.0f}°)\n"
            f"# Computed by sweeping (r, z, yaw) grid through geometric_ik()\n"
            f"# {grasp['n_reachable']} reachable (r,z) points out of {grasp['n_tested']} tested\n"
            "grasp_workspace_bounds:\n"
            f"  r_min: {grasp['r_min']:.4f}\n"
            f"  r_max: {grasp['r_max']:.4f}\n"
            f"  z_min: {grasp['z_min']:.4f}\n"
            f"  z_max: {grasp['z_max']:.4f}\n"
            "\n"
            "grasp_workspace_bounds_raw:\n"
            f"  r_min: {grasp['r_min_raw']:.4f}\n"
            f"  r_max: {grasp['r_max_raw']:.4f}\n"
            f"  z_min: {grasp['z_min_raw']:.4f}\n"
            f"  z_max: {grasp['z_max_raw']:.4f}\n"
        )

    # Write YAML
    yaml_content = (
        "# SO-ARM101 workspace bounding box\n"
        f"# Computed from {len(samples)} FK samples "
        f"({margin_pct*100:.0f}% safety margin)\n"
        f"# Home EE: ({home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f})\n"
        "#\n"
        "# Regenerate: ros2 run so_arm101_control compute_workspace\n"
        "\n"
        "workspace_bounds:\n"
        f"  x_min: {safe_min[0]:.4f}\n"
        f"  x_max: {safe_max[0]:.4f}\n"
        f"  y_min: {safe_min[1]:.4f}\n"
        f"  y_max: {safe_max[1]:.4f}\n"
        f"  z_min: {safe_min[2]:.4f}\n"
        f"  z_max: {safe_max[2]:.4f}\n"
        f"  r_min: {r_min:.4f}\n"
        f"  r_max: {r_max:.4f}\n"
        "\n"
        "workspace_bounds_raw:\n"
        f"  x_min: {raw_min[0]:.4f}\n"
        f"  x_max: {raw_max[0]:.4f}\n"
        f"  y_min: {raw_min[1]:.4f}\n"
        f"  y_max: {raw_max[1]:.4f}\n"
        f"  z_min: {raw_min[2]:.4f}\n"
        f"  z_max: {raw_max[2]:.4f}\n"
    ) + grasp_yaml

    # Determine output path
    output_path = parsed.output
    if output_path is None:
        # Try source tree config dir
        src_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', '..',
            'src', 'so_arm101_control', 'config')
        if os.path.isdir(os.path.join(src_dir, '..')):
            output_path = os.path.join(src_dir, 'workspace_bounds.yaml')
        else:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'workspace_bounds.yaml')

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        print(f'Written to: {os.path.abspath(output_path)}')
    except (OSError, IOError) as e:
        print(f'Cannot write to {output_path}: {e}')
        print()
        print(yaml_content)

    # Also write to the package source directory for symlink installs
    src_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'workspace_bounds.yaml')
    if os.path.abspath(src_yaml) != os.path.abspath(output_path):
        try:
            with open(src_yaml, 'w') as f:
                f.write(yaml_content)
            print(f'Also written to: {os.path.abspath(src_yaml)}')
        except (OSError, IOError):
            pass


if __name__ == '__main__':
    main()
