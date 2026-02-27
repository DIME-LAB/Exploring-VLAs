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
    ('Rotation',    (0.0207909, -0.0230745, 0.0948817), (-3.14159, 6.03684e-16, 1.5708),
     -1.91986, 1.91986),
    ('Pitch',       (-0.0303992, -0.0182778, -0.0542),  (-1.5708, -1.5708, 0.0),
     -1.74533, 1.74533),
    ('Elbow',       (-0.11257, -0.028, 2.46331e-16),    (-1.22818e-15, 5.75928e-16, 1.5708),
     -1.74533, 1.5708),
    ('Wrist_Pitch', (-0.1349, 0.0052, 1.65232e-16),     (3.2474e-15, 2.86219e-15, -1.5708),
     -1.65806, 1.65806),
    ('Wrist_Roll',  (0.0, -0.0611, 0.0181),             (1.5708, -9.38083e-08, 3.14159),
     -2.79253, 2.79253),
]

# Fixed transform from Wrist_Roll child (gripper) frame — this is the EE frame
# No additional offset needed; the gripper link origin IS the EE point.

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


def forward_kinematics(joint_angles):
    """Compute EE position (x,y,z) for given joint angles. Returns 3-element array."""
    T = np.eye(4)
    for i, (name, xyz, rpy, lo, hi) in enumerate(KINEMATIC_CHAIN):
        T = T @ make_transform(xyz, rpy) @ rot_z(joint_angles[i])
    return T[:3, 3]


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
        positions[i] = T[:3, 3]

    return positions


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

    # Generate random joint samples
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
    )

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


if __name__ == '__main__':
    main()
