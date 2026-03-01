#!/usr/bin/env python3
# Reference: STL mesh analysis of moving_jaw_so101_v1.stl + URDF gripper_joint FK
"""
Calibrate jaw gap ↔ gripper_joint angle mapping from STL mesh geometry.

Derives the linear model:
    jaw_gap(m) = BASELINE_JAW_GAP + JAW_GAP_RATE * gripper_joint_angle(rad)

The moving jaw tip is found from the STL mesh (extreme -Y point in jaw link
frame = finger tip). The fixed jaw tip is the TCP. The gap is the lateral
distance between them in the gripper frame at each joint angle.

Re-run this script whenever the jaw STL or gripper URDF changes.

Usage:
    python3 calibrate_jaw.py
    python3 calibrate_jaw.py --output /tmp/jaw.yaml
    ros2 run so_arm101_control calibrate_jaw
"""

import argparse
import math
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# URDF constants (from so_arm101.urdf)
# ---------------------------------------------------------------------------

# gripper_joint: parent=gripper, child=jaw
GRIPPER_JOINT_XYZ = np.array([0.0202, 0.0188, -0.0234])
GRIPPER_JOINT_RPY = (1.5708, 0.0, 0.0)  # rpy=(π/2, 0, 0)

# jaw visual origin (mesh offset in jaw link frame)
JAW_VISUAL_XYZ = np.array([0.0, 0.0, 0.0189])
JAW_VISUAL_RPY = (0.0, 0.0, 0.0)

# tcp_joint: fixed, parent=gripper, child=tcp_link
TCP_IN_GRIPPER = np.array([-0.0079, -0.000218121, -0.0981274])

# gripper_joint limits
GRIPPER_LOWER = -0.174533  # rad (-10°)
GRIPPER_UPPER = 1.74533    # rad (100°)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rpy_to_rot(roll, pitch, yaw):
    return Rz(yaw) @ Rx(pitch) @ Rx(roll)  # URDF: Rz(y) Ry(p) Rx(r)


# ---------------------------------------------------------------------------
# STL loading
# ---------------------------------------------------------------------------

def load_stl_vertices(path):
    """Load a binary STL file and return unique vertices as Nx3 numpy array."""
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]

        # Check if this might be ASCII STL
        if n_tri == 0 or b'solid' in header[:6]:
            raise ValueError(f'{path}: ASCII STL not supported, convert to binary')

        verts = np.empty((n_tri * 3, 3), dtype=np.float32)
        for i in range(n_tri):
            f.read(12)  # normal
            for j in range(3):
                verts[i * 3 + j] = struct.unpack('<fff', f.read(12))
            f.read(2)  # attribute byte count

    return np.unique(verts.astype(np.float64), axis=0)


# ---------------------------------------------------------------------------
# Jaw tip identification
# ---------------------------------------------------------------------------

def find_jaw_tip(jaw_stl_path):
    """Find the moving jaw tip in the jaw link frame.

    The jaw finger extends in the -Y direction of the jaw link frame.
    The tip is identified as the centroid of vertices near the -Y extreme,
    which gives a robust estimate even with mesh noise.

    Returns (tip_in_link_frame, n_tip_vertices, all_vertices_in_link_frame).
    """
    verts = load_stl_vertices(jaw_stl_path)

    # Transform mesh coords to jaw link frame using visual origin offset
    verts_link = verts.copy()
    verts_link[:, 0] += JAW_VISUAL_XYZ[0]
    verts_link[:, 1] += JAW_VISUAL_XYZ[1]
    verts_link[:, 2] += JAW_VISUAL_XYZ[2]

    # Jaw tip: cluster of vertices near the -Y extreme (within 2mm)
    y_min = verts_link[:, 1].min()
    tip_mask = verts_link[:, 1] < y_min + 0.002
    tip_cluster = verts_link[tip_mask]
    tip = tip_cluster.mean(axis=0)

    return tip, len(tip_cluster), verts_link


def jaw_tip_in_gripper(tip_in_link, angle):
    """Transform jaw tip from jaw link frame to gripper frame at given angle.

    Kinematic chain: gripper -> joint_origin(xyz,rpy) -> Rz(angle) -> jaw_link
    """
    joint_R = Rx(GRIPPER_JOINT_RPY[0])  # rpy=(π/2, 0, 0)
    return GRIPPER_JOINT_XYZ + joint_R @ Rz(angle) @ tip_in_link


# ---------------------------------------------------------------------------
# Gap computation and model fitting
# ---------------------------------------------------------------------------

def compute_gap_table(tip_in_link, n_points=50):
    """Compute jaw gap at evenly spaced angles across the joint range.

    Returns arrays of (angles, gaps_lateral, gaps_total).
    gap_lateral = XY distance between jaw tip and TCP in gripper frame.
    """
    angles = np.linspace(GRIPPER_LOWER, min(GRIPPER_UPPER, math.radians(60)),
                         n_points)
    gaps_lat = np.empty(n_points)
    gaps_tot = np.empty(n_points)

    for i, angle in enumerate(angles):
        tip_g = jaw_tip_in_gripper(tip_in_link, angle)
        dx = tip_g[0] - TCP_IN_GRIPPER[0]
        dy = tip_g[1] - TCP_IN_GRIPPER[1]
        dz = tip_g[2] - TCP_IN_GRIPPER[2]
        gaps_lat[i] = math.sqrt(dx**2 + dy**2)
        gaps_tot[i] = math.sqrt(dx**2 + dy**2 + dz**2)

    return angles, gaps_lat, gaps_tot


def fit_linear_model(angles, gaps):
    """Fit gap = baseline + rate * angle. Returns (baseline, rate, r_squared)."""
    A = np.vstack([np.ones_like(angles), angles]).T
    result = np.linalg.lstsq(A, gaps, rcond=None)
    baseline, rate = result[0]

    # R² goodness of fit
    predicted = baseline + rate * angles
    ss_res = np.sum((gaps - predicted) ** 2)
    ss_tot = np.sum((gaps - gaps.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return baseline, rate, r_squared


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model(baseline, rate):
    """Verify by checking known physical constraints."""
    print('\n=== Verification ===')

    # Gap at lower limit should be small but positive
    gap_lower = baseline + rate * GRIPPER_LOWER
    print(f'  Gap at lower limit ({math.degrees(GRIPPER_LOWER):.1f}°): '
          f'{gap_lower * 1000:.1f}mm')

    # Gap at angle=0
    gap_zero = baseline
    print(f'  Gap at angle=0°: {gap_zero * 1000:.1f}mm')

    # Gap at 90° (typical full open)
    gap_90 = baseline + rate * math.radians(90)
    print(f'  Gap at 90°: {gap_90 * 1000:.1f}mm')

    # Angle needed for common object widths
    print('\n  Object grip angles:')
    for label, width_mm in [('16mm (2x2 short)', 16),
                             ('20mm (2x2/2x3)', 20),
                             ('32mm (2x4 short)', 32),
                             ('40mm (2x4 long)', 40),
                             ('64mm (2x8 long)', 64)]:
        width = width_mm / 1000
        angle = (width - baseline) / rate
        if GRIPPER_LOWER <= angle <= GRIPPER_UPPER:
            print(f'    {label}: {math.degrees(angle):.1f}° ({angle:.4f} rad)')
        else:
            print(f'    {label}: OUT OF RANGE '
                  f'(needs {math.degrees(angle):.1f}°)')

    all_ok = gap_lower > 0 and rate > 0
    print(f'\n  {"PASS" if all_ok else "FAIL"}: '
          f'gap_lower={gap_lower * 1000:.1f}mm > 0, rate={rate * 1000:.1f}mm/rad > 0')
    return all_ok


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_yaml(constants, output_path):
    """Save calibration results to YAML."""
    lines = [
        '# SO-ARM101 Jaw Gap Calibration',
        '# Derived from STL mesh analysis — regenerate: python3 calibrate_jaw.py',
        f'# Model: jaw_gap = baseline + rate * gripper_joint_angle',
        f'# R² = {constants["r_squared"]:.6f}',
        f'# Jaw tip vertices used: {constants["n_tip_vertices"]}',
        '',
        'jaw_constants:',
        f'  BASELINE_JAW_GAP: {constants["baseline"]:.6f}   '
        f'# jaw gap at angle=0 (m) = {constants["baseline"] * 1000:.2f}mm',
        f'  JAW_GAP_RATE: {constants["rate"]:.6f}       '
        f'# gap increase per radian (m/rad) = {constants["rate"] * 1000:.2f}mm/rad',
    ]
    text = '\n'.join(lines) + '\n'
    with open(output_path, 'w') as f:
        f.write(text)
    print(f'\nWritten to: {output_path}')


def print_python_constants(constants):
    """Print constants for copy-paste into control_gui.py."""
    print('\n=== Python Constants (for control_gui.py) ===')
    print(f'BASELINE_JAW_GAP = {constants["baseline"]:.4f}'
          f'           # jaw gap at gripper_joint=0 (m)')
    print(f'JAW_GAP_RATE = {constants["rate"]:.4f}'
          f'               # gap increase per radian (m/rad)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description='Calibrate jaw gap model from STL mesh geometry')
    parser.add_argument('--output', type=str, default=None,
                        help='Output YAML path (default: alongside this script)')
    parser.add_argument('--jaw-stl', type=str, default=None,
                        help='Path to moving jaw STL (auto-detected if omitted)')
    parsed, _ = parser.parse_known_args(args)

    print('SO-ARM101 Jaw Gap Calibration')
    print('=' * 50)

    # --- Find jaw STL ---
    jaw_stl = parsed.jaw_stl
    if jaw_stl is None:
        # Look relative to this script -> ../../so_arm101_description/meshes/
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(pkg_dir, '..', 'so_arm101_description', 'meshes',
                         'moving_jaw_so101_v1.stl'),
            os.path.join(pkg_dir, 'meshes', 'moving_jaw_so101_v1.stl'),
        ]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c):
                jaw_stl = c
                break
        if jaw_stl is None:
            print('ERROR: Cannot find moving_jaw_so101_v1.stl. '
                  'Use --jaw-stl to specify path.')
            return 1

    print(f'\nJaw STL: {jaw_stl}')

    # --- Find jaw tip ---
    print('\nFinding jaw tip from STL mesh...')
    tip, n_tip, verts_link = find_jaw_tip(jaw_stl)
    arm_r = math.sqrt(tip[0]**2 + tip[1]**2)
    arm_len = np.linalg.norm(tip)

    print(f'  Jaw mesh: {len(verts_link)} unique vertices')
    print(f'  Y extent: [{verts_link[:, 1].min() * 1000:.1f}, '
          f'{verts_link[:, 1].max() * 1000:.1f}]mm')
    print(f'  Tip (jaw link frame): ({tip[0]:.6f}, {tip[1]:.6f}, {tip[2]:.6f})')
    print(f'  Tip cluster: {n_tip} vertices')
    print(f'  Distance from joint axis: {arm_r * 1000:.1f}mm')
    print(f'  Distance from joint origin: {arm_len * 1000:.1f}mm')

    # --- Compute gap table ---
    print('\nComputing jaw gap vs angle...')
    angles, gaps_lat, gaps_tot = compute_gap_table(tip)

    # --- Fit linear model to lateral gap ---
    baseline, rate, r_sq = fit_linear_model(angles, gaps_lat)

    print(f'\n=== Derived Constants ===')
    print(f'  BASELINE_JAW_GAP = {baseline:.6f} m  ({baseline * 1000:.2f}mm)')
    print(f'  JAW_GAP_RATE     = {rate:.6f} m/rad ({rate * 1000:.2f}mm/rad)')
    print(f'  R²               = {r_sq:.6f}')

    # --- Print sample gap table ---
    print(f'\n=== Gap Table (sample) ===')
    print(f'  {"angle":>7} | {"lateral":>9} | {"predicted":>10} | {"error":>7}')
    print(f'  {"-" * 42}')
    for deg in [-10, -5, 0, 5, 10, 15, 20, 30, 45, 60]:
        rad = math.radians(deg)
        if rad < angles[0] or rad > angles[-1]:
            continue
        # Find closest computed point
        idx = np.argmin(np.abs(angles - rad))
        actual = gaps_lat[idx]
        pred = baseline + rate * rad
        err = abs(actual - pred)
        print(f'  {deg:>6}° | {actual * 1000:>8.1f}mm | {pred * 1000:>9.1f}mm | '
              f'{err * 1000:>6.2f}mm')

    # --- TCP position info ---
    print(f'\n=== Reference Points (gripper frame) ===')
    print(f'  TCP (fixed jaw):  ({TCP_IN_GRIPPER[0]:.6f}, '
          f'{TCP_IN_GRIPPER[1]:.6f}, {TCP_IN_GRIPPER[2]:.6f})')
    tip_g0 = jaw_tip_in_gripper(tip, 0.0)
    print(f'  Jaw tip (θ=0°):   ({tip_g0[0]:.6f}, {tip_g0[1]:.6f}, {tip_g0[2]:.6f})')
    tip_g90 = jaw_tip_in_gripper(tip, math.radians(90))
    print(f'  Jaw tip (θ=90°):  ({tip_g90[0]:.6f}, {tip_g90[1]:.6f}, {tip_g90[2]:.6f})')

    # --- Verify ---
    verify_model(baseline, rate)

    # --- Print Python constants ---
    constants = {
        'baseline': round(baseline, 4),
        'rate': round(rate, 4),
        'r_squared': round(r_sq, 6),
        'n_tip_vertices': n_tip,
    }
    print_python_constants(constants)

    # --- Save YAML ---
    output_path = parsed.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__),
                                   'jaw_calibration.yaml')
    save_yaml(constants, output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
