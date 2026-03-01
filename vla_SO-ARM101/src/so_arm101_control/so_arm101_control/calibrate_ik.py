#!/usr/bin/env python3
"""
Calibrate geometric IK constants from the FK chain.

Derives all arm-plane geometry constants by running FK at specific joint
configurations and measuring joint positions / distances / angles. This
replaces the manual ad-hoc process of running FK traces and reading off
values — just re-run this script whenever the URDF changes.

The output is a YAML file that can be loaded by compute_workspace.py,
and also printed as Python constants for copy-paste verification.

Usage:
    python3 calibrate_ik.py                     # derive + verify + save
    python3 calibrate_ik.py --output /tmp/ik.yaml
    ros2 run so_arm101_control calibrate_ik     # ROS2 entry point

Reference: https://maegantucker.com/ECE4560/assignment7-so101/
Reference: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
"""

import argparse
import math
import os
import sys

import numpy as np

# Import FK chain from compute_workspace (same package)
sys.path.insert(0, os.path.dirname(__file__))
from compute_workspace import (
    KINEMATIC_CHAIN, TCP_TRANSFORM, ARM_JOINT_NAMES, JOINT_LIMITS,
    make_transform, rot_z, forward_kinematics, forward_kinematics_full,
)


# ---------------------------------------------------------------------------
# FK helpers
# ---------------------------------------------------------------------------

def fk_joint_positions(joint_angles):
    """Compute the 3D position of each joint pivot (before and after rotation).

    Returns dict mapping e.g. 'shoulder_pan_pre', 'shoulder_pan_post', ..., 'tcp'
    to 3-element numpy arrays.
    """
    positions = {}
    T = np.eye(4)
    for i, (name, xyz, rpy, lo, hi) in enumerate(KINEMATIC_CHAIN):
        T = T @ make_transform(xyz, rpy)
        positions[f'{name}_pre'] = T[:3, 3].copy()
        T = T @ rot_z(joint_angles[i])
        positions[f'{name}_post'] = T[:3, 3].copy()
    T_tcp = T @ TCP_TRANSFORM
    positions['tcp'] = T_tcp[:3, 3].copy()
    return positions


# ---------------------------------------------------------------------------
# Constant derivation
# ---------------------------------------------------------------------------

def derive_constants():
    """Derive all geometric IK constants from FK at home + gripper-down configs.

    Returns a dict of constant_name -> value (all in SI: meters, radians).
    """
    # --- 1. FK at home (all joints = 0) ---
    home = [0.0] * 5
    pos = fk_joint_positions(home)

    # X_PAN: pan joint X position in base frame
    x_pan = pos['shoulder_pan_pre'][0]

    # LIFT_R, LIFT_H: shoulder-lift pivot in arm plane
    lift_r = pos['shoulder_lift_post'][0] - x_pan
    lift_h = pos['shoulder_lift_post'][2]

    # Elbow and wrist positions at home (arm plane coords)
    r_ef = pos['elbow_flex_pre'][0] - x_pan
    h_ef = pos['elbow_flex_pre'][2]
    r_wf = pos['wrist_flex_pre'][0] - x_pan
    h_wf = pos['wrist_flex_pre'][2]

    # L_UPPER, L_LOWER: link lengths from joint-to-joint distance
    l_upper = math.sqrt((r_ef - lift_r)**2 + (h_ef - lift_h)**2)
    l_lower = math.sqrt((r_wf - r_ef)**2 + (h_wf - h_ef)**2)

    # UPPER_HOME, LOWER_HOME: arm-plane angles from horizontal at home
    upper_home = math.atan2(h_ef - lift_h, r_ef - lift_r)
    lower_home = math.atan2(h_wf - h_ef, r_wf - r_ef)
    home_bend = upper_home - lower_home

    # --- 2. Gripper-down constraint and wrist-to-TCP offset ---
    # Sweep lift/elbow configs where TCP points straight down.
    # The constraint is lift + elbow + wf = some constant (should be ~90°).
    # The wrist_flex_pre-to-TCP offset should be constant across all configs.
    down_configs = []
    dr_values = []
    dh_values = []
    sum_values = []

    for lift_deg in [0, 15, 30, 45, 60]:
        for elbow_deg in [0, -15, -30, -45, -60]:
            # Find wrist_flex that maximizes gripper-down (TCP Z = -1)
            best_wf = None
            best_down = -2.0
            for wf_deg in range(-95, 96):
                angles = [0, math.radians(lift_deg), math.radians(elbow_deg),
                          math.radians(wf_deg), 0]
                T = forward_kinematics_full(angles)
                down = -T[2, 2]
                if down > best_down:
                    best_down = down
                    best_wf = wf_deg

            if best_down < 0.999:
                continue  # not a valid gripper-down config

            wf_deg = best_wf
            angles = [0, math.radians(lift_deg), math.radians(elbow_deg),
                      math.radians(wf_deg), 0]
            p = fk_joint_positions(angles)

            # Arm-plane wrist_flex pivot and TCP positions
            r_wfp = p['wrist_flex_pre'][0] - x_pan
            h_wfp = p['wrist_flex_pre'][2]
            r_tcp = p['tcp'][0] - x_pan
            h_tcp = p['tcp'][2]

            dr_values.append(r_tcp - r_wfp)
            dh_values.append(h_tcp - h_wfp)
            sum_values.append(lift_deg + elbow_deg + wf_deg)
            down_configs.append((lift_deg, elbow_deg, wf_deg))

    # Average the measured offsets (they should be nearly identical)
    wf_tcp_dr = float(np.mean(dr_values))
    wf_tcp_dh = float(np.mean(dh_values))
    gripper_down_sum_deg = float(np.mean(sum_values))

    # Verify consistency
    dr_std = float(np.std(dr_values))
    dh_std = float(np.std(dh_values))
    sum_std = float(np.std(sum_values))

    # --- 3. Wrist roll offset from URDF ---
    # The wrist_roll joint has rpy=(1.5708, 0.0486795, 3.14159)
    # The pitch component (0.0487 rad) creates the offset:
    #   TCP_Y_yaw = (π/2 - 0.0487) + (roll - pan)
    wrist_roll_rpy = KINEMATIC_CHAIN[4][2]  # rpy of wrist_roll joint
    wrist_roll_pitch = wrist_roll_rpy[1]    # the 0.0487 offset
    wrist_roll_offset = math.pi / 2 - wrist_roll_pitch

    constants = {
        'X_PAN': x_pan,
        'LIFT_R': round(lift_r, 4),
        'LIFT_H': round(lift_h, 4),
        'L_UPPER': round(l_upper, 4),
        'L_LOWER': round(l_lower, 4),
        'UPPER_HOME_DEG': round(math.degrees(upper_home), 2),
        'LOWER_HOME_DEG': round(math.degrees(lower_home), 2),
        'HOME_BEND_DEG': round(math.degrees(home_bend), 2),
        'WF_TCP_DR': round(wf_tcp_dr, 5),
        'WF_TCP_DH': round(wf_tcp_dh, 5),
        'GRIPPER_DOWN_SUM_DEG': round(gripper_down_sum_deg, 1),
        'WRIST_ROLL_OFFSET': round(wrist_roll_offset, 4),
        'WRIST_ROLL_PITCH': round(wrist_roll_pitch, 7),
    }

    diagnostics = {
        'n_down_configs': len(down_configs),
        'dr_std_mm': round(dr_std * 1000, 4),
        'dh_std_mm': round(dh_std * 1000, 4),
        'sum_std_deg': round(sum_std, 4),
    }

    return constants, diagnostics


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_constants(constants):
    """Verify derived constants by running geometric IK round-trips.

    Imports geometric_ik and runs it with the CURRENT constants in
    compute_workspace.py. If those differ from what we just derived,
    the user should update them.
    """
    from compute_workspace import geometric_ik

    print('\n=== Round-Trip Verification (FK ∘ IK) ===')

    test_points = [
        (0.20, 0.0, 0.05, 'front center'),
        (0.25, 0.1, 0.05, 'front right'),
        (0.25, -0.1, 0.05, 'front left'),
        (0.20, 0.0, 0.02, 'front center low'),
        (0.22, 0.08, 0.04, 'mid right'),
        (0.26, 0.0, 0.02, 'far center low'),
    ]

    all_pass = True
    for x, y, z, label in test_points:
        solutions = geometric_ik(x, y, z, grasp_yaw=0.0)
        if not solutions:
            print(f'  SKIP [{label}] ({x}, {y}, {z}): no solution')
            continue

        sol = solutions[0]
        angles = [sol[n] for n in ARM_JOINT_NAMES]
        fk_pos = forward_kinematics(angles)
        err = np.linalg.norm(fk_pos - np.array([x, y, z])) * 1000

        status = 'PASS' if err < 1.0 else 'FAIL'
        if err >= 1.0:
            all_pass = False
        print(f'  {status} [{label}] err={err:.2f}mm')

    print(f'\n  Overall: {"ALL PASS" if all_pass else "SOME FAILED"}')
    return all_pass


# ---------------------------------------------------------------------------
# Compare with current values in compute_workspace.py
# ---------------------------------------------------------------------------

def compare_with_current(constants):
    """Compare derived constants with what's currently in compute_workspace.py."""
    from compute_workspace import (
        X_PAN, LIFT_R, LIFT_H, L_UPPER, L_LOWER,
        UPPER_HOME, LOWER_HOME, HOME_BEND,
        WF_TCP_DR, WF_TCP_DH, GRIPPER_DOWN_SUM, WRIST_ROLL_OFFSET,
    )

    current = {
        'X_PAN': X_PAN,
        'LIFT_R': LIFT_R,
        'LIFT_H': LIFT_H,
        'L_UPPER': L_UPPER,
        'L_LOWER': L_LOWER,
        'UPPER_HOME_DEG': round(math.degrees(UPPER_HOME), 2),
        'LOWER_HOME_DEG': round(math.degrees(LOWER_HOME), 2),
        'HOME_BEND_DEG': round(math.degrees(HOME_BEND), 2),
        'WF_TCP_DR': WF_TCP_DR,
        'WF_TCP_DH': WF_TCP_DH,
        'GRIPPER_DOWN_SUM_DEG': round(math.degrees(GRIPPER_DOWN_SUM), 1),
        'WRIST_ROLL_OFFSET': round(WRIST_ROLL_OFFSET, 4),
    }

    print('\n=== Comparison: Derived vs Current (compute_workspace.py) ===')
    print(f'  {"Constant":<25} {"Derived":<15} {"Current":<15} {"Match?"}')
    print(f'  {"-"*65}')

    all_match = True
    for key in current:
        derived = constants.get(key)
        cur = current[key]
        if derived is None:
            continue

        # Tolerance: 0.1% or 0.0001 absolute
        if isinstance(derived, float):
            match = abs(derived - cur) < max(0.001 * abs(cur), 0.0001)
        else:
            match = derived == cur

        status = 'OK' if match else 'DIFFERS'
        if not match:
            all_match = False
        print(f'  {key:<25} {derived:<15} {cur:<15} {status}')

    if all_match:
        print('\n  All constants match — no update needed.')
    else:
        print('\n  Some constants differ — consider updating compute_workspace.py.')
    return all_match


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_yaml(constants, diagnostics, output_path):
    """Save calibration results to YAML file."""
    lines = [
        '# SO-ARM101 Geometric IK Calibration',
        '# Derived from FK chain — regenerate: python3 calibrate_ik.py',
        f'# Configs tested: {diagnostics["n_down_configs"]}',
        f'# DR consistency: std={diagnostics["dr_std_mm"]:.4f}mm',
        f'# DH consistency: std={diagnostics["dh_std_mm"]:.4f}mm',
        f'# Sum consistency: std={diagnostics["sum_std_deg"]:.4f}°',
        '',
        'ik_constants:',
    ]
    for key, val in constants.items():
        lines.append(f'  {key}: {val}')

    text = '\n'.join(lines) + '\n'
    with open(output_path, 'w') as f:
        f.write(text)
    print(f'\nWritten to: {output_path}')


def print_python_constants(constants):
    """Print constants as Python code for copy-paste into compute_workspace.py."""
    print('\n=== Python Constants (for compute_workspace.py) ===')
    print(f'X_PAN = {constants["X_PAN"]}')
    print(f'LIFT_R = {constants["LIFT_R"]}')
    print(f'LIFT_H = {constants["LIFT_H"]}')
    print(f'L_UPPER = {constants["L_UPPER"]}')
    print(f'L_LOWER = {constants["L_LOWER"]}')
    print(f'UPPER_HOME = math.radians({constants["UPPER_HOME_DEG"]})')
    print(f'LOWER_HOME = math.radians({constants["LOWER_HOME_DEG"]})')
    print(f'HOME_BEND = UPPER_HOME - LOWER_HOME')
    print(f'WF_TCP_DR = {constants["WF_TCP_DR"]}')
    print(f'WF_TCP_DH = {constants["WF_TCP_DH"]}')
    print(f'GRIPPER_DOWN_SUM = math.radians({constants["GRIPPER_DOWN_SUM_DEG"]})')
    print(f'WRIST_ROLL_OFFSET = math.pi / 2 - {constants["WRIST_ROLL_PITCH"]}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description='Calibrate geometric IK constants from FK chain')
    parser.add_argument('--output', type=str, default=None,
                        help='Output YAML path (default: alongside compute_workspace.py)')
    parsed, _ = parser.parse_known_args(args)

    print('SO-ARM101 Geometric IK Calibration')
    print('=' * 50)

    # Derive constants from FK
    print('\nDeriving constants from FK chain...')
    constants, diagnostics = derive_constants()

    # Print results
    print(f'\n=== Derived Constants ===')
    print(f'  X_PAN            = {constants["X_PAN"]:.7f} m')
    print(f'  LIFT_R           = {constants["LIFT_R"]:.4f} m')
    print(f'  LIFT_H           = {constants["LIFT_H"]:.4f} m')
    print(f'  L_UPPER          = {constants["L_UPPER"]:.4f} m')
    print(f'  L_LOWER          = {constants["L_LOWER"]:.4f} m')
    print(f'  UPPER_HOME       = {constants["UPPER_HOME_DEG"]:.2f}°')
    print(f'  LOWER_HOME       = {constants["LOWER_HOME_DEG"]:.2f}°')
    print(f'  HOME_BEND        = {constants["HOME_BEND_DEG"]:.2f}°')
    print(f'  WF_TCP_DR        = {constants["WF_TCP_DR"]:.5f} m')
    print(f'  WF_TCP_DH        = {constants["WF_TCP_DH"]:.5f} m')
    print(f'  GRIPPER_DOWN_SUM = {constants["GRIPPER_DOWN_SUM_DEG"]:.1f}°')
    print(f'  WRIST_ROLL_OFFSET= {constants["WRIST_ROLL_OFFSET"]:.4f} rad')

    print(f'\n=== Consistency (across {diagnostics["n_down_configs"]} gripper-down configs) ===')
    print(f'  DR std: {diagnostics["dr_std_mm"]:.4f} mm')
    print(f'  DH std: {diagnostics["dh_std_mm"]:.4f} mm')
    print(f'  Sum std: {diagnostics["sum_std_deg"]:.4f}°')

    # Compare with current values
    compare_with_current(constants)

    # Print Python constants
    print_python_constants(constants)

    # Verify round-trip
    verify_constants(constants)

    # Save YAML
    output_path = parsed.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'ik_calibration.yaml')
    save_yaml(constants, diagnostics, output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
