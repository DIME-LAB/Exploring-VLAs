"""SO-ARM101 grasp geometry. Pure Python/numpy; no ROS, MoveIt, or GUI imports.

The calibrated five-DOF solver and ik_calibration.yaml remain in compute_workspace.
Motion transports share these exact grasp/yaw/workspace functions.
"""
import math
import os
from so_arm101_control.compute_workspace import geometric_ik

ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
GRIPPER_JOINT_NAME = 'gripper_joint'
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]

JOINT_LIMITS = {
    'shoulder_pan':    (-1.91986, 1.91986),
    'shoulder_lift':       (-1.74533, 1.74533),
    'elbow_flex':       (-1.69, 1.69),
    'wrist_flex': (-1.65806, 1.65806),
    'wrist_roll':  (-2.74385, 2.84121),
    'gripper_joint':   (-0.174533, 1.74533),
}

# Calibrated home pose for sim-side data collection (Quickstart cycle +
# Record Sim tab). Derived from real-world dataset blue_sort_v1 ep7
# frame 277 shoulders+gripper convention, sim's wrist convention, and
# elbow_flex = -shoulder_lift (parallelogram cancellation for a compact
# ready pose). _cmd_grasp_home (manual button, IK tab, Real Test tab,
# etc.) keeps using the original all-zeros + sim wrist target — only
# the data-collection path retargets to this calibrated home so the
# recorded sim trajectories share a starting pose with real-world data.
def _grasp_home_data_target():
    """Joint-angle dict (radians) for the data-collection home pose.
    Lazy import of WRIST_ROLL_URDF_PITCH avoids a circular import at
    module load."""
    from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
    return {
        'shoulder_pan':  math.radians(0.874),
        'shoulder_lift': math.radians(-14.829),
        'elbow_flex':    math.radians(14.829),
        'wrist_flex':    math.pi / 2,
        'wrist_roll':    -math.pi / 2 + WRIST_ROLL_URDF_PITCH,
    }


# Jaw geometry for single-moving-jaw gripper offset compensation.
# Derived from STL mesh analysis of moving_jaw_so101_v1.stl + FK chain.
# Linear fit: jaw_gap(m) = BASELINE_JAW_GAP + JAW_GAP_RATE * gripper_joint_angle(rad)
# At angle=0 the jaws are NOT touching — there is a 16.9mm baseline gap.
BASELINE_JAW_GAP = 0.0190           # jaw gap at gripper_joint=0 (m)
JAW_GAP_RATE = 0.0749               # gap increase per radian (m/rad)
JAW_OPEN_CLEARANCE_M = 0.005        # extra jaw gap on open beyond symmetric baseline (m)
JAW_CLOSE_CLEARANCE_M = 0.0         # extra jaw gap on close beyond symmetric baseline (m)
TCP_CLEARANCE_M = 0.001             # extra IK offset beyond grip_width/2 for jaw overhang (m)

# URDF wrist_roll joint origin pitch offset (mirrors compute_workspace.py)
_WRIST_ROLL_URDF_PITCH = 0.0487
# Inherent 90° from kinematic chain (wrist_flex yaw=-π/2, wrist_roll yaw=π)
_CHAIN_BASE_ROTATION = math.pi / 2 - _WRIST_ROLL_URDF_PITCH

# Cup mesh for MoveIt collision objects
# STL is in mm, scaled 0.001 to meters (matches cup.urdf)
# Padding factor: 1.0 = 100% actual size (no inflation)
_CUP_STL_SCALE = 0.001

# Cup body height in meters. /drop_poses publishes cup BODY-CENTER pose
# (same convention as /objects_poses_sim for legos), so every consumer
# that needs the cup base or the cup rim has to add/subtract half this.
# Sim-side (soarm101-dt extension, isaac/exts/) lifts each drop_{id}
# wrapper by this/2 when publishing; the real-world equivalent is
# aruco_camera_localizer applying its marker→cup transform to land on
# the same body-center convention.
CUP_BODY_HEIGHT_M = 0.0965
# Phase 9: kept at 1.1 (10%). Tried 1.05 — OMPL validated paths cleared
# the 5%-padded cup but physical execution still knocked it (Δ40mm sweep).
# Actual trajectory tracking error + arm-mesh vs physical-arm discrepancy
# exceeds 2mm, so 5% margin is insufficient. 10% costs some valid paths
# at close-in grasp positions, but the lego-in-cup scene filter + sync
# detach + one-shot OMPL together are now the primary correctness levers;
# padding returns to its original safety-margin role.
_CUP_COLLISION_PADDING = 1.0  # 0% default pad — empirical: 2.5% inflation
# made the carried-block path clip cup planning meshes during drop_sweep
# pan-across (sub-mm collisions at wp[14]/wp[35] that triggered OMPL RNG-
# fallback non-determinism). 0% gives tier-1 deterministic clearance for
# both grasp_move approach AND drop_sweep with a held block. Failed grasp
# attempts that DO clip are correctly rejected by tier-1 and skipped by
# the recording pipeline as failed episodes — that's the safety boundary.
# Tracking-lag absorption was the original justification for 5% padding;
# the deterministic-tier path is now slow enough (3 s / 36 °/s) that
# physics tracks within the bare cup geometry.


def _normalize_grasp_yaw(yaw, pan, current_wrist_roll=None):
    """Pick yaw or yaw±π that minimizes wrist_roll motion from the current pose.

    Gripper jaws are symmetric about the grip axis, so yaw and yaw+π
    produce equivalent grasps. Among the in-limit candidates we pick
    whichever puts the target wrist_roll closest to ``current_wrist_roll`` —
    this minimizes the swing the arm has to perform to reach the grasp.
    Falls back to the joint-range center as the reference when
    ``current_wrist_roll`` is None (preserves deterministic behavior for
    callers that don't track joint state).
    """
    wr_lo, wr_hi = JOINT_LIMITS['wrist_roll']
    ref = current_wrist_roll if current_wrist_roll is not None else (wr_lo + wr_hi) / 2

    best = None
    best_dist = float('inf')
    for candidate in (yaw, yaw + math.pi, yaw - math.pi):
        target_wr = pan + candidate - _CHAIN_BASE_ROTATION
        if not (wr_lo <= target_wr <= wr_hi):
            continue
        dist = abs(target_wr - ref)
        if dist < best_dist:
            best, best_dist = candidate, dist

    # Safety net: if no candidate respects joint limits (shouldn't happen
    # for reachable grasps), return the unfiltered center-closest pick so
    # downstream IK can still try and emit a clear failure.
    if best is None:
        wr_center = (wr_lo + wr_hi) / 2
        best, best_dist = yaw, abs(pan + yaw - _CHAIN_BASE_ROTATION - wr_center)
        for candidate in (yaw + math.pi, yaw - math.pi):
            dist = abs(pan + candidate - _CHAIN_BASE_ROTATION - wr_center)
            if dist < best_dist:
                best, best_dist = candidate, dist
    return best

# ---------------------------------------------------------------------------
# Workspace bounds — loaded from compute_workspace.py output
# ---------------------------------------------------------------------------

def _load_workspace_yaml(section_name):
    """Load a named section from workspace_bounds.yaml."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'workspace_bounds.yaml')
    bounds = {}
    in_section = False
    try:
        with open(yaml_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped == f'{section_name}:':
                    in_section = True
                    continue
                if in_section and stripped and not line[0].isspace():
                    break
                if in_section and ':' in stripped and not stripped.startswith('#'):
                    key, val = stripped.split(':', 1)
                    try:
                        bounds[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return bounds

_WS = _load_workspace_yaml('workspace_bounds')
WORKSPACE_BOUNDS = {
    'X': (_WS.get('x_min', -0.35), _WS.get('x_max', 0.35)),
    'Y': (_WS.get('y_min', -0.35), _WS.get('y_max', 0.25)),
    'Z': (_WS.get('z_min', -0.10), _WS.get('z_max', 0.45)),
}

_GWS = _load_workspace_yaml('grasp_workspace_bounds')
GRASP_WORKSPACE_BOUNDS = {
    'R_MIN': _GWS.get('r_min', 0.09),
    'R_MAX': _GWS.get('r_max', 0.31),
    'Z_MIN': _GWS.get('z_min', -0.20),
    'Z_MAX': _GWS.get('z_max', 0.07),
}


def check_grasp_reachable(x, y, z, ground_z=None):
    """Check if (x, y, z) is within the top-down grasp workspace.

    Bounds are computed by sweeping geometric_ik() over a (r, z, yaw) grid,
    so they represent the true IK-solvable region, not just FK-reachable.
    ground_z: if provided, reject targets at or below the ground plane.
    Returns (ok, reason_string). ok=True means reachable.
    """
    if ground_z is not None and z <= ground_z:
        return False, f'at/below ground: z={z:.3f}m <= ground={ground_z:.3f}m'
    r = math.sqrt(x * x + y * y)
    r_min = GRASP_WORKSPACE_BOUNDS['R_MIN']
    r_max = GRASP_WORKSPACE_BOUNDS['R_MAX']
    if r < r_min:
        return False, f'too close: r={r:.3f}m < {r_min:.3f}m'
    if r > r_max:
        return False, f'too far: r={r:.3f}m > {r_max:.3f}m'
    z_min = GRASP_WORKSPACE_BOUNDS['Z_MIN']
    z_max = GRASP_WORKSPACE_BOUNDS['Z_MAX']
    if z < z_min:
        return False, f'too low: z={z:.3f}m < {z_min:.3f}m'
    if z > z_max:
        return False, f'too high: z={z:.3f}m > {z_max:.3f}m'
    return True, ''


# Yaw offsets tried when the requested grasp_yaw has no geometric_ik solution
# for a point that passes the workspace bbox check. geometric_ik is ~100µs,
# so a 10-yaw sweep costs ~1ms vs the alternative of aborting the grasp.
# Offsets expand outward from the requested yaw. Ordering: 0 first so the
# common case (requested yaw works) is free.
_GRASP_YAW_FALLBACK_OFFSETS = (
    0.0, math.pi/8, -math.pi/8,
    math.pi/4, -math.pi/4,
    math.pi/2, -math.pi/2,
    3*math.pi/4, -3*math.pi/4, math.pi,
)


def find_reachable_grasp_yaw(poses, requested_yaw, debug=False):
    """Find a yaw for which geometric_ik returns solutions at EVERY pose.

    poses: iterable of (stage_name, x, y, z) tuples — all stages must be
        solvable at the same yaw (otherwise the arm would twist mid-grasp).
    requested_yaw: preferred yaw; tried first.

    Returns (yaw_used, {stage_name: [ik_solutions]}, debug_lines).
    Returns (None, {}, debug_lines) if no yaw in the fallback set works.
    """
    from so_arm101_control.compute_workspace import geometric_ik
    debug_lines = []
    debug_lines.append(
        f'find_reachable_grasp_yaw: requested_yaw={math.degrees(requested_yaw):+.1f}° '
        f'poses={[(s, round(x,3), round(y,3), round(z,3)) for s,x,y,z in poses]}')
    for offset in _GRASP_YAW_FALLBACK_OFFSETS:
        candidate = requested_yaw + offset
        stage_sols = {}
        stage_status = []
        all_ok = True
        for stage, px, py, pz in poses:
            sols = geometric_ik(px, py, pz, grasp_yaw=candidate)
            stage_status.append(f'{stage}={len(sols) if sols else "0"}')
            if not sols:
                all_ok = False
            else:
                stage_sols[stage] = sols
        debug_lines.append(
            f'  off={math.degrees(offset):+6.1f}° cand={math.degrees(candidate):+7.1f}° '
            f'{" ".join(stage_status)} {"✓" if all_ok else "✗"}')
        if all_ok:
            return candidate, stage_sols, debug_lines
    return None, {}, debug_lines




TCP_ABOVE_BLOCK_CLEARANCE_M = 0.00165


def object_grip_width(bbox, cross=False):
    if not bbox:
        return None
    return max(bbox['sx'], bbox['sy']) if cross else min(bbox['sx'], bbox['sy'])


def jaw_offset(width, yaw, tcp_clearance=TCP_CLEARANCE_M, enabled=True):
    if width is None or not enabled:
        return 0.0, 0.0
    half_offset = width / 2 + tcp_clearance
    return -half_offset * math.sin(yaw), half_offset * math.cos(yaw)


def gripper_angles(width, tcp_clearance=TCP_CLEARANCE_M,
                   open_clearance=JAW_OPEN_CLEARANCE_M, close_clearance=JAW_CLOSE_CLEARANCE_M):
    if width is None:
        return JOINT_LIMITS['gripper_joint'][1], JOINT_LIMITS['gripper_joint'][0]
    lo, hi = JOINT_LIMITS['gripper_joint']
    baseline = width + 2 * tcp_clearance
    opening = (baseline + open_clearance - BASELINE_JAW_GAP) / JAW_GAP_RATE
    closing = (baseline + close_clearance - BASELINE_JAW_GAP) / JAW_GAP_RATE
    return max(lo, min(opening, hi)), max(lo, min(closing, hi))


def make_grasp_plan(pose, bbox, *, cross=False, current_wrist_roll=None,
                    tcp_clearance=TCP_CLEARANCE_M, bbox_enabled=True,
                    object_z_override=0.0, approach_height=0.020,
                    approach_offset=(0.0, 0.0), drop=False):
    """Pure grasp geometry shared by GUI, ROS collection and in-process motion transports.

    Poses use bbox-centre coordinates in the robot base frame. The approach
    offset affects only the approach stage; the final target retains jaw compensation.
    """
    qz, qw = pose.get('qz', 0.0), pose.get('qw', 1.0)
    yaw = math.atan2(2 * qw * qz, 1 - 2 * qz * qz)
    if cross:
        yaw += math.pi / 2
    pan = math.atan2(pose['y'], pose['x']) if abs(pose['x']) + abs(pose['y']) > .001 else 0.0
    yaw = _normalize_grasp_yaw(yaw, pan, current_wrist_roll)
    dx, dy = jaw_offset(object_grip_width(bbox, cross), yaw, tcp_clearance, bbox_enabled)
    x, y = pose['x'] + dx, pose['y'] + dy
    z = (object_z_override if abs(object_z_override) > 1e-4 else
         pose['z'] + float((bbox or {}).get('sz', 0.0)) / 2 + TCP_ABOVE_BLOCK_CLEARANCE_M)
    z += .05 if drop else 0.0
    poses = []
    if approach_height > 0:
        poses.append(('approach', x + approach_offset[0], y + approach_offset[1], z + approach_height))
    poses.append(('final', x, y, z))
    return dict(tcp=(x, y, z), yaw=yaw, jaw_offset=(dx, dy), poses=poses)
