#!/usr/bin/env python3
"""
SO-ARM101 Control GUI — Tkinter GUI with MoveIt IK, trajectory execution,
object detection, and dual hardware/simulation modes.

Source: adapted from RoboSort/JETANK_description/jetank_control_gui.py (3133 lines)
        IK delegated to MoveIt compute_ik service (KDL solver) instead of custom scipy.
        Joint mapping from MuammerBay/SO-ARM_ROS2_URDF and SO-ARM101_MoveIt_IsaacSim.
"""

import datetime
import json
import math
import os
import random
import signal
import socket
import subprocess
import threading
import time
import tkinter as tk
import weakref
from pathlib import Path
from tkinter import ttk

from so_arm101_control.grasp_trace import tracer

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Vector3
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from visualization_msgs.msg import Marker as VisMarker, MarkerArray
from std_srvs.srv import Trigger, SetBool
from tf2_ros import Buffer as TfBuffer, TransformListener
try:
    from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetMotionPlan, GetStateValidity
    from moveit_msgs.msg import (
        PositionIKRequest, RobotState, Constraints, JointConstraint,
        MotionPlanRequest, PlanningScene as PlanningSceneMsg, CollisionObject,
        AttachedCollisionObject, AllowedCollisionEntry,
    )
    from shape_msgs.msg import SolidPrimitive, Mesh as ShapeMesh, MeshTriangle
    from moveit_msgs.srv import (
        ExecuteKnownTrajectory, ApplyPlanningScene, GetPlanningScene as GetPlanningSceneSrv,
    )
    MOVEIT_AVAILABLE = True
except ImportError:
    try:
        from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetMotionPlan, GetStateValidity
        from moveit_msgs.msg import (
            PositionIKRequest, RobotState, Constraints, JointConstraint,
            MotionPlanRequest, PlanningScene as PlanningSceneMsg, CollisionObject,
            AttachedCollisionObject, AllowedCollisionEntry,
        )
        from shape_msgs.msg import SolidPrimitive, Mesh as ShapeMesh, MeshTriangle
        from moveit_msgs.srv import (
            ApplyPlanningScene, GetPlanningScene as GetPlanningSceneSrv,
        )
        MOVEIT_AVAILABLE = True
        ExecuteKnownTrajectory = None
    except ImportError:
        MOVEIT_AVAILABLE = False
        GetPositionIK = None
        GetPositionFK = None
        GetStateValidity = None
        GetMotionPlan = None


# ---------------------------------------------------------------------------
# Debug service auto-registration (convention-based: _cmd_* → ~/cmd_name)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trajectory post-check result
# ---------------------------------------------------------------------------
# Honest encoding of what _trajectory_first_invalid_with_contacts actually
# measured. Previous return was (idx, contacts) which conflated "no info"
# with "all clear" and gave no visibility into sub-segment coverage.
from collections import namedtuple
_TrajCheckResult = namedtuple(
    '_TrajCheckResult',
    ['ok', 'bad_wp', 'bad_subidx', 'sub_t', 'contacts', 'n_wps', 'n_sub'])

# Sub-segment subsamples between consecutive waypoints during post-check.
# Module-level so a hot-reload picks up changes (class-level constants are
# not copied by _patch_methods — only methods are).
TRAJ_SUBSEGMENT_SAMPLES = 20


# ---------------------------------------------------------------------------
# Hot reload helpers
# ---------------------------------------------------------------------------

def _patch_methods(instance, new_cls):
    """Replace all methods on a running instance with those from new_cls.

    Preserves instance state (attributes, ROS2 infra, widgets). Only replaces
    method implementations. Handles regular methods, staticmethod, classmethod.
    """
    SKIP = frozenset({'__init__', '__del__', '__class__', '__dict__', '__weakref__'})
    for name, raw in new_cls.__dict__.items():
        if name in SKIP:
            continue
        if isinstance(raw, staticmethod):
            setattr(instance, name, raw.__func__)
        elif isinstance(raw, classmethod):
            setattr(instance, name, raw.__func__.__get__(new_cls, type(new_cls)))
        elif callable(raw):
            setattr(instance, name, raw.__get__(instance, type(instance)))

    # Apply default values for any new instance attributes
    for attr, default in getattr(new_cls, '_RELOAD_DEFAULTS', {}).items():
        if not hasattr(instance, attr):
            setattr(instance, attr, default)


# ---------------------------------------------------------------------------
# Joint configuration for SO-ARM101
# ---------------------------------------------------------------------------
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
# Sim-side (isaac-sim-mcp soarm101-dt extension) lifts each drop_{id}
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
_CUP_COLLISION_PADDING = 1.05  # 5% default pad — absorbs the multi-mm
# tracking-lag-induced overshoot during fast pan motions (e.g. post-drop
# grasp_home sweeps 107° at 36°/s, which exceeded the 1% margin in testing).


# --- Record Sim tab constants -----------------------------------------------
# Module-level (not class-level) so hot_reload picks them up on the running
# instance — class attributes only attach to a fresh class object on import.
REC_LEROBOT_SCRIPT = os.path.expanduser(
    "~/Projects/Exploring-VLAs/linux-env/scripts/record_sim_isaac.sh")
REC_MIRROR_SCRIPT = os.path.expanduser(
    "~/Projects/Exploring-VLAs/linux-env/scripts/joint_states_to_commands.py")
REC_DATASET_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/local")
REC_SIM_RESET_SCRIPT = os.path.expanduser(
    "~/Projects/Exploring-VLAs/vla_SO-ARM101/scripts/sim_reset.sh")
REC_MCP_HOST = "localhost"
REC_MCP_PORT = 8767
# Hardcoded scene contract — matches extension.py LEGO_USDS.
# 2 sizes per color picked → 2 episodes per scene. Skipping 2x4 (largest)
# because its grasp footprint is closer to the workspace edge and tends to
# need yaw-fallback retries.
REC_LEGOS_BY_COLOR = {
    "red":   ["red_2x2", "red_2x3"],
    "green": ["green_2x2", "green_2x3"],
    "blue":  ["blue_2x2", "blue_2x3"],
}
REC_MAX_RETRIES_PER_LEGO = 5


def _load_cup_mesh():
    """Load cup.stl as a shape_msgs/Mesh for MoveIt collision objects.

    Returns (ShapeMesh, success). Caches after first load.
    Uses trimesh for robust STL loading. The STL is in mm; vertices are
    scaled by _CUP_STL_SCALE to meters with _CUP_COLLISION_PADDING.
    """
    if hasattr(_load_cup_mesh, '_cached'):
        return _load_cup_mesh._cached, True
    try:
        import trimesh
        stl_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'so_arm101_description', 'meshes', 'cup', 'cup.stl')
        if not os.path.isfile(stl_path):
            from ament_index_python.packages import get_package_share_directory
            stl_path = os.path.join(
                get_package_share_directory('so_arm101_description'),
                'meshes', 'cup', 'cup.stl')
        raw = trimesh.load(stl_path).convex_hull  # Solid hull for clean RViz rendering
        scale = _CUP_STL_SCALE * _CUP_COLLISION_PADDING
        mesh = ShapeMesh()
        for v in raw.vertices:
            pt = Point()
            pt.x = float(v[0]) * scale
            pt.y = float(v[1]) * scale
            pt.z = float(v[2]) * scale
            mesh.vertices.append(pt)
        for f in raw.faces:
            tri = MeshTriangle()
            tri.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
            mesh.triangles.append(tri)
        _load_cup_mesh._cached = mesh
        print(f'[cup_mesh] Loaded {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices')
        return mesh, True
    except Exception as e:
        print(f'[cup_mesh] Failed to load cup.stl: {e}')
        return None, False


# Per-size lego STL meshes. 8-vertex box primitives exported from the Isaac Sim
# lego USDs (assets/legos/lego_*_<size>.usd) with origin at the bbox center so
# /objects_poses_sim pose (= body center) drives MoveIt mesh_pose directly.
# Color is not part of the geometry (studs etc. are absent; we want the
# collision envelope, not the visual), so one STL per size serves all colors.
_LEGO_SIZES = ('2x2', '2x3', '2x4')


# Vertical clearance between grasp TCP (fixed-jaw tip) and the block's top face.
# The gripper link's collision mesh has some downward-facing geometry below
# tcp_link (servo housing, jaw hinge). Without clearance, /check_state_validity
# flags gripper-vs-world-lego contact at grasp pose (~77 µm tangent penetration
# via FCL narrow-phase), which then poisons every subsequent OMPL plan with
# "start state in collision" (ec=-2). This is NOT a USD-fix compensation —
# it's the physical clearance needed between the gripper's lower-facing link
# geometry and the block's upper face. If the gripper hardware is ever
# redesigned with a cleaner TCP profile, this can drop to 0; if the block
# collision envelope is ever padded or grown, this should grow accordingly.
_GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M = 0.00165


def _load_lego_mesh(size):
    """Load lego_{size}.stl as a shape_msgs/Mesh for MoveIt collision objects.

    Returns (ShapeMesh, success). Per-size cache on the function object.
    Unlike cups, the STL is already authored in meters with its origin at
    the same point /objects_poses_sim publishes, so no scale or padding is
    applied. Mirrors _load_cup_mesh's search order (src-tree first, then
    installed share dir).
    """
    cache = getattr(_load_lego_mesh, '_cached', {})
    if size in cache:
        return cache[size], True
    try:
        import trimesh
        stl_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'so_arm101_description', 'meshes', 'lego', f'lego_{size}.stl')
        if not os.path.isfile(stl_path):
            from ament_index_python.packages import get_package_share_directory
            stl_path = os.path.join(
                get_package_share_directory('so_arm101_description'),
                'meshes', 'lego', f'lego_{size}.stl')
        raw = trimesh.load(stl_path)
        mesh = ShapeMesh()
        for v in raw.vertices:
            pt = Point()
            pt.x = float(v[0]); pt.y = float(v[1]); pt.z = float(v[2])
            mesh.vertices.append(pt)
        for f in raw.faces:
            tri = MeshTriangle()
            tri.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
            mesh.triangles.append(tri)
        cache[size] = mesh
        _load_lego_mesh._cached = cache
        print(f'[lego_mesh] Loaded {size}: {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices')
        return mesh, True
    except Exception as e:
        print(f'[lego_mesh] Failed to load lego_{size}.stl: {e}')
        return None, False


def _lego_size_from_name(name):
    """Extract '2x2'/'2x3'/'2x4' from a sim-style block name like 'red_2x3'.

    Returns None for names without a size suffix (e.g. YOLOE-style 'red_0',
    where size is unknown — vision can detect color but not size). Callers
    must fall back to bbox catalog dims (`_lookup_bbox`) when None is
    returned, NOT infer size from color.
    """
    for s in _LEGO_SIZES:
        if name.endswith(s):
            return s
    return None


def _normalize_grasp_yaw(yaw, pan):
    """Pick yaw or yaw±π that keeps wrist_roll within joint limits.

    Gripper jaws are symmetric about the grip axis, so yaw and yaw+π
    produce equivalent grasps. We pick whichever keeps wrist_roll
    closest to the center of its range.
    """
    wr_lo, wr_hi = JOINT_LIMITS['wrist_roll']
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


# ArUco marker ID -> human-readable color label for drop targets
DROP_ID_LABELS = {
    "drop_0": "red",
    "drop_1": "green",
    "drop_2": "blue",
}

# Visual marker colors for cups in RViz (RGBA, alpha<1.0 avoids ros2/rviz#875)
_CUP_VISUAL_COLORS = {
    "drop_0": (0.9, 0.15, 0.1, 0.99),   # red
    "drop_1": (0.1, 0.75, 0.2, 0.99),   # green
    "drop_2": (0.15, 0.3, 0.9, 0.99),   # blue
}
def _get_cup_stl_uri():
    """Resolve cup STL file URI for RViz MESH_RESOURCE markers."""
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(
        get_package_share_directory('so_arm101_description'),
        'meshes', 'cup', 'cup.stl')
    return 'file://' + path if os.path.isfile(path) else ''


# Phase 11-01: real-mode color → cup mapping. Inverse of _qs_auto_drop_for_lego.
# Module-level so hot-reload picks up changes (class-level constants are NOT
# copied by _patch_methods — only methods are; see comment at L89-90).
REAL_COLOR_TO_CUP = {'red': 'drop_0', 'green': 'drop_1', 'blue': 'drop_2'}

# Phase 11-01 followup: Drop Scan workflow tunables. Module-level for the
# same hot-reload reason. Times in seconds, angles in radians, variance in m².
DROP_SCAN_PAN_MIN_RAD = -math.pi / 2          # -90° (absolute lower bound)
DROP_SCAN_PAN_MAX_RAD = math.pi / 2           # +90° (absolute upper bound)
# Sweep direction: START → END in increments of STEP_SIGNED. To reverse the
# sweep direction (e.g. start from +90° and go to -90°), swap START/END and
# negate STEP_SIGNED. The MIN/MAX above stay constant — they're absolute
# bounds used for the FOV in-range filter.
DROP_SCAN_PAN_START_RAD = math.pi / 2         # +90° (begin sweep here)
DROP_SCAN_PAN_END_RAD = -math.pi / 2          # −90° (finish sweep here)
DROP_SCAN_PAN_STEP_SIGNED_RAD = -math.radians(5)  # negative = sweep right→left
# Legacy alias kept for any existing downstream readers.
DROP_SCAN_PAN_STEP_RAD = abs(DROP_SCAN_PAN_STEP_SIGNED_RAD)
DROP_SCAN_INITIAL_DURATION_S = 3.5            # safe duration for 90° initial move
DROP_SCAN_STEP_DURATION_S = 1.5               # per-5°-step duration; longer
                                              # gives the weak drives time to
                                              # actually converge wrist_roll
                                              # back to its commanded value
                                              # against shoulder_pan dynamic
                                              # disturbance (was 0.4 = drift)
DROP_SCAN_SETTLE_AFTER_STEP_S = 0.3           # let pose feed catch up
DROP_SCAN_SAMPLE_PER_STEP_S = 0.3             # quick check for new cups
DROP_SCAN_AVG_DURATION_S = 3.0                # full settle on detection
DROP_SCAN_MIN_AVG_SAMPLES = 10                # need ≥N samples to trust avg
DROP_SCAN_FOV_HALF_RAD = math.radians(25)     # cup must be within ±this of pan
DROP_SCAN_MAX_XY_VARIANCE_M2 = (0.005)**2     # 5 mm xy std dev cap


class SOArm101ControlGUI(Node):
    """ROS2 node with embedded Tkinter GUI for SO-ARM101 control."""

    def __init__(self):
        super().__init__('so_arm101_control_gui')

        # Callback group for service clients — allows responses to be
        # processed concurrently with other callbacks (requires MultiThreadedExecutor)
        self._service_cb_group = ReentrantCallbackGroup()
        # Phase 9: separate ReentrantCallbackGroup for topic subscriptions so
        # their callbacks can keep running while a service handler blocks on
        # a threading.Event (e.g. sync-refresh waiting for fresh data). With
        # the default MutuallyExclusiveCallbackGroup, a blocked service
        # handler deadlocks the executor — subscription callbacks never fire
        # because all executor threads are parked in .wait(). Research ref:
        # karelics.fi/deadlocks-in-rclpy.
        self._sub_cb_group = ReentrantCallbackGroup()

        # Hardware mode
        self.use_real_hardware = False
        self.hw_lock = threading.Lock()

        # Current joint positions (radians) — updated from joint_state_broadcaster
        self.joint_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
        self.joint_lock = threading.Lock()
        # Actual robot state — always updated from /joint_states, never blocked
        self._actual_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
        self._actual_velocities = {name: 0.0 for name in ALL_JOINT_NAMES}
        self._initial_sync_done = False  # True after first /joint_states received

        # Track last sent arm positions
        self._last_sent_arm = [0.0] * len(ARM_JOINT_NAMES)
        self._last_sent_jaw = 0.0

        # --- Action clients (proven reliable for JTC) ---
        self.arm_action_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory',
            callback_group=self._service_cb_group)
        self.gripper_action_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory',
            callback_group=self._service_cb_group)

        # Track active goals so we can cancel before sending new ones
        self._arm_goal_handle = None
        self._gripper_goal_handle = None
        self._arm_goal_lock = threading.Lock()
        self._gripper_goal_lock = threading.Lock()

        # Authoritative motion verdict. Populated by every motion primitive
        # IMMEDIATELY before it calls on_complete_event.set(). The Trigger
        # wrapper reads this after waiting on _motion_event — this is the
        # ONLY source of truth for motion success. Do not infer success from
        # "no exception raised" (the old bug: on_complete fires on both
        # success and failure paths, so the absence of an exception is not
        # evidence of motion).
        # Shape: {'ok': bool, 'outcome': str, 'msg': str, **extras}
        # Outcomes: 'completed', 'tier1_then_ompl_failed', 'ompl_mode_a',
        #   'ompl_mode_b', 'ompl_service_unavailable', 'action_server_not_ready',
        #   'goal_rejected', 'send_exception', 'result_exception',
        #   'plan_failed', 'plan_exception'
        self._last_motion_status = None

        # --- Gripper topic publisher (works fine via topic) ---
        self.gripper_traj_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        # Hardware commands (for real servo driver)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands_hw', 10)

        # --- Subscribers ---
        # High-rate subs (joint_states @ 50Hz etc.) stay on the default
        # MutuallyExclusive group. Moving them to the Reentrant group
        # oversubscribed the executor (action-client done callbacks got
        # starved, making drop_release think the gripper trajectory timed
        # out when it had actually completed). Only move subs to
        # _sub_cb_group when a service handler explicitly .wait()s for
        # them — currently that's /objects_poses_sim + /drop_poses.
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_states_callback, 10)
        self.real_js_sub = self.create_subscription(
            JointState, 'real_joint_states', self._real_js_callback, 10)
        self.ext_cmd_sub = self.create_subscription(
            JointState, 'joint_commands', self._ext_cmd_callback, 10)
        self.objects_data = {}
        self.objects_lock = threading.Lock()
        self._last_grasped_object = None  # persists through pick→drop cycle
        self.objects_sub = None  # Created by _build_grasp_tab → _update_grasp_topic

        # --- Drop target infrastructure ---
        self._drop_data = {}
        self._drop_lock = threading.Lock()
        self._drop_sub = None  # Created by _build_grasp_tab → _update_drop_topic
        self._cup_collision_names = []

        # --- Real-mode pipeline (Real Test tab; scan-then-cache cup poses) ---
        # child_frame_id ('drop_0'/'drop_1'/'drop_2') →
        # {'translation': (x,y,z), 'rotation': (qx,qy,qz,qw)} in frame 'base'.
        # Populated by Refresh Cups Pose (subscribe-once /drop_poses_real with
        # partial-cache merge: missing markers keep their previous value).
        self._cached_cup_poses = {}
        from rclpy.qos import QoSProfile, DurabilityPolicy
        self._cup_visual_pub = self.create_publisher(
            MarkerArray, '/cup_visual_markers_array',
            QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.objects_bbox = {}   # {name: {sx, sy, sz}} from bbox topic
        # Phase 9: lego block collision tracking in MoveIt planning scene.
        # _lego_collision_names tracks world CollisionObject ids; _attached_lego_name
        # holds the name currently held by the gripper (or None when table-bound).
        self._lego_collision_names = []
        self._attached_lego_name = None
        # Measured attach offset in tcp_link frame, in meters (ax, ay, az).
        # Populated by _attach_lego_to_gripper from the physical grasp result;
        # used by _cmd_drop_sweep to compute the gap→tcp shift from the
        # actual block-in-tcp pose rather than the theoretical half_gap. Lets
        # off-center grasps drop cleanly instead of relying on OMPL to find a
        # tortuous path around the 1-3 mm cup-wall clip.
        self._attached_lego_tcp_offset = None
        # MoveIt 2.5.9 correctly reports AttachedCollisionObject ↔ world
        # CollisionObject collisions (the 2.5.8 skip-bug that required the
        # 10 Hz held-lego world-pose tracker is gone — verified via
        # /check_state_validity probe). Standard AttachedCollisionObject
        # lifecycle is used in _attach_lego_to_gripper / _detach_lego_sync.
        _default_bbox = '/objects_bbox_real' if self.use_real_hardware else '/objects_bbox_sim'
        self.bbox_sub = self.create_subscription(
            String, _default_bbox, self._bbox_callback, 1)
        self.ee_pose_sub = self.create_subscription(
            PoseStamped, '/ee_pose', self._ee_pose_callback, 10)

        # MoveIt service clients + publishers
        self.ik_client = None
        self.plan_client = None
        if MOVEIT_AVAILABLE:
            from moveit_msgs.msg import DisplayTrajectory
            self.ik_client = self.create_client(
                GetPositionIK, '/compute_ik',
                callback_group=self._service_cb_group)
            self.fk_client = self.create_client(
                GetPositionFK, '/compute_fk',
                callback_group=self._service_cb_group)
            self.plan_client = self.create_client(
                GetMotionPlan, '/plan_kinematic_path',
                callback_group=self._service_cb_group)
            self.validity_client = self.create_client(
                GetStateValidity, '/check_state_validity',
                callback_group=self._service_cb_group)
            self._display_traj_pub = self.create_publisher(
                DisplayTrajectory, '/display_planned_path', 10)
            # Publish goal state to RViz MotionPlanning plugin
            self._goal_state_pub = self.create_publisher(
                RobotState, '/rviz/moveit/update_custom_goal_state', 10)
            # Switch active planning group in RViz
            self._planning_group_pub = self.create_publisher(
                String, '/rviz/moveit/select_planning_group', 10)
            self._active_planning_group = None  # Force first publish
            # Planning scene services for collision objects (ground plane, etc.)
            self._apply_scene_client = self.create_client(
                ApplyPlanningScene, '/apply_planning_scene',
                callback_group=self._service_cb_group)
            self._get_scene_client = self.create_client(
                GetPlanningSceneSrv, '/get_planning_scene',
                callback_group=self._service_cb_group)
            # MoveIt Task Constructor spike — optional, only responds when
            # control.launch.py was started with mtc:=true.
            self._mtc_run_client = self.create_client(
                Trigger, '/so_arm101_mtc/run',
                callback_group=self._service_cb_group)

        # Trajectory lock
        self._traj_lock = threading.Lock()

        # Track whether we should update sliders from joint_states
        self._slider_driven = False

        # Parameter for set_ik_target service
        self.declare_parameter('ik_target', '')
        # Parameters for jaw tuning via service calls
        self.declare_parameter('jaw_open_clearance_mm', JAW_OPEN_CLEARANCE_M * 1000)
        self.declare_parameter('jaw_close_clearance_mm', JAW_CLOSE_CLEARANCE_M * 1000)
        self.declare_parameter('tcp_clearance_mm', TCP_CLEARANCE_M * 1000)
        # Velocity scale override for _cmd_grasp_home only. When > 0,
        # replaces self.velocity_scale_var for this motion's plan. 0 =
        # no override (use the Tk default). Tunable live via ros2 param
        # set so speed-vs-lag experiments can be driven without rebuilds.
        self.declare_parameter('home_velocity_scale', 0.0)
        # Parameters for widget registry services (~/get_widget_value, ~/set_widget_value)
        self.declare_parameter('widget_id', '')
        self.declare_parameter('widget_value', '')

        # TF buffer for TCP pose lookups
        self._tf_buffer = TfBuffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # GUI
        self.running = True
        # Button registry — populated by _register_button, audited by _cmd_dump_services.
        # Every entry: {'text', 'command_name', 'tab', 'section', 'widget_class'}.
        self._button_registry = []
        # Widget registry — populated by _register_spinbox/_check/_entry/_listbox/_scale/
        # _notebook/_log_text factories. Audited by _srv_list_widgets.
        # Entries: {id: {'type', 'widget' (WeakRef), 'var', 'tab', 'section', 'label', 'writable'}}
        self._widget_registry: dict = {}
        self._widget_registry_lock = threading.Lock()
        self._widget_registry_rebuilding = False
        self._setup_gui_thread()

        # --- Debug services ---
        # Auto-register: any method named _cmd_* → ~/cmd_name (Trigger)
        # Manual services: _srv_* stay registered individually below
        self._debug_services = []
        for name in sorted(dir(self)):
            if name.startswith('_cmd_'):
                srv_name = name[5:]
                cb = self._make_trigger_callback(name)
                srv = self.create_service(Trigger, f'~/{srv_name}', cb)
                self._debug_services.append(srv)
                self.get_logger().info(f'  service: ~/{srv_name}')

        # Manual services (read/write UI fields directly)
        for name, cb in [
            ('get_joint_positions', self._srv_get_joint_positions),
            ('get_ik_target', self._srv_get_ik_target),
            ('set_ik_target', self._srv_set_ik_target),
            ('get_ee_pose', self._srv_get_ee_pose),
            ('get_tcp_pose', self._srv_get_tcp_pose),
            ('get_log', self._srv_get_log),
            ('list_commands', self._srv_list_commands),
            ('dump_services', self._srv_dump_services),
            ('get_widget_state', self._srv_get_widget_state),
            ('screenshot', self._srv_screenshot),
            ('list_widgets', self._srv_list_widgets),
            ('get_widget_value', self._srv_get_widget_value),
            ('set_widget_value', self._srv_set_widget_value),
        ]:
            self._debug_services.append(
                self.create_service(Trigger, f'~/{name}', cb))
            self.get_logger().info(f'  service: ~/{name}')

        self.get_logger().info('SO-ARM101 Control GUI initialized')

    # ------------------------------------------------------------------
    # Async service helper (thread-safe future wait)
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_future(future, timeout_sec=2.0):
        """Wait for a ROS2 future to complete by polling. Thread-safe.

        Unlike rclpy.spin_until_future_complete, this does NOT create a
        temporary executor and does NOT spin the node.  The main-thread
        executor (rclpy.spin) processes callbacks; we just poll.
        """
        end = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < end:
            time.sleep(0.01)          # 10 ms poll
        return future.result()        # None if timed-out / not done

    # ------------------------------------------------------------------
    # Button factory — central registration for audit + enforcement
    # ------------------------------------------------------------------
    # Every tk.Button / ttk.Button in this file MUST go through this helper.
    # The `command=` kwarg MUST be a bare method reference:
    #   - self._cmd_*         (auto-registers as a ~/cmd Trigger service)
    #   - self._*_btn_*       (thin UI-state wrapper that forwards to a _cmd_*)
    # Inline `command=lambda: ...` bodies are forbidden — enforced by
    # test_button_service_mapping.py (Phase 7 workstream A).

    def _register_button(self, parent, *, text, command, tab=None, section=None,
                         bg='#b0b0b0', fg='#1a1a1a', **kwargs):
        """Central button factory. Records the button in self._button_registry
        for _cmd_dump_services audit, then creates a styled tk.Button."""
        cmd_name = getattr(command, '__name__', repr(command))
        self._button_registry.append({
            'text': text,
            'command_name': cmd_name,
            'tab': tab,
            'section': section,
            'widget_class': 'tk.Button',
        })
        return tk.Button(parent, text=text, command=command,
                         bg=bg, fg=fg, **kwargs)

    # ------------------------------------------------------------------
    # Widget factory helpers (Phase 07.1) — every interactive widget
    # MUST go through these so _srv_list_widgets / _srv_get_widget_value /
    # _srv_set_widget_value can see them. Audited by test_widget_registry.py.
    # ------------------------------------------------------------------

    def _widget_registry_add(self, label, wtype, widget, var,
                             tab, section, writable):
        """Insert a widget into the registry. Duplicate labels are tolerated
        at insertion time; _finalize_widget_registry resolves collisions with
        @tab suffixes after all builders have run."""
        entry = {
            'type': wtype,
            'widget': weakref.ref(widget),
            'var': var,
            'tab': tab,
            'section': section,
            'label': label,
            'writable': writable,
        }
        with self._widget_registry_lock:
            if label in self._widget_registry and tab:
                resolved_id = f'{label}@{tab}'
            else:
                resolved_id = label
            self._widget_registry[resolved_id] = entry

    def _finalize_widget_registry(self):
        """Called after all tab builders complete. Scans for label collisions
        and retroactively adds @tab suffixes so every id is unique."""
        with self._widget_registry_lock:
            by_label: dict = {}
            for rid, entry in list(self._widget_registry.items()):
                by_label.setdefault(entry['label'], []).append(rid)
            for label, rids in by_label.items():
                if len(rids) == 1:
                    continue
                for rid in rids:
                    if '@' in rid:
                        continue
                    entry = self._widget_registry[rid]
                    tab = entry.get('tab') or '?'
                    new_id = f'{label}@{tab}'
                    if new_id == rid:
                        continue
                    self._widget_registry[new_id] = entry
                    del self._widget_registry[rid]
            self._widget_registry_rebuilding = False

    def _register_spinbox(self, parent, *, label, textvariable, from_, to,
                          tab=None, section=None, **kwargs):
        """tk.Spinbox (or ttk.Spinbox via _use_ttk=True). Writable; set coerces to float."""
        use_ttk = kwargs.pop('_use_ttk', False)
        cls = ttk.Spinbox if use_ttk else tk.Spinbox
        widget = cls(parent, textvariable=textvariable,
                     from_=from_, to=to, **kwargs)
        self._widget_registry_add(label, 'Spinbox', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_check(self, parent, *, label, variable,
                        tab=None, section=None, **kwargs):
        """tk.Checkbutton. Label doubles as the button's visible text."""
        widget = tk.Checkbutton(parent, text=label, variable=variable, **kwargs)
        self._widget_registry_add(label, 'Checkbutton', widget, variable,
                                  tab, section, writable=True)
        return widget

    def _register_entry(self, parent, *, label, textvariable,
                        tab=None, section=None, **kwargs):
        """tk.Entry. `label` is explicit (no sibling-Label heuristic)."""
        widget = tk.Entry(parent, textvariable=textvariable, **kwargs)
        self._widget_registry_add(label, 'Entry', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_listbox(self, parent, *, label, tab=None, section=None,
                          **kwargs):
        """tk.Listbox. Writable; set expects a non-negative row index."""
        widget = tk.Listbox(parent, **kwargs)
        self._widget_registry_add(label, 'Listbox', widget, None,
                                  tab, section, writable=True)
        return widget

    def _register_combobox(self, parent, *, label, textvariable, values,
                           tab=None, section=None, **kwargs):
        """ttk.Combobox (readonly). User picks one of `values`; current
        selection is bound to `textvariable` (StringVar)."""
        widget = ttk.Combobox(
            parent, textvariable=textvariable, values=values,
            state='readonly', **kwargs)
        self._widget_registry_add(label, 'Combobox', widget, textvariable,
                                  tab, section, writable=True)
        return widget

    def _register_scale(self, parent, *, label, variable, from_, to,
                        tab=None, section=None, **kwargs):
        """tk.Scale (or ttk.Scale via _use_ttk=True). Writable; coerces to float."""
        use_ttk = kwargs.pop('_use_ttk', False)
        cls = ttk.Scale if use_ttk else tk.Scale
        widget = cls(parent, variable=variable, from_=from_, to=to, **kwargs)
        self._widget_registry_add(label, 'Scale', widget, variable,
                                  tab, section, writable=True)
        return widget

    def _register_notebook(self, widget, *, label='tab'):
        """Register an existing ttk.Notebook under a reserved id.
        Caller constructs the notebook — we don't apply styling."""
        self._widget_registry_add(label, 'Notebook', widget, None,
                                  tab=None, section=None, writable=True)
        return widget

    def _register_log_text(self, widget, *, label):
        """Register a tk.Text widget as read-only log output."""
        self._widget_registry_add(label, 'Text', widget, None,
                                  tab=None, section=None, writable=False)
        return widget

    # ------------------------------------------------------------------
    # Shared UI cluster: Reset Arm / Randomize / Reset Gripper / Open / Close
    # duplicated between FK and IK tabs (Phase 7 Plan 07-03 / D-16..D-18).
    # ------------------------------------------------------------------

    def _build_arm_btn_row(self, parent, *, tab, reset_cmd):
        """Build the `[Reset Arm] [Reset Grasp Home] [Randomize]` button row.

        tab: 'FK' or 'IK' — used for dump_services tagging.
        reset_cmd: which command 'Reset Arm' invokes (different on FK vs IK).
        """
        self._register_button(parent, text='Reset Arm', tab=tab, section='Arm',
                              command=reset_cmd).pack(side=tk.LEFT, padx=5)
        if tab == 'FK':
            self._register_button(parent, text='Reset Grasp Home', tab=tab,
                                  section='Arm',
                                  command=self._cmd_reset_grasp_home).pack(
                side=tk.LEFT, padx=5)
        randomize_cmd = (self._cmd_randomize_arm if tab == 'FK'
                         else self._cmd_ik_randomize)
        self._register_button(parent, text='Randomize', tab=tab, section='Arm',
                              command=randomize_cmd).pack(side=tk.LEFT, padx=5)

    def _build_gripper_btn_row(self, parent, *, tab):
        """Build `[Reset Gripper] [Open] [Close]` row into `parent`."""
        self._register_button(parent, text='Reset Gripper', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_zero).pack(
            side=tk.LEFT, padx=5)
        self._register_button(parent, text='Open', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_open).pack(
            side=tk.LEFT, padx=5)
        self._register_button(parent, text='Close', tab=tab,
                              section='Gripper',
                              command=self._cmd_gripper_close).pack(
            side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # Debug service helpers
    # ------------------------------------------------------------------

    def _make_trigger_callback(self, method_name):
        """Factory: returns a Trigger callback that dispatches to tkinter thread.

        If the _cmd_* method sets self._motion_event before returning,
        the callback waits for that event too (up to 30s), so the service
        response is deferred until the full motion sequence completes.
        """
        def _callback(request, response):
            done_event = threading.Event()
            result = {'ok': True, 'msg': ''}

            def _run():
                try:
                    # Clear any stale motion event / error / status
                    self._motion_event = None
                    self._cmd_error = None
                    self._last_motion_status = None
                    # Tag the next trajectory dump with this cmd's name so
                    # post-hoc log analysis can match dumps to commands.
                    # _cmd_foo → tag 'foo'. Anything _execute_full_trajectory
                    # writes from here until the next _cmd_* gets this tag.
                    if method_name.startswith('_cmd_'):
                        self._last_motion_tag = method_name[5:]
                    getattr(self, method_name)()
                    # Check if the command signaled failure synchronously
                    # (geometric IK reject, missing service, etc). Async
                    # motion verdict is read later after _motion_event fires.
                    if self._cmd_error:
                        result['ok'] = False
                        result['msg'] = self._cmd_error
                    else:
                        # Placeholder — overwritten below by motion verdict
                        # if _motion_event was registered. For commands that
                        # don't set _motion_event (pure sync cmds), this
                        # stands as the final message.
                        result['msg'] = f'{method_name} dispatched'
                except Exception as e:
                    result['ok'] = False
                    result['msg'] = str(e)
                finally:
                    done_event.set()

            if getattr(self, '_gui_ready', False):
                self.root.after(0, _run)
            else:
                result['ok'] = False
                result['msg'] = 'GUI not available'
                done_event.set()

            # Wait for the _cmd_* method to return (interruptible by shutdown)
            while not done_event.is_set() and self.running:
                done_event.wait(timeout=0.5)

            # If the command registered a motion event, wait for motion too.
            # Timeout prevents permanent hang if async callback chain breaks.
            motion_evt = getattr(self, '_motion_event', None)
            motion_registered = motion_evt is not None
            motion_timed_out = False
            if motion_registered:
                elapsed = 0.0
                timeout_s = 60.0
                while not motion_evt.is_set() and self.running and elapsed < timeout_s:
                    motion_evt.wait(timeout=0.5)
                    elapsed += 0.5
                if not motion_evt.is_set():
                    motion_timed_out = True
                    result['ok'] = False
                    result['msg'] = f'Motion timed out after {timeout_s:.0f}s'
                self._motion_event = None

            # Read authoritative motion verdict. Motion primitives set
            # _last_motion_status right before on_complete_event.set(),
            # so by the time we get here it reflects the actual outcome.
            if motion_registered and not motion_timed_out:
                status = getattr(self, '_last_motion_status', None)
                if status is None:
                    # Primitive didn't populate status — treat as failure.
                    # Every on_complete_event.set() site MUST write status.
                    result['ok'] = False
                    result['msg'] = (f'{method_name}: motion completed but '
                                     'no _last_motion_status was reported')
                else:
                    result['ok'] = bool(status.get('ok', False))
                    msg = status.get('msg') or status.get('outcome') \
                        or 'unknown'
                    result['msg'] = f'{method_name}: {msg}'

            # Check for deferred errors (set by background threads after _run returned)
            cmd_err = getattr(self, '_cmd_error', None)
            if cmd_err:
                result['ok'] = False
                result['msg'] = cmd_err
                self._cmd_error = None

            response.success = result['ok']
            response.message = result['msg']
            return response
        return _callback

    def _srv_list_commands(self, request, response):
        """Return all available _cmd_* command names for agent discovery."""
        commands = sorted(n[5:] for n in dir(self) if n.startswith('_cmd_'))
        response.success = True
        response.message = ', '.join(commands)
        return response

    def _srv_get_widget_state(self, request, response):
        """Return the runtime state of a widget from _button_registry.

        Pass the widget id via the `ik_target` parameter. Matches on:
          - Button text (e.g. 'Reset Arm') — first match wins across tabs.
          - Bound method name (e.g. '_cmd_zero_arm', '_ik_btn_set_joints').
          - 'text@tab' disambiguator (e.g. 'Reset Arm@FK').

        Returns JSON-like `key=value, ...` pairs in response.message:
          id=_cmd_zero_arm, text=Reset Arm, tab=FK, section=Arm,
          enabled=true, visible=true, state=normal

        Usage:
          ros2 param set /so_arm101_control_gui ik_target 'Reset Arm@FK'
          ros2 service call /so_arm101_control_gui/get_widget_state std_srvs/srv/Trigger {}
        """
        hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not hint:
            response.success = False
            response.message = ('no widget id. Set param first: '
                                "ros2 param set /so_arm101_control_gui "
                                "ik_target '<button text or _cmd_ name>'")
            return response

        # Allow 'text@tab' disambiguation
        qualifier = None
        if '@' in hint:
            hint, qualifier = hint.rsplit('@', 1)

        match = None
        for entry in self._button_registry:
            if qualifier and entry.get('tab') != qualifier:
                continue
            if entry.get('text') == hint or entry.get('command_name') == hint:
                match = entry
                break

        if match is None:
            response.success = False
            response.message = (f'widget not found: {hint!r}'
                                + (f' (tab={qualifier!r})' if qualifier else ''))
            return response

        # Walk the tkinter tree to find the live widget by recreating the
        # lookup: registry stores text+tab+section, not the widget ref (by
        # design — avoids WeakRef bookkeeping for a GUI that doesn't churn).
        # Find the button by scanning winfo_children recursively for a
        # tk.Button whose text matches.
        def _find_button(parent, text, tab_hint=None):
            for child in parent.winfo_children():
                if isinstance(child, tk.Button) and child.cget('text') == text:
                    return child
                found = _find_button(child, text, tab_hint)
                if found is not None:
                    return found
            return None

        widget = _find_button(self.root, match['text'])
        if widget is None:
            response.success = False
            response.message = (f'widget {match["text"]!r} registered but not '
                                'currently in the tree — did the tab rebuild?')
            return response

        try:
            state = str(widget.cget('state'))
        except tk.TclError:
            state = 'unknown'
        try:
            enabled = state != 'disabled'
        except Exception:
            enabled = True
        try:
            visible = bool(widget.winfo_viewable())
        except Exception:
            visible = True

        parts = [
            f'id={match.get("command_name")}',
            f'text={match.get("text")}',
            f'tab={match.get("tab") or "-"}',
            f'section={match.get("section") or "-"}',
            f'enabled={"true" if enabled else "false"}',
            f'visible={"true" if visible else "false"}',
            f'state={state}',
        ]
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_screenshot(self, request, response):
        """Save a PNG of the main GUI window; return its path + dimensions.

        Pass the desired output path via the `ik_target` parameter. When
        empty, defaults to /tmp/so_arm101_control_gui.png.

        Uses xdotool + ImageMagick `import` for reliability across tk
        versions. Falls back to `import -window root` if xdotool fails.

        Usage:
          ros2 param set /so_arm101_control_gui ik_target '/tmp/gui.png'
          ros2 service call /so_arm101_control_gui/screenshot std_srvs/srv/Trigger {}

        Response message format: path=/tmp/gui.png, width=1280, height=720
        """
        import subprocess
        target = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not target:
            target = '/tmp/so_arm101_control_gui.png'

        # Resolve window id via xdotool (match the tkinter window title)
        title = self.root.title() if getattr(self, 'root', None) else 'SO-ARM101'
        window_id = None
        try:
            out = subprocess.check_output(
                ['xdotool', 'search', '--name', title],
                encoding='utf-8', timeout=3)
            ids = [line.strip() for line in out.splitlines() if line.strip()]
            window_id = ids[0] if ids else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            window_id = None

        try:
            if window_id:
                subprocess.check_call(
                    ['import', '-window', window_id, target], timeout=5)
            else:
                # Fallback: whole root display
                subprocess.check_call(['import', '-window', 'root', target],
                                       timeout=5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            response.success = False
            response.message = f'screenshot failed: {e}'
            return response

        # Read dimensions via `identify` if available; otherwise skip.
        width, height = 0, 0
        try:
            out = subprocess.check_output(
                ['identify', '-format', '%w %h', target],
                encoding='utf-8', timeout=3).strip()
            w, h = out.split()
            width, height = int(w), int(h)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, ValueError):
            pass

        response.success = True
        response.message = f'path={target}, width={width}, height={height}'
        return response

    # ------------------------------------------------------------------
    # Widget registry services (Phase 07.1) — list/get/set any registered
    # widget by name. Agent-facing full CLI-level control surface.
    # ------------------------------------------------------------------

    def _widget_rebuilding_response(self, response):
        if self._widget_registry_rebuilding:
            response.success = False
            response.message = 'GUI rebuilding — retry in 1s'
            return True
        return False

    @staticmethod
    def _widget_live_value(entry):
        """Read the current value of a widget entry. Returns (value_str, extra_str)."""
        widget = entry['widget']() if entry['widget'] else None
        var = entry.get('var')
        wtype = entry['type']
        if widget is None:
            return '(widget freed)', ''
        try:
            if wtype == 'Spinbox' or wtype == 'Scale':
                val = var.get() if var is not None else widget.get()
                return f'{val}', ''
            if wtype == 'Entry':
                val = var.get() if var is not None else widget.get()
                return f'{val}', ''
            if wtype == 'Checkbutton':
                val = bool(var.get()) if var is not None else False
                return ('true' if val else 'false'), ''
            if wtype == 'Listbox':
                sel = widget.curselection()
                idx = int(sel[0]) if sel else -1
                count = widget.size()
                text = widget.get(idx) if idx >= 0 else ''
                items = [widget.get(i) for i in range(count)]
                return text, f'index={idx}, count={count}, items=[{"|".join(items)}]'
            if wtype == 'Combobox':
                val = var.get() if var is not None else widget.get()
                values = list(widget.cget('values') or ())
                return f'{val}', f'values=[{"|".join(values)}]'
            if wtype == 'Notebook':
                active = widget.select()
                return widget.tab(active, 'text') if active else '', ''
            if wtype == 'Text':
                raw = widget.get('1.0', tk.END)
                if len(raw) > 10240:
                    raw = '…(truncated)…' + raw[-10240:]
                # Compact summary in extra field
                lines = raw.rstrip('\n').split('\n')
                summary = f'chars={len(raw)}, lines={len(lines)}'
                return raw, summary
        except tk.TclError as e:
            return f'(tcl error: {e})', ''
        return '(unknown type)', ''

    def _build_list_widgets_output(self):
        """Build the markdown table for _srv_list_widgets."""
        lines = []
        lines.append('# Widgets (auto-generated from list_widgets)')
        lines.append('')
        lines.append('> Regenerated from control_gui.py via '
                     '`ros2 service call /so_arm101_control_gui/list_widgets '
                     'std_srvs/srv/Trigger {}`. Do NOT hand-edit.')
        lines.append('')
        lines.append('## Widgets')
        lines.append('')
        lines.append('| Tab | Section | Label | Type | Current Value | Writable |')
        lines.append('|-----|---------|-------|------|---------------|----------|')
        with self._widget_registry_lock:
            items = sorted(self._widget_registry.items(),
                           key=lambda kv: (kv[1].get('tab') or '',
                                           kv[1].get('section') or '',
                                           kv[1].get('label') or ''))
            rows = []
            agent_only = []
            for rid, entry in items:
                value, _extra = self._widget_live_value(entry)
                # Truncate value for table readability
                vshort = value if len(value) <= 30 else value[:27] + '…'
                # Escape | to keep table rendering
                vshort = vshort.replace('|', '\\|').replace('\n', ' ')
                row = (
                    f'| {entry.get("tab") or "-"} '
                    f'| {entry.get("section") or "-"} '
                    f'| `{rid}` '
                    f'| {entry["type"]} '
                    f'| {vshort} '
                    f'| {"yes" if entry["writable"] else "no"} |'
                )
                if entry.get('tab') or entry.get('section'):
                    rows.append(row)
                else:
                    agent_only.append(row)
            lines.extend(rows)
            lines.append('')
            lines.append('## Agent-only widgets (no tab/section)')
            lines.append('')
            lines.append('| Label | Type | Current Value | Writable |')
            lines.append('|-------|------|---------------|----------|')
            for row in agent_only:
                # Strip the tab/section columns for this sub-table
                parts = [p.strip() for p in row.split('|')]
                # parts: ['', tab, section, label, type, value, writable, '']
                lines.append(f'| {parts[3]} | {parts[4]} | {parts[5]} | {parts[6]} |')
        return '\n'.join(lines)

    def _srv_list_widgets(self, request, response):
        """Return the full widget inventory as markdown."""
        if self._widget_rebuilding_response(response):
            return response
        try:
            response.message = self._build_list_widgets_output()
            response.success = True
        except Exception as e:
            response.success = False
            response.message = f'list_widgets error: {e}'
        return response

    def _srv_get_widget_value(self, request, response):
        """Read a widget's current value by id (passed via widget_id param).

        Also accepts button IDs — if widget_id matches a button in
        _button_registry, returns the same state _srv_get_widget_state
        would return (D-11 uniform API).
        """
        if self._widget_rebuilding_response(response):
            return response
        wid = self.get_parameter('widget_id').get_parameter_value().string_value.strip()
        if not wid:
            response.success = False
            response.message = ('no widget_id. Set param first: '
                                "ros2 param set /so_arm101_control_gui widget_id '<id>'")
            return response

        # If it's a button, delegate to the existing widget-state service
        for btn in self._button_registry:
            if btn.get('command_name') == wid or btn.get('text') == wid:
                # Temporarily reuse get_widget_state path via ik_target
                old_param = self.get_parameter('ik_target').get_parameter_value().string_value
                try:
                    from rclpy.parameter import Parameter as RclParam
                    self.set_parameters([RclParam('ik_target', RclParam.Type.STRING, wid)])
                    return self._srv_get_widget_state(request, response)
                finally:
                    self.set_parameters([RclParam('ik_target', RclParam.Type.STRING, old_param)])

        with self._widget_registry_lock:
            entry = self._widget_registry.get(wid)
        if entry is None:
            response.success = False
            response.message = f'widget {wid!r} not found — call ~/list_widgets for the full inventory'
            return response

        value, extra = self._widget_live_value(entry)
        parts = [
            f'id={wid}',
            f'type={entry["type"]}',
            f'tab={entry.get("tab") or "-"}',
            f'section={entry.get("section") or "-"}',
            f'writable={"true" if entry["writable"] else "false"}',
            f'value={value}',
        ]
        if extra:
            parts.append(f'extra={extra}')
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_set_widget_value(self, request, response):
        """Write a value to a registered widget by id.

        Reads widget_id and widget_value params. Coerces the value to the
        widget's expected type (D-15..D-18). Dispatches the write on the
        GUI thread and waits up to 1.0s for completion (D-19).
        """
        if self._widget_rebuilding_response(response):
            return response
        wid = self.get_parameter('widget_id').get_parameter_value().string_value.strip()
        raw = self.get_parameter('widget_value').get_parameter_value().string_value
        if not wid:
            response.success = False
            response.message = ('no widget_id. Set both params first: '
                                "ros2 param set ... widget_id '<id>' && "
                                "ros2 param set ... widget_value '<value>'")
            return response

        # Buttons reject writes
        for btn in self._button_registry:
            if btn.get('command_name') == wid or btn.get('text') == wid:
                response.success = False
                response.message = (f'{wid!r} is a button — use '
                                    f'`ros2 service call ~/{btn.get("command_name","<cmd>")[5:]}`')
                return response

        with self._widget_registry_lock:
            entry = self._widget_registry.get(wid)
        if entry is None:
            response.success = False
            response.message = f'widget {wid!r} not found'
            return response
        if not entry['writable']:
            response.success = False
            response.message = f'widget {wid!r} is read-only'
            return response

        widget_ref = entry['widget']
        widget = widget_ref() if widget_ref else None
        var = entry.get('var')
        wtype = entry['type']
        if widget is None:
            response.success = False
            response.message = f'widget {wid!r} freed — retry after next refresh'
            return response

        # ---- Auto-switch tab so the write is visible in the GUI ----
        # When the target widget is tagged with a tab (FK/IK/Grasp/RViz),
        # switch the main notebook to that tab BEFORE writing, so a human
        # watching the GUI sees the change in context. Skip for the notebook
        # itself (setting 'tab' IS the switch) and for untagged widgets.
        target_tab = entry.get('tab')
        if target_tab and wtype != 'Notebook':
            with self._widget_registry_lock:
                nb_entry = self._widget_registry.get('tab')
            if nb_entry is not None:
                nb_widget = nb_entry['widget']() if nb_entry['widget'] else None
                if nb_widget is not None:
                    try:
                        tabs = [nb_widget.tab(t, 'text') for t in nb_widget.tabs()]
                        if target_tab in tabs:
                            def _switch_tab():
                                try:
                                    nb_widget.select(tabs.index(target_tab))
                                except tk.TclError:
                                    pass
                            if getattr(self, '_gui_ready', False):
                                self.root.after(0, _switch_tab)
                    except tk.TclError:
                        pass

        # ---- Coerce per type ----
        done = threading.Event()
        outcome = {'ok': False, 'msg': ''}

        def _do_write():
            try:
                if wtype in ('Spinbox', 'Scale'):
                    try:
                        val = float(raw)
                    except ValueError:
                        outcome['msg'] = f'{wid!r} expects float, got {raw!r}'
                        return
                    # Clamp to widget's configured range if available
                    try:
                        lo = float(widget.cget('from'))
                        hi = float(widget.cget('to'))
                        if val < lo or val > hi:
                            self._append_log(
                                f'set_widget_value: {wid} clamped {val} to '
                                f'[{lo}, {hi}]', 'warn')
                            val = max(lo, min(val, hi))
                    except (tk.TclError, ValueError):
                        pass
                    if var is not None:
                        var.set(val)
                    else:
                        widget.set(val)
                    # tkinter Scale's `command` only fires on user drag, not
                    # on var.set(val). For arm/IK joint sliders that means
                    # _on_slider doesn't run, so self.joint_positions stays
                    # stale and plan_execute reads the wrong target. Fire
                    # _on_slider manually for joint-name-matched widgets.
                    if wtype == 'Scale':
                        joint_name = wid.split('@')[0]  # strip tab suffix if any
                        if joint_name in ALL_JOINT_NAMES and hasattr(
                                self, '_on_slider'):
                            try:
                                self._on_slider(joint_name, float(val))
                            except Exception:
                                pass
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {val} to {wid}'
                    return
                if wtype == 'Checkbutton':
                    low = raw.strip().lower()
                    if low in ('true', '1', 'yes', 'on'):
                        b = True
                    elif low in ('false', '0', 'no', 'off', ''):
                        b = False
                    else:
                        outcome['msg'] = f'{wid!r} expects bool (true/false/1/0/yes/no), got {raw!r}'
                        return
                    if var is not None:
                        var.set(b)
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {"true" if b else "false"} to {wid}'
                    return
                if wtype == 'Entry':
                    if var is not None:
                        var.set(raw)
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, raw)
                    outcome['ok'] = True
                    outcome['msg'] = f'wrote {raw!r} to {wid}'
                    return
                if wtype == 'Listbox':
                    try:
                        idx = int(raw)
                    except ValueError:
                        outcome['msg'] = f'{wid!r} expects int row index, got {raw!r}'
                        return
                    count = widget.size()
                    if idx < 0 or idx >= count:
                        outcome['msg'] = f'row {idx} out of range (count={count})'
                        return
                    widget.selection_clear(0, tk.END)
                    widget.selection_set(idx)
                    widget.see(idx)
                    outcome['ok'] = True
                    outcome['msg'] = f'selected row {idx} in {wid}'
                    return
                if wtype == 'Notebook':
                    # Find the tab with matching text
                    tabs = [widget.tab(t, 'text') for t in widget.tabs()]
                    if raw not in tabs:
                        outcome['msg'] = (f'tab {raw!r} not found; '
                                          f'available: {", ".join(tabs)}')
                        return
                    widget.select(tabs.index(raw))
                    outcome['ok'] = True
                    outcome['msg'] = f'switched to tab {raw}'
                    return
                if wtype == 'Combobox':
                    values = list(widget.cget('values') or ())
                    if raw not in values:
                        outcome['msg'] = (
                            f'{raw!r} not in combobox values '
                            f'[{", ".join(values)}]')
                        return
                    if var is not None:
                        var.set(raw)
                    else:
                        widget.set(raw)
                    outcome['ok'] = True
                    outcome['msg'] = f'selected {raw!r} in {wid}'
                    return
                outcome['msg'] = f'unsupported widget type {wtype}'
            except Exception as e:
                outcome['msg'] = f'write failed: {e}'
            finally:
                done.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do_write)
        else:
            response.success = False
            response.message = 'GUI not ready'
            return response

        if not done.wait(timeout=1.0):
            response.success = False
            response.message = f'write to {wid} timed out after 1.0s'
            return response

        response.success = outcome['ok']
        response.message = outcome['msg'] or ('ok' if outcome['ok'] else 'unknown error')
        return response

    def _srv_get_joint_positions(self, request, response):
        """Return current joint positions as name=value pairs."""
        with self.joint_lock:
            pairs = [f'{n}={self.joint_positions[n]:.6f}'
                     for n in ALL_JOINT_NAMES]
        response.success = True
        response.message = ', '.join(pairs)
        return response

    def _srv_get_ik_target(self, request, response):
        """Read current IK target fields (XYZ + quaternion)."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                parts = []
                for axis in ['X', 'Y', 'Z']:
                    parts.append(f'{axis}={self.xyz_vars[axis].get():.6f}')
                for comp in ['Roll', 'shoulder_lift', 'Yaw']:
                    parts.append(f'{comp}={self.rpy_vars[comp].get():.1f}')
                qx, qy, qz, qw = self._rpy_deg_to_quat(
                    self.rpy_vars['Roll'].get(),
                    self.rpy_vars['shoulder_lift'].get(),
                    self.rpy_vars['Yaw'].get())
                parts.extend([f'qx={qx:.6f}', f'qy={qy:.6f}',
                              f'qz={qz:.6f}', f'qw={qw:.6f}'])
                result['msg'] = ', '.join(parts)
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    def _srv_set_ik_target(self, request, response):
        """Write IK target fields. Pass key=value pairs in request.message.
        Example: ros2 service call ... '{message: "x=0.12, z=0.15, qw=1.0"}'
        Supported keys: x, y, z, qx, qy, qz, qw (case-insensitive)."""
        done_event = threading.Event()
        result = {'ok': True, 'msg': ''}

        # Parse key=value pairs from the trigger message field
        raw = getattr(request, 'message', '') if hasattr(request, 'message') else ''
        # Trigger doesn't have a message field on request — use a workaround:
        # We'll accept the values from the service call's yaml string that gets
        # stuffed into the Trigger request. But Trigger.Request has no fields.
        # So we need to use a different approach — pass via ROS param or topic.
        # Actually, let's use a simple convention: the caller sets a parameter first.

        # Better approach: read from a latched parameter
        # For now, parse from the node's parameter
        raw = self.get_parameter('ik_target').get_parameter_value().string_value

        if not raw:
            response.success = False
            response.message = (
                'Set param first: ros2 param set /so_arm101_control_gui '
                'ik_target "x=0.12,y=0.0,z=0.15,qw=1.0" '
                'then call this service')
            return response

        # Parse
        updates = {}
        for part in raw.replace(' ', '').split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                updates[k.lower()] = float(v)

        def _write():
            try:
                axis_map = {'x': 'X', 'y': 'Y', 'z': 'Z'}
                rpy_map = {'roll': 'Roll', 'pitch': 'shoulder_lift', 'yaw': 'Yaw'}
                for k, v in updates.items():
                    if k in axis_map and axis_map[k] in self.xyz_vars:
                        self.xyz_vars[axis_map[k]].set(v)
                    elif k in rpy_map and rpy_map[k] in self.rpy_vars:
                        self.rpy_vars[rpy_map[k]].set(v)
                set_keys = ', '.join(f'{k}={v:.4f}' for k, v in updates.items())
                result['msg'] = f'Set: {set_keys}'
            except Exception as e:
                result['ok'] = False
                result['msg'] = str(e)
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _write)
        else:
            result['ok'] = False
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = result['ok']
        response.message = result['msg']
        return response

    def _srv_get_ee_pose(self, request, response):
        """Read current End-Effector pose values (gripper link)."""
        parts = []
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            parts.append(f'{key}={self.ee_labels[key].get()}')
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_get_tcp_pose(self, request, response):
        """Look up tcp_link pose in base frame via TF2."""
        try:
            t = self._tf_buffer.lookup_transform(
                'base', 'tcp_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p = t.transform.translation
            q = t.transform.rotation
            response.success = True
            response.message = (
                f'X={p.x:.6f}, Y={p.y:.6f}, Z={p.z:.6f}, '
                f'qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}')
        except Exception as e:
            response.success = False
            response.message = f'TF lookup failed: {e}'
        return response

    def _srv_get_log(self, request, response):
        """Read last 20 lines from Process Log."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                content = self._process_log.get('1.0', 'end').strip()
                lines = content.split('\n')
                result['msg'] = '\n'.join(lines) if lines else '(empty)'
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    # ------------------------------------------------------------------
    # Controller command publishing (via action interface)
    # ------------------------------------------------------------------

    def _send_arm_goal(self, positions, duration_s=0.5, blocking=False):
        """Send arm joint positions via FollowJointTrajectory action.

        If blocking=True, waits for the trajectory to complete (or fail)
        before returning.  Must be called from a background thread when
        blocking — never from the tkinter or rclpy main thread.
        Returns True on success, False on failure (only meaningful when
        blocking; fire-and-forget always returns None).
        """
        if not self.arm_action_client.server_is_ready():
            self._append_log('arm_controller action server not ready', 'warn')
            return False if blocking else None

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [positions.get(n, 0.0) for n in ARM_JOINT_NAMES]
        point.velocities = [0.0] * len(ARM_JOINT_NAMES)
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Cancel previous goal if still active
        with self._arm_goal_lock:
            if self._arm_goal_handle is not None:
                try:
                    self._arm_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._arm_goal_handle = None

        future = self.arm_action_client.send_goal_async(goal)

        if not blocking:
            future.add_done_callback(self._arm_goal_response)
            return None

        # Blocking path: wait for goal acceptance, then for result
        timeout = duration_s + 10.0
        goal_handle = self._wait_future(future, timeout_sec=5.0)
        if goal_handle is None or not goal_handle.accepted:
            self._append_log('Arm goal rejected or timed out', 'warn')
            return False
        with self._arm_goal_lock:
            self._arm_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=timeout)
        if result is None:
            self._append_log('Arm trajectory timed out', 'warn')
            return False
        if result.status == 4:  # GoalStatus.STATUS_SUCCEEDED
            return True
        self._append_log(f'Arm trajectory finished with status {result.status}', 'warn')
        return False

    def _arm_goal_response(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                with self._arm_goal_lock:
                    self._arm_goal_handle = goal_handle
        except Exception:
            pass

    def _send_gripper_goal(self, jaw_position, duration_s=0.5, blocking=False):
        """Send gripper position via FollowJointTrajectory action.

        If blocking=True, waits for the trajectory to complete before
        returning.  Must be called from a background thread when blocking.
        Returns True/False when blocking, None otherwise.
        """
        if not self.gripper_action_client.server_is_ready():
            self._append_log('gripper_controller action server not ready', 'warn')
            return False if blocking else None

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['gripper_joint']
        point = JointTrajectoryPoint()
        point.positions = [jaw_position]
        point.velocities = [0.0]
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        with self._gripper_goal_lock:
            if self._gripper_goal_handle is not None:
                try:
                    self._gripper_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._gripper_goal_handle = None

        future = self.gripper_action_client.send_goal_async(goal)

        if not blocking:
            future.add_done_callback(self._gripper_goal_response)
            return None

        timeout = duration_s + 10.0
        goal_handle = self._wait_future(future, timeout_sec=5.0)
        if goal_handle is None or not goal_handle.accepted:
            self._append_log('Gripper goal rejected or timed out', 'warn')
            return False
        with self._gripper_goal_lock:
            self._gripper_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=timeout)
        if result is None:
            self._append_log('Gripper trajectory timed out', 'warn')
            return False
        if result.status == 4:
            return True
        self._append_log(f'Gripper trajectory finished with status {result.status}', 'warn')
        return False

    def _gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                with self._gripper_goal_lock:
                    self._gripper_goal_handle = goal_handle
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Joint state feedback
    # ------------------------------------------------------------------

    def _joint_states_callback(self, msg):
        """Update internal state and GUI sliders from joint_state_broadcaster."""
        # Always track actual robot state (never blocked by _slider_driven)
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self._actual_positions and i < len(msg.position):
                    self._actual_positions[name] = msg.position[i]
            # Velocities — used by _wait_arm_at_rest() to synchronize
            # TF+block-pose reads at attach-time (avoids Layer-B-lag
            # baked into AttachedCollisionObject.mesh_poses[0]).
            if msg.velocity:
                for i, name in enumerate(msg.name):
                    if i < len(msg.velocity):
                        self._actual_velocities[name] = msg.velocity[i]
        # On first message, seed joint_positions from actual robot state
        if not self._initial_sync_done:
            self._initial_sync_done = True
            with self.joint_lock:
                for i, name in enumerate(msg.name):
                    if name in self.joint_positions and i < len(msg.position):
                        self.joint_positions[name] = msg.position[i]
                positions = dict(self.joint_positions)
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._sync_all_sliders, positions)
            self._publish_goal_state()
            self.get_logger().info(
                f'Initial joint sync from /joint_states: '
                f'{", ".join(f"{n}={v:.3f}" for n, v in positions.items())}')
            return
        if self._slider_driven:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)
        # Update sliders and goal state to reflect actual robot state
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_all_sliders, positions)
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def _setup_gui_thread(self):
        # macOS requires tkinter on the main thread (Cocoa constraint).
        # Defer _run_gui to be called from main() instead of a daemon thread.
        pass

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title('SO-ARM101 Control')
        self.root.geometry('580x780')
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Status bar
        self.status_var = tk.StringVar(value='Mode: Simulation')
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(status_frame, textvariable=self.status_var, anchor='w',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.hw_var = tk.StringVar(value='sim')
        tk.Radiobutton(status_frame, text='Real Hardware', variable=self.hw_var,
                       value='real', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)
        tk.Radiobutton(status_frame, text='Simulation', variable=self.hw_var,
                       value='sim', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)

        # Ground plane collision toggle (common to all tabs)
        scene_frame = tk.Frame(self.root)
        scene_frame.pack(fill=tk.X, padx=5, pady=1)
        self._ground_plane_var = tk.BooleanVar(value=True)
        self._register_check(scene_frame, label='Ground Plane',
                             section='Scene',
                             variable=self._ground_plane_var,
                             command=self._cmd_toggle_ground_plane).pack(side=tk.LEFT)
        tk.Label(scene_frame, text='  Z:').pack(side=tk.LEFT)
        self._ground_z_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(scene_frame, label='Ground Z',
                               section='Scene',
                               textvariable=self._ground_z_var,
                               from_=-0.5, to=0.5, increment=0.01,
                               width=6).pack(side=tk.LEFT)
        # Publish ground plane on startup after MoveIt is ready
        self.root.after(3000, self._cmd_toggle_ground_plane)

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self._notebook = notebook
        self._register_notebook(notebook, label='tab')

        self._build_individual_tab(notebook)
        self._build_arm_control_tab(notebook)
        self._build_grasp_tab(notebook)
        self._build_quickstart_tab(notebook)
        self._build_real_test_tab(notebook)
        self._build_record_sim_tab(notebook)
        self._build_display_tab(notebook)

        # Auto-populate IK fields when switching to IK tab
        notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

        # --- Log Panel (bottom) ---
        self._build_log_panel()

        # Hot reload keybindings
        self.root.bind('<Control-r>', lambda e: self._hot_reload_logic())
        self.root.bind('<Control-Shift-R>', lambda e: self._hot_reload_gui())

        self._finalize_widget_registry()
        self._gui_ready = True
        self.root.mainloop()
        self._gui_ready = False

    # ------------------------------------------------------------------
    # Tab 1: Individual Joint Control
    # ------------------------------------------------------------------

    def _build_individual_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='FK')

        self.sliders = {}
        self.slider_labels = {}

        # --- Arm Section ---
        arm_frame = ttk.LabelFrame(frame, text='Arm')
        arm_frame.pack(fill=tk.X, padx=10, pady=(5, 2))

        for name in ARM_JOINT_NAMES:
            lo, hi = JOINT_LIMITS[name]
            row = tk.Frame(arm_frame)
            row.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(row, text=name, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

            var = tk.DoubleVar(value=0.0)
            slider = self._register_scale(
                row, label=name, tab='FK', section='Joint Sliders',
                variable=var, from_=lo, to=hi,
                orient=tk.HORIZONTAL, resolution=0.001, length=300,
                command=lambda val, n=name: self._on_slider(n, float(val)))
            slider.pack(side=tk.LEFT, padx=5)

            lbl = tk.Label(row, text='0.000', width=8)
            lbl.pack(side=tk.LEFT)

            self.sliders[name] = var
            self.slider_labels[name] = lbl

        arm_btn_frame = tk.Frame(arm_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_arm_btn_row(arm_btn_frame, tab='FK',
                                reset_cmd=self._cmd_zero_arm)

        # --- Gripper Section ---
        gripper_frame = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row = tk.Frame(gripper_frame)
        jaw_row.pack(fill=tk.X, padx=5, pady=2)

        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

        jaw_var = tk.DoubleVar(value=0.0)
        jaw_slider = self._register_scale(
            jaw_row, label=GRIPPER_JOINT_NAME, tab='FK', section='Joint Sliders',
            variable=jaw_var, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        jaw_slider.pack(side=tk.LEFT, padx=5)

        jaw_lbl = tk.Label(jaw_row, text='0.000', width=8)
        jaw_lbl.pack(side=tk.LEFT)

        self.sliders[GRIPPER_JOINT_NAME] = jaw_var
        self.slider_labels[GRIPPER_JOINT_NAME] = jaw_lbl

        gripper_btn_frame = tk.Frame(gripper_frame)
        gripper_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_gripper_btn_row(gripper_btn_frame, tab='FK')

        # --- Action Buttons (always visible) ---
        action_frame = tk.Frame(frame)
        action_frame.pack(fill=tk.X, padx=10, pady=(8, 5))
        self.set_joints_btn = self._register_button(
            action_frame, text='Set Joints', tab='FK', section='Actions',
            command=self._cmd_set_joints)
        self.set_joints_btn.pack(side=tk.LEFT, padx=5)
        self.execute_btn = self._register_button(
            action_frame, text='Plan & Execute', tab='FK', section='Actions',
            command=self._cmd_plan_execute)
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        tk.Label(action_frame, text='Speed:', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.velocity_scale_var = tk.DoubleVar(value=0.5)
        self._last_speed_val = 0.5
        vcmd = (self.root.register(self._validate_speed), '%P')
        self.velocity_scale_spin = self._register_spinbox(
            action_frame, label='Speed', tab='FK', section='Actions',
            textvariable=self.velocity_scale_var,
            from_=0.1, to=1.0, increment=0.1,
            width=4, font=('Arial', 9), validate='all', validatecommand=vcmd)
        self.velocity_scale_spin.pack(side=tk.LEFT)

    def _validate_speed(self, value_str):
        """Validate speed spinbox: allow empty (during editing) and 0.1-1.0."""
        if value_str == '':
            return True
        try:
            v = float(value_str)
            if v > 1.0:
                self._append_log('Speed must be between 0.1 and 1.0 (100% of joint limits)', 'warn')
                return False
            if v < 0.1:
                self._append_log('Speed must be between 0.1 and 1.0', 'warn')
                return False
            if v != self._last_speed_val:
                self._last_speed_val = v
                pct = int(v * 100)
                self._append_log(f'Speed set to {v:.1f} ({pct}% of joint limits)')
            return True
        except ValueError:
            return False

    def _on_slider(self, joint_name, value):
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions[joint_name] = value
        if joint_name in self.slider_labels:
            self.slider_labels[joint_name].config(text=f'{value:.3f}')
        # Sync IK tab jaw label (sliders share the same DoubleVar)
        if joint_name == GRIPPER_JOINT_NAME and hasattr(self, '_ik_jaw_label'):
            self._ik_jaw_label.config(text=f'{value:.3f}')
        # Switch planning group based on which joint is being moved
        if joint_name == GRIPPER_JOINT_NAME:
            self._select_planning_group('gripper')
        else:
            self._select_planning_group('arm')
        # Update RViz goal state only — robot doesn't move until button click
        self._publish_goal_state()

    def _cmd_zero_arm(self):
        """Reset arm joints to zero."""
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                self.joint_positions[name] = 0.0
        for name in ARM_JOINT_NAMES:
            if name in self.sliders:
                self.sliders[name].set(0.0)
                self.slider_labels[name].config(text='0.000')
        self._publish_goal_state()
        self.status_var.set('Arm zeroed')

    def _cmd_reset_grasp_home(self):
        """Set slider targets to the grasp-home pose (gripper pointing down).

        Mirrors _cmd_zero_arm: sets joint_positions + sliders + goal state,
        but with the grasp_home config instead of zeros. Does NOT execute
        motion — only stages the target (same discipline as Reset Arm).
        """
        from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
        target = {name: 0.0 for name in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()
        self.status_var.set('Grasp home target staged')

    def _cmd_gripper_zero(self):
        """Reset gripper to zero (closed)."""
        self._gripper_command(0.0)
        self.status_var.set('Gripper zeroed')

    def _select_planning_group(self, group_name):
        """Switch the active planning group in RViz.

        On first call, cycles through gripper→arm to clear startup markers.
        """
        if not hasattr(self, '_planning_group_pub'):
            return
        from std_msgs.msg import String
        # First call: cycle gripper→arm to clear green startup markers
        if not hasattr(self, '_planning_group_initialized'):
            self._planning_group_initialized = True
            msg = String()
            msg.data = 'gripper'
            self._planning_group_pub.publish(msg)
        if hasattr(self, '_active_planning_group') and self._active_planning_group == group_name:
            return
        self._active_planning_group = group_name
        msg = String()
        msg.data = group_name
        self._planning_group_pub.publish(msg)
        # Republish goal state after RViz processes the group switch
        if getattr(self, '_gui_ready', False):
            self.root.after(150, self._publish_goal_state)

    def _cmd_randomize_arm(self):
        """Set arm joints to a random collision-free configuration."""
        self._select_planning_group('arm')
        self.status_var.set('Finding random valid state...')

        def _find_valid():
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate random joint values within limits
                positions = {}
                for name in ARM_JOINT_NAMES:
                    lo, hi = JOINT_LIMITS[name]
                    positions[name] = random.uniform(lo, hi)

                # Check validity via MoveIt
                if MOVEIT_AVAILABLE and hasattr(self, 'validity_client') \
                        and self.validity_client.service_is_ready():
                    req = GetStateValidity.Request()
                    req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                    req.robot_state.joint_state.position = [
                        positions[n] for n in ARM_JOINT_NAMES]
                    req.group_name = 'arm'
                    future = self.validity_client.call_async(req)
                    self._wait_future(future, timeout_sec=1.0)
                    if future.result() is not None and not future.result().valid:
                        continue  # collision — retry
                # Valid (or no validity service available) — apply it
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._apply_random_arm, positions)
                return

            # Exhausted attempts — apply last one anyway
            self._append_log(f'No collision-free state after {max_attempts} attempts', 'warn')
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._apply_random_arm, positions)

        threading.Thread(target=_find_valid, daemon=True).start()

    def _apply_random_arm(self, positions):
        """Apply validated random arm positions to sliders and goal state."""
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions.update(positions)
        for name in ARM_JOINT_NAMES:
            val = positions[name]
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')
        self._publish_goal_state()
        self.status_var.set('Arm randomized (valid)')

    # ------------------------------------------------------------------
    # Planning Mode
    # ------------------------------------------------------------------


    def _publish_goal_state(self):
        """Publish current slider positions (arm + gripper) as goal state to RViz."""
        if not MOVEIT_AVAILABLE or not hasattr(self, '_goal_state_pub'):
            return
        goal_state = RobotState()
        goal_state.is_diff = True  # Required by MotionPlanning plugin
        goal_state.joint_state.name = list(ALL_JOINT_NAMES)
        with self.joint_lock:
            goal_state.joint_state.position = [
                self.joint_positions[n] for n in ALL_JOINT_NAMES]
        self._goal_state_pub.publish(goal_state)

    def _cmd_set_joints(self):
        """Send current slider positions directly to arm + gripper controllers."""
        with self.joint_lock:
            positions = dict(self.joint_positions)
        self._send_arm_goal(positions, duration_s=0.5)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.5)
        if self.use_real_hardware:
            self._send_hw_command(positions)
        joints_str = ', '.join(f'{n}: {positions[n]:.3f}' for n in ARM_JOINT_NAMES)
        jaw_str = f'{GRIPPER_JOINT_NAME}: {jaw:.3f}'
        self.status_var.set('Joints set')
        self._append_log(f'Set Joints → {joints_str}, {jaw_str}')
        # Allow joint_states to sync sliders again after robot reaches goal
        self.root.after(1000, self._clear_slider_driven)

    def _clear_slider_driven(self):
        """Allow joint_states callback to sync sliders again."""
        self._slider_driven = False

    # ------------------------------------------------------------------
    # Collision-free IK planning: geometric IK → collision check → MoveIt path
    # ------------------------------------------------------------------

    def _plan_collision_free_execute(self, x, y, z, grip_angle,
                                            wrist_roll, on_complete=None,
                                            lock_pan=None, duration_s=3.0):
        """Solve geometric IK, then plan a collision-free path via MoveIt.

        Full pipeline: geometric_ik → collision check on solutions →
        MoveIt path planning around obstacles → trajectory execution.

        Args:
            x, y, z: TCP target position in base frame (meters).
            grip_angle: gripper orientation constraint (radians).
                0 = horizontal, π/4 = 45° down, π/2 = straight down.
            wrist_roll: fixed wrist_roll angle (radians).
            on_complete: optional threading.Event to set when done.
            lock_pan: optional shoulder_pan value (radians) to overwrite in
                the chosen IK solution before dispatch. Used by drop_sweep
                to keep the base joint at drop_point's pan — geometric IK
                on the half_gap-shifted tcp_target gives a slightly
                different pan than the cup-center pan drop_point used,
                producing a visible ~1° base yaw during the sweep that has
                no physical justification. Mirrors the "single yaw across
                stages" pattern in find_reachable_grasp_yaw (line 7212).
        Returns immediately; execution is async.
        """
        from so_arm101_control.compute_workspace import geometric_ik
        # Switch RViz panel to arm group so the Goal State ghost shows the
        # arm target (gripper commands set it to "gripper").
        self._select_planning_group('arm')

        solutions = geometric_ik(x, y, z, grip_angle=grip_angle,
                                 wrist_roll=wrist_roll)
        if not solutions:
            self._append_log(
                f'IK: no solution for ({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
            # Write status BEFORE on_complete.set() so QS runner's
            # _qs_wait_for_step sees the failure verdict (not a stale None
            # that reads as success).
            self._last_motion_status = {
                'ok': False, 'outcome': 'ik_no_solution',
                'msg': (f'geometric IK returned no solutions for '
                        f'({x:.3f}, {y:.3f}, {z:.3f})')}
            if on_complete:
                on_complete.set()
            return

        # Pick first collision-free solution
        chosen = None
        for i, sol in enumerate(solutions):
            config = 'elbow-up' if i == 0 else 'elbow-down'
            if self._check_state_valid(sol):
                chosen = sol
                self._append_log(f'  IK: {config} collision-free')
                break
            self._append_log(f'  IK: {config} collides', 'warn')

        if chosen is None:
            self._append_log('IK: all solutions collide', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'ik_all_collide',
                'msg': 'all geometric IK solutions collided with scene'}
            if on_complete:
                on_complete.set()
            return

        # Optional pan-lock: overwrite the IK-chosen shoulder_pan with the
        # caller-provided value. drop_sweep uses this to suppress the
        # visible base yaw caused by IK solving on the shifted tcp_target
        # vs. drop_point's cup-center pan. Re-validate with locked pan
        # since the small angular difference can move the gripper into
        # a new collision (rare but possible \u2014 surface as ik_pan_lock_collide).
        if lock_pan is not None:
            ik_pan = chosen['shoulder_pan']
            chosen['shoulder_pan'] = float(lock_pan)
            delta_deg = math.degrees(lock_pan - ik_pan)
            self._append_log(
                f'  pan-lock: {math.degrees(ik_pan):+.2f}\u00b0 \u2192 '
                f'{math.degrees(lock_pan):+.2f}\u00b0 (\u0394{delta_deg:+.2f}\u00b0)')
            if not self._check_state_valid(chosen):
                self._append_log(
                    'IK: pan-locked solution collides', 'warn')
                self._last_motion_status = {
                    'ok': False, 'outcome': 'ik_pan_lock_collide',
                    'msg': (f'pan-locked IK solution at '
                            f'{math.degrees(lock_pan):.1f}\u00b0 collides')}
                if on_complete:
                    on_complete.set()
                return

        self._append_log(
            f'IK target: pan={math.degrees(chosen["shoulder_pan"]):.1f}\u00b0 '
            f'lift={math.degrees(chosen["shoulder_lift"]):.1f}\u00b0 '
            f'elbow={math.degrees(chosen["elbow_flex"]):.1f}\u00b0 '
            f'wrist_flex={math.degrees(chosen["wrist_flex"]):.1f}\u00b0 '
            f'wrist_roll={math.degrees(chosen["wrist_roll"]):.1f}\u00b0')

        # ONE IK path: unified motion primitive handles direct joint-space
        # interpolation + OMPL-with-validated-retry fallback. Same pattern as
        # _cmd_grasp_home.
        if on_complete is None:
            on_complete = threading.Event()
        self._joint_space_collision_free_execute(
            chosen, on_complete_event=on_complete, duration_s=duration_s)

    # ------------------------------------------------------------------
    # FK tab Plan & Execute (MoveIt collision-aware path planning)
    # ------------------------------------------------------------------

    def _cmd_plan_execute(self, target=None, on_complete=None, planner_id='',
                          planning_time=10.0):
        """Plan and execute via MoveIt with collision avoidance.

        Args:
            target: dict of joint name → angle. If None, reads from slider
                positions (GUI button mode).
            on_complete: optional threading.Event to set when done
                (programmatic/service mode).
            planner_id: OMPL planner to use (e.g. 'RRTstar' for optimized
                path). Empty string = default (RRTConnect).
            planning_time: max seconds for planner (default 10s, use 30s
                for optimizing planners like RRTstar).
        """
        if not MOVEIT_AVAILABLE or self.plan_client is None:
            self.status_var.set('MoveIt not available')
            self._last_motion_status = {
                'ok': False, 'outcome': 'moveit_unavailable',
                'msg': 'MoveIt not available'}
            if on_complete:
                on_complete.set()
            return
        if not self.plan_client.service_is_ready():
            self.status_var.set('Planning service not ready...')
            self._append_log('/plan_kinematic_path service not ready', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_service_not_ready',
                'msg': '/plan_kinematic_path service not ready'}
            if on_complete:
                on_complete.set()
            return

        # If a lego is attached, re-snap its tcp_link-local pose from Isaac's
        # current /objects_poses_sim reading. Arm should be at rest here (any
        # previous motion has completed), so the re-seat happens on synchronized
        # TF+pose streams — no Layer-B-lag baked in. Fixes the "attached lego
        # offset from gripper" RViz visualization bug.
        if getattr(self, '_attached_lego_name', None):
            try:
                self._refresh_attached_pose()
            except Exception as exc:
                self._append_log(f'refresh_attached_pose failed: {exc}', 'warn')

        self.root.after(0, lambda: self.execute_btn.config(state=tk.DISABLED))
        self._set_status('Planning...')
        # Tell the RViz MotionPlanning panel we're planning for the arm.
        # Gripper commands published "gripper" to /rviz/moveit/select_planning_group
        # (control_gui.py:6102). Without this switch-back, RViz's Goal State ghost
        # stays locked to the gripper group and renders no orange arm ghost.
        self._select_planning_group('arm')
        # Phase 9: if no on_complete was passed (button click or Trigger
        # service), create one and register as _motion_event so the service
        # wrapper waits for the real motion verdict instead of returning
        # "dispatched" immediately.
        if on_complete is None:
            on_complete = threading.Event()
            self._motion_event = on_complete
        self._plan_execute_on_complete = on_complete

        if target is None:
            with self.joint_lock:
                target = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
                self._execute_jaw_target = self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0)
        else:
            # Sync sliders so RViz ghost updates to match the target
            with self.joint_lock:
                for n in ARM_JOINT_NAMES:
                    self.joint_positions[n] = target[n]
            self._execute_jaw_target = None

        goal_str = ', '.join(f'{n}: {target[n]:.3f}' for n in ARM_JOINT_NAMES)
        self._append_log(f'Plan & Execute \u2192 {goal_str}')

        constraints = Constraints()
        for name in ARM_JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target[name]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.pipeline_id = 'ompl'
        mpr.planner_id = planner_id
        attempts_var = getattr(self, '_planning_attempts_var', None)
        mpr.num_planning_attempts = attempts_var.get() if attempts_var else 50
        mpr.allowed_planning_time = planning_time
        vel_scale = self.velocity_scale_var.get()
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = vel_scale
        mpr.goal_constraints.append(constraints)

        # Phase 9: populate start_state from live joint positions. Without
        # this MoveIt emits "Found empty JointState message" errors and
        # silently aborts planning (verified: 4 such errors per call when
        # start_state was default-constructed; no trajectory returned).
        # _ompl_plan_sync already does this; FK-tab plan_execute was the
        # one remaining path missing it.
        start_state = RobotState()
        start_state.joint_state.name = list(ALL_JOINT_NAMES)
        with self.joint_lock:
            start_state.joint_state.position = [
                float(self._actual_positions.get(
                    n, self.joint_positions.get(n, 0.0)))
                for n in ALL_JOINT_NAMES]
        mpr.start_state = start_state

        request.motion_plan_request = mpr

        future = self.plan_client.call_async(request)
        future.add_done_callback(self._plan_and_execute_callback)

    def _plan_and_execute_callback(self, future):
        """Handle planning result — display trajectory in RViz, then execute."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            resp = future.result()
        except Exception as e:
            self._set_status(f'Planning failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_exception',
                'msg': f'planning exception: {e!r}'}
            if on_complete:
                on_complete.set()
            return

        error_code = resp.motion_plan_response.error_code.val
        if error_code != 1:
            self._set_status(f'Planning failed (error {error_code})')
            self._append_log(f'Planning failed (error {error_code})', 'warn')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_failed',
                'msg': f'plan_kinematic_path error_code={error_code}',
                'error_code': error_code}
            if on_complete:
                on_complete.set()
            return

        robot_trajectory = resp.motion_plan_response.trajectory
        pt = resp.motion_plan_response.planning_time
        n_pts = len(robot_trajectory.joint_trajectory.points)
        self._append_log(f'Plan found ({pt:.3f}s), {n_pts} points')

        # Mode B post-check — subsamples between waypoints for collisions
        # OMPL's ~3.6°-spaced discrete check can miss. Same discipline the
        # grasp path runs at _ompl_plan_validate_execute (line 5883).
        # Without this, OMPL-validated trajectories can still clip cups at
        # sub-segment states and get executed. Parity closed 2026-04-24.
        chk = self._trajectory_first_invalid_with_contacts(
            robot_trajectory.joint_trajectory)
        if not chk.ok:
            contact_summary = '; '.join(
                f'{c.contact_body_1}↔{c.contact_body_2}(d={c.depth*1000:.1f}mm)'
                for c in chk.contacts[:5]) or 'no contact info'
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            self._set_status(f'Plan rejected by Mode B: {where}')
            self._append_log(
                f'  Mode B rejected {where} '
                f'({chk.n_wps} wps + {chk.n_sub} subs): {contact_summary}',
                'warn')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'plan_mode_b_rejected',
                'msg': f'Mode B rejected {where}: {contact_summary}',
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t}
            if on_complete:
                on_complete.set()
            return
        self._append_log(
            f'  Mode B validated ({chk.n_wps} wps + {chk.n_sub} subs clear)')

        # Display trajectory in RViz
        from moveit_msgs.msg import DisplayTrajectory
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(robot_trajectory)
        self._display_traj_pub.publish(display_msg)

        # Execute via arm_controller
        if not self.arm_action_client.server_is_ready():
            self._set_status('Arm controller not ready')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'action_server_not_ready',
                'msg': 'arm_controller action server not ready'}
            if on_complete:
                on_complete.set()
            return

        self._set_status(f'Executing ({n_pts} points)...')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory
        send_future = self.arm_action_client.send_goal_async(goal)
        send_future.add_done_callback(self._execute_response)

        # Send gripper to its goal position too (only from GUI button path)
        if getattr(self, '_execute_jaw_target', None) is not None:
            self._send_gripper_goal(self._execute_jaw_target, duration_s=1.0)

    def _execute_response(self, future):
        """Handle trajectory execution acceptance."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            goal_handle = future.result()
        except Exception as e:
            self._set_status(f'Execution failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'send_exception',
                'msg': f'action send_goal exception: {e!r}'}
            if on_complete:
                on_complete.set()
            return
        if not goal_handle.accepted:
            self._set_status('Execution rejected')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            self._last_motion_status = {
                'ok': False, 'outcome': 'goal_rejected',
                'msg': 'arm action server rejected the goal'}
            if on_complete:
                on_complete.set()
            return
        self._set_status('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._execute_result)

    def _execute_result(self, future):
        """Handle trajectory execution result."""
        on_complete = getattr(self, '_plan_execute_on_complete', None)
        try:
            res = future.result()
            ec = getattr(getattr(res, 'result', None), 'error_code', None)
            ok = (ec is None) or (ec == 0)
            if ok:
                self._set_status('Execution complete')
                self._append_log('Trajectory execution complete')
                self._last_motion_status = {
                    'ok': True, 'outcome': 'completed',
                    'msg': 'plan_execute trajectory complete',
                    'error_code': ec}
            else:
                self._set_status(f'Execution error: ec={ec}')
                self._append_log(f'Trajectory error_code={ec}', 'error')
                self._last_motion_status = {
                    'ok': False, 'outcome': 'trajectory_error',
                    'msg': f'trajectory error_code={ec}',
                    'error_code': ec}
        except Exception as e:
            self._set_status(f'Execution error: {e}')
            self._append_log(f'Execution error: {e}', 'error')
            self._last_motion_status = {
                'ok': False, 'outcome': 'result_exception',
                'msg': f'result callback exception: {e!r}'}
        self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
        # Let joint_states sync sliders back to actual robot position
        self.root.after(1000, self._clear_slider_driven)
        if on_complete:
            on_complete.set()

    def _set_status(self, text):
        """Thread-safe status bar update."""
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self.status_var.set, text)

    # ------------------------------------------------------------------
    # Log Panel
    # ------------------------------------------------------------------

    def _build_log_panel(self):
        """Build the bottom log panel with Process Log and System Errors tabs."""
        self._log_outer = outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, padx=5, pady=(2, 5), expand=False)

        log_notebook = ttk.Notebook(outer)
        log_notebook.pack(fill=tk.BOTH, expand=True)
        self._log_notebook = log_notebook
        self._register_notebook(log_notebook, label='log_tab')

        # Process Log tab — text + buttons inside
        proc_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(proc_frame, text='Process Log')
        proc_btn = tk.Frame(proc_frame)
        proc_btn.pack(side=tk.RIGHT, fill=tk.Y)
        self._register_button(proc_btn, text='Clear', tab='Logs', section='Process', width=6,
                              command=self._log_btn_clear).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        self._register_button(proc_btn, text='Copy', tab='Logs', section='Process', width=6,
                              command=self._log_btn_copy).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        proc_scroll = tk.Scrollbar(proc_frame)
        proc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._process_log = tk.Text(proc_frame, height=6, wrap=tk.WORD,
                                     font=('Consolas', 9), state=tk.DISABLED,
                                     bg='#1e1e1e', fg='#d4d4d4', borderwidth=0)
        self._register_log_text(self._process_log, label='process_log')
        self._process_log.config(yscrollcommand=proc_scroll.set)
        proc_scroll.config(command=self._process_log.yview)
        self._process_log.tag_configure('info', foreground='#d4d4d4')
        self._process_log.tag_configure('warn', foreground='#ffa500')
        self._process_log.tag_configure('error', foreground='#ff4444')
        self._process_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # System Errors tab — text + buttons inside
        err_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(err_frame, text='System Errors')
        err_btn = tk.Frame(err_frame)
        err_btn.pack(side=tk.RIGHT, fill=tk.Y)
        self._register_button(err_btn, text='Clear', tab='Logs', section='Errors', width=6,
                              command=self._log_btn_clear).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        self._register_button(err_btn, text='Copy', tab='Logs', section='Errors', width=6,
                              command=self._log_btn_copy).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        err_scroll = tk.Scrollbar(err_frame)
        err_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._error_log = tk.Text(err_frame, height=6, wrap=tk.WORD,
                                   font=('Consolas', 9), state=tk.DISABLED,
                                   bg='#1e1e1e', fg='#ff6b6b', borderwidth=0)
        self._register_log_text(self._error_log, label='error_log')
        self._error_log.config(yscrollcommand=err_scroll.set)
        err_scroll.config(command=self._error_log.yview)
        self._error_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Capture stderr and uncaught exceptions into System Errors
        self._setup_stderr_capture()

    def _get_active_log_widget(self):
        """Return the Text widget for the currently selected log tab."""
        idx = self._log_notebook.index(self._log_notebook.select())
        return self._process_log if idx == 0 else self._error_log

    def _log_btn_clear(self):
        widget = self._get_active_log_widget()
        widget.config(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        widget.config(state=tk.DISABLED)

    def _log_btn_copy(self):
        widget = self._get_active_log_widget()
        content = widget.get('1.0', tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._append_log('Log copied to clipboard')

    def _setup_stderr_capture(self):
        """Redirect stderr and sys.excepthook to System Errors log."""
        if hasattr(self, '_stderr_captured'):
            return  # already redirected (e.g. after GUI rebuild)
        self._stderr_captured = True
        import sys

        gui = self  # prevent closure issues
        original_stderr = sys.stderr

        class StderrRedirector:
            def __init__(self):
                self._buffer = ''

            def write(self, text):
                original_stderr.write(text)  # still print to terminal
                if not text.strip():
                    return
                self._buffer += text
                if '\n' in self._buffer:
                    lines = self._buffer.split('\n')
                    for line in lines[:-1]:
                        if line.strip():
                            gui._append_system_error(line)
                    self._buffer = lines[-1]

            def flush(self):
                if self._buffer.strip():
                    gui._append_system_error(self._buffer)
                    self._buffer = ''
                original_stderr.flush()

        sys.stderr = StderrRedirector()

        original_excepthook = sys.excepthook

        def _excepthook(exc_type, exc_value, exc_tb):
            import traceback
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            gui._append_system_error(tb_str)
            original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook

    def _append_system_error(self, text):
        """Thread-safe append to System Errors log only."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        def _do():
            if not hasattr(self, '_error_log'):
                return
            self._error_log.config(state=tk.NORMAL)
            self._error_log.insert(tk.END, line)
            self._error_log.see(tk.END)
            self._error_log.config(state=tk.DISABLED)
            # Auto-switch to System Errors tab
            self._log_notebook.select(1)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do)

    def _append_log(self, text, level='info'):
        """Thread-safe log append to Process Log. level: 'info', 'warn', 'error'.

        When level is 'warn' or 'error', also sets _cmd_error so the
        trigger service callback can report success=False.

        ALSO routes to the node's rclpy logger so every GUI-level motion
        event (Attached, Detached, Grasp Home, OMPL post-check, etc.)
        lands in the launch rosout log with a wall-clock timestamp.
        Without this, those events only existed in the in-memory Tk log
        buffer (ring-truncated to a few dozen lines), making post-hoc
        reconstruction of a run impossible. If you're grep'ing
        /tmp/so_arm101_mtc.log for a sequence, these lines are now there.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        if level in ('warn', 'error'):
            self._cmd_error = text

        # Persist to rosout. Use the level mapping so grep-by-severity works.
        try:
            logger = self.get_logger()
            if level == 'error':
                logger.error(text)
            elif level == 'warn':
                logger.warn(text)
            else:
                logger.info(text)
        except Exception:
            pass  # Node may not be fully initialized during teardown

        def _do_append():
            if not hasattr(self, '_process_log'):
                return
            self._process_log.config(state=tk.NORMAL)
            self._process_log.insert(tk.END, line, level)
            self._process_log.see(tk.END)
            self._process_log.config(state=tk.DISABLED)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do_append)

    # ------------------------------------------------------------------
    # Hot Reload (Ctrl+R = logic, Ctrl+Shift+R = logic + GUI rebuild)
    # ------------------------------------------------------------------

    def _hot_reload_logic(self):
        """Reload control_gui + compute_workspace modules and patch methods.

        Preserves all ROS2 infra, widgets, locks, and runtime state.
        Only method implementations and module-level constants are updated.

        Reloads from the SOURCE directory (not build/) so edits to src/
        take effect without running colcon build first.
        """
        import importlib
        import importlib.util
        import sys

        try:
            # Resolve source directory from the build path
            # build/.../control_gui.py → src/.../control_gui.py
            src_dir = None
            mod_file = sys.modules.get('so_arm101_control.control_gui')
            if mod_file and hasattr(mod_file, '__file__'):
                build_path = mod_file.__file__
                # Replace /build/ with /src/ in the path
                if '/build/' in build_path:
                    src_path = build_path.replace('/build/', '/src/')
                    if os.path.isfile(src_path):
                        src_dir = os.path.dirname(src_path)

            def _reload_from_source(mod_name):
                """Reload a module from src/ instead of build/."""
                mod = sys.modules.get(mod_name)
                if mod is None:
                    return None
                if src_dir:
                    filename = os.path.basename(mod.__file__)
                    src_file = os.path.join(src_dir, filename)
                    if os.path.isfile(src_file):
                        mod.__file__ = src_file
                        # Also update the loader's path so reload reads the right file
                        spec = importlib.util.spec_from_file_location(mod_name, src_file)
                        mod.__spec__ = spec
                        mod.__loader__ = spec.loader
                return importlib.reload(mod)

            # Reload compute_workspace first (control_gui imports from it)
            _reload_from_source('so_arm101_control.compute_workspace')

            # Reload control_gui
            new_mod = _reload_from_source('so_arm101_control.control_gui')

            # Patch all methods on this running instance
            _patch_methods(self, new_mod.SOArm101ControlGUI)

            # Register any newly-added _cmd_* methods as Trigger services.
            # Without this, new debug services introduced via hot-reload stay
            # callable as Python methods but are invisible to ROS2 service calls.
            existing_names = set()
            for s in self._debug_services:
                sn = getattr(s, 'srv_name', '') or ''
                existing_names.add(sn.rsplit('/', 1)[-1])
            new_count = 0
            for name in sorted(dir(self)):
                if not name.startswith('_cmd_'):
                    continue
                srv_name = name[5:]
                if srv_name in existing_names:
                    continue
                cb = self._make_trigger_callback(name)
                srv = self.create_service(Trigger, f'~/{srv_name}', cb)
                self._debug_services.append(srv)
                new_count += 1
            if new_count:
                self._append_log(
                    f'HOT RELOAD: registered {new_count} new _cmd_* service(s)', 'info')

            src_label = ' (from src/)' if src_dir else ' (from build/)'
            self._append_log(f'HOT RELOAD: logic reloaded{src_label}', 'info')
        except Exception as e:
            import traceback
            self._append_log(f'HOT RELOAD FAILED: {e}', 'error')
            self._append_system_error(traceback.format_exc())

    def _save_gui_state(self):
        """Snapshot all GUI variable values before a GUI rebuild."""
        state = {}

        # FK tab
        state['sliders'] = {n: v.get() for n, v in self.sliders.items()}
        state['velocity_scale'] = self.velocity_scale_var.get()
        state['slider_driven'] = self._slider_driven
        state['last_speed_val'] = self._last_speed_val

        # IK tab
        state['xyz'] = {k: v.get() for k, v in self.xyz_vars.items()}
        state['rpy'] = {k: v.get() for k, v in self.rpy_vars.items()}
        state['ik_last_valid'] = dict(self._ik_last_valid)
        state['ik_valid'] = self._ik_valid
        state['ik_planned_target'] = self._ik_planned_target
        state['ik_status'] = self.ik_status_var.get()
        state['ee_labels'] = {k: v.get() for k, v in self.ee_labels.items()}

        # Grasp tab
        state['grasp_topic'] = self._grasp_topic_var.get()
        state['bbox_topic'] = self._bbox_topic_var.get()
        state['bbox_enabled'] = self._bbox_enabled_var.get()
        state['grasp_arm_duration'] = self._grasp_arm_duration_var.get()
        state['grasp_approach_height'] = self._grasp_approach_height_var.get()
        state['grasp_obj_z'] = self._grasp_obj_z_var.get()
        state['grasp_cross'] = self._grasp_cross_var.get()
        state['grasp_grip_close'] = self._grasp_grip_close_var.get()
        state['grasp_grip_open'] = self._grasp_grip_open_var.get()
        state['grasp_grip_duration'] = self._grasp_grip_duration_var.get()
        state['jaw_open_clearance'] = self._jaw_open_clearance_var.get()
        state['jaw_close_clearance'] = self._jaw_close_clearance_var.get()
        state['tcp_clearance'] = self._tcp_clearance_var.get()
        state['drop_hover_above_rim'] = self._drop_hover_above_rim_var.get()

        # Object listbox
        sel = self.obj_listbox.curselection()
        state['obj_selection'] = sel[0] if sel else None

        # Log contents
        state['process_log'] = self._process_log.get('1.0', 'end-1c')
        state['error_log'] = self._error_log.get('1.0', 'end-1c')
        state['log_tab'] = self._log_notebook.index(self._log_notebook.select())

        # Active notebook tab
        state['active_tab'] = self._notebook.index(self._notebook.select())

        return state

    def _restore_gui_state(self, state):
        """Restore GUI variable values after a GUI rebuild."""
        # FK tab
        for name, val in state.get('sliders', {}).items():
            if name in self.sliders:
                self.sliders[name].set(val)
                if name in self.slider_labels:
                    self.slider_labels[name].config(text=f'{val:.3f}')
        if 'velocity_scale' in state:
            self.velocity_scale_var.set(state['velocity_scale'])
        self._slider_driven = state.get('slider_driven', False)
        self._last_speed_val = state.get('last_speed_val', 0.5)

        # IK tab — suppress IK solves during batch restore
        self._ik_trace_active = False
        for k, v in state.get('xyz', {}).items():
            if k in self.xyz_vars:
                self.xyz_vars[k].set(v)
        for k, v in state.get('rpy', {}).items():
            if k in self.rpy_vars:
                self.rpy_vars[k].set(v)
        self._ik_trace_active = True
        self._ik_last_valid = state.get('ik_last_valid', {})
        self._ik_valid = state.get('ik_valid', True)
        self._ik_planned_target = state.get('ik_planned_target')
        self.ik_status_var.set(state.get('ik_status', 'Ready'))
        for k, v in state.get('ee_labels', {}).items():
            if k in self.ee_labels:
                self.ee_labels[k].set(v)

        # Grasp tab
        self._grasp_topic_var.set(state.get('grasp_topic', ''))
        self._bbox_topic_var.set(state.get('bbox_topic', ''))
        self._bbox_enabled_var.set(state.get('bbox_enabled', True))
        self._grasp_arm_duration_var.set(state.get('grasp_arm_duration', 2.5))
        self._grasp_approach_height_var.set(state.get('grasp_approach_height', 0.020))
        self._grasp_obj_z_var.set(state.get('grasp_obj_z', 0.0))
        self._grasp_cross_var.set(state.get('grasp_cross', False))
        self._grasp_grip_close_var.set(state.get('grasp_grip_close', -10.0))
        self._grasp_grip_open_var.set(state.get('grasp_grip_open', 100.0))
        self._grasp_grip_duration_var.set(state.get('grasp_grip_duration', 3.0))
        self._jaw_open_clearance_var.set(state.get('jaw_open_clearance', 5.0))
        self._jaw_close_clearance_var.set(state.get('jaw_close_clearance', 0.0))
        self._tcp_clearance_var.set(state.get('tcp_clearance', 1.0))
        self._drop_hover_above_rim_var.set(state.get('drop_hover_above_rim', 0.050))

        # Repopulate listbox from current objects_data
        self._populate_object_list()
        if state.get('obj_selection') is not None:
            idx = state['obj_selection']
            if idx < self.obj_listbox.size():
                self.obj_listbox.selection_set(idx)

        # Restore log contents
        self._process_log.config(state=tk.NORMAL)
        self._process_log.insert('1.0', state.get('process_log', ''))
        self._process_log.see(tk.END)
        self._process_log.config(state=tk.DISABLED)

        self._error_log.config(state=tk.NORMAL)
        self._error_log.insert('1.0', state.get('error_log', ''))
        self._error_log.see(tk.END)
        self._error_log.config(state=tk.DISABLED)

        # Restore active tabs
        if state.get('log_tab') is not None:
            try:
                self._log_notebook.select(state['log_tab'])
            except Exception:
                pass
        if state.get('active_tab') is not None:
            try:
                self._notebook.select(state['active_tab'])
            except Exception:
                pass

    def _hot_reload_gui(self):
        """Rebuild all GUI tabs with new code. Preserves state and ROS2 infra."""
        try:
            state = self._save_gui_state()
            self._hot_reload_logic()

            self._gui_ready = False
            self._ik_debounce_id = None
            # Widget registry is keyed by label + tab. Tabs are about to be
            # rebuilt from scratch, so the current entries are stale.
            # Flag services to short-circuit with "retry in 1s" until finalize.
            with self._widget_registry_lock:
                self._widget_registry.clear()
            self._widget_registry_rebuilding = True

            # Destroy notebook and log panel
            self._notebook.destroy()
            self._log_outer.destroy()

            # Recreate notebook
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
            self._notebook = notebook
            self._register_notebook(notebook, label='tab')

            # Rebuild tabs with error-safe wrappers
            for builder, tab_name in [
                (self._build_individual_tab, 'FK'),
                (self._build_arm_control_tab, 'IK'),
                (self._build_grasp_tab, 'Grasp'),
                (self._build_quickstart_tab, 'Quickstart'),
                (self._build_real_test_tab, 'Real Test'),
                (self._build_record_sim_tab, 'Record Sim'),
                (self._build_display_tab, 'RViz'),
            ]:
                try:
                    builder(notebook)
                except Exception as e:
                    err_frame = ttk.Frame(notebook)
                    notebook.add(err_frame, text=f'{tab_name} (ERROR)')
                    tk.Label(err_frame, text=f'Build error:\n{e}',
                             fg='red', wraplength=500, justify=tk.LEFT
                             ).pack(padx=10, pady=10)

            notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

            # Rebuild log panel
            self._build_log_panel()

            # Restore all state
            self._restore_gui_state(state)

            self._finalize_widget_registry()
            self._gui_ready = True
            self._append_log('HOT RELOAD: GUI rebuilt (Ctrl+Shift+R)', 'info')

        except Exception as e:
            import traceback
            self._gui_ready = True
            self._append_log(f'HOT RELOAD GUI FAILED: {e}', 'error')
            self._append_system_error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Tab 2: Arm Control (IK via MoveIt)
    # ------------------------------------------------------------------

    def _build_arm_control_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='IK')
        self._ik_tab_frame = frame

        # --- Arm (XYZ + RPY) side by side ---
        coord_frame = ttk.LabelFrame(frame, text='Arm')
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        self.xyz_vars = {}
        self._ik_spinboxes = {}    # field -> Spinbox widget (for color changes)
        self._ik_last_valid = {}   # field -> last value that produced valid IK
        self._ik_solve_gen = 0     # generation counter for async IK results
        self._ik_solve_lock = threading.Lock()  # prevent concurrent rclpy.spin
        self._ik_debounce_id = None
        self._ik_trace_active = True  # guard to suppress traces during programmatic updates

        columns_row = tk.Frame(coord_frame)
        columns_row.pack(fill=tk.X, padx=5, pady=2)

        # Left column: XYZ
        xyz_col = tk.Frame(columns_row)
        xyz_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(xyz_col, text='Position', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        for label, default in [('X', 0.12), ('Y', 0.0), ('Z', 0.15)]:
            lo, hi = WORKSPACE_BOUNDS[label]
            row = tk.Frame(xyz_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{label}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = self._register_spinbox(
                row, label=label, tab='IK', section='Position',
                textvariable=var, from_=lo, to=hi,
                increment=0.01, width=8, format='%.3f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=label: self._on_ik_var_changed(f))
            self.xyz_vars[label] = var
            self._ik_spinboxes[label] = spin
            self._ik_last_valid[label] = default

        # Right column: RPY
        rpy_col = tk.Frame(columns_row)
        rpy_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(rpy_col, text='Orientation', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        self.rpy_vars = {}
        for comp, default in [('Roll', 0.0), ('shoulder_lift', 0.0), ('Yaw', 0.0)]:
            row = tk.Frame(rpy_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{comp[0]}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = self._register_spinbox(
                row, label=comp, tab='IK', section='Orientation',
                textvariable=var, from_=-180.0, to=180.0,
                increment=1.0, width=8, format='%.1f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=comp: self._on_ik_var_changed(f))
            self.rpy_vars[comp] = var
            self._ik_spinboxes[comp] = spin
            self._ik_last_valid[comp] = default

        # Buttons inside Arm frame
        arm_btn_frame = tk.Frame(coord_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._build_arm_btn_row(arm_btn_frame, tab='IK',
                                reset_cmd=self._cmd_ik_reset)

        # --- Gripper Section (shares DoubleVar with FK tab) ---
        gripper_frame2 = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame2.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row2 = tk.Frame(gripper_frame2)
        jaw_row2.pack(fill=tk.X, padx=5, pady=2)
        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row2, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))
        # Reuse the FK tab's DoubleVar so both sliders stay in sync automatically
        jaw_var_shared = self.sliders[GRIPPER_JOINT_NAME]
        self._ik_jaw_slider = self._register_scale(
            jaw_row2, label=f'{GRIPPER_JOINT_NAME}@IK',
            tab='IK', section='Joint Sliders',
            variable=jaw_var_shared, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        self._ik_jaw_slider.pack(side=tk.LEFT, padx=5)
        self._ik_jaw_label = tk.Label(jaw_row2, text='0.000', width=8)
        self._ik_jaw_label.pack(side=tk.LEFT)

        gripper_btn_frame2 = tk.Frame(gripper_frame2)
        gripper_btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        self._build_gripper_btn_row(gripper_btn_frame2, tab='IK')

        # --- Action buttons: Set Joints / Plan & Execute ---
        ik_btn_frame = tk.Frame(frame)
        ik_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self._register_button(ik_btn_frame, text='Set Joints', tab='IK', section='Actions',
                              command=self._ik_btn_set_joints).pack(side=tk.LEFT, padx=5)
        self._register_button(ik_btn_frame, text='Plan & Execute', tab='IK', section='Actions',
                              command=self._ik_btn_plan_execute).pack(side=tk.LEFT, padx=5)

        # IK state tracking
        self.ik_status_var = tk.StringVar(value='Ready')
        self._ik_valid = True
        self._ik_planned_target = None

        # Hidden EE pose storage — used by services
        self.ee_labels = {}
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            self.ee_labels[key] = tk.StringVar(value='---')

    def _build_grasp_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Grasp')

        # --- Object Source ---
        topic_frame = ttk.LabelFrame(frame, text='Object Source')
        topic_frame.pack(fill=tk.X, padx=10, pady=5)

        topic_row = tk.Frame(topic_frame)
        topic_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(topic_row, text='Topic:', anchor='w').pack(side=tk.LEFT)
        default_topic = '/objects_poses_real' if self.use_real_hardware else '/objects_poses_sim'
        self._grasp_topic_var = tk.StringVar(value=default_topic)
        self._register_entry(topic_row, label='Grasp Topic', tab='Grasp', section='Topic',
                             textvariable=self._grasp_topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self._register_button(topic_row, text='Update Topic', tab='Grasp', section='Topic',
                              command=self._cmd_grasp_update_topic).pack(side=tk.RIGHT, padx=(2, 0))
        self._register_button(topic_row, text='Refresh Objects', tab='Grasp', section='Topic',
                              command=self._cmd_grasp_refresh).pack(side=tk.RIGHT, padx=(2, 0))

        opts_row = tk.Frame(topic_frame)
        opts_row.pack(fill=tk.X, padx=5, pady=2)
        default_bbox_topic = '/objects_bbox_real' if self.use_real_hardware else '/objects_bbox_sim'
        tk.Label(opts_row, text='BBox:', anchor='w').pack(side=tk.LEFT)
        self._bbox_topic_var = tk.StringVar(value=default_bbox_topic)
        self._register_entry(opts_row, label='BBox Topic', tab='Grasp', section='Topic',
                             textvariable=self._bbox_topic_var, width=22).pack(
            side=tk.LEFT, padx=(5, 5))
        self._bbox_enabled_var = tk.BooleanVar(value=True)
        self._register_check(opts_row, label='TCP offset', tab='Grasp', section='Topic',
                             variable=self._bbox_enabled_var).pack(side=tk.LEFT, padx=5)

        # --- Detected Objects ---
        obj_frame = ttk.LabelFrame(frame, text='Detected Objects')
        obj_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obj_listbox = self._register_listbox(
            obj_frame, label='Detected Objects', tab='Grasp',
            section='Detected Objects',
            height=2, font=('Consolas', 9),
            selectbackground='#d0d0d0', selectforeground='#1a1a1a')
        self.obj_listbox.pack(fill=tk.X, padx=5, pady=2)

        # --- Arm | Gripper columns ---
        ctrl_cols = ttk.Frame(frame)
        ctrl_cols.pack(fill=tk.X, padx=10, pady=5)

        # Left column: Arm
        arm_col = ttk.LabelFrame(ctrl_cols, text='Arm')
        arm_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        arm_dur_row = tk.Frame(arm_col)
        arm_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(arm_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_arm_duration_var = tk.DoubleVar(value=2.5)
        self._register_spinbox(arm_dur_row, label='Arm Duration (s)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_arm_duration_var,
                               from_=0.5, to=10.0, increment=0.5,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        approach_row = tk.Frame(arm_col)
        approach_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(approach_row, text='Approach height (m):', anchor='w').pack(side=tk.LEFT)
        # Phase 9: 20mm chosen over the original 50mm to avoid a grasp-
        # race: if descent's OMPL plan fails, grasp_move still returns
        # success while the arm is stuck at approach height — gripper_close
        # then attaches the block at the wrong tcp offset (trace showed
        # block_in_tcp.z=0.050 instead of ~0). Keeping approach_h small
        # bounds the worst-case attach offset to ~20mm rather than 50mm.
        # Reachability at 20mm is 100% across BLOCK_RANDOM bounds
        # (compute_grasp_reachability.py). 50mm fails 7% of cells.
        self._grasp_approach_height_var = tk.DoubleVar(value=0.020)
        self._register_spinbox(approach_row, label='Approach height (m)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_approach_height_var,
                               from_=0.00, to=0.20, increment=0.01,
                               width=6, format='%.2f').pack(side=tk.LEFT, padx=(5, 0))

        obj_z_row = tk.Frame(arm_col)
        obj_z_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(obj_z_row, text='Object Z (m):', anchor='w').pack(side=tk.LEFT)
        self._grasp_obj_z_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(obj_z_row, label='Object Z (m)',
                               tab='Grasp', section='Arm',
                               textvariable=self._grasp_obj_z_var,
                               from_=-0.05, to=0.20, increment=0.005,
                               width=6, format='%.3f').pack(side=tk.LEFT, padx=(5, 0))

        self._grasp_cross_var = tk.BooleanVar(value=False)
        self._register_check(arm_col, label='Cross-axis grasp',
                             tab='Grasp', section='Arm',
                             variable=self._grasp_cross_var).pack(
            fill=tk.X, padx=5, pady=1)

        self._register_button(arm_col, text='Home', tab='Grasp', section='Arm',
                              command=self._cmd_grasp_home).pack(fill=tk.X, padx=5, pady=2)

        # grasp_home velocity_scale override. 0.0 = no override (use the
        # planning default); anything > 0 swaps velocity_scale_var for
        # just the grasp_home plan, then restores. Lets the lag-vs-speed
        # knob live in the GUI so it's tunable without ros2 param set.
        # Mirrors home_velocity_scale ROS2 parameter via _cmd_apply_home_speed.
        home_speed_row = tk.Frame(arm_col)
        home_speed_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(home_speed_row, text='Home speed scale:',
                 anchor='w').pack(side=tk.LEFT)
        self._home_speed_var = tk.DoubleVar(value=0.0)
        self._register_spinbox(home_speed_row, label='Home speed scale',
                               tab='Grasp', section='Arm',
                               textvariable=self._home_speed_var,
                               from_=0.0, to=1.0, increment=0.05,
                               format='%.2f', width=6).pack(side=tk.LEFT, padx=(5, 0))
        self._register_button(home_speed_row, text='Apply', tab='Grasp',
                              section='Arm',
                              command=self._cmd_apply_home_speed).pack(
                              side=tk.LEFT, padx=(5, 0))

        self._grasp_move_btn = self._register_button(
            arm_col, text='Move to Grab', tab='Grasp', section='Arm',
            command=self._cmd_grasp_move)
        self._grasp_move_btn.pack(fill=tk.X, padx=5, pady=2)

        # Right column: Gripper
        grip_col = ttk.LabelFrame(ctrl_cols, text='Gripper')
        grip_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        _jaw_min_deg = math.degrees(JOINT_LIMITS['gripper_joint'][0])
        _jaw_max_deg = math.degrees(JOINT_LIMITS['gripper_joint'][1])
        grip_range_row = tk.Frame(grip_col)
        grip_range_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_range_row, text='Range:', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_close_var = tk.DoubleVar(value=_jaw_min_deg)
        self._register_spinbox(grip_range_row, label='Grip Close',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_close_var,
                               from_=_jaw_min_deg, to=_jaw_max_deg,
                               increment=5, width=5, format='%.0f').pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(grip_range_row, text='-').pack(side=tk.LEFT)
        self._grasp_grip_open_var = tk.DoubleVar(value=_jaw_max_deg)
        self._register_spinbox(grip_range_row, label='Grip Open',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_open_var,
                               from_=_jaw_min_deg, to=_jaw_max_deg,
                               increment=5, width=5, format='%.0f').pack(side=tk.LEFT)
        tk.Label(grip_range_row, text='\u00b0').pack(side=tk.LEFT)

        grip_dur_row = tk.Frame(grip_col)
        grip_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_duration_var = tk.DoubleVar(value=3.0)
        self._register_spinbox(grip_dur_row, label='Grip Duration (s)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._grasp_grip_duration_var,
                               from_=0.2, to=5.0, increment=0.1,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        clearance_row = tk.Frame(grip_col)
        clearance_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(clearance_row, text='Open clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_open_clearance_var = tk.DoubleVar(value=JAW_OPEN_CLEARANCE_M * 1000)
        self._register_spinbox(clearance_row, label='Open clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._jaw_open_clearance_var,
                               from_=-5.0, to=20.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        close_cl_row = tk.Frame(grip_col)
        close_cl_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(close_cl_row, text='Close clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_close_clearance_var = tk.DoubleVar(value=JAW_CLOSE_CLEARANCE_M * 1000)
        self._register_spinbox(close_cl_row, label='Close clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._jaw_close_clearance_var,
                               from_=-10.0, to=10.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        tcp_clear_row = tk.Frame(grip_col)
        tcp_clear_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(tcp_clear_row, text='TCP clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._tcp_clearance_var = tk.DoubleVar(value=TCP_CLEARANCE_M * 1000)
        self._register_spinbox(tcp_clear_row, label='TCP clearance (mm)',
                               tab='Grasp', section='Gripper',
                               textvariable=self._tcp_clearance_var,
                               from_=-5.0, to=10.0, increment=0.5,
                               width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        grip_btn_row1 = tk.Frame(grip_col)
        grip_btn_row1.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(grip_btn_row1, text='Grasp Open', tab='Grasp', section='Gripper',
                              command=self._cmd_gripper_open_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(grip_btn_row1, text='Grasp Close', tab='Grasp', section='Gripper',
                              command=self._cmd_gripper_close_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        grip_btn_row2 = tk.Frame(grip_col)
        grip_btn_row2.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(grip_btn_row2, text='Open', tab='Grasp', section='Gripper Range',
                              command=self._cmd_gripper_open_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(grip_btn_row2, text='Close', tab='Grasp', section='Gripper Range',
                              command=self._cmd_gripper_close_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- Drop Targets ---
        drop_src_frame = ttk.LabelFrame(frame, text='Drop Source')
        drop_src_frame.pack(fill=tk.X, padx=10, pady=5)

        drop_topic_row = tk.Frame(drop_src_frame)
        drop_topic_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_topic_row, text='Topic:', anchor='w').pack(side=tk.LEFT)
        self._drop_topic_var = tk.StringVar(value='/drop_poses')
        self._register_entry(drop_topic_row, label='Drop Topic',
                             tab='Grasp', section='Drop Source',
                             textvariable=self._drop_topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self._register_button(drop_topic_row, text='Update Drop Topic', tab='Grasp', section='Drop Source',
                              command=self._drop_btn_update_topic).pack(side=tk.RIGHT, padx=(2, 0))
        self._register_button(drop_topic_row, text='Refresh Drops', tab='Grasp', section='Drop Source',
                              command=self._cmd_drop_refresh).pack(side=tk.RIGHT, padx=(2, 0))

        drop_list_frame = ttk.LabelFrame(frame, text='Drop Targets')
        drop_list_frame.pack(fill=tk.X, padx=10, pady=5)
        self._drop_listbox = self._register_listbox(
            drop_list_frame, label='Drop Targets',
            tab='Grasp', section='Drop Targets',
            height=2, font=('Consolas', 9),
            selectbackground='#d0d0d0', selectforeground='#1a1a1a')
        self._drop_listbox.pack(fill=tk.X, padx=5, pady=2)

        drop_frame = ttk.LabelFrame(frame, text='Drop')
        drop_frame.pack(fill=tk.X, padx=10, pady=5)

        drop_dur_row = tk.Frame(drop_frame)
        drop_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_dur_row, text='Sweep duration (s):', anchor='w').pack(side=tk.LEFT)
        # 3.0 s default matches grasp_home / grasp_move / drop_point —
        # gives the controller time to track the wrist_flex sweep without
        # accumulating the multi-degree lag that bites at higher rates.
        self._drop_duration_var = tk.DoubleVar(value=3.0)
        self._register_spinbox(drop_dur_row, label='Sweep Duration (s)',
                               tab='Grasp', section='Drop',
                               textvariable=self._drop_duration_var,
                               from_=0.5, to=10.0, increment=0.5,
                               width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        # Hover above cup rim for drop sweep — the block center lands at
        # cup_center_z + CUP_BODY_HEIGHT_M/2 + this value, i.e. the cup's
        # rim height + this clearance. Default 30 mm is tuned for the
        # current 97 mm cup: enough clearance that the block falls past
        # the rim cleanly, small enough that drop impact is minimal.
        drop_hover_row = tk.Frame(drop_frame)
        drop_hover_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(drop_hover_row, text='Hover above rim (m):',
                 anchor='w').pack(side=tk.LEFT)
        # Default 50 mm: gives the attached block's bounding box clearance
        # above the cup rim during the wrist_flex sweep (90°→45°). At 30 mm,
        # the block swept arc clipped cup walls at 1-3 mm; OMPL used to
        # paper over this with tortuous paths, deterministic planner
        # surfaces it honestly. User-tunable if a specific lego/cup combo
        # needs more/less clearance.
        self._drop_hover_above_rim_var = tk.DoubleVar(value=0.050)
        self._register_spinbox(drop_hover_row, label='Hover Above Rim (m)',
                               tab='Grasp', section='Drop',
                               textvariable=self._drop_hover_above_rim_var,
                               from_=0.00, to=0.15, increment=0.005,
                               width=8, format='%.3f').pack(side=tk.LEFT, padx=(5, 0))

        drop_btn_row = tk.Frame(drop_frame)
        drop_btn_row.pack(fill=tk.X, padx=5, pady=2)
        self._register_button(drop_btn_row, text='Point to Drop', tab='Grasp', section='Drop',
                              command=self._cmd_drop_point).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self._register_button(drop_btn_row, text='Sweep to Drop', tab='Grasp', section='Drop',
                              command=self._cmd_drop_sweep).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        self._register_button(drop_btn_row, text='Release', tab='Grasp', section='Drop',
                              command=self._cmd_drop_release).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- MTC Spike (disabled) ---
        # The Pick & Drop button is commented out; backend (_cmd_mtc_pick_place,
        # _mtc_run_client, and the so_arm101_mtc C++ node) is kept intact so
        # we can re-enable this quickly for the IK iteration spike. Toggle the
        # three lines below ON to restore the button.
        # mtc_frame = ttk.LabelFrame(frame, text='MTC (spike)')
        # mtc_frame.pack(fill=tk.X, padx=10, pady=5)
        # mtc_btn_row = tk.Frame(mtc_frame)
        # mtc_btn_row.pack(fill=tk.X, padx=5, pady=2)
        # self._register_button(mtc_btn_row, text='MTC Pick & Drop',
        #                       tab='Grasp', section='MTC',
        #                       command=self._cmd_mtc_pick_place).pack(
        #                           side=tk.LEFT, fill=tk.X, expand=True)

        # Start drop subscription immediately
        self._update_drop_topic(self._drop_topic_var.get())

        # Initial subscription to default topic
        self._cmd_grasp_update_topic()

    # ------------------------------------------------------------------
    # Quickstart tab — high-level player for the full pick-drop lifecycle.
    # Each step just delegates to the matching _cmd_* handler on the Grasp
    # tab, so all settings (sweep duration, hover, padding, topic, grip
    # duration, …) continue to flow through from the Grasp tab untouched.
    # ------------------------------------------------------------------

    # Pick-and-drop step sequence. Each tuple: (label, callable, kwargs).
    # The callable is bound on the class, so we reference it by name and
    # resolve with getattr(self, name) in the runner — keeps this table
    # readable and independent of method order.
    _QS_SEQUENCE = [
        ('move to home',        '_cmd_grasp_home',            {'skip_if_home': True}),
        ('grasp open',          '_cmd_gripper_open_for_object', {}),
        ('grasp move',          '_cmd_grasp_move',            {}),
        ('grasp close',         '_cmd_gripper_close_for_object', {}),
        ('return home (carry)', '_cmd_grasp_home',            {}),
        ('point to drop cup',   '_cmd_drop_point',            {}),
        ('drop sweep',          '_cmd_drop_sweep',            {}),
        ('release',             '_cmd_drop_release',          {}),
        ('return home',         '_cmd_grasp_home',            {}),
    ]

    def _qs_auto_drop_for_lego(self, lego_name):
        """Return the /drop_poses child_frame_id of the cup matching this
        lego's color, or None if it can't be inferred.

        Per CLAUDE.md § Robot Facts the ArUco marker ID mapping is
        red → 0, green → 1, blue → 2.
        """
        if not lego_name:
            return None
        color = lego_name.split('_', 1)[0].lower()
        return {'red': 'drop_0', 'green': 'drop_1',
                'blue': 'drop_2'}.get(color)

    def _build_quickstart_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Quickstart')

        # Player state (thread/events).
        self._qs_state = 'idle'             # 'idle' | 'running' | 'paused'
        self._qs_thread = None
        # set() = running; clear() = pause-wall (next step blocks on wait)
        self._qs_resume_evt = threading.Event()
        self._qs_resume_evt.set()
        # set() = abort at next step boundary
        self._qs_abort_evt = threading.Event()
        self._qs_status_var = tk.StringVar(value='Idle')
        self._qs_step_var = tk.StringVar(value='—')

        # ===== TOP: Refresh-all bar =====
        top_bar = ttk.Frame(frame)
        top_bar.pack(fill=tk.X, padx=10, pady=(8, 4))
        self._register_button(
            top_bar, text='🔄  Refresh all (objects + drops)',
            tab='Quickstart', section='Top',
            command=self._cmd_qs_refresh_all,
        ).pack(fill=tk.X, ipady=4)

        # ===== BODY: two columns =====
        body = ttk.Frame(frame)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- LEFT column: object listbox ---
        left = ttk.LabelFrame(body, text='Detected objects  (pick one)')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._qs_listbox = tk.Listbox(
            left, height=16, exportselection=False,
            font=('TkFixedFont', 10))
        self._qs_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- RIGHT column: player + steps ---
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        ctrl_frame = ttk.LabelFrame(right, text='Pick & Drop Player')
        ctrl_frame.pack(fill=tk.X, padx=0, pady=(0, 4))
        ctrl_row = ttk.Frame(ctrl_frame)
        ctrl_row.pack(fill=tk.X, padx=5, pady=5)
        self._register_button(
            ctrl_row, text='▶ Play',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_play,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=2)
        self._register_button(
            ctrl_row, text='⏸ Pause',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_pause,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2), ipady=2)
        self._register_button(
            ctrl_row, text='⏮ Restart',
            tab='Quickstart', section='Player',
            command=self._cmd_qs_restart,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=2)
        status_row = ttk.Frame(ctrl_frame)
        status_row.pack(fill=tk.X, padx=5, pady=(0, 5))
        tk.Label(status_row, text='Status:', anchor='w').pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self._qs_status_var,
                 anchor='w', font=('TkDefaultFont', 9, 'bold')).pack(
                     side=tk.LEFT, padx=(5, 15))
        tk.Label(status_row, text='Step:', anchor='w').pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self._qs_step_var,
                 anchor='w').pack(side=tk.LEFT, padx=(5, 0))

        # Individual step buttons — right column below Player
        steps_frame = ttk.LabelFrame(
            right, text='Individual steps  (Grasp-tab handlers)')
        steps_frame.pack(fill=tk.BOTH, expand=True)
        for (label, method_name, kwargs) in self._QS_SEQUENCE:
            row = ttk.Frame(steps_frame)
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=f'{label}', anchor='w', width=22).pack(side=tk.LEFT)
            self._register_button(
                row, text='Run',
                tab='Quickstart', section=label,
                command=lambda m=method_name, kw=kwargs: self._qs_run_step_oneshot(m, kw)
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # (MTC spike re-enable: uncomment a button row here pointing at
        # self._cmd_mtc_pick_place when iterating on the MTC task graph.)

        # Initial populate — use refresh_all so legos AND drops get a fresh
        # pass on first tab build. Deferred via root.after so the grasp tab
        # (source of the legacy listbox) is already constructed.
        self.root.after(800, self._cmd_qs_refresh_all)

    # ---- Quickstart helpers ----

    def _cmd_qs_refresh_all(self):
        """Hit both Grasp-tab refresh buttons (objects AND drops), then
        repopulate our local listbox once the topic subscriptions have
        received fresh data. Non-blocking — schedules the local repopulate
        via root.after(1000, ...) because _cmd_grasp_refresh uses 500 ms
        and _cmd_drop_refresh waits up to 2 s for fresh /drop_poses data.
        """
        self._append_log('Quickstart: refresh (objects + drops)')
        try:
            self._cmd_grasp_refresh()
        except Exception as exc:
            self._append_log(f'Quickstart refresh objects failed: {exc}', 'warn')
        try:
            self._cmd_drop_refresh()
        except Exception as exc:
            self._append_log(f'Quickstart refresh drops failed: {exc}', 'warn')
        # Our local listbox mirrors self.objects_data which _cmd_grasp_refresh
        # repopulates after ~500 ms — wait a bit longer for safety.
        self.root.after(1200, self._qs_refresh_objects)

    def _cmd_qs_select(self):
        """Select an entry in the Quickstart listbox by name (ik_target param)
        or first item. Mirrors _cmd_grasp_select but targets _qs_listbox so
        _cmd_qs_play passes its _qs_get_selected_object() guard when driven
        headlessly over ROS services.
        """
        if not hasattr(self, '_qs_listbox') or self._qs_listbox.size() == 0:
            self._append_log('Quickstart: no objects to select — run qs_refresh_all first', 'warn')
            self._cmd_error = 'qs listbox empty'
            return
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        target_idx = 0
        if name_hint and '=' not in name_hint:
            matched = False
            for i in range(self._qs_listbox.size()):
                if self._qs_listbox.get(i).split('  ')[0] == name_hint:
                    target_idx = i
                    matched = True
                    break
            if not matched:
                self._append_log(f'Quickstart: "{name_hint}" not in listbox — defaulting to first', 'warn')
        self._qs_listbox.selection_clear(0, tk.END)
        self._qs_listbox.selection_set(target_idx)
        self._qs_listbox.see(target_idx)
        picked = self._qs_listbox.get(target_idx).split('  ')[0]
        self._append_log(f'Quickstart selected: {picked}')

    def _qs_refresh_objects(self):
        """Populate the Quickstart listbox from the shared objects_data dict
        (same source as the Grasp tab's listbox). Preserves selection if the
        previously-selected entry still exists.
        """
        prev = None
        if self._qs_listbox.curselection():
            prev = self._qs_listbox.get(self._qs_listbox.curselection()[0]).split('  ')[0]
        self._qs_listbox.delete(0, tk.END)
        with self.objects_lock:
            names = sorted(self.objects_data.keys())
        restore_idx = None
        for i, n in enumerate(names):
            pose = self.objects_data.get(n, {})
            x, y, z = pose.get('x', 0), pose.get('y', 0), pose.get('z', 0)
            self._qs_listbox.insert(
                tk.END, f'{n:<18s}  x={x:+.3f} y={y:+.3f} z={z:+.3f}')
            if n == prev:
                restore_idx = i
        if restore_idx is not None:
            self._qs_listbox.selection_set(restore_idx)

    def _qs_get_selected_object(self):
        sel = self._qs_listbox.curselection()
        if not sel:
            return None
        return self._qs_listbox.get(sel[0]).split('  ')[0]

    def _qs_set_status(self, status, step=None):
        """Thread-safe status/step update via Tk's event loop."""
        def _update():
            self._qs_status_var.set(status)
            if step is not None:
                self._qs_step_var.set(step)
        self.root.after(0, _update)

    def _qs_is_at_home(self, tol_deg=2.0):
        """Check whether the arm joints are already at the grasp-home pose."""
        from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
        target = {n: 0.0 for n in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        tol = math.radians(tol_deg)
        with self.joint_lock:
            for n in ARM_JOINT_NAMES:
                live = self._actual_positions.get(n, self.joint_positions.get(n, 0.0))
                if abs(live - target[n]) > tol:
                    return False
        return True

    def _qs_sync_grasp_listbox(self, obj_name):
        """Mirror the Quickstart selection into the Grasp tab's obj_listbox
        so every _cmd_* downstream reads the same selection. Waits for
        population if the listbox is empty (_cmd_grasp_refresh populates
        asynchronously via root.after — 0.5 s was not enough).
        """
        if not hasattr(self, 'obj_listbox') or self.obj_listbox.size() == 0:
            self._cmd_grasp_refresh()
            deadline = time.time() + 3.0
            while time.time() < deadline and (
                    not hasattr(self, 'obj_listbox') or
                    self.obj_listbox.size() == 0):
                time.sleep(0.1)
        for i in range(self.obj_listbox.size()):
            if self.obj_listbox.get(i).split('  ')[0] == obj_name:
                self.obj_listbox.selection_clear(0, tk.END)
                self.obj_listbox.selection_set(i)
                return True
        return False

    def _qs_sync_drop_listbox(self, drop_name):
        """Same pattern for the drop listbox — _cmd_drop_refresh waits up
        to 2 s for fresh /drop_poses, so wait that long for population."""
        if not hasattr(self, '_drop_listbox') or self._drop_listbox.size() == 0:
            self._cmd_drop_refresh()
            deadline = time.time() + 3.0
            while time.time() < deadline and (
                    not hasattr(self, '_drop_listbox') or
                    self._drop_listbox.size() == 0):
                time.sleep(0.1)
        for i in range(self._drop_listbox.size()):
            entry = self._drop_listbox.get(i).split(' [')[0]
            if entry == drop_name:
                self._drop_listbox.selection_clear(0, tk.END)
                self._drop_listbox.selection_set(i)
                return True
        return False

    def _qs_wait_for_step(self, timeout_s=60.0):
        """Wait for the most recent _cmd_* to complete, gated on pause.

        The existing _cmd_* handlers set self._motion_event when they fire
        a motion. Event firing means "something finished" — it does NOT mean
        success. The verdict is in self._last_motion_status (set by
        _ompl_plan_validate_execute on both the OK path and the Mode B /
        plan-failed paths). Without consulting it, a Mode B post-check
        rejection silently propagates as "step passed" and the runner
        marches into the next step — which is how the cup-knocking plans
        used to execute anyway after being flagged.

        Returns True only when the event fires AND status.ok is truthy
        (or status is missing entirely — e.g. for synchronous _cmd_*s that
        never call _ompl_plan_validate_execute, like gripper_open/close
        which just clear a wait flag).
        """
        evt = getattr(self, '_motion_event', None)
        if evt is None:
            return True  # nothing to wait on
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._qs_abort_evt.is_set():
                return False
            # Soft pause: block here until resume
            if not self._qs_resume_evt.wait(timeout=0.2):
                continue
            if evt.wait(timeout=0.2):
                status = getattr(self, '_last_motion_status', None)
                if status is not None and status.get('ok') is False:
                    # Surface WHY we halted so the log trail reads end-to-end.
                    outcome = status.get('outcome', 'unknown')
                    msg = status.get('msg', '')
                    self._append_log(
                        f'Quickstart: step failed ({outcome}): {msg}', 'err')
                    return False
                return True
        return False

    # ---- Step-runner entry points ----

    def _qs_run_step_oneshot(self, method_name, kwargs):
        """Run a single step from the individual-step buttons. Non-blocking
        on the Tk thread; the step itself may spawn its own motion thread.
        Skips if the Player is currently running (to avoid interleaving)."""
        if self._qs_state == 'running':
            self._append_log(
                'Quickstart step ignored: Player is running — use Pause first', 'warn')
            return
        self._qs_set_status('Single step', step=method_name)
        threading.Thread(
            target=self._qs_execute_step,
            args=(method_name, kwargs),
            daemon=True).start()

    def _qs_execute_step(self, method_name, kwargs):
        """Run one step's handler. Handles the skip-if-home and auto-cup-
        selection policies before dispatching to the underlying _cmd_*.
        Returns True on success, False on abort/error.
        """
        # skip_if_home shortcut for grasp_home
        if kwargs.get('skip_if_home') and self._qs_is_at_home():
            self._append_log('Already at home — skipping grasp_home')
            return True

        # Pre-dispatch policies for steps that need a target selection
        if method_name in ('_cmd_gripper_open_for_object',
                          '_cmd_grasp_move',
                          '_cmd_gripper_close_for_object'):
            lego = self._qs_get_selected_object()
            if lego is None:
                self._append_log('Quickstart: no object selected', 'err')
                return False
            if not self._qs_sync_grasp_listbox(lego):
                self._append_log(
                    f'Quickstart: "{lego}" not in Grasp listbox — try Refresh', 'err')
                return False

        if method_name in ('_cmd_drop_point', '_cmd_drop_sweep',
                          '_cmd_drop_release'):
            lego = self._qs_get_selected_object()
            drop = self._qs_auto_drop_for_lego(lego) if lego else None
            if drop is None:
                self._append_log(
                    f'Quickstart: could not auto-select cup for "{lego}" — '
                    'implement _qs_auto_drop_for_lego', 'err')
                return False
            if not self._qs_sync_drop_listbox(drop):
                self._append_log(
                    f'Quickstart: "{drop}" not in drop listbox', 'err')
                return False

        # Clear motion event AND last motion status before dispatch so
        # _qs_wait_for_step reads THIS step's verdict, not the previous
        # step's (or the previous run's — which can leak across qs_play
        # invocations because _last_motion_status is an instance attr,
        # not a per-call local). _make_trigger_callback at line ~845
        # does the same clear on the ROS-service entry path; QS calls
        # handlers directly, so we need to clear here too.
        self._motion_event = threading.Event()
        self._last_motion_status = None
        self._cmd_error = None
        # Tag the next trajectory dump with this step's cmd name.
        if method_name.startswith('_cmd_'):
            self._last_motion_tag = method_name[5:]

        handler = getattr(self, method_name, None)
        if handler is None:
            self._append_log(f'Quickstart: no such handler {method_name}', 'err')
            return False
        try:
            handler()
        except Exception as exc:
            self._append_log(f'Quickstart step {method_name} raised: {exc}', 'err')
            return False

        ok = self._qs_wait_for_step(timeout_s=60.0)
        if not ok and self._qs_abort_evt.is_set():
            return False
        return ok

    # ---- Player controls ----

    def _cmd_qs_play(self):
        """Play: start the lifecycle from the first step, OR resume from pause."""
        if self._qs_state == 'paused':
            self._qs_set_status('Running')
            self._qs_resume_evt.set()
            self._qs_state = 'running'
            return
        if self._qs_state == 'running':
            self._append_log('Quickstart already running', 'warn')
            return
        if self._qs_get_selected_object() is None:
            self._append_log('Quickstart Play: select an object first', 'err')
            return
        self._qs_abort_evt.clear()
        self._qs_resume_evt.set()
        self._qs_state = 'running'
        self._qs_thread = threading.Thread(target=self._qs_run, daemon=True)
        self._qs_thread.start()

    def _cmd_qs_pause(self):
        if self._qs_state != 'running':
            self._append_log('Quickstart Pause: not running', 'warn')
            return
        self._qs_resume_evt.clear()
        self._qs_state = 'paused'
        self._qs_set_status('Paused')

    def _cmd_qs_restart(self):
        """Abort current cycle and return to idle. Does NOT restart a new
        cycle automatically — user presses Play again (intentional, so a
        mid-motion abort doesn't race into a fresh motion command).
        """
        self._qs_abort_evt.set()
        self._qs_resume_evt.set()  # unblock any paused wait so the runner exits
        self._qs_state = 'idle'
        self._qs_set_status('Aborted', step='—')

    def _qs_run(self):
        """Lifecycle runner — sequential dispatch of _QS_SEQUENCE steps."""
        try:
            # Kick both refreshes on the Tk thread so listboxes repopulate
            # before the first selection-dependent step. Sleep past the
            # root.after(1200) repopulate delay in _qs_refresh_all.
            self._qs_set_status('Running', step='refreshing objects + drops')
            self.root.after(0, self._cmd_qs_refresh_all)
            time.sleep(1.5)

            self._qs_set_status('Running', step='(starting)')
            for i, (label, method_name, kwargs) in enumerate(self._QS_SEQUENCE, start=1):
                if self._qs_abort_evt.is_set():
                    self._append_log('Quickstart aborted')
                    return
                self._qs_resume_evt.wait()  # honor pause
                self._qs_set_status('Running', step=f'{i}/{len(self._QS_SEQUENCE)}: {label}')
                ok = self._qs_execute_step(method_name, kwargs)
                if not ok:
                    if self._qs_abort_evt.is_set():
                        self._qs_set_status('Aborted', step=label)
                    else:
                        self._qs_set_status('ERROR', step=label)
                        self._append_log(f'Quickstart halted at: {label}', 'err')
                    return
            self._qs_set_status('Complete', step='✓ done')
            self._append_log('Quickstart: pick-and-drop cycle complete')
        finally:
            self._qs_state = 'idle'

    # ------------------------------------------------------------------
    # Tab: Real Test  (real-mode pipeline: YOLOE legos + ArUco cups)
    # ------------------------------------------------------------------
    # Mirrors the Quickstart flow but sources poses from the real-camera
    # detection stack (/objects_poses_real, /drop_poses_real) instead of
    # sim ground truth. Cups are detected ONCE per cycle at a scan pose
    # and cached; legos refresh every cycle. Built in isolation here for
    # end-to-end validation, will refactor into existing tabs once stable.

    def _ensure_real_state(self):
        """Lazy-init Real Test tab state vars on the live instance.

        __init__ runs once per process, so when hot-reload introduces new
        state vars they don't materialize on the running instance until
        they're accessed. This helper is called at the top of the tab
        builder and every _cmd_real_* handler so the running session
        picks up the new vars without a full relaunch.
        """
        if not hasattr(self, '_cached_cup_poses'):
            self._cached_cup_poses = {}
        if not hasattr(self, '_cached_lego_poses'):
            # Same shape as _cached_cup_poses + objects_data:
            # {child_frame_id (e.g. 'red_lego_0') →
            #  {'x','y','z','qx','qy','qz','qw'}} in frame 'base'.
            # Populated by Refresh Legos Pose at grasp_home (where wrist
            # camera looks down at workspace). Consumed by Run Real Pick
            # & Drop instead of the live /objects_poses_real feed.
            self._cached_lego_poses = {}
        if not hasattr(self, '_real_selected_color_var'):
            # Bound to the color dropdown in Real Test → Run.
            # '' = nothing selected (also the empty-state when no color has
            # both a cached lego AND a cached cup).
            self._real_selected_color_var = tk.StringVar(value='')
        if not hasattr(self, '_real_sweep_range_deg_var'):
            # Drop Scan sweep range in degrees from start point. Default 90°
            # (instead of full 180°) so the user can constrain the sweep to
            # the side where cups are likely to be.
            self._real_sweep_range_deg_var = tk.IntVar(value=90)
        if not hasattr(self, '_real_sweep_reversed_var'):
            # False = canonical direction (start at +90°, sweep right→left
            # toward -90°). True = swap (start at -90°, sweep left→right).
            self._real_sweep_reversed_var = tk.BooleanVar(value=False)

    def _build_real_test_tab(self, notebook):
        self._ensure_real_state()
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Real Test')

        # ===== Section: Setup =====
        # Three primitive motions used to position the arm before scanning,
        # plus the wrist-roll sign toggle that decides which pose makes the
        # cup ArUco markers visible to the wrist camera.
        setup = ttk.LabelFrame(frame, text='Setup')
        setup.pack(fill=tk.X, padx=10, pady=(8, 4))

        row1 = ttk.Frame(setup)
        row1.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row1, text='Grasp Home', tab='Real Test', section='Setup',
            command=self._cmd_grasp_home,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=3)
        self._register_button(
            row1, text='Open Gripper', tab='Real Test', section='Setup',
            command=self._cmd_gripper_open,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=3)

        # Row 2: explicit wrist_roll set buttons. User picks +90° (markers
        # in FOV / scan pose) or −90° (jaw-clearance / drop pose) BEFORE
        # pressing Drop Scan. Drop Scan locks whatever wrist_roll is at
        # the moment it starts and preserves it through the entire sweep.
        row2 = ttk.Frame(setup)
        row2.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row2, text='Roll +90°  (scan pose)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_wrist_roll_plus,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=3)
        self._register_button(
            row2, text='Roll −90°  (drop pose)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_wrist_roll_minus,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), ipady=3)

        # Row 3: sweep range + direction swap.
        row3 = ttk.Frame(setup)
        row3.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row3, text='Sweep range (°):', anchor='w').pack(
            side=tk.LEFT, padx=(0, 2))
        self._register_spinbox(
            row3, label='Sweep range (°)',
            tab='Real Test', section='Setup',
            textvariable=self._real_sweep_range_deg_var,
            from_=5, to=180, increment=5, width=5,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._register_button(
            row3, text='↔  Swap direction',
            tab='Real Test', section='Setup',
            command=self._cmd_real_swap_sweep_direction,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0), ipady=2)

        # Row 4: Drop Scan on its own row.
        row4 = ttk.Frame(setup)
        row4.pack(fill=tk.X, padx=5, pady=4)
        self._register_button(
            row4, text='▶  Drop Scan  (sweeps configured range, locks current roll)',
            tab='Real Test', section='Setup',
            command=self._cmd_real_drop_scan,
        ).pack(fill=tk.X, expand=True, ipady=3)

        # ===== Section: Calibration =====
        # Two scan-then-cache pipelines, one per detection pass:
        #   - Cups via /drop_poses_real (ArUco), scanned at drop_point_green
        #     pose where wrist camera sees all 3 cup markers in one frame.
        #   - Legos via /objects_poses_real (YOLOE), scanned at grasp_home
        #     pose where wrist camera looks down at the workspace. YOLOE
        #     only sees what's in FOV at scan time — caching freezes that
        #     view so the loop can iterate it even after the arm moves
        #     away from home for grasp/drop motions.
        #
        # Each refresh partial-merges captured detections into its cache
        # (missing markers/legos keep their previous cached value), pushes
        # the merged cache to MoveIt as collision objects, and updates the
        # corresponding live data structure (_drop_data / objects_data) +
        # listbox so existing _cmd_* paths read the cached values.
        cal = ttk.LabelFrame(frame, text='Calibration  (scan-then-cache)')
        cal.pack(fill=tk.X, padx=10, pady=4)

        self._register_button(
            cal,
            text='🔄  Refresh Cups Pose  (subscribe-once /drop_poses_real)',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_refresh_cups_pose,
        ).pack(fill=tk.X, padx=5, pady=(4, 1), ipady=3)

        self._real_cups_status_var = tk.StringVar(value='Cups: (none)')
        tk.Label(
            cal, textvariable=self._real_cups_status_var,
            anchor='w', font=('TkFixedFont', 9), fg='#333',
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        self._register_button(
            cal,
            text='🔄  Refresh Legos Pose  (subscribe-once /objects_poses_real)',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_refresh_legos_pose,
        ).pack(fill=tk.X, padx=5, pady=(4, 1), ipady=3)

        self._real_legos_status_var = tk.StringVar(value='Legos: (none)')
        tk.Label(
            cal, textvariable=self._real_legos_status_var,
            anchor='w', font=('TkFixedFont', 9), fg='#333',
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        clear_row = ttk.Frame(cal)
        clear_row.pack(fill=tk.X, padx=5, pady=(2, 5))
        self._register_button(
            clear_row, text='Clear all caches',
            tab='Real Test', section='Calibration',
            command=self._cmd_real_clear_cache,
        ).pack(side=tk.RIGHT, padx=(0, 3))

        # ===== Section: Steps (manual debug) =====
        # Individual step buttons that run ONE _QS_SEQUENCE step each, with
        # the cache-pose re-injection + listbox selection that the full Run
        # path does — but stop after the step. Lets you debug "why does
        # grasp_move fail?" by stepping through grasp_open → grasp_move →
        # close → ... without short-circuiting on the first failure.
        # Each button shares the dropdown's selected color (closest cached
        # lego of that color is picked) and forces real topics on entry.
        steps = ttk.LabelFrame(frame, text='Steps  (manual debug)')
        steps.pack(fill=tk.X, padx=10, pady=4)

        steps_row1 = ttk.Frame(steps)
        steps_row1.pack(fill=tk.X, padx=5, pady=(4, 2))
        for label, cmd in [
            ('Grasp Home',  self._cmd_real_grasp_home),
            ('Grasp Open',  self._cmd_real_grasp_open),
            ('Grasp Move',  self._cmd_real_grasp_move),
            ('Grasp Close', self._cmd_real_grasp_close),
        ]:
            self._register_button(
                steps_row1, text=label, tab='Real Test', section='Steps',
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=2)

        steps_row2 = ttk.Frame(steps)
        steps_row2.pack(fill=tk.X, padx=5, pady=(2, 5))
        for label, cmd in [
            ('Drop Point',   self._cmd_real_drop_point),
            ('Drop Sweep',   self._cmd_real_drop_sweep),
            ('Release',      self._cmd_real_drop_release),
        ]:
            self._register_button(
                steps_row2, text=label, tab='Real Test', section='Steps',
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=2)

        # ===== Section: Run =====
        # Color dropdown (populated by _real_refresh_color_dropdown from the
        # intersection of cached lego colors AND cached cup colors) + a Run
        # button that picks ONE lego of the selected color. Cycle is full
        # _QS_SEQUENCE; on success the lego is evicted from cache and the
        # dropdown shrinks accordingly.
        run_frame = ttk.LabelFrame(frame, text='Run')
        run_frame.pack(fill=tk.X, padx=10, pady=(4, 8))

        color_row = ttk.Frame(run_frame)
        color_row.pack(fill=tk.X, padx=5, pady=(4, 2))
        tk.Label(color_row, text='Drop color:', anchor='w').pack(
            side=tk.LEFT, padx=(0, 5))
        self._real_color_combobox = self._register_combobox(
            color_row, label='Real Test color',
            tab='Real Test', section='Run',
            textvariable=self._real_selected_color_var,
            values=[], width=10,
        )
        self._real_color_combobox.pack(side=tk.LEFT)

        self._real_run_button = self._register_button(
            run_frame, text='▶  Pick & Drop Selected Color',
            tab='Real Test', section='Run',
            command=self._cmd_real_run_one_color,
        )
        self._real_run_button.pack(fill=tk.X, padx=5, pady=(2, 4), ipady=4)
        # Start disabled — the first Refresh that yields a non-empty
        # intersection re-enables it via _real_refresh_color_dropdown.
        self._real_run_button.config(state='disabled')

        self._real_run_status_var = tk.StringVar(value='Idle')
        status_row = ttk.Frame(run_frame)
        status_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(status_row, text='Status:', anchor='w').pack(side=tk.LEFT)
        tk.Label(
            status_row, textvariable=self._real_run_status_var,
            anchor='w', font=('TkDefaultFont', 9, 'bold'),
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Hint text — pre-flight assumption surface for the new flow.
        hint = (
            'Pre-flight: Drop Point Green → Refresh Cups Pose → Grasp Home '
            '→ Refresh Legos Pose. Then pick a color and press Run. '
            'Successful drops evict that lego from cache automatically.'
        )
        tk.Label(
            frame, text=hint, anchor='w', justify=tk.LEFT,
            fg='#555', font=('TkDefaultFont', 8),
            wraplength=900,
        ).pack(fill=tk.X, padx=12, pady=(8, 4))

        # Self-recovery on hot-reload: instance cache survives a GUI rebuild,
        # but the freshly-constructed combobox is initialized empty + button
        # disabled. Re-run the dropdown helper so the rebuilt widget reflects
        # the live cache state without forcing the user to press Refresh.
        # Wrapped in root.after so it runs after the rest of the tab finishes
        # building (helper accesses self._real_color_combobox).
        self.root.after(0, self._real_refresh_color_dropdown)

    # ------------------------------------------------------------------
    # Real Test tab — command handlers
    # ------------------------------------------------------------------

    def _cmd_real_toggle_wrist_roll_sign(self):
        """Deprecated by Roll +90° / Roll −90° buttons. Kept as stub so the
        auto-registered ~/real_toggle_wrist_roll_sign service doesn't error
        if invoked. Will be removed on next Phase 11 cleanup pass."""
        self._append_log(
            'Real: toggle_wrist_roll_sign deprecated — '
            'use Roll +90° / Roll −90° buttons instead', 'warn')

    # --- Drop Scan workflow (Phase 11-01 followup, replaces Drop Point Green) ---
    # Sweeps shoulder_pan from -90° to +90° in small steps. At each step,
    # samples /drop_poses_real briefly. When a not-yet-cached cup appears in
    # the camera's view zone (target_pan derived from cup XY is within
    # ±FOV_HALF of current sweep pan), the arm visual-servos to point at
    # that cup, then averages the cup pose over AVG_DURATION seconds of
    # nearly-stationary samples. Builds up _cached_cup_poses one cup at a
    # time. Bookended by grasp_home before/after.

    # Drop Scan tunables live at module level (DROP_SCAN_*) so hot-reload
    # picks up changes — see comment at the top of this file about
    # _patch_methods not copying class-level constants.

    def _set_yoloe_overlay(self, enable: bool, timeout_s: float = 0.5) -> bool:
        """Toggle the merged-viewer YOLOE overlay via aruco_camera_localizer
        service. Lazy-creates the client on first call so this works after
        hot-reload (which doesn't re-run __init__). Best-effort — every
        rclpy/tk failure path is swallowed because this is called from the
        `finally` block of `_real_drop_scan_thread` and must never raise
        (otherwise it masks the real exception or stalls cleanup)."""
        try:
            client = getattr(self, '_yoloe_overlay_client', None)
            if client is None:
                client = self.create_client(
                    SetBool, '/aruco_viewer/yoloe_overlay_enable')
                self._yoloe_overlay_client = client
            if not client.wait_for_service(timeout_sec=timeout_s):
                self._append_log(
                    'YOLOE overlay toggle skipped — '
                    '/aruco_viewer/yoloe_overlay_enable not advertised', 'warn')
                return False
            req = SetBool.Request()
            req.data = bool(enable)
            client.call_async(req)
            return True
        except Exception as e:
            try:
                self._append_log(
                    f'YOLOE overlay toggle errored ({type(e).__name__}): {e}',
                    'warn')
            except Exception:
                pass
            return False

    def _cmd_real_drop_scan(self):
        """Replacement for Drop Point Green: sweep -90° → +90° on
        shoulder_pan, detect each cup as it enters view, average its pose
        over 3s, cache, push to scene + dropdown. Bookended by grasp_home.

        Uses the existing Drop Topic flow (/drop_poses_real) as the
        per-cup pose source — visual servoing is "single corrective pan
        per detection," not continuous closed-loop.
        """
        self._ensure_real_state()
        if self._qs_state == 'running':
            self._append_log(
                'Real: drop_scan cannot start — a run is already in progress',
                'warn')
            return
        # Force real topics BEFORE spawning the worker thread. rclpy's
        # destroy_subscription/create_subscription calls can race with
        # MultiThreadedExecutor worker threads if they happen on a
        # daemon thread mid-spin (InvalidHandle crash). The service-
        # callback thread that's calling us here is still an executor
        # worker thread but the idempotent-by-actual-topic logic in
        # _real_ensure_real_topics now makes destroy/create a once-per-
        # process event, so the race window collapses.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_drop_scan_thread, daemon=True).start()

    def _real_drop_scan_thread(self):
        try:
            self._qs_abort_evt.clear()
            self._qs_resume_evt.set()  # release pause gate (default unset)
            self._qs_state = 'running'
            self._real_set_status('Drop Scan', step='preparing')
            self._append_log('Real drop_scan: starting')
            # Suppress YOLOE overlay in the wrist-camera view so the user
            # sees only ArUco drawings during the sweep. Restored in finally.
            self._set_yoloe_overlay(False)

            # --- Phase 0: pre-scan setup (gripper open + canonical scan pose) ---
            # The wrist must be at +90° wrist_roll (camera-down "scan pose")
            # AND the gripper must be fully open before the sweep starts —
            # otherwise we sweep with the previous run's residual orientation
            # / a partially-closed gripper occluding markers. Hardcoded so the
            # user can't accidentally launch a scan with the arm in drop pose.
            SCAN_WRIST_ROLL_RAD = math.pi / 2  # +90° = scan pose
            self._real_set_status('Drop Scan', step='opening gripper')
            self._append_log('Real drop_scan: opening gripper before sweep')
            self._cmd_gripper_open()
            evt = getattr(self, '_motion_event', None)
            if evt is not None:
                evt.wait(timeout=5.0)

            self._real_set_status('Drop Scan', step='flipping wrist to scan pose')
            self._append_log(
                f'Real drop_scan: flipping wrist_roll → '
                f'{math.degrees(SCAN_WRIST_ROLL_RAD):+.1f}° (scan pose)')
            if not self._real_set_wrist_roll(SCAN_WRIST_ROLL_RAD):
                self._append_log(
                    'Real drop_scan: wrist_roll flip failed', 'err')
                self._real_set_status('Halted', step='wrist_roll flip failed')
                return

            # --- Phase 1: scan_home, locking the scan-pose wrist_roll ---
            scan_wrist_roll = SCAN_WRIST_ROLL_RAD
            self._append_log(
                f'Real drop_scan: locking wrist_roll = '
                f'{math.degrees(scan_wrist_roll):+.1f}° for entire sweep')
            self._real_set_status('Drop Scan', step='scan_home')
            if not self._real_scan_home_preserve_roll(scan_wrist_roll):
                self._append_log(
                    'Real drop_scan: scan_home failed', 'err')
                self._real_set_status('Halted', step='scan_home failed')
                return

            from so_arm101_control.compute_workspace import X_PAN
            visited_ids = set()

            # --- Derive sweep parameters from UI vars ---
            # range_deg sets how far from start the sweep goes; reversed_var
            # picks which direction. Default: 90° from +90° start, going
            # right→left (so end = 0°). Clamp range to [5, 180].
            range_deg = max(5, min(180,
                int(self._real_sweep_range_deg_var.get())))
            range_rad = math.radians(range_deg)
            reversed_dir = bool(self._real_sweep_reversed_var.get())
            if reversed_dir:
                sweep_start_rad = DROP_SCAN_PAN_MIN_RAD          # -90°
                sweep_step_signed_rad = +DROP_SCAN_PAN_STEP_RAD  # +5°
                sweep_end_rad = sweep_start_rad + range_rad
            else:
                sweep_start_rad = DROP_SCAN_PAN_MAX_RAD          # +90°
                sweep_step_signed_rad = -DROP_SCAN_PAN_STEP_RAD  # -5°
                sweep_end_rad = sweep_start_rad - range_rad
            # Clamp end to absolute joint bounds (range_deg can drive it past).
            sweep_end_rad = max(DROP_SCAN_PAN_MIN_RAD,
                                min(DROP_SCAN_PAN_MAX_RAD, sweep_end_rad))
            self._append_log(
                f'Real drop_scan: range={range_deg}° '
                f'direction={"left→right" if reversed_dir else "right→left"} '
                f'start={math.degrees(sweep_start_rad):+.1f}° '
                f'end={math.degrees(sweep_end_rad):+.1f}°')

            # --- Phase 2a: move to sweep start with safe duration ---
            # The initial move from home (pan=0) to start is up to 90° of
            # travel — send with a generous duration so we don't exceed the
            # drive's tracking envelope (~30°/s sustained per CLAUDE.md).
            # Subsequent 5° increments use the short duration safely.
            self._real_set_status(
                'Drop Scan',
                step=f'moving to sweep start ({math.degrees(sweep_start_rad):+.0f}°)')
            self._append_log(
                f'Real drop_scan: moving to start pan='
                f'{math.degrees(sweep_start_rad):+.1f}°')
            if not self._real_drop_scan_motion(
                    self._real_drop_scan_pan_target(
                        sweep_start_rad, scan_wrist_roll),
                    3.5):
                self._append_log(
                    'Real drop_scan: initial move to start failed', 'err')
                self._real_set_status('Halted', step='initial pan failed')
                return

            # --- Phase 2b: sweep start → end in step_signed increments ---
            # Loop continues while signed remaining distance is non-negative
            # (works in both directions since step_signed carries the sign).
            pan = sweep_start_rad
            first_iter = True
            sweep_sign = 1.0 if sweep_step_signed_rad > 0 else -1.0
            while sweep_sign * (sweep_end_rad - pan) >= -1e-6:
                if self._qs_abort_evt.is_set():
                    self._append_log('Real drop_scan: aborted', 'warn')
                    self._real_set_status('Aborted')
                    return
                self._real_set_status(
                    'Drop Scan',
                    step=f'sweep pan={math.degrees(pan):+.1f}° '
                         f'cached={len(visited_ids)}/3')
                # Skip motion on first iter — we're already at pan_min from
                # the initial move above. Subsequent iters do the small step.
                if not first_iter:
                    target = self._real_drop_scan_pan_target(
                        pan, scan_wrist_roll)
                    if not self._real_drop_scan_motion(
                            target, DROP_SCAN_STEP_DURATION_S):
                        self._append_log(
                            f'Real drop_scan: pan motion to '
                            f'{math.degrees(pan):+.1f}° failed', 'err')
                        self._real_set_status('Halted', step='pan motion failed')
                        return
                first_iter = False
                time.sleep(DROP_SCAN_SETTLE_AFTER_STEP_S)

                # Sample briefly — looking for new cups in view zone
                samples = self._real_sample_drop_poses(
                    DROP_SCAN_SAMPLE_PER_STEP_S)
                for drop_id, pose_list in samples.items():
                    if drop_id in visited_ids:
                        continue
                    if not pose_list:
                        continue
                    # Compute target pan from latest sample
                    p = pose_list[-1]
                    target_pan = math.atan2(-p['y'], p['x'] - X_PAN)
                    # In view zone? (within ±FOV_HALF of current pan)
                    if abs(target_pan - pan) > DROP_SCAN_FOV_HALF_RAD:
                        continue
                    # Out of pan range? skip
                    if (target_pan < DROP_SCAN_PAN_MIN_RAD or
                            target_pan > DROP_SCAN_PAN_MAX_RAD):
                        continue
                    self._append_log(
                        f'Real drop_scan: detected {drop_id} → target_pan='
                        f'{math.degrees(target_pan):+.1f}° (sweep at '
                        f'{math.degrees(pan):+.1f}°)')
                    self._real_set_status(
                        'Drop Scan',
                        step=f'settling on {drop_id}')

                    # Visual servo: single corrective pan to target. Once we
                    # commit to servoing on a drop_id we never re-target it,
                    # success or failure — otherwise a flaky cup keeps
                    # pulling the sweep back and the scan never advances.
                    servo_target = self._real_drop_scan_pan_target(
                        target_pan, scan_wrist_roll)
                    if not self._real_drop_scan_motion(
                            servo_target, DROP_SCAN_STEP_DURATION_S + 0.1):
                        self._append_log(
                            f'Real drop_scan: servo to {drop_id} failed — '
                            'marking visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        continue
                    time.sleep(0.3)  # let arm settle before averaging

                    # 3-sec average
                    avg_samples = self._real_sample_drop_poses(
                        DROP_SCAN_AVG_DURATION_S, filter_id=drop_id)
                    drop_samples = avg_samples.get(drop_id, [])
                    if len(drop_samples) < DROP_SCAN_MIN_AVG_SAMPLES:
                        self._append_log(
                            f'Real drop_scan: {drop_id} settle failed — only '
                            f'{len(drop_samples)} samples '
                            f'(need ≥{DROP_SCAN_MIN_AVG_SAMPLES}) — '
                            'marking visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        # Resume sweep from servoed pan
                        pan = target_pan
                        continue
                    avg_pose, xy_var = self._real_average_pose_samples(
                        drop_samples)
                    if xy_var > DROP_SCAN_MAX_XY_VARIANCE_M2:
                        self._append_log(
                            f'Real drop_scan: {drop_id} xy variance '
                            f'{xy_var*1e6:.1f} mm² too high — marking '
                            'visited, sweep continues', 'warn')
                        visited_ids.add(drop_id)
                        pan = target_pan
                        continue
                    self._cached_cup_poses[drop_id] = avg_pose
                    visited_ids.add(drop_id)
                    self._append_log(
                        f'Real drop_scan: cached {drop_id} '
                        f'(avg {len(drop_samples)} samples, '
                        f'xy_std={1000*math.sqrt(xy_var):.2f} mm)')
                    pan = target_pan  # resume from servoed pos

                pan += DROP_SCAN_PAN_STEP_SIGNED_RAD

            # --- Phase 3: push cached to MoveIt + visual + dropdown ---
            self._real_set_status('Drop Scan', step='applying scene')
            self._remove_cup_collision_objects()
            time.sleep(0.5)  # planning-scene monitor round-trip
            if self._cached_cup_poses:
                self._add_cup_collision_objects(
                    cups_dict=dict(self._cached_cup_poses))
                self._publish_cup_visual_markers(
                    cups_dict=dict(self._cached_cup_poses))
            # Mirror into _drop_data + listbox so existing flows pick up.
            with self._drop_lock:
                self._drop_data.update(self._cached_cup_poses)
            self.root.after(0, self._populate_drop_list)
            # Status label per cup
            def _update_label():
                marks = []
                for n in ('drop_0', 'drop_1', 'drop_2'):
                    glyph = '✓' if n in self._cached_cup_poses else '✗'
                    marks.append(f'{n}:{glyph}')
                if hasattr(self, '_real_cups_status_var'):
                    self._real_cups_status_var.set(
                        'Cups: ' + '   '.join(marks))
            self.root.after(0, _update_label)
            self._real_refresh_color_dropdown()
            # Report on the AUTHORITATIVE cache (visited_ids now includes
            # cups that were attempted but rejected for high variance / low
            # sample count, so it's not a "cached" count anymore).
            cached_names = sorted(self._cached_cup_poses.keys())
            attempted_only = sorted(visited_ids - set(cached_names))
            summary = (
                f'Real drop_scan: cached {len(cached_names)} cup(s): '
                f'{cached_names}')
            if attempted_only:
                summary += f' (attempted but rejected: {attempted_only})'
            self._append_log(summary)

            # --- Phase 4: return to scan_home (preserve scan-start wrist_roll) ---
            self._real_set_status('Drop Scan', step='scan_home (return)')
            if not self._real_scan_home_preserve_roll(scan_wrist_roll):
                self._append_log('Real drop_scan: return scan_home failed',
                                 'warn')
            self._real_set_status(
                'Complete',
                step=f'✓ {len(self._cached_cup_poses)} cup(s) cached')
        finally:
            self._set_yoloe_overlay(True)
            self._qs_state = 'idle'

    # --- Drop scan helpers ---

    def _real_drop_scan_pan_target(self, pan_rad, wrist_roll_rad):
        """Build a joint target with shoulder_pan = pan_rad and wrist_roll
        LOCKED at `wrist_roll_rad` (the captured-at-scan-start value, NOT
        the live current value). Other arm joints fixed at scan-home values
        (shoulder_lift=0, elbow_flex=0, wrist_flex=π/2).

        Locking wrist_roll to a captured constant — rather than reading
        current /joint_states each step — prevents drift from compounding
        across the 36-step sweep. The controller must keep correcting back
        to the captured value rather than capturing whatever drift occurred."""
        target = {n: 0.0 for n in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = float(wrist_roll_rad)
        target['shoulder_pan'] = float(pan_rad)
        return target

    def _real_scan_home_preserve_roll(self, wrist_roll_rad=None):
        """Move arm to a scan-ready pose (pan=0, lift=0, elbow=0, wrist_flex=π/2)
        but PRESERVE wrist_roll at `wrist_roll_rad` (defaults to current value
        from /joint_states). Used by Drop Scan so whatever wrist_roll the user
        set via the Roll +90° / Roll −90° buttons is honored throughout the
        sweep — drop_scan no longer flips the wrist itself.

        Safe duration computed from max joint delta (~30°/s sustained on the
        SO-ARM101 drives, min 1.5s).
        """
        with self.joint_lock:
            current = dict(self.joint_positions)
        if wrist_roll_rad is None:
            wrist_roll_rad = current['wrist_roll']
        target = {n: 0.0 for n in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        target['wrist_roll'] = float(wrist_roll_rad)
        max_delta = max(abs(target[n] - current[n]) for n in ARM_JOINT_NAMES)
        SAFE_RATE_RAD_PER_S = math.radians(30)
        duration = max(1.5, max_delta / SAFE_RATE_RAD_PER_S + 0.5)
        self._append_log(
            f'Real: scan_home (preserve wrist_roll='
            f'{math.degrees(wrist_roll_rad):+.1f}°, '
            f'max_Δ={math.degrees(max_delta):.1f}°, dur={duration:.1f}s)')
        return bool(self._send_arm_goal(
            target, duration_s=duration, blocking=True))

    def _real_set_wrist_roll(self, target_rad):
        """Move ONLY wrist_roll to target_rad; all other joints stay at
        current /joint_states values. Safe duration based on the wrist_roll
        delta (other joints have ~0 delta so they don't bound it).
        """
        self._ensure_real_state()
        with self.joint_lock:
            current = dict(self.joint_positions)
        target = dict(current)
        target['wrist_roll'] = float(target_rad)
        delta = abs(target_rad - current['wrist_roll'])
        SAFE_RATE_RAD_PER_S = math.radians(30)
        duration = max(1.5, delta / SAFE_RATE_RAD_PER_S + 0.5)
        self._append_log(
            f'Real: wrist_roll → {math.degrees(target_rad):+.1f}° '
            f'(Δ={math.degrees(delta):.1f}°, dur={duration:.1f}s)')
        return bool(self._send_arm_goal(
            target, duration_s=duration, blocking=True))

    def _cmd_real_swap_sweep_direction(self):
        """Toggle the Drop Scan sweep direction. False (default) = canonical
        right→left starting at +90°. True = swap, left→right starting at -90°.
        Read by `_real_drop_scan_thread` at scan start."""
        self._ensure_real_state()
        new_val = not self._real_sweep_reversed_var.get()
        self._real_sweep_reversed_var.set(new_val)
        direction = 'left→right (start -90°)' if new_val else 'right→left (start +90°)'
        self._append_log(f'Real: sweep direction → {direction}')

    def _cmd_real_wrist_roll_plus(self):
        """Move wrist_roll to +90° (markers-in-FOV / scan pose)."""
        threading.Thread(
            target=self._real_set_wrist_roll, args=(math.pi / 2,),
            daemon=True).start()

    def _cmd_real_wrist_roll_minus(self):
        """Move wrist_roll to −90° (jaw-clearance / drop pose)."""
        threading.Thread(
            target=self._real_set_wrist_roll, args=(-math.pi / 2,),
            daemon=True).start()

    def _real_drop_scan_motion(self, target, duration_s):
        """Send arm goal blocking. Returns True on success."""
        self._append_log(
            f'  drop_scan_motion: dispatching pan='
            f'{math.degrees(target["shoulder_pan"]):+.1f}° dur={duration_s:.1f}s')
        ok = bool(self._send_arm_goal(
            target, duration_s=duration_s, blocking=True))
        self._append_log(
            f'  drop_scan_motion: returned ok={ok}')
        return ok

    def _real_sample_drop_poses(self, duration_s, filter_id=None):
        """Subscribe to /drop_poses_real for `duration_s`, collect all
        TF transforms by child_frame_id. Returns {drop_id: [pose_dict, ...]}.
        Each pose_dict has 'x','y','z','qx','qy','qz','qw'.
        If filter_id is given, only collect that drop_id (saves work)."""
        samples = {}
        def _cb(msg):
            for tf in msg.transforms:
                fid = tf.child_frame_id
                if filter_id is not None and fid != filter_id:
                    continue
                samples.setdefault(fid, []).append({
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                })
        sub = self.create_subscription(
            TFMessage, '/drop_poses_real', _cb, 50,
            callback_group=self._sub_cb_group)
        try:
            time.sleep(duration_s)
        finally:
            self.destroy_subscription(sub)
        return samples

    def _real_average_pose_samples(self, samples):
        """Component-wise mean of xyz + normalized-mean quaternion. Returns
        (avg_pose_dict, xy_variance_m2). Quaternion sign-flips canonicalized
        before averaging to handle the q=-q ambiguity (otherwise opposite
        signs cancel)."""
        n = len(samples)
        sx = sum(s['x'] for s in samples) / n
        sy = sum(s['y'] for s in samples) / n
        sz = sum(s['z'] for s in samples) / n
        # xy variance (squared 2D distance from mean)
        xy_var = sum((s['x']-sx)**2 + (s['y']-sy)**2 for s in samples) / n
        # Quat mean: canonicalize sign vs first sample to avoid ±q cancellation
        ref = samples[0]
        qx = qy = qz = qw = 0.0
        for s in samples:
            sign = 1.0 if (s['qw']*ref['qw'] + s['qx']*ref['qx'] +
                           s['qy']*ref['qy'] + s['qz']*ref['qz']) >= 0 else -1.0
            qx += sign * s['qx']
            qy += sign * s['qy']
            qz += sign * s['qz']
            qw += sign * s['qw']
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
        return ({
            'x': sx, 'y': sy, 'z': sz,
            'qx': qx/norm, 'qy': qy/norm, 'qz': qz/norm, 'qw': qw/norm,
        }, xy_var)

    def _cmd_real_drop_point_green(self):
        """Deprecated by Drop Scan (Phase 11-01 followup). Kept as a
        deprecation stub so the auto-registered ~/real_drop_point_green
        service doesn't error if invoked during the transition. Will be
        deleted on next Phase 11 cleanup pass."""
        self._append_log(
            'Real: drop_point_green is deprecated — use Drop Scan instead',
            'warn')

    def _cmd_real_refresh_cups_pose(self):
        """Subscribe-once to /drop_poses_real, partial-merge captured
        transforms into _cached_cup_poses (missing markers keep their
        previous cached value), push the merged set to MoveIt as collision
        objects, and update _drop_data + _drop_listbox so the Grasp tab's
        drop selection mirrors the cache.
        """
        self._ensure_real_state()
        # Force all relevant topics to /*_real BEFORE the refresh so live
        # callbacks during/after the refresh write real-source data into
        # _drop_data / objects_data / objects_bbox. Without this, sim subs
        # continuously overwrite _drop_data and any subsequent code path
        # that publishes visual markers without an explicit cups_dict
        # (e.g. _cmd_apply_collision_padding → _refresh_display_markers)
        # would re-publish sim cup positions.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_refresh_cups_thread, daemon=True).start()

    def _real_refresh_cups_thread(self):
        evt = threading.Event()
        captured = {}

        def _once_cb(msg):
            if evt.is_set():
                return
            for tf in msg.transforms:
                captured[tf.child_frame_id] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }
            evt.set()

        topic = '/drop_poses_real'
        self._append_log(f'Real: subscribing once to {topic} (3s timeout)')
        sub = self.create_subscription(
            TFMessage, topic, _once_cb, 10,
            callback_group=self._sub_cb_group)
        try:
            ok = evt.wait(timeout=3.0)
            if not ok:
                self._append_log(
                    f'Real: no message on {topic} within 3s — is '
                    'aruco_camera_localizer running and seeing markers?',
                    'warn')
                return
            # Partial-cache merge: only update what was captured this scan.
            self._cached_cup_poses.update(captured)
            captured_names = sorted(captured.keys())
            kept_previous = [
                n for n in sorted(self._cached_cup_poses.keys())
                if n not in captured]
            self._append_log(
                f'Real: refreshed cups: captured={captured_names}, '
                f'kept_previous={kept_previous}, '
                f'total_cached={len(self._cached_cup_poses)}')
            # Mirror into _drop_data + listbox so existing drop_point/sweep/
            # release flows (which read from _drop_data) pick up the cache.
            with self._drop_lock:
                self._drop_data.update(self._cached_cup_poses)
            self.root.after(0, self._populate_drop_list)
            # WIPE-then-PUSH mirrors _cmd_drop_refresh's working pattern —
            # MoveIt's ApplyPlanningScene ADD-on-existing-id is not cleanly
            # idempotent, so repeat refreshes silently fail without prior
            # REMOVE. 500 ms gap = planning-scene monitor round-trip.
            self._remove_cup_collision_objects()
            time.sleep(0.5)
            self._add_cup_collision_objects(
                cups_dict=dict(self._cached_cup_poses))
            # Visual markers must use the SAME cached snapshot as the
            # collision — without the explicit cups_dict, the publisher
            # would read self._drop_data which the (sim) /drop_poses sub
            # continuously overwrites, causing visual+collision divergence.
            self._publish_cup_visual_markers(
                cups_dict=dict(self._cached_cup_poses))
            self._append_log(
                f'Real: published cup visual markers from cache '
                f'({len(self._cached_cup_poses)} cups)')
            # Status label: ✓ for cached, ✗ for missing.
            def _update_label():
                marks = []
                for n in ('drop_0', 'drop_1', 'drop_2'):
                    glyph = '✓' if n in self._cached_cup_poses else '✗'
                    marks.append(f'{n}:{glyph}')
                self._real_cups_status_var.set(
                    'Cups: ' + '   '.join(marks))
            self.root.after(0, _update_label)
            self._real_refresh_color_dropdown()
        finally:
            self.destroy_subscription(sub)

    def _cmd_real_refresh_legos_pose(self):
        """Subscribe-once to /objects_poses_real, partial-merge captured
        lego TFs into _cached_lego_poses (missing legos keep their previous
        cached value), push the merged set to MoveIt as collision objects,
        and update objects_data + obj_listbox so the Grasp tab's lego
        selection mirrors the cache.

        Run this AT GRASP_HOME pose — that's the only pose where the wrist
        camera looks down at the workspace. YOLOE only sees what's in FOV
        at scan time, which is why the cache is essential for the run loop:
        once the arm leaves home for a grasp, the live /objects_poses_real
        feed only contains whatever's near tcp_link.
        """
        self._ensure_real_state()
        # Same reasoning as Refresh Cups Pose — force topics before the
        # refresh so live callbacks during the cycle write real-source data.
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_refresh_legos_thread, daemon=True).start()

    def _real_refresh_legos_thread(self):
        evt = threading.Event()
        captured = {}

        def _once_cb(msg):
            if evt.is_set():
                return
            for tf in msg.transforms:
                captured[tf.child_frame_id] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }
            evt.set()

        topic = '/objects_poses_real'
        self._append_log(f'Real: subscribing once to {topic} (3s timeout)')
        sub = self.create_subscription(
            TFMessage, topic, _once_cb, 10,
            callback_group=self._sub_cb_group)
        try:
            ok = evt.wait(timeout=3.0)
            if not ok:
                self._append_log(
                    f'Real: no message on {topic} within 3s — is '
                    'localize_yoloe running and seeing legos in FOV?',
                    'warn')
                return
            # Partial-cache merge: only update what was captured this scan.
            self._cached_lego_poses.update(captured)
            captured_names = sorted(captured.keys())
            kept_previous = [
                n for n in sorted(self._cached_lego_poses.keys())
                if n not in captured]
            self._append_log(
                f'Real: refreshed legos: captured={captured_names}, '
                f'kept_previous={kept_previous}, '
                f'total_cached={len(self._cached_lego_poses)}')
            # Mirror into objects_data + listbox so existing grasp_move /
            # gripper_close_for_object flows (which read objects_data)
            # pick up the cache. Live /objects_poses_real callbacks may
            # overwrite during the run — re-injection right before the
            # cycle in _real_run_one_color_thread guards against that.
            with self.objects_lock:
                self.objects_data.update(self._cached_lego_poses)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._qs_refresh_objects)
            # WIPE-then-PUSH mirrors _cmd_grasp_refresh's working pattern.
            # MoveIt's ApplyPlanningScene ADD on an existing id is not
            # cleanly idempotent — repeat refreshes silently fail unless
            # the prior collision objects are removed first. The 500 ms
            # gap is the round-trip the planning-scene monitor needs to
            # see the REMOVE before the ADD lands against a clean state.
            self._remove_lego_collision_objects()
            time.sleep(0.5)
            self._add_lego_collision_objects(
                legos_dict=dict(self._cached_lego_poses))
            self.root.after(0, self._real_update_legos_status_label)
            self._real_refresh_color_dropdown()
        finally:
            self.destroy_subscription(sub)

    def _real_update_legos_status_label(self):
        """Refresh the 'Legos: N cached  (red:R blue:B green:G)' label.
        Extracted so the post-drop eviction path can reuse it without
        duplicating the tally logic."""
        tally = {'red': 0, 'blue': 0, 'green': 0}
        for name in self._cached_lego_poses:
            color = name.split('_', 1)[0].lower()
            if color in tally:
                tally[color] += 1
        total = len(self._cached_lego_poses)
        breakdown = '  '.join(f'{c}:{n}' for c, n in tally.items())
        if hasattr(self, '_real_legos_status_var'):
            self._real_legos_status_var.set(
                f'Legos: {total} cached  ({breakdown})')

    def _cmd_real_clear_cache(self):
        """Clear every Real Test tab cache and reset to defaults.

        Inverse of both Refresh Cups Pose, Refresh Legos Pose, and the
        wrist-roll toggle. Walks all state mutated by this tab's handlers
        so no ghost state remains.

        Cleared:
          - _cached_cup_poses + _cached_lego_poses (the scan caches)
          - _drop_data + objects_data entries that were mirrored from the
            caches (selective by name — leaves entries from live topic
            subscriptions alone, since those refill on the next message)
          - drop_listbox + obj_listbox (re-populated from trimmed data)
          - cup + lego collision objects in MoveIt planning scene
          - cup visual markers in RViz (idempotent DELETE)
          - Status labels (Cups + Legos + Run)

        Not cleared:
          - QS state (handled separately by Quickstart Restart)
        """
        self._ensure_real_state()
        cached_cup_names = list(self._cached_cup_poses.keys())
        cached_lego_names = list(self._cached_lego_poses.keys())

        self._cached_cup_poses.clear()
        self._cached_lego_poses.clear()

        # Selectively pop the cached names from _drop_data / objects_data
        # so the listboxes don't keep showing cups/legos we no longer trust.
        # Live subscriptions will refill these on the next msg.
        if cached_cup_names:
            with self._drop_lock:
                for name in cached_cup_names:
                    self._drop_data.pop(name, None)
            self.root.after(0, self._populate_drop_list)
        if cached_lego_names:
            with self.objects_lock:
                for name in cached_lego_names:
                    self.objects_data.pop(name, None)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._qs_refresh_objects)

        # Wipe the FULL sim-side state too — objects_data + _drop_data
        # carry residual entries from any live subscription that fired
        # since the last refresh. The user explicitly asked for clear
        # cache to clear EVERYTHING (sim + real) so subsequent refreshes
        # start from a known-empty state instead of inheriting half-stale
        # poses that mix sim and real frames.
        with self.objects_lock:
            sim_lego_count = len(self.objects_data)
            self.objects_data.clear()
        with self._drop_lock:
            sim_cup_count = len(self._drop_data)
            self._drop_data.clear()
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._populate_drop_list)
        self.root.after(0, self._qs_refresh_objects)

        self._remove_cup_collision_objects()
        self._remove_lego_collision_objects()
        # Visual markers (RViz colored cups). Idempotent — publishes DELETE
        # regardless of source (this tab's Refresh, sim's _cmd_drop_refresh,
        # or never added). Doesn't depend on _show_visual_var; clearing the
        # cache always clears the visual representation.
        self._delete_visual_markers()

        # Reset subs back to sim defaults — without this, the next refresh
        # button press in the Grasp/QS tab would still pull from /*_real
        # because subs are sticky after a Real Test session.
        self._sim_ensure_sim_topics()

        if hasattr(self, '_real_cups_status_var'):
            self._real_cups_status_var.set('Cups: (none)')
        if hasattr(self, '_real_legos_status_var'):
            self._real_legos_status_var.set('Legos: (none)')
        if hasattr(self, '_real_run_status_var'):
            self._real_run_status_var.set('Idle')

        self._append_log(
            f'Cleared ALL caches: real({len(cached_cup_names)} cups, '
            f'{len(cached_lego_names)} legos) + '
            f'sim({sim_cup_count} cups, {sim_lego_count} legos), '
            f'collisions removed, subs reset to sim, wrist_roll sign reset')
        self._real_refresh_color_dropdown()

    # --- Color-driven single-pick helper (Phase 11-01) ---
    # Color → cup mapping lives at module level (REAL_COLOR_TO_CUP) so
    # hot-reload picks up changes — class-level constants don't survive
    # _patch_methods.

    def _real_ensure_real_topics(self):
        """Force Grasp Topic + BBox Topic + Drop Topic to /*_real equivalents
        and re-subscribe so sim callbacks can't leak data into objects_data /
        objects_bbox / _drop_data during a Real Test cycle (D-07).

        Idempotent based on the SUBSCRIPTION's own `topic_name`, NOT the
        StringVar widget value. The widget can disagree with the live
        subscription (the original D-07 finding — _build_grasp_tab inits
        the sub on the sim default before _restore_gui_state flips the
        StringVar to real, no auto-re-sub). Reading sub.topic_name
        bypasses that. Re-subscribing every call is the previous
        implementation but it races destroy_subscription/create_subscription
        against MultiThreadedExecutor worker threads currently servicing
        the same sub — produces `InvalidHandle: destruction was requested`
        crashes mid-spin. The idempotent path makes the destroy/create
        race a once-per-process event instead of once-per-scan.

        Must be called from the tk thread (touches StringVars + buttons).

        Why this is essential:
        - `_objects_callback` (sim sub) overwrites `objects_data` AND auto-
          schedules `_add_lego_collision_objects` on >3 mm moves → sim
          poses leak into both objects_data AND the MoveIt planning scene.
        - `_bbox_callback` (sim sub) leaves `_lookup_bbox` returning sim
          catalog values instead of YOLOE's real catalog.
        - `_drop_callback` (sim sub) overwrites `_drop_data`, which
          `_publish_cup_visual_markers` reads → visual cups in RViz show
          sim positions instead of real-cached positions, causing
          visual-vs-collision divergence in the user's view.
        """
        REAL_OBJECTS = '/objects_poses_real'
        REAL_BBOX = '/objects_bbox_real'
        REAL_DROP = '/drop_poses_real'

        def _sub_topic(sub_attr):
            sub = getattr(self, sub_attr, None)
            if sub is None:
                return None
            try:
                return sub.topic_name.lstrip('/').rstrip('/')
            except Exception:
                return None

        def _norm(topic):
            return topic.lstrip('/').rstrip('/')

        # --- Grasp Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('objects_sub') != _norm(REAL_OBJECTS):
            prev_obj = self._grasp_topic_var.get().strip()
            self._grasp_topic_var.set(REAL_OBJECTS)
            self._cmd_grasp_update_topic()
            self._append_log(
                f'Real: forced grasp_topic {prev_obj!r} → {REAL_OBJECTS!r} '
                '(re-subscribed)')
        else:
            # Sub already on real — just sync the widget StringVar so the UI
            # doesn't lie about which topic is live.
            if self._grasp_topic_var.get().strip() != REAL_OBJECTS:
                self._grasp_topic_var.set(REAL_OBJECTS)

        # --- BBox Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('bbox_sub') != _norm(REAL_BBOX):
            prev_bbox = self._bbox_topic_var.get().strip()
            self._bbox_topic_var.set(REAL_BBOX)
            if hasattr(self, 'bbox_sub') and self.bbox_sub is not None:
                self.destroy_subscription(self.bbox_sub)
            self.bbox_sub = self.create_subscription(
                String, REAL_BBOX, self._bbox_callback, 1)
            # Clear stale sim bbox; next /objects_bbox_real msg refills it
            # within ~1 s (YOLOE publishes continuously).
            self.objects_bbox = {}
            self._append_log(
                f'Real: forced bbox_topic {prev_bbox!r} → {REAL_BBOX!r} '
                '(re-subscribed)')
        else:
            if self._bbox_topic_var.get().strip() != REAL_BBOX:
                self._bbox_topic_var.set(REAL_BBOX)

        # --- Drop Topic: re-subscribe only if sub topic isn't already real ---
        if _sub_topic('_drop_sub') != _norm(REAL_DROP):
            prev_drop = self._drop_topic_var.get().strip()
            self._drop_topic_var.set(REAL_DROP)
            # _update_drop_topic clears _drop_data + removes cup collisions —
            # those will be re-added by the next Refresh Cups Pose. The
            # call ALSO publishes "Drop topic: ..." log line for visibility.
            self._update_drop_topic(REAL_DROP)
            self._append_log(
                f'Real: forced drop_topic {prev_drop!r} → {REAL_DROP!r} '
                '(re-subscribed)')
        else:
            if self._drop_topic_var.get().strip() != REAL_DROP:
                self._drop_topic_var.set(REAL_DROP)

    def _sim_ensure_sim_topics(self):
        """Mirror of `_real_ensure_real_topics` for the SIM side.

        Force `objects_sub`, `bbox_sub`, and `_drop_sub` to the sim defaults
        (`/objects_poses_sim`, `/objects_bbox_sim`, `/drop_poses`). Idempotent
        on each sub's `topic_name` so the destroy/create race is at most a
        once-per-process event.

        Called from `_cmd_grasp_refresh` and `_cmd_drop_refresh` so that
        sim refresh buttons reliably switch back to sim topics even after a
        prior `_real_ensure_real_topics` swap. Without this, sim refresh
        leaks: it clears objects_data + cup_collision and waits for the
        live topic to refill — but the topic is still on `/*_real`, so
        sim collision objects come from real-mode poses. Closes the
        D-08 sim/real state-leak loop the user described.
        """
        SIM_OBJECTS = '/objects_poses_sim'
        SIM_BBOX = '/objects_bbox_sim'
        SIM_DROP = '/drop_poses'

        def _sub_topic(sub_attr):
            sub = getattr(self, sub_attr, None)
            if sub is None:
                return None
            try:
                return sub.topic_name.lstrip('/').rstrip('/')
            except Exception:
                return None

        def _norm(topic):
            return topic.lstrip('/').rstrip('/')

        # Grasp Topic → /objects_poses_sim
        if _sub_topic('objects_sub') != _norm(SIM_OBJECTS):
            prev = self._grasp_topic_var.get().strip() \
                if hasattr(self, '_grasp_topic_var') else ''
            if hasattr(self, '_grasp_topic_var'):
                self._grasp_topic_var.set(SIM_OBJECTS)
            self._cmd_grasp_update_topic()
            self._append_log(
                f'Sim: forced grasp_topic {prev!r} → {SIM_OBJECTS!r} '
                '(re-subscribed)')
        elif hasattr(self, '_grasp_topic_var') \
                and self._grasp_topic_var.get().strip() != SIM_OBJECTS:
            self._grasp_topic_var.set(SIM_OBJECTS)

        # BBox Topic → /objects_bbox_sim
        if _sub_topic('bbox_sub') != _norm(SIM_BBOX):
            prev = self._bbox_topic_var.get().strip() \
                if hasattr(self, '_bbox_topic_var') else ''
            if hasattr(self, '_bbox_topic_var'):
                self._bbox_topic_var.set(SIM_BBOX)
            if hasattr(self, 'bbox_sub') and self.bbox_sub is not None:
                self.destroy_subscription(self.bbox_sub)
            self.bbox_sub = self.create_subscription(
                String, SIM_BBOX, self._bbox_callback, 1)
            self.objects_bbox = {}
            self._append_log(
                f'Sim: forced bbox_topic {prev!r} → {SIM_BBOX!r} '
                '(re-subscribed)')
        elif hasattr(self, '_bbox_topic_var') \
                and self._bbox_topic_var.get().strip() != SIM_BBOX:
            self._bbox_topic_var.set(SIM_BBOX)

        # Drop Topic → /drop_poses
        if _sub_topic('_drop_sub') != _norm(SIM_DROP):
            prev = self._drop_topic_var.get().strip() \
                if hasattr(self, '_drop_topic_var') else ''
            if hasattr(self, '_drop_topic_var'):
                self._drop_topic_var.set(SIM_DROP)
            self._update_drop_topic(SIM_DROP)
            self._append_log(
                f'Sim: forced drop_topic {prev!r} → {SIM_DROP!r} '
                '(re-subscribed)')
        elif hasattr(self, '_drop_topic_var') \
                and self._drop_topic_var.get().strip() != SIM_DROP:
            self._drop_topic_var.set(SIM_DROP)

    def _real_inject_active_pair(self, lego_name, cup_name):
        """Re-inject the active lego + cup pose into objects_data and
        _drop_data from the cache. Called BEFORE EACH _QS_SEQUENCE step
        so a late real-topic callback (or any other writer) can't drift
        the pose mid-cycle. Cheap (~µs) — idempotent .update() per call.
        """
        with self._drop_lock:
            self._drop_data[cup_name] = dict(
                self._cached_cup_poses[cup_name])
        with self.objects_lock:
            self.objects_data[lego_name] = dict(
                self._cached_lego_poses[lego_name])

    # --- Per-step debug buttons (Phase 11-01 followup) ---
    # Each _cmd_real_<step> button runs ONE _QS_SEQUENCE step in real-mode
    # so the user can debug step-by-step without losing the cache-pose
    # injection guarantees. Auto-registered as ~/real_<step> services via
    # the _cmd_* convention.

    def _real_run_step(self, label, method_name, kwargs=None):
        """Common dispatcher for individual Real Test step buttons.

        Pre-flight: dropdown color is selected, at least one cached lego
        of that color exists, and (for cup steps) the matching cup is
        cached. Forces real topics, picks closest cached lego of color,
        re-injects pose for active lego+cup, dispatches via
        `_qs_execute_step` which handles obj_listbox / drop_listbox sync
        per the existing per-step policies.

        Runs the actual motion in a background thread so the GUI doesn't
        freeze. Logs ✓/✗ outcome at completion.
        """
        self._ensure_real_state()
        color = self._real_selected_color_var.get()
        if not color:
            self._append_log(
                f'Real step "{label}": no color selected — pick one first',
                'warn')
            return
        if not self._cached_lego_poses:
            self._append_log(
                f'Real step "{label}": lego cache empty — Refresh Legos first',
                'err')
            return
        candidates = [
            (math.hypot(p['x'], p['y']), name)
            for name, p in self._cached_lego_poses.items()
            if name.split('_', 1)[0].lower() == color
        ]
        if not candidates:
            self._append_log(
                f'Real step "{label}": no cached legos of color {color!r}',
                'err')
            return
        candidates.sort()
        _, lego = candidates[0]
        cup_name = self._qs_auto_drop_for_lego(lego)

        # Force real topics + re-inject (idempotent if already real).
        self._real_ensure_real_topics()
        if cup_name and cup_name in self._cached_cup_poses:
            self._real_inject_active_pair(lego, cup_name)
        else:
            # Lego-only inject for steps that don't need a cup
            with self.objects_lock:
                self.objects_data[lego] = dict(
                    self._cached_lego_poses[lego])
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._populate_drop_list)
        self.root.after(0, self._qs_refresh_objects)

        def _worker():
            time.sleep(0.6)  # let tk flush listbox repopulation
            if not self._qs_select_lego(lego):
                self._append_log(
                    f'Real step "{label}": {lego} not in qs listbox', 'err')
                return
            self._append_log(
                f'Real step "{label}": {lego}'
                + (f' → {cup_name}' if cup_name else ''))
            ok = self._qs_execute_step(method_name, kwargs or {})
            self._append_log(
                f'Real step "{label}": {"✓" if ok else "✗"}')
        threading.Thread(target=_worker, daemon=True).start()

    def _cmd_real_grasp_home(self):
        """Real-mode Grasp Home (debug step) — moves arm to home pose."""
        self._real_run_step('grasp home', '_cmd_grasp_home')

    def _cmd_real_grasp_open(self):
        """Real-mode Grasp Open (debug step) — opens gripper sized for
        the dropdown color's closest cached lego."""
        self._real_run_step('grasp open', '_cmd_gripper_open_for_object')

    def _cmd_real_grasp_move(self):
        """Real-mode Grasp Move (debug step) — IK to dropdown color's
        closest cached lego pose. Uses cached pose only, never live topic."""
        self._real_run_step('grasp move', '_cmd_grasp_move')

    def _cmd_real_grasp_close(self):
        """Real-mode Grasp Close (debug step) — closes gripper on the
        active lego, attaches to gripper if grasp succeeds."""
        self._real_run_step('grasp close', '_cmd_gripper_close_for_object')

    def _cmd_real_drop_point(self):
        """Real-mode Drop Point (debug step) — pans arm to face the
        cached cup matching the dropdown color."""
        self._real_run_step('drop point', '_cmd_drop_point')

    def _cmd_real_drop_sweep(self):
        """Real-mode Drop Sweep (debug step) — IK-plans a sweep from
        carry pose to over the cached cup. Requires a lego attached."""
        self._real_run_step('drop sweep', '_cmd_drop_sweep')

    def _cmd_real_drop_release(self):
        """Real-mode Release (debug step) — opens gripper to release
        the carried lego into the cup."""
        self._real_run_step('release', '_cmd_drop_release')

    def _real_refresh_color_dropdown(self):
        """Recompute combobox values = intersection(cached lego colors,
        cached cup colors) and update widget + Run button enable state.

        Thread-safe — mutates tk widgets via root.after(0, ...). Called on
        every cache mutation: refresh cups, refresh legos, clear cache,
        post-drop evict.

        Wraps the body in try/except so a silent failure (e.g. missing
        attribute after partial hot-reload) surfaces in the log instead of
        being swallowed by the daemon thread that called us.
        """
        try:
            if not hasattr(self, '_real_color_combobox'):
                # Tab not built yet (early hot-reload race) — no widget to update.
                return
            lego_colors = {
                n.split('_', 1)[0].lower() for n in self._cached_lego_poses
            }
            lego_colors &= set(REAL_COLOR_TO_CUP)  # drop unknowns silently
            cup_colors_present = {
                color for color, cup_name in REAL_COLOR_TO_CUP.items()
                if cup_name in self._cached_cup_poses
            }
            available = sorted(lego_colors & cup_colors_present)

            def _apply():
                cb = getattr(self, '_real_color_combobox', None)
                btn = getattr(self, '_real_run_button', None)
                if cb is None:
                    return
                cb['values'] = available
                current = self._real_selected_color_var.get()
                if available:
                    if current not in available:
                        self._real_selected_color_var.set(available[0])
                    if btn is not None:
                        btn.config(state='normal')
                else:
                    self._real_selected_color_var.set('')
                    if btn is not None:
                        btn.config(state='disabled')
            self.root.after(0, _apply)
            self._append_log(f'Real: dropdown colors = {available}')
        except Exception as e:
            self._append_log(
                f'Real: dropdown refresh failed — {type(e).__name__}: {e}',
                'err')

    def _cmd_real_run_one_color(self):
        """Single-pick handler — picks ONE lego of the dropdown's selected
        color and runs it through the full _QS_SEQUENCE with cache-only
        IK inputs (no live /objects_poses_real or /drop_poses topics
        consulted mid-cycle).

        Pre-flight checked:
          - A color is selected in the dropdown.
          - Cup pose cache is non-empty.
          - Lego pose cache is non-empty.
          - No existing QS / Real run is in progress.

        On success: evicts the dropped lego from cache + dropdown shrinks.
        On failure: cache untouched, user can Refresh + retry.
        """
        self._ensure_real_state()
        color = self._real_selected_color_var.get()
        if not color:
            self._append_log(
                'Real: no color selected — Refresh first, then pick a color',
                'warn')
            return
        if not self._cached_cup_poses:
            self._append_log(
                'Real: cup pose cache is empty — Refresh Cups Pose first',
                'err')
            return
        if not self._cached_lego_poses:
            self._append_log(
                'Real: lego pose cache is empty — Refresh Legos Pose first '
                '(at grasp_home pose, where wrist camera sees workspace)',
                'err')
            return
        if self._qs_state == 'running':
            self._append_log(
                'Real: a run is already in progress — '
                'use Quickstart Restart to abort it first', 'warn')
            return
        # Force real topics on the tk thread BEFORE spawning the worker.
        # Worker thread re-injects from cache, but only the real-topic sub
        # prevents sim callbacks from clobbering objects_data / planning
        # scene between IK steps (D-07).
        self._real_ensure_real_topics()
        threading.Thread(
            target=self._real_run_one_color_thread,
            args=(color,), daemon=True).start()

    def _real_run_one_color_thread(self, color):
        try:
            self._append_log(
                'Real: single-pick mode — using cache only, '
                'no live topic source')
            self._real_set_status(f'Running ({color})', step='preparing')
            self._qs_abort_evt.clear()
            self._qs_resume_evt.set()
            self._qs_state = 'running'

            # Pick the closest-to-base lego of the chosen color.
            # Tiebreak: alphabetical on cache key for determinism.
            candidates = [
                (math.hypot(p['x'], p['y']), name, p)
                for name, p in self._cached_lego_poses.items()
                if name.split('_', 1)[0].lower() == color
            ]
            if not candidates:
                self._append_log(
                    f'Real: no cached legos of color {color!r} '
                    '(cache may have been mutated since dropdown refresh)',
                    'err')
                self._real_set_status('Error', step='no candidates')
                return
            candidates.sort(key=lambda t: (t[0], t[1]))
            distance, lego, _pose = candidates[0]

            cup_name = self._qs_auto_drop_for_lego(lego)
            if cup_name is None or cup_name not in self._cached_cup_poses:
                # Defense-in-depth — dropdown filter should prevent this.
                self._append_log(
                    f'Real: {lego} → cup {cup_name!r} not in cache', 'err')
                self._real_set_status('Error', step='cup missing')
                return

            self._append_log(
                f'Real: picked {lego} (dist={distance:.3f} m, '
                f'{len(candidates)} candidate(s) of color {color!r}) '
                f'→ {cup_name}')

            # Re-inject cached poses into the live data structures right
            # before the cycle. _cmd_grasp_move / _cmd_drop_sweep read these,
            # and a stray live-topic callback could otherwise overwrite
            # them with whatever YOLOE last saw (which is partial — the
            # camera no longer sees the full workspace once the arm leaves
            # grasp_home). Cache is the authoritative source here.
            with self._drop_lock:
                self._drop_data[cup_name] = dict(
                    self._cached_cup_poses[cup_name])
            with self.objects_lock:
                self.objects_data.update(self._cached_lego_poses)
            self.root.after(0, self._populate_object_list)
            self.root.after(0, self._populate_drop_list)
            self.root.after(0, self._qs_refresh_objects)
            time.sleep(0.6)  # let tk flush listbox repopulation

            if not self._qs_select_lego(lego):
                self._append_log(
                    f'Real: {lego} not in qs listbox after re-inject — '
                    'check _qs_refresh_objects path', 'err')
                self._real_set_status('Error', step='select failed')
                return

            ok_all = True
            failed_label = None
            for i, (label, method_name, kwargs) in enumerate(
                    self._QS_SEQUENCE, start=1):
                if self._qs_abort_evt.is_set():
                    ok_all = False
                    break
                self._qs_resume_evt.wait()  # honor pause
                # Defense in depth: re-inject cached poses RIGHT BEFORE
                # each step so any stray callback (even from /objects_poses_real
                # — YOLOE may have re-detected with a slightly drifted pose)
                # can't influence the IK target. Cache is the only source.
                self._real_inject_active_pair(lego, cup_name)
                self._real_set_status(
                    f'Running ({color})',
                    step=f'{i}/{len(self._QS_SEQUENCE)}: {label} ({lego})')
                ok = self._qs_execute_step(method_name, kwargs)
                if not ok:
                    ok_all = False
                    failed_label = label
                    self._append_log(
                        f'Real: cycle for {lego} halted at "{label}"', 'err')
                    break

            if self._qs_abort_evt.is_set():
                self._real_set_status('Aborted', step=f'({lego})')
                # No eviction — abort during a successful sequence still
                # shouldn't strip the lego from cache (we don't know if the
                # drop completed atomically).
            elif ok_all:
                self._real_set_status(
                    'Complete', step=f'✓ {lego} → {cup_name}')
                self._append_log(
                    f'Real: cycle complete — {lego} dropped in {cup_name}')
                self._real_evict_lego_from_cache(lego)
            else:
                self._real_set_status(
                    'Halted', step=f'✗ {lego} at "{failed_label}"')
                # No eviction (D-06).
        finally:
            self._qs_state = 'idle'

    def _real_set_status(self, status, step=None):
        """Thread-safe status update for the Real Test tab.

        Mirrors `_append_log`'s `_gui_ready` guard so we never call
        `root.after` from a daemon thread when the tk mainloop has exited
        (or hasn't fully started). Without the guard, any failure inside a
        `_real_*_thread` worker that touches the status label crashes the
        worker with `RuntimeError: main thread is not in main loop` instead
        of running the `finally` cleanup."""
        if not getattr(self, '_gui_ready', False):
            return
        def _update():
            text = status if step is None else f'{status} — {step}'
            self._real_run_status_var.set(text)
        try:
            self.root.after(0, _update)
        except RuntimeError:
            pass

    def _real_evict_lego_from_cache(self, lego_name):
        """Remove a successfully-dropped lego from cache + planning scene.

        Called only on the success path of _real_run_one_color_thread (D-05).
        Failed cycles MUST NOT call this — leaving the lego cached lets
        the user retry after a Refresh (D-06).

        Steps:
          1. Pop from _cached_lego_poses + objects_data (both must drop the
             entry, else the listbox keeps showing a phantom block).
          2. Wipe-then-push lego collisions in MoveIt scene with the now-
             smaller cache. Same 500 ms gap as _real_refresh_legos_thread
             (planning-scene monitor round-trip — see HANDOFF.json
             anti-patterns: ApplyPlanningScene ADD-on-existing-id is not
             cleanly idempotent).
          3. Refresh listboxes + Legos status label + dropdown.
        """
        popped = self._cached_lego_poses.pop(lego_name, None)
        if popped is None:
            self._append_log(
                f'Real: evict noop — {lego_name} not in cache', 'warn')
            return
        with self.objects_lock:
            self.objects_data.pop(lego_name, None)
        # Wipe-then-push to reflect the smaller cache. _add_lego_collision_objects
        # writes _lego_collision_names from scratch so we don't need to
        # surgically remove just one id.
        self._remove_lego_collision_objects()
        time.sleep(0.5)
        if self._cached_lego_poses:
            self._add_lego_collision_objects(
                legos_dict=dict(self._cached_lego_poses))
        self.root.after(0, self._populate_object_list)
        self.root.after(0, self._qs_refresh_objects)
        self.root.after(0, self._real_update_legos_status_label)
        self._real_refresh_color_dropdown()
        self._append_log(
            f'Real: evicted {lego_name} from cache '
            f'(remaining: {len(self._cached_lego_poses)})')

    def _qs_select_lego(self, lego_name):
        """Select a lego in the Quickstart listbox by name. Used by both
        the Real loop and as a generalization of the per-step QS policy.
        Returns False if the listbox is empty or the name isn't present.
        """
        if not hasattr(self, '_qs_listbox') or self._qs_listbox.size() == 0:
            return False
        for i in range(self._qs_listbox.size()):
            if self._qs_listbox.get(i).split('  ')[0] == lego_name:
                self._qs_listbox.selection_clear(0, tk.END)
                self._qs_listbox.selection_set(i)
                return True
        return False

    # ==================================================================
    # Record Sim — closed-loop dataset recording in Isaac Sim
    # ==================================================================
    #
    # Orchestrates the full recording session:
    # 1) spawn lerobot-record subprocess + /joint_commands_lerobot mirror
    # 2) loop scenes (per scene = randomize → run K episodes, one per
    #    pickable lego of the chosen color)
    # 3) per episode: optional reset-to-grasp_home → drive QS pick-place →
    #    wait for "pick-and-drop cycle complete" sentinel
    # 4) on QS halt → re-randomize same scene, retry same lego (up to
    #    REC_MAX_RETRIES_PER_LEGO before skipping)
    # 5) on Stop or session done → SIGINT subprocesses, finalize dataset
    #
    # All blocking work runs on a daemon thread; UI updates marshaled via
    # root.after(0, ...). State stored in self._rec_state dict guarded by
    # self._rec_lock.
    #
    # Constants live at MODULE level (defined just above this class) so
    # hot_reload picks up changes without a full restart — class-level
    # attributes only attach to a fresh class object created on import,
    # not to the running instance's existing __class__.

    def _build_record_sim_tab(self, notebook):
        """Tab: closed-loop sim recording. Spawns lerobot-record + mirror,
        runs a configurable number of pick-and-drop episodes against the
        chosen color, re-randomizing the scene every K episodes."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Record Sim')

        # --- State (lazy init — survives hot-reload) ---
        if not hasattr(self, '_rec_lock'):
            self._rec_lock = threading.Lock()
        if not hasattr(self, '_rec_state'):
            self._rec_state = {
                'phase': 'idle',          # idle|starting|running|paused|stopping
                'episode': 0,             # 1-indexed during run, 0 idle
                'n_episodes': 0,
                'scene': 0,
                'last_lego': '',
                'last_verdict': '',
                'dataset_dir': '',
                'lerobot_proc': None,
                'mirror_proc': None,
                'thread': None,
                'stop_evt': threading.Event(),
                'pause_evt': threading.Event(),  # set = run, clear = pause
                'total_retries': 0,
            }
            self._rec_state['pause_evt'].set()  # not paused on init

        # --- Tk vars (bound to widgets, persisted via widget registry) ---
        if not hasattr(self, '_rec_episodes_var'):
            self._rec_episodes_var = tk.IntVar(value=16)
        if not hasattr(self, '_rec_color_var'):
            self._rec_color_var = tk.StringVar(value='blue')
        if not hasattr(self, '_rec_dataset_var'):
            self._rec_dataset_var = tk.StringVar(value='')
        if not hasattr(self, '_rec_randomize_legos_var'):
            self._rec_randomize_legos_var = tk.BooleanVar(value=True)
        if not hasattr(self, '_rec_randomize_cups_var'):
            self._rec_randomize_cups_var = tk.BooleanVar(value=False)
        if not hasattr(self, '_rec_reset_arm_var'):
            self._rec_reset_arm_var = tk.BooleanVar(value=True)
        if not hasattr(self, '_rec_pause_between_var'):
            self._rec_pause_between_var = tk.BooleanVar(value=False)
        if not hasattr(self, '_rec_status_var'):
            self._rec_status_var = tk.StringVar(value='IDLE')
        if not hasattr(self, '_rec_progress_var'):
            self._rec_progress_var = tk.StringVar(value='—')
        if not hasattr(self, '_rec_dataset_path_var'):
            self._rec_dataset_path_var = tk.StringVar(value='—')

        # ===== Section: Settings =====
        settings = ttk.LabelFrame(frame, text='Settings')
        settings.pack(fill=tk.X, padx=10, pady=(8, 4))

        row1 = ttk.Frame(settings); row1.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row1, text='Episodes:').pack(side=tk.LEFT, padx=(0, 4))
        self._register_spinbox(
            row1, label='Episodes', tab='Record Sim', section='Settings',
            textvariable=self._rec_episodes_var,
            from_=1, to=10000, increment=1, width=6,
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(row1, text='Block color:').pack(side=tk.LEFT, padx=(0, 4))
        color_cb = ttk.Combobox(
            row1, textvariable=self._rec_color_var,
            values=list(REC_LEGOS_BY_COLOR.keys()),
            state='readonly', width=8,
        )
        color_cb.pack(side=tk.LEFT, padx=(0, 12))
        # Register so widget services can read/set it.
        self._widget_registry_add(
            'Block color', 'Combobox', color_cb,
            self._rec_color_var, tab='Record Sim', section='Settings',
            writable=True)

        row2 = ttk.Frame(settings); row2.pack(fill=tk.X, padx=5, pady=4)
        tk.Label(row2, text='Dataset name:').pack(side=tk.LEFT, padx=(0, 4))
        ds_entry = ttk.Entry(row2, textvariable=self._rec_dataset_var, width=40)
        ds_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._widget_registry_add(
            'Dataset name', 'Entry', ds_entry,
            self._rec_dataset_var, tab='Record Sim', section='Settings',
            writable=True)
        tk.Label(row2, text='(blank → auto rec_<HHMMSS>)',
                 fg='#888').pack(side=tk.LEFT)

        # Per-episode actions
        actions = ttk.LabelFrame(frame, text='Per-episode actions')
        actions.pack(fill=tk.X, padx=10, pady=4)
        for label, var in [
            ('Randomize legos (per scene = every K episodes)',
             self._rec_randomize_legos_var),
            ('Randomize cups (per scene)',
             self._rec_randomize_cups_var),
            ('Reset arm to grasp_home (per episode)',
             self._rec_reset_arm_var),
            ('Pause briefly between episodes (~2.5s)',
             self._rec_pause_between_var),
        ]:
            self._register_check(
                actions, label=label, tab='Record Sim', section='Actions',
                variable=var,
            ).pack(anchor=tk.W, padx=8, pady=1)

        # ===== Section: Control =====
        control = ttk.LabelFrame(frame, text='Control')
        control.pack(fill=tk.X, padx=10, pady=4)
        ctrl_row = ttk.Frame(control); ctrl_row.pack(fill=tk.X, padx=5, pady=6)
        self._rec_btn_start = self._register_button(
            ctrl_row, text='▶ Start', tab='Record Sim', section='Control',
            command=self._cmd_rec_start,
        )
        self._rec_btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=2, ipady=4)
        self._rec_btn_pause = self._register_button(
            ctrl_row, text='⏸ Pause', tab='Record Sim', section='Control',
            command=self._cmd_rec_pause,
        )
        self._rec_btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=2, ipady=4)
        self._rec_btn_stop = self._register_button(
            ctrl_row, text='⏹ Stop', tab='Record Sim', section='Control',
            command=self._cmd_rec_stop,
        )
        self._rec_btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='↻ Reset Scene', tab='Record Sim', section='Control',
            command=self._cmd_rec_reset_scene,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)
        self._register_button(
            ctrl_row, text='📁 Open Dataset', tab='Record Sim',
            section='Control', command=self._cmd_rec_open_dataset,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, ipady=4)

        # ===== Section: Status =====
        status = ttk.LabelFrame(frame, text='Status')
        status.pack(fill=tk.X, padx=10, pady=4)
        st_row1 = ttk.Frame(status); st_row1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row1, text='State:', width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(st_row1, textvariable=self._rec_status_var,
                 font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)

        st_row2 = ttk.Frame(status); st_row2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row2, text='Progress:', width=10, anchor=tk.W).pack(
            side=tk.LEFT)
        tk.Label(st_row2, textvariable=self._rec_progress_var).pack(
            side=tk.LEFT)

        st_row3 = ttk.Frame(status); st_row3.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(st_row3, text='Dataset:', width=10, anchor=tk.W).pack(
            side=tk.LEFT)
        tk.Label(st_row3, textvariable=self._rec_dataset_path_var,
                 fg='#0066cc').pack(side=tk.LEFT)

        # Initial button-enabled-ness
        self._rec_update_button_state()

    # -- Helpers -----------------------------------------------------------

    def _rec_set_status(self, status_text=None, progress_text=None,
                        dataset_text=None):
        """Marshal a status update onto the Tk thread."""
        def _apply():
            if status_text is not None:
                self._rec_status_var.set(status_text)
            if progress_text is not None:
                self._rec_progress_var.set(progress_text)
            if dataset_text is not None:
                self._rec_dataset_path_var.set(dataset_text)
            self._rec_update_button_state()
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _rec_update_button_state(self):
        """Enable/disable Start/Pause/Stop based on phase."""
        phase = self._rec_state['phase']
        is_idle = phase == 'idle'
        is_running = phase in ('running',)
        is_paused = phase == 'paused'
        try:
            self._rec_btn_start.configure(
                state=(tk.NORMAL if is_idle else tk.DISABLED))
            self._rec_btn_pause.configure(
                state=(tk.NORMAL if (is_running or is_paused) else tk.DISABLED),
                text=('▶ Resume' if is_paused else '⏸ Pause'))
            self._rec_btn_stop.configure(
                state=(tk.NORMAL if not is_idle else tk.DISABLED))
        except Exception:
            pass

    def _rec_mcp_call(self, cmd_type, params=None, timeout=120):
        """Call an Isaac Sim MCP tool via socket 8767. Returns response dict
        or None on failure. Used for randomize_object_poses, randomize_cups."""
        msg = json.dumps({"type": cmd_type, "params": params or {}})
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((REC_MCP_HOST, REC_MCP_PORT))
            sock.sendall(msg.encode("utf-8"))
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                try:
                    return json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
            sock.close()
        except Exception as exc:
            self._append_log(f'Record: MCP {cmd_type} failed: {exc}', 'warn')
        return None

    def _rec_call_trigger(self, srv_name, timeout=20):
        """Call a /so_arm101_control_gui/<srv_name> Trigger service from this
        node's executor. Returns the Trigger response or None."""
        cli = self.create_client(Trigger, f'~/{srv_name}')
        if not cli.wait_for_service(timeout_sec=5):
            return None
        fut = cli.call_async(Trigger.Request())
        # Spin until done — we're on a background thread, the executor runs
        # in main, so use spin_until_future_complete with the current
        # node + a timeout.
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < timeout:
            time.sleep(0.05)
        if fut.done():
            return fut.result()
        return None

    def _rec_topic_has_publisher(self, topic_name):
        """Return True if any node currently publishes `topic_name`.
        Uses the running node's discovery — no daemon dependency."""
        try:
            infos = self.get_publishers_info_by_topic(topic_name)
            return len(infos) > 0
        except Exception:
            return False

    def _rec_dataset_repo_id(self):
        """Resolve dataset repo_id from the entry; auto-name if blank."""
        name = (self._rec_dataset_var.get() or '').strip()
        if not name:
            name = f"rec_{datetime.datetime.now():%H%M%S}"
        if '/' not in name:
            name = f"local/{name}"
        return name

    # -- Commands ----------------------------------------------------------

    def _cmd_rec_start(self):
        """Spawn lerobot-record + mirror, then start the orchestration thread."""
        with self._rec_lock:
            if self._rec_state['phase'] != 'idle':
                self._append_log('Record: already running', 'warn')
                return
            self._rec_state['phase'] = 'starting'
            self._rec_state['stop_evt'].clear()
            self._rec_state['pause_evt'].set()
            self._rec_state['episode'] = 0
            self._rec_state['scene'] = 0
            self._rec_state['total_retries'] = 0
            self._rec_state['last_verdict'] = ''
            self._rec_state['last_lego'] = ''
            self._rec_state['n_episodes'] = int(self._rec_episodes_var.get())

        repo_id = self._rec_dataset_repo_id()
        ds_path = os.path.join(REC_DATASET_ROOT, repo_id.split('/', 1)[-1])
        if os.path.exists(ds_path):
            self._append_log(
                f'Record: dataset {ds_path} already exists; pick a new name',
                'err')
            with self._rec_lock:
                self._rec_state['phase'] = 'idle'
            self._rec_set_status(status_text='IDLE')
            return

        self._rec_state['dataset_dir'] = ds_path
        self._rec_set_status(
            status_text='STARTING (spawning subprocesses)',
            progress_text=f'0/{self._rec_state["n_episodes"]}',
            dataset_text=ds_path)
        self._append_log(f'Record: starting → {repo_id}')

        # Mirror dedup: if /joint_commands_lerobot already has a publisher
        # (mirror left over from a prior session or started manually),
        # reuse it instead of spawning a duplicate. We check the topic
        # publisher count rather than process names because the topic is
        # the actual contract the recorder needs.
        existing_mirror = self._rec_topic_has_publisher(
            '/joint_commands_lerobot')
        if existing_mirror:
            self._append_log(
                'Record: reusing existing /joint_commands_lerobot publisher')
            self._rec_state['mirror_proc'] = None
        else:
            try:
                mp = subprocess.Popen(
                    ['python3', '-u', REC_MIRROR_SCRIPT],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True)
                self._rec_state['mirror_proc'] = mp
            except Exception as exc:
                self._append_log(
                    f'Record: mirror spawn failed: {exc}', 'err')
                self._rec_finalize(error=True)
                return

        # Spawn lerobot-record (pixi-Jazzy via bash wrapper).
        cmd = [
            'bash', REC_LEROBOT_SCRIPT,
            f'--dataset.repo_id={repo_id}',
            f'--dataset.num_episodes={self._rec_state["n_episodes"]}',
            '--dataset.episode_time_s=120',
            '--dataset.reset_time_s=2',
            '--dataset.single_task=sort blue blocks',
            '--dataset.push_to_hub=false',
            '--display_data=false',
        ]
        try:
            log_path = f"/tmp/rec_{datetime.datetime.now():%H%M%S}.log"
            log_fd = open(log_path, 'w')
            lp = subprocess.Popen(
                cmd, stdout=log_fd, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True)
            self._rec_state['lerobot_proc'] = lp
            self._rec_state['lerobot_log'] = log_path
        except Exception as exc:
            self._append_log(f'Record: lerobot spawn failed: {exc}', 'err')
            self._rec_finalize(error=True)
            return

        # Start orchestration thread.
        t = threading.Thread(target=self._rec_loop_thread, daemon=True)
        self._rec_state['thread'] = t
        t.start()

    def _cmd_rec_pause(self):
        """Toggle pause/resume the orchestration loop. lerobot keeps running."""
        with self._rec_lock:
            phase = self._rec_state['phase']
            if phase == 'running':
                self._rec_state['phase'] = 'paused'
                self._rec_state['pause_evt'].clear()
                self._append_log('Record: paused (lerobot still recording)')
                self._rec_set_status(status_text='PAUSED')
            elif phase == 'paused':
                self._rec_state['phase'] = 'running'
                self._rec_state['pause_evt'].set()
                self._append_log('Record: resumed')
                self._rec_set_status(status_text='RUNNING')

    def _cmd_rec_stop(self):
        """Signal the loop to stop, then SIGINT subprocesses."""
        with self._rec_lock:
            if self._rec_state['phase'] == 'idle':
                return
            self._rec_state['phase'] = 'stopping'
            self._rec_state['stop_evt'].set()
            self._rec_state['pause_evt'].set()  # release pause if blocked
        self._rec_set_status(status_text='STOPPING')
        self._append_log('Record: stopping (waiting for clean tear-down…)')
        # Run the rest on a thread so we don't block tk.
        threading.Thread(target=self._rec_finalize_thread, daemon=True).start()

    def _cmd_rec_reset_scene(self):
        """Trigger sim_reset.sh in a subprocess. Idempotent — safe any time."""
        self._append_log('Record: sim_reset')
        try:
            subprocess.Popen(
                ['bash', REC_SIM_RESET_SCRIPT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception as exc:
            self._append_log(f'Record: reset failed: {exc}', 'err')

    def _cmd_rec_open_dataset(self):
        """xdg-open the most recent dataset folder (or the configured one)."""
        target = self._rec_state.get('dataset_dir') or REC_DATASET_ROOT
        if not os.path.exists(target):
            self._append_log(f'Record: no dataset at {target}', 'warn')
            return
        try:
            subprocess.Popen(['xdg-open', target],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._append_log(f'Record: open failed: {exc}', 'err')

    # -- Subprocess teardown ----------------------------------------------

    def _rec_finalize_thread(self):
        """Background: SIGINT subprocesses, wait for clean exit, then idle."""
        for key in ('lerobot_proc', 'mirror_proc'):
            proc = self._rec_state.get(key)
            if proc and proc.poll() is None:
                try:
                    # SIGINT propagates through start_new_session
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    try:
                        proc.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
        # Wait up to ~10s for clean exit, then escalate.
        deadline = time.time() + 10.0
        for key in ('lerobot_proc', 'mirror_proc'):
            proc = self._rec_state.get(key)
            if not proc:
                continue
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self._rec_finalize(error=False)

    def _rec_finalize(self, error=False):
        """Reset state to idle. Safe to call from any thread."""
        with self._rec_lock:
            ep_done = self._rec_state['episode']
            ep_target = self._rec_state['n_episodes']
            self._rec_state['phase'] = 'idle'
            self._rec_state['lerobot_proc'] = None
            self._rec_state['mirror_proc'] = None
            self._rec_state['stop_evt'].clear()
            self._rec_state['pause_evt'].set()
        msg = 'IDLE (error)' if error else 'IDLE'
        # Final summary stays in the progress label so the user can see
        # how many episodes landed before they stopped / it completed.
        if ep_target > 0:
            summary = f'last run: {ep_done}/{ep_target} episodes'
        else:
            summary = '—'
        self._rec_set_status(status_text=msg, progress_text=summary)
        self._append_log(f'Record: finalized ({msg}, {summary})')

    # -- Main loop --------------------------------------------------------

    def _rec_loop_thread(self):
        """Orchestration loop. Runs on a daemon background thread."""
        st = self._rec_state
        legos = list(REC_LEGOS_BY_COLOR[self._rec_color_var.get()])
        legos_per_scene = len(legos)
        n_target = st['n_episodes']

        # Warm-up wait — give lerobot ~10s to subscribe + emit first frame.
        for _ in range(50):
            if st['stop_evt'].is_set():
                return
            if st.get('lerobot_proc') and st['lerobot_proc'].poll() is None:
                # Probe its log for "Recording episode" sentinel
                log_path = st.get('lerobot_log', '')
                if log_path and os.path.exists(log_path):
                    try:
                        with open(log_path) as f:
                            if 'Recording episode 0' in f.read():
                                break
                    except Exception:
                        pass
            time.sleep(0.2)

        with self._rec_lock:
            if st['phase'] != 'starting':
                return
            st['phase'] = 'running'
        self._rec_set_status(status_text='RUNNING')

        episode_idx = 0  # 0-indexed
        scene_idx = 0
        while episode_idx < n_target and not st['stop_evt'].is_set():
            st['pause_evt'].wait()  # block while paused

            # Start of scene?
            if episode_idx % legos_per_scene == 0:
                scene_idx += 1
                st['scene'] = scene_idx
                self._rec_set_status(
                    status_text='RANDOMIZING',
                    progress_text=(
                        f'Episode {episode_idx + 1}/{n_target}, '
                        f'scene {scene_idx}'))
                if self._rec_randomize_legos_var.get():
                    self._rec_mcp_call('randomize_object_poses', {})
                if self._rec_randomize_cups_var.get():
                    self._rec_mcp_call('randomize_cups', {})
                time.sleep(1.0)  # let physics settle

            lego = legos[episode_idx % legos_per_scene]
            st['last_lego'] = lego

            verdict = self._rec_run_episode_with_retry(lego, episode_idx, n_target)

            if verdict == 'ABORT':
                break
            st['last_verdict'] = verdict
            episode_idx += 1
            st['episode'] = episode_idx

            if self._rec_pause_between_var.get():
                time.sleep(2.5)

        # Clean exit: signal stop, finalize.
        if not st['stop_evt'].is_set():
            self._append_log(f'Record: completed {episode_idx} episodes')
            self._cmd_rec_stop()

    def _rec_run_episode_with_retry(self, lego, episode_idx, n_target):
        """Run one episode for `lego`, re-randomizing on QS halt up to
        REC_MAX_RETRIES_PER_LEGO times. Returns 'PASS' / 'SKIP' / 'ABORT'."""
        st = self._rec_state
        for attempt in range(REC_MAX_RETRIES_PER_LEGO):
            if st['stop_evt'].is_set():
                return 'ABORT'
            st['pause_evt'].wait()

            self._rec_set_status(
                status_text='RUNNING',
                progress_text=(
                    f'Ep {episode_idx + 1}/{n_target}, scene {st["scene"]}, '
                    f'lego {lego} (try {attempt + 1})'))

            # Optional reset to grasp_home before each episode.
            if self._rec_reset_arm_var.get():
                self._rec_call_trigger('grasp_home', timeout=20)

            verdict = self._rec_run_one_qs_cycle(lego)
            if verdict == 'PASS':
                return 'PASS'
            if verdict == 'ABORT':
                return 'ABORT'
            # FAIL: re-randomize and retry
            st['total_retries'] += 1
            self._append_log(
                f'Record: episode {episode_idx + 1} failed on {lego} '
                f'(attempt {attempt + 1}); re-randomizing')
            self._rec_mcp_call('randomize_object_poses', {})
            time.sleep(1.0)

        # Exhausted retries — log and skip.
        self._append_log(
            f'Record: lego {lego} unreachable after '
            f'{REC_MAX_RETRIES_PER_LEGO} retries; advancing', 'warn')
        return 'SKIP'

    def _rec_run_one_qs_cycle(self, lego, timeout_s=90):
        """Drive one QuickStart pick-and-drop cycle for the named lego.
        Returns 'PASS' / 'FAIL' / 'ABORT'.

        Per Q1 (Always PASS — never verify), success is the
        'pick-and-drop cycle complete' sentinel in get_log. No pose check.
        """
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        st = self._rec_state

        # 1) Set ik_target.
        params_cli = self.create_client(
            SetParameters, '~/set_parameters')
        params_cli.wait_for_service(timeout_sec=5)
        p = Parameter(name='ik_target',
                      value=ParameterValue(
                          type=ParameterType.PARAMETER_STRING,
                          string_value=lego))
        fut = params_cli.call_async(SetParameters.Request(parameters=[p]))
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10:
            time.sleep(0.05)

        # 2) qs_refresh_all → sleep → qs_select → sleep
        if st['stop_evt'].is_set():
            return 'ABORT'
        self._rec_call_trigger('qs_refresh_all', timeout=15)
        time.sleep(2.5)
        if st['stop_evt'].is_set():
            return 'ABORT'
        self._rec_call_trigger('qs_select', timeout=15)
        time.sleep(0.5)

        # 3) Snapshot pre-cycle log line count.
        log_resp = self._rec_call_trigger('get_log', timeout=10)
        if log_resp is None:
            return 'FAIL'
        pre_lines = len(log_resp.message.split('\\n'))

        # 4) qs_play.
        if st['stop_evt'].is_set():
            return 'ABORT'
        self._rec_call_trigger('qs_play', timeout=15)

        # 5) Poll get_log for terminal sentinel.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if st['stop_evt'].is_set():
                return 'ABORT'
            st['pause_evt'].wait()
            time.sleep(2.0)
            log_resp = self._rec_call_trigger('get_log', timeout=10)
            if log_resp is None:
                continue
            new = '\\n'.join(log_resp.message.split('\\n')[pre_lines:])
            if 'pick-and-drop cycle complete' in new:
                return 'PASS'
            if ('Quickstart halted' in new or
                    'Quickstart aborted' in new):
                return 'FAIL'
        return 'FAIL'

    # ==================================================================
    # End Record Sim
    # ==================================================================

    def _build_display_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='RViz')

        # --- Cups ---
        cup_frame = ttk.LabelFrame(frame, text='Cups')
        cup_frame.pack(fill=tk.X, padx=10, pady=5)

        self._show_visual_var = tk.BooleanVar(value=True)
        self._register_check(cup_frame, label='Visual (colored cups)',
                             tab='RViz', section='Cups',
                             variable=self._show_visual_var,
                             command=self._toggle_visual_markers).pack(anchor='w', padx=5, pady=2)

        pad_row = tk.Frame(cup_frame)
        pad_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(pad_row, text='Collision padding %:', anchor='w').pack(side=tk.LEFT)
        # 5% default — matches the global _CUP_COLLISION_PADDING=1.05.
        # Absorbs the multi-mm tracking-lag overshoot during fast pan
        # motions (post-drop grasp_home, etc). 1% wasn't enough; 5%
        # confirmed clean by user in repeated cycles.
        self._collision_padding_var = tk.IntVar(value=5)
        self._register_spinbox(pad_row, label='Collision padding %',
                               tab='RViz', section='Cups',
                               textvariable=self._collision_padding_var,
                               from_=0, to=50, increment=5, width=5).pack(side=tk.LEFT, padx=(5, 0))
        self._register_button(pad_row, text='Apply', tab='RViz', section='Cups',
                              command=self._cmd_apply_collision_padding).pack(side=tk.LEFT, padx=(5, 0))

        # --- Planning ---
        plan_frame = ttk.LabelFrame(frame, text='Planning')
        plan_frame.pack(fill=tk.X, padx=10, pady=5)

        grip_row = tk.Frame(plan_frame)
        grip_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_row, text='Drop grip angle (deg):', anchor='w').pack(side=tk.LEFT)
        self._drop_grip_angle_var = tk.IntVar(value=45)
        self._register_spinbox(grip_row, label='Drop grip angle (deg)',
                               tab='RViz', section='Planning',
                               textvariable=self._drop_grip_angle_var,
                               from_=0, to=90, increment=5, width=5).pack(side=tk.LEFT, padx=(5, 0))

        attempts_row = tk.Frame(plan_frame)
        attempts_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(attempts_row, text='Planning attempts:', anchor='w').pack(side=tk.LEFT)
        self._planning_attempts_var = tk.IntVar(value=50)
        self._register_spinbox(attempts_row, label='Planning attempts',
                               tab='RViz', section='Planning',
                               textvariable=self._planning_attempts_var,
                               from_=1, to=200, increment=10, width=5).pack(side=tk.LEFT, padx=(5, 0))

        # --- Info ---
        info_frame = ttk.LabelFrame(frame, text='Info')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(info_frame, text='Collision planning is always ON regardless of display',
                 anchor='w', fg='#555555').pack(anchor='w', padx=5, pady=1)
        tk.Label(info_frame, text='Changes to planning params take effect on next motion',
                 anchor='w', fg='#555555').pack(anchor='w', padx=5, pady=1)

    # ------------------------------------------------------------------
    # IK tab: tab-change, FK, spinbox IK, buttons
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rpy_deg(qx, qy, qz, qw):
        """Convert quaternion to Roll/shoulder_lift/Yaw in degrees."""
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @staticmethod
    def _rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
        """Convert Roll/shoulder_lift/Yaw in degrees to quaternion (x, y, z, w)."""
        r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        cr, cp, cy = math.cos(r / 2), math.cos(p / 2), math.cos(y / 2)
        sr, sp, sy = math.sin(r / 2), math.sin(p / 2), math.sin(y / 2)
        return (sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy)

    def _get_ik_target_quat(self):
        """Read XYZ + RPY from spinboxes, return (x, y, z, qx, qy, qz, qw)."""
        x = self.xyz_vars['X'].get()
        y = self.xyz_vars['Y'].get()
        z = self.xyz_vars['Z'].get()
        qx, qy, qz, qw = self._rpy_deg_to_quat(
            self.rpy_vars['Roll'].get(),
            self.rpy_vars['shoulder_lift'].get(),
            self.rpy_vars['Yaw'].get())
        return x, y, z, qx, qy, qz, qw

    def _on_tab_changed(self, event):
        """Auto-populate IK spinboxes from current EE pose when switching to IK tab."""
        tab_text = self._notebook.tab(self._notebook.select(), 'text')
        if tab_text != 'IK':
            return
        self._compute_fk_to_spinboxes()
        with self.joint_lock:
            self._ik_planned_target = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
        self._ik_valid = True

    def _compute_fk_to_spinboxes(self, joint_positions=None):
        """Compute FK and populate IK spinboxes. Uses current joints if None."""
        if not MOVEIT_AVAILABLE or not hasattr(self, 'fk_client'):
            return
        if not self.fk_client.service_is_ready():
            return

        if joint_positions is None:
            with self.joint_lock:
                positions = [self.joint_positions[n] for n in ARM_JOINT_NAMES]
        else:
            positions = [joint_positions[n] for n in ARM_JOINT_NAMES]

        def _call_fk():
            with self._ik_solve_lock:
                request = GetPositionFK.Request()
                request.header.frame_id = 'base'
                request.header.stamp = self.get_clock().now().to_msg()
                request.fk_link_names = ['tcp_link']
                request.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                request.robot_state.joint_state.position = list(positions)

                future = self.fk_client.call_async(request)
                self._wait_future(future, timeout_sec=2.0)

            if future.result() is None:
                return
            resp = future.result()
            if resp.error_code.val != 1 or not resp.pose_stamped:
                return

            p = resp.pose_stamped[0].pose.position
            o = resp.pose_stamped[0].pose.orientation
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._populate_ik_spinboxes,
                               p.x, p.y, p.z, o.x, o.y, o.z, o.w)

        threading.Thread(target=_call_fk, daemon=True).start()

    def _populate_ik_spinboxes(self, x, y, z, qx, qy, qz, qw):
        """Set IK spinbox values from FK result and mark state as valid."""
        self._ik_trace_active = False  # suppress IK solves during batch update
        self.xyz_vars['X'].set(round(x, 3))
        self.xyz_vars['Y'].set(round(y, 3))
        self.xyz_vars['Z'].set(round(z, 3))
        r, p, ya = self._quat_to_rpy_deg(qx, qy, qz, qw)
        self.rpy_vars['Roll'].set(round(r, 1))
        self.rpy_vars['shoulder_lift'].set(round(p, 1))
        self.rpy_vars['Yaw'].set(round(ya, 1))
        self._ik_trace_active = True
        # Store as last valid
        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()
        # Mark valid, clear red
        self._ik_valid = True
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

    def _cmd_ik_reset(self):
        """Reset IK tab: zero arm and populate spinboxes from resulting FK."""
        self._cmd_zero_arm()
        def _after_zero():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(500, _after_zero)

    # --- Spinbox IK: debounced, serialized ---

    def _on_ik_var_changed(self, field):
        """Any IK spinbox variable changed. Debounce then solve IK."""
        if not self._ik_trace_active:
            self.get_logger().debug(f'IK trace suppressed: {field}')
            return  # suppress during programmatic batch updates
        self.get_logger().info(f'IK var changed: {field}')
        if self._ik_debounce_id is not None:
            self.root.after_cancel(self._ik_debounce_id)
        self._ik_debounce_id = self.root.after(
            150, lambda: self._ik_solve_interactive(field))

    def _ik_solve_interactive(self, changed_field, revert_on_fail=False):
        """Compute IK from current spinbox values. Serialized via lock."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self.get_logger().warn('IK solve skipped: MoveIt not available')
            return
        if not self.ik_client.service_is_ready():
            self.get_logger().warn('IK solve skipped: /compute_ik not ready')
            return

        self._ik_debounce_id = None
        x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()
        self.get_logger().info(
            f'IK solve starting: {changed_field} -> ({x:.3f}, {y:.3f}, {z:.3f})')

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        seeds = [
            list(current_joints),
            [math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0,
             0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]

        self._ik_solve_gen += 1
        gen = self._ik_solve_gen

        def _make_ik_request(seed, avoid_collisions):
            request = GetPositionIK.Request()
            ik_req = PositionIKRequest()
            ik_req.group_name = 'arm'
            ik_req.avoid_collisions = avoid_collisions
            robot_state = RobotState()
            robot_state.joint_state.name = list(ARM_JOINT_NAMES)
            robot_state.joint_state.position = list(seed)
            ik_req.robot_state = robot_state
            pose = PoseStamped()
            pose.header.frame_id = 'base'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            ik_req.pose_stamped = pose
            ik_req.timeout.sec = 0
            ik_req.timeout.nanosec = 500_000_000  # 0.5s cap per request
            request.ik_request = ik_req
            return request

        def _extract_joints(result):
            sol = result.solution.joint_state
            target = {}
            for i, name in enumerate(sol.name):
                if name in ARM_JOINT_NAMES and i < len(sol.position):
                    target[name] = sol.position[i]
            return target

        def _solve():
            if not self._ik_solve_lock.acquire(blocking=False):
                self.get_logger().info('IK solve: lock busy, skipping')
                return  # another solve running, debounce will retry
            try:
                # Pass 1: collision-aware IK (valid + safe)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=True)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass1 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK SUCCESS (collision-free)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_success,
                                           target, gen)
                        return

                # Pass 2: collision-ignored IK (reachable but colliding)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=False)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass2 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK COLLISION (goal shown red)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_fail_with_goal,
                                           target, changed_field, gen)
                        return

                # Truly unreachable — no solution even ignoring collisions
                self.get_logger().info('IK UNREACHABLE (all seeds failed)')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._ik_interactive_fail,
                                   changed_field, revert_on_fail, gen)
            finally:
                self._ik_solve_lock.release()

        threading.Thread(target=_solve, daemon=True).start()

    def _ik_interactive_success(self, target, gen):
        """IK succeeded — update goal state, clear red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = True
        self._ik_planned_target = dict(target)

        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()

        # Clear all red
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

        # Update sliders + goal state
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

    def _ik_interactive_fail_with_goal(self, target, changed_field, gen):
        """IK collision — publish colliding goal (RViz shows red), mark field red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False
        self._ik_planned_target = None  # not executable

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')

        # Publish the colliding solution so goal robot shows RED in RViz
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

        self._append_log(
            f'IK collision — {changed_field} causes collision', 'warn')

    def _ik_interactive_fail(self, changed_field, revert_on_fail, gen):
        """IK truly unreachable — mark field red, log warning."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')
        self._append_log(
            f'IK unreachable — {changed_field} value out of workspace', 'warn')

    # --- IK buttons (always enabled, show warning if invalid) ---

    def _ik_btn_set_joints(self):
        """Send currently planned IK joints directly to controllers."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._cmd_set_joints()

    def _ik_btn_plan_execute(self):
        """Plan & Execute from currently planned IK joints via MoveIt."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._cmd_plan_execute()

    # --- IK services (for programmatic access) ---

    def _cmd_ik_set_joints(self):
        """Service: send IK joints to controllers."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._cmd_set_joints()
        else:
            self._compute_ik_full(mode='set_joints')

    def _cmd_ik_plan_execute(self):
        """Service: plan & execute IK joints."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._cmd_plan_execute()
        else:
            self._compute_ik_full(mode='plan_execute')

    def _cmd_ik_randomize(self):
        """Randomize arm goal state, compute FK, populate spinboxes."""
        self._cmd_randomize_arm()
        def _fk_after_randomize():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            joints_str = ', '.join(f'{n}: {pos[n]:.3f}' for n in ARM_JOINT_NAMES)
            self._append_log(f'Randomized goal: {joints_str}')
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(1500, _fk_after_randomize)

    # --- Full IK solver (for services, 6 seeds + 2 passes) ---

    # ------------------------------------------------------------------
    # IK pipeline — three entry points (Phase 7 Plan 07-04 / D-10..D-13):
    #
    #   Solvers (return candidate joint dicts):
    #     _solve_grasp_ik   — geometric, top-down grasp poses only (~100µs)
    #     _solve_free_ik    — MoveIt KDL multi-seed for freeform xyz+quaternion
    #
    #   Execution wrapper (solve + collision-check + execute):
    #     _plan_collision_free_execute — geometric + MoveIt path planning
    #                                    (used by drop sweep, planned grasp)
    #
    #   Post-solve helpers (shared by both solvers):
    #     _check_state_valid  — collision validity via MoveIt planning scene
    #     _ik_apply_and_act   — sync sliders, publish goal state, dispatch mode
    #
    # Motion entry points:
    #   _execute_trajectory — direct-send for trajectory-shaped moves
    #   _cmd_plan_execute   — MoveIt-planned collision-aware execution
    #   _cmd_set_joints     — 0.5s slider-quick-move (bypasses trajectory by
    #                         design — quick-move semantics differ)
    # ------------------------------------------------------------------

    def _solve_grasp_ik(self, x, y, z, grasp_yaw, mode=None):
        """Geometric top-down grasp IK + collision check (no MoveIt fallback).

        Fast analytical solver (~100µs). Validates each solution against the
        MoveIt planning scene. On success, dispatches to _ik_apply_and_act(mode).
        Runs on a background daemon thread (matches historic _try_geometric).
        """
        def _try_geometric():
            with self._ik_solve_lock:
                ground_z = (self._ground_z_var.get()
                            if hasattr(self, '_ground_z_var') else None)
                ok, reason = check_grasp_reachable(x, y, z, ground_z=ground_z)
                if not ok:
                    self._append_log(
                        f'Grasp rejected: {reason} '
                        f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                    return

                yaw_used, stage_sols, dbg_lines = find_reachable_grasp_yaw(
                    [('grasp', x, y, z)], grasp_yaw)
                if yaw_used is None:
                    self._append_log(
                        f'Grasp unreachable: no geometric IK at any yaw '
                        f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                    for line in dbg_lines:
                        self._append_log(line, 'warn')
                    return
                solutions = stage_sols['grasp']
                if yaw_used != grasp_yaw:
                    self._append_log(
                        f'Grasp yaw fallback: '
                        f'{math.degrees(grasp_yaw):.1f}° → '
                        f'{math.degrees(yaw_used):.1f}° '
                        f'(Δ{math.degrees(yaw_used - grasp_yaw):+.1f}°)')

                for i, sol in enumerate(solutions):
                    config = 'elbow-up' if i == 0 else 'elbow-down'
                    if self._check_state_valid(sol):
                        self._append_log(
                            f'Geometric IK: {config}, '
                            f'wrist_roll='
                            f'{math.degrees(sol["wrist_roll"]):.1f}°')
                        self._ik_apply_and_act(sol, mode)
                        return
                    self._append_log(
                        f'Geometric IK: {config} collides', 'warn')

                self._append_log(
                    f'Grasp unreachable: all geometric IK solutions collide '
                    f'at yaw {math.degrees(yaw_used):.1f}° '
                    f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

        threading.Thread(target=_try_geometric, daemon=True).start()

    def _compute_ik_full(self, mode=None, target_pose=None, grasp_yaw=None):
        """IK routing layer: geometric for grasps, MoveIt KDL for freeform.

        See _solve_grasp_ik (grasp_yaw path) and _solve_free_ik (freeform
        xyz+quaternion path). Both solvers call _ik_apply_and_act on success
        to sync sliders and dispatch the execution mode.

        target_pose: optional (x, y, z, qx, qy, qz, qw) to bypass spinboxes.
        grasp_yaw: desired jaw-line direction (rad) for top-down alignment.
        """
        if target_pose is not None:
            x, y, z, qx, qy, qz, qw = target_pose
        else:
            x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()

        if grasp_yaw is not None:
            self._solve_grasp_ik(x, y, z, grasp_yaw, mode=mode)
            return

        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return
        self._solve_free_ik(x, y, z, qx, qy, qz, qw, mode, grasp_yaw)

    def _solve_free_ik(self, x, y, z, qx, qy, qz, qw, mode, grasp_yaw):
        """MoveIt multi-seed IK solver — used for non-grasp moves only."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        bearing = math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0

        seeds = [
            list(current_joints),
            [bearing, 0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]
        for _ in range(3):
            seeds.append([random.uniform(*JOINT_LIMITS[n]) for n in ARM_JOINT_NAMES])

        def _try_seeds():
            with self._ik_solve_lock:
                for avoid_col in [True, False]:
                    for seed in seeds:
                        request = GetPositionIK.Request()
                        ik_req = PositionIKRequest()
                        ik_req.group_name = 'arm'
                        ik_req.avoid_collisions = avoid_col

                        robot_state = RobotState()
                        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                        robot_state.joint_state.position = list(seed)
                        ik_req.robot_state = robot_state

                        pose = PoseStamped()
                        pose.header.frame_id = 'base'
                        pose.header.stamp = self.get_clock().now().to_msg()
                        pose.pose.position.x = x
                        pose.pose.position.y = y
                        pose.pose.position.z = z
                        pose.pose.orientation.x = qx
                        pose.pose.orientation.y = qy
                        pose.pose.orientation.z = qz
                        pose.pose.orientation.w = qw
                        ik_req.pose_stamped = pose
                        ik_req.timeout.sec = 0
                        ik_req.timeout.nanosec = 500_000_000  # 0.5s cap
                        request.ik_request = ik_req

                        future = self.ik_client.call_async(request)
                        self._wait_future(future, timeout_sec=2.0)

                        if future.result() is None:
                            continue
                        resp = future.result()
                        if resp.error_code.val == 1:
                            sol = resp.solution.joint_state
                            target = {}
                            for i, name in enumerate(sol.name):
                                if name in ARM_JOINT_NAMES \
                                        and i < len(sol.position):
                                    target[name] = sol.position[i]
                            self._ik_apply_and_act(target, mode)
                            return

            self._append_log(
                f'IK failed for ({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

        threading.Thread(target=_try_seeds, daemon=True).start()

    def _check_state_valid(self, target):
        """Check if a joint state is collision-free via MoveIt's planning scene."""
        valid, _ = self._check_state_valid_with_contacts(target)
        return valid

    def _check_state_valid_with_contacts(self, target):
        """Like _check_state_valid, but also returns contacts on invalid states.
        Returns (valid: bool, contacts: list). Empty list on valid / timeout /
        no-checker-available. Used by tier1 segment check so failures report
        which links collided.

        Phase 9: gripper_joint is included in the state using the live
        actual position. Without it, MoveIt substitutes the default (0 rad
        = 19mm baseline jaw gap), which doesn't match reality when the
        gripper is carrying a block (~0.11 rad = 27mm gap) or fully open
        (~1.0 rad = 94mm gap). The mismatch caused OMPL to reject plans
        our tier1 check accepted — they disagreed on jaw-world collisions.
        """
        if not MOVEIT_AVAILABLE or not hasattr(self, 'validity_client') \
                or not self.validity_client.service_is_ready():
            return True, []
        req = GetStateValidity.Request()
        # Include gripper_joint using live position so jaws have correct
        # spread in the collision check. Fall back to the target's own
        # gripper value if the caller supplied one, then live position,
        # then 0 as a last resort.
        with self.joint_lock:
            gj_live = self._actual_positions.get(
                GRIPPER_JOINT_NAME,
                self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0))
        gj = target.get(GRIPPER_JOINT_NAME, gj_live)
        req.robot_state.joint_state.name = list(ALL_JOINT_NAMES)
        req.robot_state.joint_state.position = [
            target.get(n, 0.0) for n in ARM_JOINT_NAMES] + [float(gj)]
        # is_diff=True: merge this joint_state onto the scene's stored
        # robot_state rather than replacing it. Without this, default-
        # constructed robot_state fields (including attached_collision_objects)
        # clobber the scene's attachments for the duration of this probe,
        # making any AttachedCollisionObject installed via apply_planning_scene
        # silently invisible — this was the real cause of Phase 9's
        # "300mm cube invisible to /check_state_validity" observation and
        # of the drop_sweep passing the held block through cup meshes.
        req.robot_state.is_diff = True
        req.group_name = 'arm'
        future = self.validity_client.call_async(req)
        self._wait_future(future, timeout_sec=1.0)
        res = future.result()
        if res is None:
            return True, []
        return bool(res.valid), list(res.contacts)

    def _ik_apply_and_act(self, target, mode):
        """Apply IK solution to sliders/goal state and optionally execute."""
        def _apply():
            self._slider_driven = True
            self._select_planning_group('arm')
            with self.joint_lock:
                for name in ARM_JOINT_NAMES:
                    if name in target:
                        self.joint_positions[name] = target[name]
            for name in ARM_JOINT_NAMES:
                if name in target and name in self.sliders:
                    self.sliders[name].set(target[name])
                    self.slider_labels[name].config(text=f'{target[name]:.3f}')
            self._publish_goal_state()
            self._ik_planned_target = dict(target)
            self._ik_valid = True
            if mode == 'set_joints':
                self._cmd_set_joints()
            elif mode == 'plan_execute':
                self._cmd_plan_execute()
            elif mode == 'grasp_approach':
                # Route through the unified motion primitive so the approach
                # path is collision-checked against the scene (catches fingers
                # clipping ground/cups on low/edge blocks).
                duration = getattr(self, '_grasp_arm_duration', 2.0)
                final_joints = getattr(self, '_grasp_final_joints', None)
                approach_done = threading.Event()

                def _wait_then_descend():
                    approach_done.wait(timeout=60.0)
                    if final_joints is not None:
                        self._append_log('Approach complete, descending to grasp')
                        self._ik_apply_and_act(final_joints, 'grasp_execute')
                    else:
                        evt = getattr(self, '_grasp_motion_event', None)
                        if evt:
                            evt.set()

                threading.Thread(target=_wait_then_descend, daemon=True).start()
                self._joint_space_collision_free_execute(
                    target, on_complete_event=approach_done,
                    duration_s=duration)
            elif mode == 'grasp_execute':
                duration = getattr(self, '_grasp_arm_duration', 2.0)
                evt = getattr(self, '_grasp_motion_event', None) \
                    or threading.Event()
                self._joint_space_collision_free_execute(
                    target, on_complete_event=evt, duration_s=duration)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _apply)

    def _set_ik_status(self, text):
        self.ik_status_var.set(text)
        self._append_log(text)

    def _ee_pose_callback(self, msg):
        if getattr(self, '_gui_ready', False):
            p = msg.pose.position
            o = msg.pose.orientation
            self.root.after(0, self._update_ee_display,
                           p.x, p.y, p.z, o.x, o.y, o.z, o.w)

    def _update_ee_display(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.ee_labels['X'].set(f'{x:.4f}')
        self.ee_labels['Y'].set(f'{y:.4f}')
        self.ee_labels['Z'].set(f'{z:.4f}')
        self.ee_labels['qx'].set(f'{qx:.4f}')
        self.ee_labels['qy'].set(f'{qy:.4f}')
        self.ee_labels['qz'].set(f'{qz:.4f}')
        self.ee_labels['qw'].set(f'{qw:.4f}')

    def _objects_callback(self, msg):
        new_any_big_move = False
        with self.objects_lock:
            for tf in msg.transforms:
                name = tf.child_frame_id
                prior = self.objects_data.get(name)
                self.objects_data[name] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }
                # Detect meaningful pose change for scene-sync scheduling
                if prior is not None:
                    dx = self.objects_data[name]['x'] - prior['x']
                    dy = self.objects_data[name]['y'] - prior['y']
                    dz = self.objects_data[name]['z'] - prior['z']
                    if (dx*dx + dy*dy + dz*dz) ** 0.5 > 0.003:  # >3mm
                        new_any_big_move = True
        # Signal any pending grasp_refresh that fresh data has arrived.
        refresh_evt = getattr(self, '_objects_refresh_event', None)
        if refresh_evt is not None and not refresh_evt.is_set():
            refresh_evt.set()

        # Phase 9: rate-limited live sync — if any block moved >3mm since last
        # seen, schedule a lego-scene resync (but at most once per 0.5s).
        # Never resync while a lego is attached (its world entry is stale by design).
        # Phase 11-01 followup: also skip resync while a Quickstart/Real run
        # OR Drop Scan is active. During those flows the cached/frozen poses
        # are the authoritative source — letting live PnP noise (especially
        # angular-dependent during a pan sweep) re-add the legos at drifting
        # base-frame coordinates makes the collision meshes appear to move
        # with shoulder_pan even though they're physically stationary.
        if new_any_big_move:
            self._ensure_lego_state()
            if self._attached_lego_name:
                return
            if getattr(self, '_qs_state', 'idle') == 'running':
                return  # frozen during run/scan — re-inject path owns the scene
            now = self.get_clock().now().nanoseconds / 1e9
            last = getattr(self, '_lego_sync_last_t', 0.0)
            if now - last > 0.5:
                self._lego_sync_last_t = now
                # Mode B suspect: lego re-add mid-cycle can invalidate an
                # OMPL plan made against the pre-resync scene.
                tracer.event('objects_callback_readd_scheduled',
                             attached=self._attached_lego_name)
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._add_lego_collision_objects)

    def _bbox_callback(self, msg):
        """Cache world-aligned bounding boxes from /objects_bbox_sim."""
        import json
        try:
            self.objects_bbox = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _lookup_bbox(self, name):
        """Resolve a bbox dict for an object name with multi-tier fallback.

        Three lookup tiers, tried in order:
          1. Exact match (e.g. 'red_lego_2x4' from sim ground truth, or
             'red' from a color-only catalog entry).
          2. Color-only key (e.g. 'red_0' → 'red').
          3. Any key starting with 'color_' (e.g. 'red_0' → 'red_2x4').
             This handles the case where the catalog feeding our subscription
             only has size-specific entries (no explicit color entries) —
             the per-color size convention means any matching entry has the
             right dims.

        Two bbox topics exist (/objects_bbox_sim from Isaac Sim, /objects_bbox_real
        from the YOLOE-side catalog) with different key sets. This function
        is agnostic to which is feeding the subscription. Returns None only
        when no key matches the color prefix at all.
        """
        if not name:
            return None
        # Tier 1: exact match.
        bbox = self.objects_bbox.get(name)
        if bbox:
            return bbox
        color = name.split('_', 1)[0].lower()
        # Tier 2: color-only key (catalog has explicit 'red'/'green'/'blue').
        bbox = self.objects_bbox.get(color)
        if bbox:
            return bbox
        # Tier 3: any key starting with '<color>_' (size-specific entries).
        prefix = f'{color}_'
        for cat_name, cat_bbox in self.objects_bbox.items():
            if cat_name.lower().startswith(prefix):
                return cat_bbox
        return None

    def _build_lego_geometry(self, name):
        """Resolve collision geometry for a lego by name.

        Real-mode (YOLOE) detections have no size suffix because vision can
        only recover color, not size. So this helper does NOT infer size
        from color (that would bake a sim-only assumption into real code).
        Instead:

          1. If name has a size suffix (sim convention 'red_2x3'), try to
             load the per-size STL mesh — most accurate geometry.
          2. Otherwise (or if mesh load fails), fall back to a SolidPrimitive
             box with dimensions read from the bbox catalog via _lookup_bbox.
             The catalog is the user-owned config — anything that needs
             dims should consult it, not infer from naming patterns.

        Returns ('mesh', shape_msgs.Mesh) or ('box', SolidPrimitive) or
        (None, None) when no entry matches.
        """
        size = _lego_size_from_name(name)
        if size is not None:
            lego_mesh, mesh_ok = _load_lego_mesh(size)
            if mesh_ok:
                return ('mesh', lego_mesh)
        # Fallback: SolidPrimitive box from bbox catalog dims.
        bbox = self._lookup_bbox(name)
        if bbox is None:
            return (None, None)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [
            float(bbox.get('sx', 0.020)),
            float(bbox.get('sy', 0.016)),
            float(bbox.get('sz', 0.011)),
        ]
        return ('box', box)

    # ------------------------------------------------------------------
    # Drop target subscription + helpers
    # ------------------------------------------------------------------

    def _drop_callback(self, msg):
        """Populate _drop_data from /drop_poses TFMessage.

        Cache-only. _cmd_drop_refresh republishes visuals and collisions
        on a fixed time delay (matching _cmd_grasp_refresh); no event
        signaling happens here.
        """
        with self._drop_lock:
            for tf in msg.transforms:
                name = tf.child_frame_id
                self._drop_data[name] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }

    def _update_drop_topic(self, topic='/drop_poses'):
        """Switch drop subscription to topic."""
        if self._drop_sub is not None:
            self.destroy_subscription(self._drop_sub)
        with self._drop_lock:
            self._drop_data.clear()
        self._remove_cup_collision_objects()
        self._drop_sub = self.create_subscription(
            TFMessage, topic, self._drop_callback, 10,
            callback_group=self._sub_cb_group)
        self._append_log(f'Drop topic: {topic}')

    def _populate_drop_list(self):
        """Refresh drop_listbox from _drop_data (call on GUI thread)."""
        if not hasattr(self, '_drop_listbox'):
            return
        self._drop_listbox.delete(0, tk.END)
        with self._drop_lock:
            for name, pos in self._drop_data.items():
                label = DROP_ID_LABELS.get(name, "")
                label_str = f" [{label}]" if label else ""
                self._drop_listbox.insert(
                    tk.END,
                    f'{name}{label_str}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')
        count = self._drop_listbox.size()
        if count > 0:
            self._append_log(f'Drop targets refreshed: {count} found')

    def _acm_allow_ground_rest(self, obj_ids):
        """Update the ACM so each object id in obj_ids is allowed to touch
        `ground_plane`. Physically legitimate: legos and cups REST on the
        table surface (which is what `ground_plane` actually models — its
        top face is at z=0, coincident with lego/cup bottom faces). Without
        these ACM entries, every resting object is flagged as colliding
        with ground_plane at wp[0] of every plan, and attached legos
        inherit the same contact at grasp time.

        Mirrors the one-off `base ↔ ground_plane` allowance that
        `_cmd_toggle_ground_plane` installs (control_gui.py ~6997-7000) —
        this is the symmetric entry for everything else that legitimately
        rests on the surface.

        Idempotent: adding an id already allowed is a no-op.
        """
        if not MOVEIT_AVAILABLE or not hasattr(self, '_get_scene_client'):
            return
        if not obj_ids:
            return
        if not self._get_scene_client.wait_for_service(timeout_sec=3.0):
            return
        if not self._apply_scene_client.wait_for_service(timeout_sec=3.0):
            return

        get_req = GetPlanningSceneSrv.Request()
        get_req.components.components = 128  # ALLOWED_COLLISION_MATRIX
        future = self._get_scene_client.call_async(get_req)
        self._wait_future(future, timeout_sec=3.0)
        if future.result() is None:
            return
        acm = future.result().scene.allowed_collision_matrix

        if 'ground_plane' not in acm.entry_names:
            # ground_plane row doesn't exist yet — _cmd_toggle_ground_plane
            # installs it on startup; if it hasn't run we can't wire anything
            # up yet. Caller will get another chance on the next add.
            return
        gp_idx = acm.entry_names.index('ground_plane')

        changed = False
        for obj_id in obj_ids:
            if obj_id in acm.entry_names:
                idx = acm.entry_names.index(obj_id)
                if not acm.entry_values[idx].enabled[gp_idx]:
                    acm.entry_values[idx].enabled[gp_idx] = True
                    changed = True
                if not acm.entry_values[gp_idx].enabled[idx]:
                    acm.entry_values[gp_idx].enabled[idx] = True
                    changed = True
            else:
                # Extend every existing row with one more column for obj_id,
                # defaulting to disallow contact.
                for entry in acm.entry_values:
                    entry.enabled.append(False)
                # Flip the ground_plane row's new column to allow contact.
                acm.entry_values[gp_idx].enabled[-1] = True
                # New row for obj_id: disallow everything except ground_plane
                # (allowed) and self (harmless, conventionally true).
                row = AllowedCollisionEntry()
                row.enabled = [False] * len(acm.entry_names)
                row.enabled[gp_idx] = True
                row.enabled.append(True)  # self-entry
                acm.entry_names.append(obj_id)
                acm.entry_values.append(row)
                changed = True

        if not changed:
            return

        acm_scene = PlanningSceneMsg()
        acm_scene.is_diff = True
        acm_scene.allowed_collision_matrix = acm
        req = ApplyPlanningScene.Request()
        req.scene = acm_scene
        fut = self._apply_scene_client.call_async(req)
        self._wait_future(fut, timeout_sec=5.0)

    def _add_cup_collision_objects(self, cups_dict=None):
        """Add cup collision objects to MoveIt planning scene.

        Reads cup poses from _drop_data by default. If cups_dict is provided
        (e.g. real-mode cached cups from Refresh Cups Pose), uses that
        snapshot instead — bypasses the live /drop_poses subscription so
        the planning scene reflects the user-frozen scan, not whatever the
        live topic is currently publishing.

        Uses the CAD cup mesh (via trimesh) for accurate collision geometry.
        Falls back to SolidPrimitive.CYLINDER if mesh loading fails.
        """
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        # Cylinder fallback dimensions (mm → m, with padding)
        CUP_RADIUS = 0.039 * _CUP_COLLISION_PADDING
        CUP_HEIGHT = 0.0965 * _CUP_COLLISION_PADDING

        cup_mesh, mesh_ok = _load_cup_mesh()

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log('apply_planning_scene not available for cups', 'warn')
                return
            scene = PlanningSceneMsg()
            scene.is_diff = True
            if cups_dict is not None:
                drop_items = dict(cups_dict)
            else:
                with self._drop_lock:
                    drop_items = dict(self._drop_data)
            for name, pos in drop_items.items():
                co = CollisionObject()
                co.header.frame_id = 'base'
                co.id = f'cup_{name}'
                co.operation = CollisionObject.ADD
                p = Pose()
                p.position.x = pos['x']
                p.position.y = pos['y']
                # Phase 9: honor cup orientation from /drop_poses. Previously
                # forced identity quaternion, which meant a rotated cup in
                # Isaac showed up axis-aligned in MoveIt — collision checks
                # missed the asymmetric flare of the rim and the drop IK
                # offset computed against a wrong orientation.
                p.orientation.x = float(pos.get('qx', 0.0))
                p.orientation.y = float(pos.get('qy', 0.0))
                p.orientation.z = float(pos.get('qz', 0.0))
                p.orientation.w = float(pos.get('qw', 1.0))
                # /drop_poses publishes cup BODY-CENTER (z = base + half_height).
                if mesh_ok:
                    # Cup STL mesh origin is at the cup's BASE. To place its
                    # base on the ground, set mesh_pose.z = cup_base_z =
                    # cup_center_z − half_height.
                    p.position.z = pos['z'] - CUP_BODY_HEIGHT_M / 2.0
                    co.meshes.append(cup_mesh)
                    co.mesh_poses.append(p)
                else:
                    # SolidPrimitive.CYLINDER is center-based (same convention
                    # as /drop_poses now), so no Z adjustment.
                    p.position.z = pos['z']
                    cyl = SolidPrimitive()
                    cyl.type = SolidPrimitive.CYLINDER
                    cyl.dimensions = [CUP_HEIGHT, CUP_RADIUS]
                    co.primitives.append(cyl)
                    co.primitive_poses.append(p)
                scene.world.collision_objects.append(co)
            if not scene.world.collision_objects:
                return
            self._cup_collision_names = [co.id for co in scene.world.collision_objects]
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            if future.result() is not None and future.result().success:
                label = 'mesh' if mesh_ok else 'cylinder'
                self._append_log(f'Added {len(drop_items)} cup collision objects ({label})')
                # Cups physically rest on the table (= ground_plane top at
                # z=0). Allow the rest-contact in the ACM so post-checks
                # don't flag every wp[0] as invalid.
                self._acm_allow_ground_rest(self._cup_collision_names)
            else:
                self._append_log('Failed to add cup collision objects', 'warn')

        threading.Thread(target=_apply, daemon=True).start()

    def _remove_cup_collision_objects(self):
        """Remove all cup collision objects from planning scene."""
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                return
            scene = PlanningSceneMsg()
            scene.is_diff = True
            for name in list(getattr(self, '_cup_collision_names', [])):
                co = CollisionObject()
                co.header.frame_id = 'base'
                co.id = name
                co.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(co)
            if scene.world.collision_objects:
                req = ApplyPlanningScene.Request()
                req.scene = scene
                future = self._apply_scene_client.call_async(req)
                self._wait_future(future, timeout_sec=5.0)
                self._cup_collision_names.clear()

        threading.Thread(target=_apply, daemon=True).start()

    # ------------------------------------------------------------------
    # Phase 9: lego block collision tracking in MoveIt planning scene
    # ------------------------------------------------------------------
    # Blocks enter the scene at grasp_refresh as world CollisionObjects.
    # On successful gripper_close_for_object they convert to AttachedCollisionObject
    # on tcp_link — MoveIt then tracks the carried block's world pose via FK, so
    # plans during drop_sweep respect the full carried-block envelope.
    # On drop_release the attached object is removed from the scene entirely.

    @staticmethod
    def _pose7_to_mat(x, y, z, qx, qy, qz, qw):
        """Build 4×4 homogeneous transform from translation + unit quaternion."""
        import numpy as np
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        m = np.eye(4)
        m[0, 0] = 1 - 2*(yy + zz); m[0, 1] = 2*(xy - wz); m[0, 2] = 2*(xz + wy)
        m[1, 0] = 2*(xy + wz); m[1, 1] = 1 - 2*(xx + zz); m[1, 2] = 2*(yz - wx)
        m[2, 0] = 2*(xz - wy); m[2, 1] = 2*(yz + wx); m[2, 2] = 1 - 2*(xx + yy)
        m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
        return m

    @staticmethod
    def _mat_to_pose7(m):
        """Extract translation + unit quaternion from 4×4 homogeneous transform."""
        x, y, z = float(m[0, 3]), float(m[1, 3]), float(m[2, 3])
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = (tr + 1.0) ** 0.5 * 2
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = ((1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = ((1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = ((1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return x, y, z, qx, qy, qz, qw

    def _ensure_lego_state(self):
        """Lazy-init Phase 9 state so hot-reload (which skips __init__) still works."""
        if not hasattr(self, '_lego_collision_names'):
            self._lego_collision_names = []
        if not hasattr(self, '_attached_lego_name'):
            self._attached_lego_name = None

    def _add_lego_collision_objects(self, legos_dict=None):
        """Add mesh CollisionObjects for every detected block to the planning scene.

        Reads block poses from objects_data by default. If legos_dict is
        provided (e.g. real-mode cached legos from Refresh Legos Pose),
        uses that snapshot instead — bypasses the live /objects_poses_real
        subscription so the planning scene reflects the user-frozen scan,
        not whatever the live topic is currently publishing (only the
        legos in YOLOE's current FOV).

        Keyed `lego_{name}`. Geometry from the per-size lego STL
        (`_load_lego_mesh`). Mirrors _add_cup_collision_objects.
        The STL origin was preserved from the Isaac Sim USD prim origin so
        no Z shift is applied to the published pose — unlike cups, which
        publish body-center and need mesh_pose.z = pos['z'] − H/2.
        """
        self._ensure_lego_state()
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log('apply_planning_scene not available for legos', 'warn')
                return
            if legos_dict is not None:
                blocks = dict(legos_dict)
            else:
                with self.objects_lock:
                    blocks = dict(self.objects_data)
            if not blocks:
                return
            # Snapshot cup positions so we can filter legos that have landed
            # inside one. An in-cup lego is already covered by the cup's own
            # collision object; adding its box as a separate world collision
            # causes false start-state collisions for grasp_home after a
            # successful drop (gripper parked above cup ↔ lego-in-cup).
            with self._drop_lock:
                cups = dict(self._drop_data)
            # Cup rim dimensions from the loader (no padding — we want actual
            # geometry for the "inside cup?" test).
            CUP_R = 0.039
            CUP_H = 0.0965
            def _inside_any_cup(p):
                """Returns True if block's XY is inside any cup's footprint
                AND block_z is at-or-above cup base. No upper Z bound — a
                block perched ABOVE the cup rim (e.g. balanced on a convex-
                hull cup top during drop) is still 'cup territory' as far
                as collision is concerned. The cup's own collision object
                + the gripper's own padding handle that region; adding a
                separate lego box on top of the cup causes phantom start-
                state collisions for the post-drop grasp_home."""
                lx, ly, lz = float(p.get('x', 0)), float(p.get('y', 0)), float(p.get('z', 0))
                # /drop_poses is cup BODY-CENTER; "at-or-above cup base" is
                # therefore lz >= (cup_center_z − half_height) − tol.
                for cpos in cups.values():
                    cx, cy, cz = float(cpos['x']), float(cpos['y']), float(cpos['z'])
                    dx, dy = lx - cx, ly - cy
                    cup_base_z = cz - CUP_BODY_HEIGHT_M / 2.0
                    if (dx*dx + dy*dy) ** 0.5 <= CUP_R + 0.005:
                        if lz >= cup_base_z - 0.005:
                            return True
                return False
            add_scene = PlanningSceneMsg(); add_scene.is_diff = True
            remove_scene = PlanningSceneMsg(); remove_scene.is_diff = True
            added = []
            geom_kinds = {'mesh': 0, 'box': 0}
            skipped_in_cup = []
            # Idempotent-REMOVE pattern: for every lego we DON'T want in
            # the world scene (attached + in-cup), issue an unconditional
            # REMOVE. Pre-Phase 9 we relied on self._lego_collision_names
            # bookkeeping, which diverged from MoveIt's scene across
            # attach/detach cycles and left phantom lego_blue_2x4 as a world
            # object after a drop. The always-REMOVE approach is correct;
            # the Phase-9-era belief that MoveIt Humble "tolerates REMOVE of
            # non-existent objects" is only half true: the scene is updated
            # correctly (missing stays missing), but the service still
            # returns success=False whenever any REMOVE targets a
            # non-existent id. Bundling ADDs and REMOVEs in one diff
            # therefore poisons the ADD-success signal. Split them: send
            # the REMOVEs in their own diff and ignore its success flag,
            # then send the ADDs separately and trust that flag.
            def _remove_diff(lego_id):
                rm = CollisionObject()
                rm.header.frame_id = 'base'
                rm.id = lego_id
                rm.operation = CollisionObject.REMOVE
                remove_scene.world.collision_objects.append(rm)
            for name, pose in blocks.items():
                lego_id = f'lego_{name}'
                # Skip the currently-held one — it's tracked by the held-lego
                # world-pose syncer at 10Hz. Adding an outdated pose here
                # from Isaac Sim's /objects_poses_sim (which shows the block
                # at its original table position, not in the gripper) would
                # clobber the live TCP-tracked pose.
                if name == self._attached_lego_name:
                    continue  # DO NOT remove — we own this id now
                # Skip legos that have landed inside a cup. The cup collision
                # object already occupies that space; adding the lego on top
                # causes spurious gripper-vs-lego false positives for any
                # motion starting or ending above the cup rim.
                if _inside_any_cup(pose):
                    skipped_in_cup.append(name)
                    _remove_diff(lego_id)  # unconditional REMOVE
                    continue
                kind, geom = self._build_lego_geometry(name)
                if kind is None:
                    continue
                co = CollisionObject()
                co.header.frame_id = 'base'
                co.id = lego_id
                co.operation = CollisionObject.ADD
                p = Pose()
                p.position.x = float(pose['x'])
                p.position.y = float(pose['y'])
                p.position.z = float(pose['z'])
                p.orientation.x = float(pose.get('qx', 0.0))
                p.orientation.y = float(pose.get('qy', 0.0))
                p.orientation.z = float(pose.get('qz', 0.0))
                p.orientation.w = float(pose.get('qw', 1.0))
                if kind == 'mesh':
                    co.meshes.append(geom)
                    co.mesh_poses.append(p)
                    geom_kinds['mesh'] += 1
                else:  # 'box'
                    co.primitives.append(geom)
                    co.primitive_poses.append(p)
                    geom_kinds['box'] += 1
                add_scene.world.collision_objects.append(co)
                added.append(co.id)
            if not add_scene.world.collision_objects and not remove_scene.world.collision_objects:
                return
            # Pass 1: REMOVEs (success flag discarded — REMOVE-missing-id
            # returns success=False in Humble but the scene state is still
            # what we want).
            if remove_scene.world.collision_objects:
                rm_req = ApplyPlanningScene.Request()
                rm_req.scene = remove_scene
                rm_fut = self._apply_scene_client.call_async(rm_req)
                self._wait_future(rm_fut, timeout_sec=5.0)
            # Pass 2: ADDs (success flag trusted).
            if not add_scene.world.collision_objects:
                if skipped_in_cup:
                    self._append_log(
                        f'All legos in-cup ({len(skipped_in_cup)}) — '
                        f'none added to scene: {skipped_in_cup}')
                return
            req = ApplyPlanningScene.Request()
            req.scene = add_scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            res = future.result()
            if res is not None and res.success:
                self._lego_collision_names = added
                note = f' (skipped {len(skipped_in_cup)} in-cup: {skipped_in_cup})' if skipped_in_cup else ''
                # Geometry-kind tally: mesh = sim-style names with size suffix
                # (precise STL geometry); box = YOLOE-style names where size
                # is unknown (bbox catalog dims, primitive box).
                kind_summary = (
                    f"{geom_kinds['mesh']} mesh + {geom_kinds['box']} box"
                    if geom_kinds['mesh'] and geom_kinds['box']
                    else 'mesh' if geom_kinds['mesh']
                    else 'box')
                self._append_log(
                    f'Added {len(added)} lego collision objects '
                    f'({kind_summary}){note}')
                # Legos physically rest on the table (= ground_plane top at
                # z=0). Allow the rest-contact in the ACM so post-checks
                # don't flag every wp[0] as invalid.
                self._acm_allow_ground_rest(added)
            else:
                reason = 'timeout' if res is None else 'server success=False'
                self._append_log(
                    f'Failed to add lego collision objects: {reason} '
                    f'[ADD={len(added)}]', 'warn')

        threading.Thread(target=_apply, daemon=True).start()

    def _remove_single_lego_from_world(self, obj_name):
        """Remove one lego's world CollisionObject (does not touch attached).
        Called at grasp_move entry so gripper can approach without MoveIt
        flagging the target-being-grasped as an obstacle.
        """
        self._ensure_lego_state()
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return
        name_id = f'lego_{obj_name}'

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                return
            scene = PlanningSceneMsg()
            scene.is_diff = True
            co = CollisionObject()
            co.header.frame_id = 'base'
            co.id = name_id
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=3.0)
            if name_id in self._lego_collision_names:
                self._lego_collision_names.remove(name_id)

        threading.Thread(target=_apply, daemon=True).start()

    def _remove_lego_collision_objects(self):
        """Remove all lego_* collision objects from the world (not attached).

        Queries the live planning scene for any `lego_*`-prefixed ids and
        unions with `_lego_collision_names` before removing — defends
        against orphan legos from earlier add cycles whose tracker was
        overwritten by a subsequent add (the tracker is replaced wholesale
        per add, so prior ids leak if not removed first).

        Phase 11-01 followup: discovered when Real Test → Clear all caches
        left 8 sim-named lego collisions in the planning scene because
        they'd been added before the topic-switch fix landed and the
        tracker was overwritten by the post-fix add of 5 real-named legos.
        """
        self._ensure_lego_state()
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                return
            # Discover orphan lego_* ids in the live scene.
            scene_lego_ids = set()
            if hasattr(self, '_get_scene_client') and \
                    self._get_scene_client.wait_for_service(timeout_sec=2.0):
                get_req = GetPlanningSceneSrv.Request()
                get_req.components.components = 8  # WORLD_OBJECT_NAMES
                fut = self._get_scene_client.call_async(get_req)
                self._wait_future(fut, timeout_sec=3.0)
                if fut.result() is not None:
                    for co in fut.result().scene.world.collision_objects:
                        if co.id.startswith('lego_'):
                            scene_lego_ids.add(co.id)
            # Union of tracker + live discovery — remove everything.
            tracked_ids = set(self._lego_collision_names)
            all_ids = tracked_ids | scene_lego_ids
            if not all_ids:
                return
            scene = PlanningSceneMsg()
            scene.is_diff = True
            for name in sorted(all_ids):
                co = CollisionObject()
                co.header.frame_id = 'base'
                co.id = name
                co.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(co)
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            self._lego_collision_names.clear()
            # Log only if we caught orphans — silent in the normal case.
            orphans = scene_lego_ids - tracked_ids
            if orphans:
                self._append_log(
                    f'Removed {len(all_ids)} lego collision objects '
                    f'(incl. {len(orphans)} orphan(s) not in tracker: '
                    f'{sorted(orphans)})')

        threading.Thread(target=_apply, daemon=True).start()

    def _wait_arm_at_rest(self, timeout_sec=1.5, vel_threshold_rad_s=0.02,
                           dwell_sec=0.15):
        """Block until all arm joint velocities are below threshold for
        dwell_sec continuous time. Returns True if settled, False on timeout.

        Purpose: synchronize the TF and /objects_poses_sim streams before
        reading them in _attach_lego_to_gripper / _refresh_attached_pose.
        When the arm is moving, Layer-B lag (documented in
        isaac-sim-mcp/docs/DEBUG-GUIDE.md § 4) makes /joint_states-derived
        TF disagree with Isaac physics' object poses — baking a 10-50 mm
        offset into the attached lego's mesh_poses[0] that then looks like
        the block hovering offset from the gripper in RViz. Waiting for
        zero-velocity eliminates the mismatch.
        """
        import time
        t0 = time.time()
        settled_since = None
        while time.time() - t0 < timeout_sec:
            with self.joint_lock:
                velocities = [abs(self._actual_velocities.get(n, 0.0))
                              for n in ARM_JOINT_NAMES]
            max_vel = max(velocities) if velocities else 0.0
            if max_vel < vel_threshold_rad_s:
                if settled_since is None:
                    settled_since = time.time()
                elif time.time() - settled_since >= dwell_sec:
                    return True
            else:
                settled_since = None
            time.sleep(0.02)
        return False

    def _attach_lego_to_gripper(self, obj_name):
        """On successful grasp close, convert world CO → AttachedCollisionObject.

        Pose is computed as block-in-tcp_link via TF lookup at attach time,
        so MoveIt tracks the carried block via FK thereafter.
        """
        self._ensure_lego_state()
        if not obj_name:
            return False
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return False
        bbox = self._lookup_bbox(obj_name)
        if not bbox:
            self._append_log(f'Attach skipped: no bbox for {obj_name}', 'warn')
            return False
        # Wait for arm to settle so TF (from /joint_states) and block pose
        # (from /objects_poses_sim) reflect the same static state — prevents
        # Layer-B-lag from baking a visible offset into the attached
        # collision object (see _wait_arm_at_rest docstring).
        settled = self._wait_arm_at_rest(timeout_sec=1.5)
        if not settled:
            self._append_log(
                'Attach: arm still moving after 1.5s wait — offset may be '
                'inaccurate', 'warn')
        with self.objects_lock:
            block = dict(self.objects_data.get(obj_name, {}))
        if not block:
            self._append_log(f'Attach skipped: no pose for {obj_name}', 'warn')
            return False

        # Model: tcp_link is the fixed-jaw tip. After grasp_move + gripper_close,
        # the block is pinched between the two jaws with:
        #   - Block side-center (vertical mid-height) at tcp.Z (no vertical offset)
        #   - Block center XY offset from tcp by the jaw half-width (set by
        #     _compute_jaw_offset during grasp_move)
        # We compute the actual block-in-tcp transform via TF — this gives the
        # correct XY offset AND the block's orientation relative to tcp (which
        # determines how the BOX collision volume is oriented in tcp's frame).
        import numpy as np
        try:
            t = self._tf_buffer.lookup_transform('base', 'tcp_link', rclpy.time.Time())
        except Exception as e:
            self._append_log(f'Attach failed: TF lookup tcp_link: {e}', 'warn')
            return False
        tr = t.transform
        tcp_mat = self._pose7_to_mat(
            tr.translation.x, tr.translation.y, tr.translation.z,
            tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        bx = float(block['x']); by = float(block['y'])
        # Isaac Sim reports the block origin at its CENTER, not its bottom —
        # verified empirically: all three brick sizes (2x2/2x3/2x4) report
        # z≈8mm when resting on ground, matching the 12.78mm-tall brick's
        # center, not half-height above bottom. The previous "+ sz/2"
        # adjustment baked a 6.4mm upward offset into the stored local
        # transform, which then propagated as a 6.4mm Z error in the
        # world-tracked MoveIt projection (Phase 9 verified).
        bz = float(block['z'])
        block_mat = self._pose7_to_mat(
            bx, by, bz,
            float(block.get('qx', 0.0)), float(block.get('qy', 0.0)),
            float(block.get('qz', 0.0)), float(block.get('qw', 1.0)))
        block_in_tcp = np.linalg.inv(tcp_mat) @ block_mat
        ax, ay, az, aqx, aqy, aqz, aqw = self._mat_to_pose7(block_in_tcp)
        # Sanity: block should be within one grip-width of tcp. Reject if grasp
        # clearly failed (e.g., FC-1 rejected grasp_move but close fired anyway).
        dist_mm = float((ax*ax + ay*ay + az*az) ** 0.5) * 1000.0
        if dist_mm > 80.0:
            self._append_log(
                f'Attach skipped: tcp-to-block {dist_mm:.0f}mm > 80mm '
                f'(grasp_move likely failed — no object in gripper)', 'warn')
            return False

        def _apply_scene(scene_msg, label):
            req = ApplyPlanningScene.Request()
            req.scene = scene_msg
            try:
                future = self._apply_scene_client.call_async(req)
                self._wait_future(future, timeout_sec=5.0)
            except Exception as exc:
                self._append_log(f'{label}: exception: {exc}', 'warn')
                return False
            res = future.result() if future else None
            if res is None:
                self._append_log(f'{label}: timed out (5s)', 'warn')
                return False
            if not res.success:
                self._append_log(f'{label}: apply returned success=False', 'warn')
                return False
            return True

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log('apply_planning_scene not available for attach', 'warn')
                return
            # Two-phase apply — Humble quirk: any REMOVE of a non-existent id
            # in the diff returns success=False for the ENTIRE call, even if
            # every ADD succeeded. _remove_single_lego_from_world ran on a
            # background thread before us and usually completed, so bundling
            # a world REMOVE in the same diff as our attach ADD would poison
            # the response flag.  Split: phase 1 fires the world REMOVE
            # (ignore result, idempotent), phase 2 does the attached ADD
            # with explicit block_in_tcp pose — response flag is trustworthy.
            name_id = f'lego_{obj_name}'

            # Phase 1: world REMOVE (best-effort; ignore the success flag)
            rm_scene = PlanningSceneMsg()
            rm_scene.is_diff = True
            rm = CollisionObject()
            rm.id = name_id
            rm.operation = CollisionObject.REMOVE
            rm_scene.world.collision_objects.append(rm)
            try:
                fut = self._apply_scene_client.call_async(
                    ApplyPlanningScene.Request(scene=rm_scene))
                self._wait_future(fut, timeout_sec=3.0)
            except Exception:
                pass  # idempotent — world may have been clean already

            # Phase 2: attached ADD with explicit link-local pose.
            # Geometry path matches world-CO path in _add_lego_collision_objects:
            # mesh-by-size for sim names, primitive box from bbox catalog for
            # YOLOE-style names where vision can't recover size.
            kind, geom = self._build_lego_geometry(obj_name)
            if kind is None:
                self._append_log(
                    f'Attach failed: no geometry for {obj_name} '
                    f'(no mesh, no bbox catalog entry)', 'warn')
                return
            add_scene = PlanningSceneMsg()
            add_scene.is_diff = True
            add_scene.robot_state.is_diff = True
            aco = AttachedCollisionObject()
            aco.link_name = 'tcp_link'
            aco.object.id = name_id
            aco.object.header.frame_id = 'tcp_link'
            aco.object.operation = CollisionObject.ADD
            block_pose = Pose()
            block_pose.position.x = float(ax)
            block_pose.position.y = float(ay)
            block_pose.position.z = float(az)
            block_pose.orientation.x = float(aqx)
            block_pose.orientation.y = float(aqy)
            block_pose.orientation.z = float(aqz)
            block_pose.orientation.w = float(aqw)
            if kind == 'mesh':
                aco.object.meshes.append(geom)
                aco.object.mesh_poses.append(block_pose)
            else:  # 'box'
                aco.object.primitives.append(geom)
                aco.object.primitive_poses.append(block_pose)
            aco.touch_links = ['tcp_link', 'gripper', 'jaw']
            add_scene.robot_state.attached_collision_objects.append(aco)

            # Claim attached-state BEFORE the apply so a racing
            # /objects_poses_sim → _add_lego_collision_objects hop doesn't
            # re-add the world copy underneath us. Rolled back on failure.
            prev_attached = self._attached_lego_name
            prev_offset = self._attached_lego_tcp_offset
            self._attached_lego_name = obj_name
            # Snapshot physical grasp offset in tcp_link frame. drop_sweep
            # reads this to place the block center at the cup target, not
            # tcp_link — the 1-2 mm delta between measured |ax| and the
            # theoretical half_gap is exactly what OMPL used to flail on.
            self._attached_lego_tcp_offset = (float(ax), float(ay), float(az))
            if not _apply_scene(add_scene, 'attach_lego'):
                self._attached_lego_name = prev_attached
                self._attached_lego_tcp_offset = prev_offset
                return
            self._append_log(
                f'Attach OK: {obj_name} → tcp_link at '
                f'({ax*1000:+.1f}, {ay*1000:+.1f}, {az*1000:+.1f}) mm '
                f'(magnitude {dist_mm:.1f} mm)', 'info')

            # A freshly-grasped object inherits the table-level pose from
            # which it was picked up — its bottom face is at z=0, touching
            # ground_plane. MoveIt treats attached objects as part of the
            # robot's collision volume, so attached_lego ↔ ground_plane
            # becomes a robot-world contact and gets reported unless we
            # declare it allowed. Same ACM row as the world copy uses; the
            # id is the same in both cases.
            self._acm_allow_ground_rest([name_id])

            if name_id in self._lego_collision_names:
                self._lego_collision_names.remove(name_id)
            tracer.event('attach_applied',
                         obj_name=obj_name,
                         block_in_tcp=(ax, ay, az),
                         block_in_tcp_quat=(aqx, aqy, aqz, aqw),
                         bbox=(float(bbox['sx']), float(bbox['sy']),
                               float(bbox['sz'])),
                         dist_mm=dist_mm,
                         mode='attached_explicit')
            self._append_log(
                f'Attached {name_id} to tcp_link '
                f'({bbox["sx"]*1000:.0f}×{bbox["sy"]*1000:.0f}×{bbox["sz"]*1000:.0f}mm) '
                f'offset=({ax*1000:+.1f},{ay*1000:+.1f},{az*1000:+.1f})mm')

        # SYNCHRONOUS: attach MUST complete before the caller signals its
        # motion-done event. Previously this dispatched _apply on a daemon
        # thread and returned immediately, so _cmd_gripper_close_for_object
        # fired evt.set() while the attach's phase-1 REMOVE + phase-2 ADD
        # were still in flight. The QS runner advanced to "Grasp Home",
        # planned it, and the post-check ran against a scene that still
        # had the world copy of the lego at its table pose — producing the
        # classic lego_*↔ground_plane(d=0.0mm) false-positive at wp[0].
        # Running _apply inline makes the attach a synchronous barrier.
        # Callers are already on a background thread (_send in
        # _cmd_gripper_close_for_object), so there's no Tk deadlock risk.
        _apply()
        return True

    # ------------------------------------------------------------------
    # AttachedCollisionObject lifecycle — standard MoveIt 2.5.9 path.
    # Pre-2.5.9 required a 10 Hz world-tracker workaround (see Phase 9
    # notes in prior revisions). The bug is fixed: /check_state_validity
    # correctly reports attached ↔ world collision pairs, so the workaround
    # is gone and touch_links on the AttachedCollisionObject handles the
    # gripper-vs-held-block allow semantics.
    # ------------------------------------------------------------------

    def _refresh_attached_pose(self):
        """Re-snapshot the attached lego's tcp_link-local offset from Isaac's
        current /objects_poses_sim reading. The attached body is rigid once
        set, but the physical block slides/settles in the jaws over time
        (gravity + compliance in Isaac). Calling this right before plan
        operations keeps MoveIt's collision geometry aligned with reality.
        One-shot — no timer.

        Returns True if the attached pose was updated, False if there was
        nothing to refresh (no lego attached, no bbox, no fresh pose, or
        the block appears to have slipped out of the gripper).
        """
        if not getattr(self, '_attached_lego_name', None):
            return False
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return False
        obj_name = self._attached_lego_name
        bbox = self._lookup_bbox(obj_name)
        if not bbox:
            return False
        # Synchronize TF and block pose streams — same rationale as
        # _attach_lego_to_gripper. If arm still has residual velocity,
        # refresh produces an offset-biased pose identical to the original
        # attach bug we're fixing.
        self._wait_arm_at_rest(timeout_sec=0.8)
        with self.objects_lock:
            block = dict(self.objects_data.get(obj_name, {}))
        if not block:
            return False

        import numpy as np
        try:
            t = self._tf_buffer.lookup_transform(
                'base', 'tcp_link', rclpy.time.Time())
        except Exception:
            return False
        tr = t.transform
        tcp_mat = self._pose7_to_mat(
            tr.translation.x, tr.translation.y, tr.translation.z,
            tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        block_mat = self._pose7_to_mat(
            float(block['x']), float(block['y']), float(block['z']),
            float(block.get('qx', 0.0)), float(block.get('qy', 0.0)),
            float(block.get('qz', 0.0)), float(block.get('qw', 1.0)))
        block_in_tcp = np.linalg.inv(tcp_mat) @ block_mat
        ax, ay, az, aqx, aqy, aqz, aqw = self._mat_to_pose7(block_in_tcp)
        dist_mm = float((ax*ax + ay*ay + az*az) ** 0.5) * 1000.0
        if dist_mm > 80.0:
            self._append_log(
                f'Refresh skipped: block {dist_mm:.0f}mm from tcp_link '
                f'(likely slipped from gripper)', 'warn')
            return False

        name_id = f'lego_{obj_name}'
        # Humble quirk: a single diff with REMOVE+ADD of the same attached
        # id is a no-op (success=True but the old attached stays put). Two
        # sequential diffs — REMOVE then ADD — actually re-seats the pose.
        rm_scene = PlanningSceneMsg()
        rm_scene.is_diff = True
        rm_scene.robot_state.is_diff = True
        aco_rm = AttachedCollisionObject()
        aco_rm.link_name = 'tcp_link'
        aco_rm.object.id = name_id
        aco_rm.object.operation = CollisionObject.REMOVE
        rm_scene.robot_state.attached_collision_objects.append(aco_rm)

        add_scene = PlanningSceneMsg()
        add_scene.is_diff = True
        add_scene.robot_state.is_diff = True
        kind, geom = self._build_lego_geometry(obj_name)
        if kind is None:
            return False
        aco = AttachedCollisionObject()
        aco.link_name = 'tcp_link'
        aco.object.id = name_id
        aco.object.header.frame_id = 'tcp_link'
        aco.object.operation = CollisionObject.ADD
        pose = Pose()
        pose.position.x = float(ax); pose.position.y = float(ay)
        pose.position.z = float(az)
        pose.orientation.x = float(aqx); pose.orientation.y = float(aqy)
        pose.orientation.z = float(aqz); pose.orientation.w = float(aqw)
        if kind == 'mesh':
            aco.object.meshes.append(geom)
            aco.object.mesh_poses.append(pose)
        else:  # 'box'
            aco.object.primitives.append(geom)
            aco.object.primitive_poses.append(pose)
        aco.touch_links = ['tcp_link', 'gripper', 'jaw']
        add_scene.robot_state.attached_collision_objects.append(aco)

        try:
            fut = self._apply_scene_client.call_async(
                ApplyPlanningScene.Request(scene=rm_scene))
            self._wait_future(fut, timeout_sec=3.0)
            # Ignore REMOVE result: returns False if the attached didn't
            # exist yet (e.g., first refresh right after grasp).
            fut = self._apply_scene_client.call_async(
                ApplyPlanningScene.Request(scene=add_scene))
            self._wait_future(fut, timeout_sec=3.0)
            res = fut.result()
            if not (res and res.success):
                return False
        except Exception:
            return False
        tracer.event('attached_pose_refreshed',
                     obj_name=obj_name,
                     block_in_tcp=(ax, ay, az),
                     block_in_tcp_quat=(aqx, aqy, aqz, aqw),
                     dist_mm=dist_mm)
        return True

    def _cmd_detach_lego(self):
        """Manual/test-harness detach — force-remove every known lego from the
        robot's attached list, regardless of _attached_lego_name's current value.
        Useful after a crashed cycle left a stale AttachedCollisionObject.
        """
        self._ensure_lego_state()
        self._attached_lego_name = None
        self._attached_lego_tcp_offset = None
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                return
            # Best-effort: REMOVE every (color × size) lego as attached.
            colors = ('red', 'green', 'blue')
            sizes = ('2x2', '2x3', '2x4')
            scene = PlanningSceneMsg()
            scene.is_diff = True
            for c in colors:
                for s in sizes:
                    aco = AttachedCollisionObject()
                    aco.link_name = 'tcp_link'
                    aco.object.id = f'lego_{c}_{s}'
                    aco.object.operation = CollisionObject.REMOVE
                    scene.robot_state.attached_collision_objects.append(aco)
            scene.robot_state.is_diff = True
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            self._attached_lego_name = None
            self._attached_lego_tcp_offset = None
            self._append_log('Detached all legos (force cleanup)')

        threading.Thread(target=_apply, daemon=True).start()
        # Repopulate world legos shortly after
        self.root.after(300, self._add_lego_collision_objects)

    def _detach_lego(self):
        """Non-blocking detach wrapper. Spawns the apply in a thread so
        callers that don't need to wait (e.g. _cmd_grasp_refresh) aren't
        blocked. For drop_release, use `_detach_lego_sync` instead — it
        waits for the planning-scene apply to commit BEFORE returning,
        which is required so the subsequent grasp_home plans against a
        scene without the (now phantom) attached block.
        """
        threading.Thread(target=self._detach_lego_sync, daemon=True).start()

    def _detach_lego_sync(self):
        """Blocking release of the currently-held lego.

        Sends AttachedCollisionObject op=REMOVE and a matching world
        CollisionObject REMOVE in one scene diff. MoveIt detaches the
        lego; the next /objects_poses_sim tick refreshes its world pose
        from Isaac Sim truth (wherever it actually landed after release).
        """
        self._ensure_lego_state()
        name = self._attached_lego_name
        if not name:
            return False
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            return False
        if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
            self._append_log('apply_planning_scene not available for detach', 'warn')
            return False
        obj_id = f'lego_{name}'
        scene = PlanningSceneMsg()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        # Detach from gripper
        aco = AttachedCollisionObject()
        aco.link_name = 'tcp_link'
        aco.object.id = obj_id
        aco.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects.append(aco)
        # Remove the world copy MoveIt would put back at detach location —
        # /objects_poses_sim will re-add at ground truth pose next tick.
        rm = CollisionObject()
        rm.id = obj_id
        rm.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(rm)
        try:
            req = ApplyPlanningScene.Request()
            req.scene = scene
            fut = self._apply_scene_client.call_async(req)
            self._wait_future(fut, timeout_sec=3.0)
        except Exception as exc:
            self._append_log(f'Detach apply exception: {exc}', 'warn')
        self._attached_lego_name = None
        self._attached_lego_tcp_offset = None
        self._cycle_detach_seen = True
        tracer.event('release_applied', obj_name=name)
        self._append_log(f'Detached {obj_id}')
        return True

    def _publish_cup_visual_markers(self, cups_dict=None):
        """Publish colored cup meshes as RViz visual markers.

        Reads cup poses from `self._drop_data` by default. If `cups_dict`
        is provided (e.g. real-mode cached cups from Refresh Cups Pose),
        uses that snapshot instead — bypasses the live /drop_poses
        subscription so the visual matches the user-frozen scan, not
        whatever the live (potentially sim) topic is currently publishing.

        Mirrors the cups_dict pattern in `_add_cup_collision_objects` so
        visual + collision share the same source-of-truth in real mode.
        """
        if not hasattr(self, '_cup_visual_pub'):
            from rclpy.qos import QoSProfile, DurabilityPolicy
            self._cup_visual_pub = self.create_publisher(
                MarkerArray, '/cup_visual_markers_array',
                QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if cups_dict is not None:
            drop_items = dict(cups_dict)
        else:
            with self._drop_lock:
                drop_items = dict(self._drop_data)
        if not drop_items:
            return
        scale = _CUP_STL_SCALE * _CUP_COLLISION_PADDING
        ma = MarkerArray()
        for i, (name, pos) in enumerate(drop_items.items()):
            r, g, b, a = _CUP_VISUAL_COLORS.get(name, (0.5, 0.5, 0.5, 0.99))
            m = VisMarker()
            m.header.frame_id = 'base'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'cup_visual'
            m.id = i
            m.type = VisMarker.MESH_RESOURCE
            m.action = VisMarker.ADD
            m.mesh_resource = _get_cup_stl_uri()
            m.pose.position.x = pos['x']
            m.pose.position.y = pos['y']
            # Cup STL origin is at base. /drop_poses publishes body-center,
            # so shift down by half height to render the mesh base on ground.
            m.pose.position.z = pos['z'] - CUP_BODY_HEIGHT_M / 2.0
            # Phase 9: honor cup orientation from /drop_poses (was identity)
            m.pose.orientation.x = float(pos.get('qx', 0.0))
            m.pose.orientation.y = float(pos.get('qy', 0.0))
            m.pose.orientation.z = float(pos.get('qz', 0.0))
            m.pose.orientation.w = float(pos.get('qw', 1.0))
            m.scale = Vector3(x=scale, y=scale, z=scale)
            from std_msgs.msg import ColorRGBA
            m.color = ColorRGBA(r=r, g=g, b=b, a=a)
            m.lifetime = Duration(sec=0)
            m.frame_locked = True
            ma.markers.append(m)
        self._cup_visual_pub.publish(ma)

    def _cmd_apply_collision_padding(self):
        """Update collision padding and re-add collision objects + visual markers."""
        global _CUP_COLLISION_PADDING
        pct = self._collision_padding_var.get()
        _CUP_COLLISION_PADDING = 1.0 + pct / 100.0
        # Clear cached mesh so it rebuilds with new padding
        if hasattr(_load_cup_mesh, '_cached'):
            del _load_cup_mesh._cached
        self._append_log(f'Collision padding: {pct}% ({_CUP_COLLISION_PADDING:.2f}x)')
        # Re-add collision objects and visual markers with new padding
        self._add_cup_collision_objects()
        self.root.after(300, self._refresh_display_markers)

    def _cmd_apply_home_speed(self):
        """Sync _home_speed_var into the home_velocity_scale ROS2 param so
        _cmd_grasp_home picks it up on the next motion. 0.0 = no override
        (use the planning default velocity_scale_var); values in (0, 1.0]
        override velocity_scale for the grasp_home plan only.
        """
        from rclpy.parameter import Parameter as RclParam
        v = float(self._home_speed_var.get())
        self.set_parameters([RclParam('home_velocity_scale', RclParam.Type.DOUBLE, v)])
        label = 'no override' if v <= 0.0 else f'{v:.2f}'
        self._append_log(f'Home speed scale: {label}')

    def _refresh_display_markers(self):
        """Republish visual markers based on current toggle state."""
        if getattr(self, '_show_visual_var', None) and self._show_visual_var.get():
            self._publish_cup_visual_markers()

    def _toggle_visual_markers(self):
        """Toggle colored cup visual markers on/off."""
        if self._show_visual_var.get():
            self._publish_cup_visual_markers()
        else:
            self._delete_visual_markers()


    def _delete_visual_markers(self):
        """Delete cup visual markers."""
        if not hasattr(self, '_cup_visual_pub'):
            return
        ma = MarkerArray()
        for i in range(3):
            m = VisMarker()
            m.header.frame_id = 'base'
            m.ns = 'cup_visual'
            m.id = i
            m.action = VisMarker.DELETE
            ma.markers.append(m)
        self._cup_visual_pub.publish(ma)


    def _get_selected_drop_pose(self):
        """Return (name, x, y, z) for the selected drop listbox entry, or None."""
        if not hasattr(self, '_drop_listbox'):
            return None
        sel = self._drop_listbox.curselection()
        if not sel:
            self._append_log('No drop target selected', 'warn')
            return None
        text = self._drop_listbox.get(sel[0])
        # Strip label annotation: "drop_0 [red]  (...)" → "drop_0"
        raw = text.split('  ')[0]  # "drop_0 [red]" or "drop_0"
        name = raw.split(' [')[0]  # "drop_0"
        with self._drop_lock:
            pose = self._drop_data.get(name)
        if pose is None:
            self._append_log(f'Drop target {name!r} not in data', 'warn')
            return None
        return name, pose['x'], pose['y'], pose['z']

    def _cmd_grasp_update_topic(self):
        """Switch object subscription to topic from GUI entry and auto-refresh."""
        new_topic = self._grasp_topic_var.get().strip()
        if not new_topic:
            return
        # Destroy old subscription and create new one
        if hasattr(self, 'objects_sub') and self.objects_sub is not None:
            self.destroy_subscription(self.objects_sub)
        with self.objects_lock:
            self.objects_data.clear()
        self.objects_sub = self.create_subscription(
            TFMessage, new_topic, self._objects_callback, 10,
            callback_group=self._sub_cb_group)
        # Update button text
        if hasattr(self, '_grasp_move_btn'):
            if new_topic == '/drop_poses':
                self._grasp_move_btn.config(text='Move to Drop')
            else:
                self._grasp_move_btn.config(text='Move to Grab')
        self._append_log(f'Grasp topic: {new_topic}')

    def _cmd_grasp_refresh(self):
        """Resync object state from /objects_poses_sim. Async — schedules
        listbox repopulate on the tk thread via root.after(500, ...).
        Callers that need guaranteed-fresh state should sleep ~1s after.

        Forces sub topics back to sim before pulling — without this, a prior
        Real Test session leaves `objects_sub` on /objects_poses_real and
        the 500 ms refill grabs real-mode poses into the sim collision
        scene (the D-08 leak the user reported)."""
        if not hasattr(self, 'obj_listbox'):
            return
        if tracer.is_active():
            tracer.close_cycle('user_canceled',
                               note='grasp_refresh while cycle open')
        # Reset subs to sim BEFORE clearing — otherwise the next live
        # callback could refill objects_data with real-mode data before
        # the timed re-populate fires.
        self._sim_ensure_sim_topics()
        with self.objects_lock:
            self.objects_data.clear()
        self.obj_listbox.delete(0, tk.END)
        self._ensure_lego_state()
        if self._attached_lego_name:
            self._detach_lego_sync()
        self._remove_lego_collision_objects()
        if getattr(self, '_gui_ready', False):
            self.root.after(500, self._populate_object_list)
            self.root.after(700, self._add_lego_collision_objects)

    def _populate_object_list(self):
        if not hasattr(self, 'obj_listbox'):
            return
        self.obj_listbox.delete(0, tk.END)
        with self.objects_lock:
            for name, pos in self.objects_data.items():
                self.obj_listbox.insert(
                    tk.END,
                    f'{name}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')
        count = self.obj_listbox.size()
        if count > 0:
            self._append_log(f'Objects refreshed: {count} found')

    def _cmd_grasp_home(self):
        """Move arm to grasp-ready home: gripper pointing down.

        Routes through _joint_space_collision_free_execute (tier1 per-
        waypoint validity check + OMPL fallback) — same discipline as
        drop_sweep / drop_point. Previously used _cmd_plan_execute which
        only ran OMPL one-shot without the 2%-resolution pre-check, so a
        trajectory that clipped the held lego against a cup could slip
        through OMPL's sampling gap.
        """
        target = {name: 0.0 for name in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
        target['wrist_roll'] = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        self._append_log(f'Grasp Home: wrist_flex=90° wrist_roll={math.degrees(target["wrist_roll"]):.1f}°')
        tracer.event('grasp_home_start', target=dict(target))
        evt = threading.Event()
        self._motion_event = evt
        # Emit grasp_home_done when motion settles. Close the trace cycle
        # iff a detach was seen in this cycle — that signals this is the
        # post-drop return-to-home, the final leg of the pick-place.
        # Pre-grasp and post-pick grasp_home calls leave _cycle_detach_seen
        # at False, so they just event and don't close.
        def _emit_home_done():
            evt.wait(timeout=30.0)
            tracer.event('grasp_home_done')
            if getattr(self, '_cycle_detach_seen', False) and tracer.is_active():
                tracer.close_cycle('completed')
                self._cycle_detach_seen = False
        threading.Thread(target=_emit_home_done, daemon=True).start()
        # Speed override: if home_velocity_scale param > 0, temporarily
        # swap velocity_scale_var for this motion's plan, then restore.
        # Lets the speed-vs-lag experiment sweep grasp_home speed live
        # via `ros2 param set` without affecting other motions.
        hvs = float(self.get_parameter('home_velocity_scale').value)
        saved_vs = None
        if hvs > 0.0 and hasattr(self, 'velocity_scale_var'):
            saved_vs = self.velocity_scale_var.get()
            self.velocity_scale_var.set(hvs)
            self._append_log(f'  velocity_scale override: {hvs:.2f} (was {saved_vs:.2f})')

        # grasp_home: deterministic-only (tier1 linear + tier2 retract-pan-settle).
        # OMPL fallback disabled — RRTConnect's RNG variance has produced cup-
        # clipping trajectories from far-pan start poses. If both tiers fail,
        # surface loudly rather than roll the dice.
        self._joint_space_collision_free_execute(
            target, on_complete_event=evt, duration_s=3.0,
            allow_ompl_fallback=False)

        if saved_vs is not None:
            self.velocity_scale_var.set(saved_vs)

    # Tier-2 intermediate pose — all non-pan joints lifted above the cup
    # plane. shoulder_pan is held at whatever the current segment needs
    # (start_pan for the retract, target_pan for the settle).
    _NEUTRAL_NON_PAN_JOINTS = {
        'shoulder_lift': -0.2,
        'elbow_flex': 0.0,
        'wrist_flex': math.pi / 2,
        'wrist_roll': -math.pi / 2,
    }

    def _joint_space_collision_free_execute(self, target, on_complete_event,
                                             duration_s=3.0, waypoints=50,
                                             allow_ompl_fallback=True):
        """Tiered deterministic planner with opt-in OMPL fallback.

        Tier 1: linear joint-space interp, per-waypoint validity check.
            Dispatches the SAME (waypoints+1)-point trajectory it validated
            via _execute_full_trajectory — so FollowJointTrajectory's spline
            only interpolates between closely-spaced validated points. This
            is what broke the old tier1 (_LEGACY_TIER1 below): it validated
            N points but executed (start, end, duration), so the controller's
            spline could diverge and clip cups between.
        Tier 2: retract-pan-settle decomposition. Three linear sub-segments
            (retract above cup plane, pan across, settle to target). Handles
            the common "need to pan past a cup" case geometrically, without
            sampling.
        Fallback: OMPL via _ompl_plan_validate_execute. Opt-in per caller
            through allow_ompl_fallback. grasp_home passes False —
            RRTConnect's RNG variance has produced cup-clipping plans from
            far-pan start poses, and deterministic tiers must suffice.

        Both tiers share _check_state_valid_with_contacts with the Mode B
        OMPL post-check, so there is no planner-vs-check coherence gap.
        """
        # Every motion = arm-group plan. Switch the RViz panel so the Goal
        # State ghost highlights the right group (gripper_command set it to
        # 'gripper' — without this, grasp_home and drops keep no ghost).
        self._select_planning_group('arm')
        # Snapshot current (physics) positions before advancing
        # self.joint_positions to the target — tier1/tier2 interpolate from
        # the physical start, not from the freshly-overwritten slider target.
        with self.joint_lock:
            current = {n: self._actual_positions.get(
                n, self.joint_positions.get(n, 0.0))
                       for n in ARM_JOINT_NAMES}
            for n in ARM_JOINT_NAMES:
                if n in target:
                    self.joint_positions[n] = target[n]
        # Sync sliders so RViz's interactive markers land on the goal.
        for n in ARM_JOINT_NAMES:
            if n in target and n in self.sliders:
                self.sliders[n].set(target[n])
                if n in self.slider_labels:
                    self.slider_labels[n].config(text=f'{target[n]:.3f}')
        self._publish_goal_state()

        target_full = {n: target.get(n, current[n]) for n in ARM_JOINT_NAMES}

        # Already-at-target short-circuit (0.5° tol on every joint).
        ALREADY_AT_TARGET_TOL = math.radians(0.5)
        max_delta = max(abs(target_full[n] - current[n]) for n in ARM_JOINT_NAMES)
        if max_delta < ALREADY_AT_TARGET_TOL:
            self._append_log(
                f'  already at target (max Δ={math.degrees(max_delta):.2f}° '
                f'< 0.5°) — no motion dispatched')
            tracer.event('tier1_noop',
                         max_delta_deg=math.degrees(max_delta),
                         target=dict(target_full), current=dict(current))
            self._last_motion_status = {
                'ok': True, 'outcome': 'already_at_target',
                'msg': (f'already at target '
                        f'(max Δ={math.degrees(max_delta):.2f}°)')}
            on_complete_event.set()
            return True

        # Tier 1: linear joint-space interp + per-waypoint check.
        traj = self._plan_linear_joint_path(
            current, target_full, duration_s, n_samples=waypoints)
        if traj is not None:
            tracer.event('planner_used', which='linear',
                         n_wps=len(traj.points))
            self._append_log(
                f'  tier1 linear: {len(traj.points)} wps clean, executing')
            self._execute_full_trajectory(traj, on_complete_event)
            return True

        # Tier 2: retract-pan-settle decomposition.
        traj2 = self._plan_retract_pan_settle(
            current, target_full, duration_s, n_samples=waypoints)
        if traj2 is not None:
            tracer.event('planner_used', which='retract_pan_settle',
                         n_wps=len(traj2.points))
            self._append_log(
                f'  tier2 retract-pan-settle: {len(traj2.points)} wps clean, '
                f'executing')
            self._execute_full_trajectory(traj2, on_complete_event)
            return True

        # Fallback: OMPL (opt-in).
        if not allow_ompl_fallback:
            tracer.event('planner_used', which='none')
            self._append_log(
                '  REFUSED: tier1 + tier2 both collided; OMPL fallback '
                'disabled (deterministic-only mode)', 'err')
            self._last_motion_status = {
                'ok': False, 'outcome': 'tiered_planner_exhausted',
                'msg': ('tier1 linear and tier2 retract-pan-settle both '
                        'collided; OMPL fallback disabled')}
            on_complete_event.set()
            return False

        tracer.event('planner_used', which='ompl_fallback')
        self._append_log(
            '  tier1 + tier2 both collided; falling back to OMPL')
        return self._ompl_plan_validate_execute(target_full, on_complete_event)

    def _plan_linear_joint_path(self, current, target_full, duration_s,
                                 n_samples=50):
        """Deterministic linear joint-space interp + per-waypoint check.

        Builds an (n_samples+1)-point JointTrajectory with monotonic
        time_from_start. Validates every waypoint with
        _check_state_valid_with_contacts (same checker the OMPL Mode B
        post-check uses — no coherence gap).

        Returns a JointTrajectory on success, None on first collision.
        Pure float arithmetic: q_i = start + (i/N) * (target - start) —
        same input always produces the same output.
        """
        jt = JointTrajectory()
        jt.joint_names = list(ARM_JOINT_NAMES)
        for i in range(n_samples + 1):
            alpha = i / n_samples
            q = {n: current[n] + alpha * (target_full[n] - current[n])
                 for n in ARM_JOINT_NAMES}
            valid, contacts = self._check_state_valid_with_contacts(q)
            if not valid:
                summary = '; '.join(
                    f'{c.contact_body_1}↔{c.contact_body_2}'
                    f'(d={c.depth*1000:.1f}mm)'
                    for c in contacts[:3]) or 'no contact info'
                self._append_log(
                    f'  tier1 linear: wp[{i}]/α={alpha:.2f} collides — '
                    f'{summary}', 'warn')
                return None
            pt = JointTrajectoryPoint()
            pt.positions = [q[n] for n in ARM_JOINT_NAMES]
            t = alpha * duration_s
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            jt.points.append(pt)
        return jt

    def _plan_retract_pan_settle(self, current, target_full, duration_s,
                                   n_samples=50):
        """Tier 2: three-segment geometric decomposition.

        Seg A: (current) → (current_pan, NEUTRAL)   [retract above cup plane]
        Seg B: (current_pan, NEUTRAL) → (target_pan, NEUTRAL)  [pan across]
        Seg C: (target_pan, NEUTRAL) → (target)     [settle to final config]

        Each sub-segment runs through _plan_linear_joint_path. On success,
        segments are concatenated into one JointTrajectory with adjusted
        time_from_start; duplicate joining waypoints are dropped.

        Returns None on any sub-segment collision — the caller then falls
        through to OMPL (or reports failure, if fallback disabled).
        """
        neutral_at_start = {
            n: current[n] if n == 'shoulder_pan'
            else self._NEUTRAL_NON_PAN_JOINTS[n]
            for n in ARM_JOINT_NAMES}
        neutral_at_target = {
            n: target_full[n] if n == 'shoulder_pan'
            else self._NEUTRAL_NON_PAN_JOINTS[n]
            for n in ARM_JOINT_NAMES}

        # Degenerate case: current already at neutral non-pan → seg A is
        # a no-op and the whole decomposition reduces to a pan + settle
        # that's essentially the direct path tier1 already rejected.
        if all(abs(current[n] - neutral_at_start[n]) < math.radians(2.0)
               for n in ARM_JOINT_NAMES if n != 'shoulder_pan'):
            self._append_log(
                '  tier2: already near neutral, decomposition degenerate',
                'warn')
            return None

        seg_dur = duration_s / 3.0
        seg_samples = max(10, n_samples // 3)

        jt_a = self._plan_linear_joint_path(
            current, neutral_at_start, seg_dur, n_samples=seg_samples)
        if jt_a is None:
            self._append_log('  tier2: seg A (retract) collides', 'warn')
            return None
        jt_b = self._plan_linear_joint_path(
            neutral_at_start, neutral_at_target, seg_dur,
            n_samples=seg_samples)
        if jt_b is None:
            self._append_log('  tier2: seg B (pan across) collides', 'warn')
            return None
        jt_c = self._plan_linear_joint_path(
            neutral_at_target, target_full, seg_dur, n_samples=seg_samples)
        if jt_c is None:
            self._append_log('  tier2: seg C (settle) collides', 'warn')
            return None

        concat = JointTrajectory()
        concat.joint_names = list(ARM_JOINT_NAMES)
        t_offset = 0.0
        for seg_idx, jt in enumerate([jt_a, jt_b, jt_c]):
            for pt_idx, pt in enumerate(jt.points):
                if seg_idx > 0 and pt_idx == 0:
                    continue  # drop duplicate of prev segment's end point
                new_pt = JointTrajectoryPoint()
                new_pt.positions = list(pt.positions)
                t_local = (pt.time_from_start.sec
                           + pt.time_from_start.nanosec * 1e-9)
                t = t_local + t_offset
                new_pt.time_from_start.sec = int(t)
                new_pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
                concat.points.append(new_pt)
            t_offset += seg_dur
        return concat

    def _joint_space_collision_free_execute_LEGACY_TIER1(
            self, target, on_complete_event, duration_s=3.0, waypoints=50):
        """Legacy tier1-first path kept for reference. Do not call — see the
        new version above for the always-OMPL discipline."""
        with self.joint_lock:
            current = {n: self._actual_positions.get(n, self.joint_positions.get(n, 0.0))
                       for n in ARM_JOINT_NAMES}

        target_full = {n: target.get(n, current[n]) for n in ARM_JOINT_NAMES}

        # Short-circuit: if current is already within 0.5° of target on every
        # joint, there's nothing to do. Previously this would still dispatch
        # a zero-delta 51-waypoint trajectory, return success instantly, and
        # confuse the user into thinking a motion completed ("robot didn't
        # move but service said OK"). Now we explicitly flag the no-op case
        # in _last_motion_status so the caller can distinguish "moved to
        # target" from "already at target".
        ALREADY_AT_TARGET_TOL = math.radians(0.5)  # 0.0087 rad
        max_delta = max(abs(target_full[n] - current[n]) for n in ARM_JOINT_NAMES)
        if max_delta < ALREADY_AT_TARGET_TOL:
            self._append_log(
                f'  already at target (max Δ={math.degrees(max_delta):.2f}° '
                f'< 0.5°) — no motion dispatched')
            tracer.event('tier1_noop',
                         max_delta_deg=math.degrees(max_delta),
                         target=dict(target_full), current=dict(current))
            self._last_motion_status = {
                'ok': True, 'outcome': 'already_at_target',
                'msg': (f'already at target '
                        f'(max Δ={math.degrees(max_delta):.2f}°)')}
            on_complete_event.set()
            return True

        def _segment_valid(a, b, N):
            """Return (ok, invalid_info, contacts). ok=True if all wps valid."""
            for i in range(N + 1):
                alpha = i / N
                q = {n: a[n] + alpha * (b[n] - a[n]) for n in ARM_JOINT_NAMES}
                valid, contacts = self._check_state_valid_with_contacts(q)
                if not valid:
                    return False, (i, alpha), contacts
            return True, None, []

        # Step 1: try direct joint-space interpolation (fast, deterministic).
        ok, bad, bad_contacts = _segment_valid(current, target_full, waypoints)
        tracer.event('tier1_check_done',
                     ok=bool(ok),
                     bad_wp=(None if ok else bad[0]),
                     bad_alpha=(None if ok else bad[1]),
                     waypoints=waypoints,
                     target=dict(target_full),
                     current=dict(current),
                     contacts=([{
                         'a': c.contact_body_1, 'b': c.contact_body_2,
                         'depth_mm': c.depth * 1000,
                     } for c in bad_contacts[:10]] if not ok else []))
        if ok:
            self._append_log(
                f'  joint-space direct: {waypoints+1} wps clean, executing')
            # DIAG: dump the exact waypoints tier1 validated
            try:
                import json as _json
                _wps = []
                for i in range(waypoints + 1):
                    alpha = i / waypoints
                    q = {n: current[n] + alpha * (target_full[n] - current[n])
                         for n in ARM_JOINT_NAMES}
                    _wps.append({'t': alpha * duration_s,
                                 'positions': [q[n] for n in ARM_JOINT_NAMES]})
                with open('/tmp/last_arm_trajectory.json', 'w') as _f:
                    _json.dump({'source': 'tier1',
                                'joint_names': list(ARM_JOINT_NAMES),
                                'points': _wps}, _f)
            except Exception:
                pass
            def _tier1_done():
                # _execute_trajectory runs a UI animation thread and calls
                # on_complete after it returns. It doesn't currently wire
                # into the action-client result callback, so we can only
                # report "animation finished" — NOT that the physical
                # controller converged. For now, treat finishing the
                # animation as success; _send_arm_goal handles the actual
                # controller dispatch internally.
                # TODO: if we need controller-level verdict for tier1, wire
                # _send_arm_goal to surface the action result.
                self._last_motion_status = {
                    'ok': True, 'outcome': 'completed',
                    'msg': f'tier1 joint-space ({waypoints+1} wps) complete'}
                on_complete_event.set()
            self._execute_trajectory(
                target_full, duration_s=duration_s,
                on_complete=_tier1_done)
            return True

        # Step 2: direct path collides — fall back to OMPL, one-shot.
        self._append_log(
            f'  joint-space direct: wp[{bad[0]}]@α={bad[1]:.2f} collides — '
            f'falling back to OMPL (one-shot)')
        return self._ompl_plan_validate_execute(
            target_full, on_complete_event)

    def _ompl_plan_validate_execute(self, target, on_complete_event):
        """OMPL path planning + waypoint validation. One shot, no retries.

        Previously this method retried up to 10× with small goal perturbations,
        on the (false) premise that OMPL is deterministic per (start, goal,
        scene). RRTConnect is sampling-based and non-deterministic — perturbing
        the goal was just re-rolling the dice with extra latency. In the prior
        session retries cost ~50s per failure and failed anyway (6/6).

        Strategy: plan once; validate every waypoint; on failure classify
        (Mode A: OMPL returned no plan, Mode B: plan invalid per our checker)
        and surface the diagnostic loudly. Returns True iff a valid plan was
        scheduled for execution.
        """
        if not MOVEIT_AVAILABLE or self.plan_client is None \
                or not self.plan_client.service_is_ready():
            self._append_log('OMPL: service not available', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'ompl_service_unavailable',
                'msg': 'OMPL plan service not available'}
            on_complete_event.set()
            return False

        with self.joint_lock:
            current = {n: self._actual_positions.get(
                n, self.joint_positions.get(n, 0.0))
                       for n in ALL_JOINT_NAMES}

        tracer.snapshot_scene('before_ompl_plan', self)
        traj = self._ompl_plan_sync(current, target, tolerance=0.01)

        if traj is None or not traj.joint_trajectory.points:
            ec_info = getattr(self, '_last_ompl_error', None)
            tracer.snapshot_scene('after_ompl_returns_modeA', self)
            tracer.event('ompl_attempt', attempt=0, perturb='none',
                         mode='A', ec_info=repr(ec_info))
            tracer.event('ompl_exhausted', n_attempts=1, mode='A',
                         ec_info=repr(ec_info))
            self._append_log(
                f'  OMPL Mode A: planner returned no plan ({ec_info})', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'ompl_mode_a',
                'msg': f'OMPL Mode A: no plan ({ec_info})'}
            on_complete_event.set()
            return False

        tracer.snapshot_scene('after_ompl_returns', self)
        chk = self._trajectory_first_invalid_with_contacts(
            traj.joint_trajectory)
        if not chk.ok:
            contact_summary = '; '.join(
                f'{c.contact_body_1}↔{c.contact_body_2}(d={c.depth*1000:.1f}mm)'
                for c in chk.contacts[:5]) or 'no contact info'
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            tracer.snapshot_scene('after_waypoint_check_modeB', self)
            tracer.event('ompl_attempt', attempt=0, perturb='none',
                         mode='B',
                         bad_wp=chk.bad_wp, bad_subidx=chk.bad_subidx,
                         sub_t=chk.sub_t,
                         n_wps=chk.n_wps, n_sub=chk.n_sub,
                         contacts=[{
                             'a': c.contact_body_1,
                             'b': c.contact_body_2,
                             'depth_mm': c.depth * 1000,
                         } for c in chk.contacts[:10]])
            tracer.event('ompl_exhausted', n_attempts=1, mode='B',
                         bad_wp=chk.bad_wp,
                         n_wps=chk.n_wps)
            self._append_log(
                f'  OMPL post-check: {where} invalid '
                f'(checked {chk.n_wps} wps + {chk.n_sub} sub-states): '
                f'{contact_summary}', 'warn')
            self._last_motion_status = {
                'ok': False, 'outcome': 'ompl_mode_b',
                'msg': (f'post-check rejected {where}: {contact_summary}'),
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t}
            on_complete_event.set()
            return False

        tracer.event('ompl_validated', attempt=0, perturb='none',
                     n_wps=chk.n_wps, n_sub=chk.n_sub)
        # Honest log: we checked every waypoint AND N intermediate states
        # per segment. "Validated" here means collision-free at discrete
        # sample points under the current /planning_scene, NOT proof of
        # clearance at infinitesimal resolution — see TRAJ_SUBSEGMENT_SAMPLES.
        self._append_log(
            f'  OMPL post-check: {chk.n_wps} wps + {chk.n_sub} sub-states '
            f'clear ({TRAJ_SUBSEGMENT_SAMPLES}/seg)')
        self._execute_full_trajectory(
            traj.joint_trajectory, on_complete_event)
        return True

    def _ompl_plan_sync(self, start_joint_dict, target_joint_dict, tolerance=0.01):
        """Blocking OMPL plan call. Returns RobotTrajectory or None.

        tolerance: joint goal tolerance (rad). Default 0.01 matches the
        _cmd_plan_execute interface.
        """
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.pipeline_id = 'ompl'
        attempts_var = getattr(self, '_planning_attempts_var', None)
        mpr.num_planning_attempts = attempts_var.get() if attempts_var else 50
        # 10s (was 5s). Phase 9 data: tier1 check sometimes rejects the
        # linear-interp path at a 0.03mm cup-rim clip, and the 5s OMPL
        # budget was not always enough for RRTConnect to find an alternative.
        # One-shot plan → 10s keeps worst-case latency ≤10s (vs the old
        # retry loop's 50s) while drastically reducing sampling variance.
        mpr.allowed_planning_time = 10.0
        vel_scale = self.velocity_scale_var.get() \
            if hasattr(self, 'velocity_scale_var') else 0.5
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = vel_scale

        start_state = RobotState()
        start_state.joint_state.name = list(ALL_JOINT_NAMES)
        start_state.joint_state.position = [
            float(start_joint_dict.get(n, 0.0)) for n in ALL_JOINT_NAMES]
        mpr.start_state = start_state

        constraints = Constraints()
        for name in ARM_JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(target_joint_dict.get(name, 0.0))
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        mpr.goal_constraints.append(constraints)
        req.motion_plan_request = mpr

        future = self.plan_client.call_async(req)
        self._wait_future(future, timeout_sec=10.0)
        if future.result() is None:
            self._last_ompl_error = ('no_response', None)
            return None
        r = future.result()
        ec = r.motion_plan_response.error_code.val
        if ec != 1:
            self._last_ompl_error = (f'ec={ec}', r.motion_plan_response.planning_time)
            return None
        self._last_ompl_error = None
        return r.motion_plan_response.trajectory

    def _trajectory_all_valid(self, joint_trajectory):
        """Return True if every waypoint of joint_trajectory is collision-free.

        Uses /check_state_validity (same service as _check_state_valid, but
        operates on a trajectory point instead of a dict).
        """
        if not joint_trajectory.points:
            return False
        names = list(joint_trajectory.joint_names)
        for pt in joint_trajectory.points:
            state = {n: p for n, p in zip(names, pt.positions)}
            if not self._check_state_valid(state):
                return False
        return True

    def _trajectory_first_invalid_with_contacts(self, joint_trajectory,
                                                 subsamples=None):
        """Find first invalid state on a trajectory. Checks every waypoint
        AND `subsamples` interpolated states between each consecutive pair.
        Returns a dict on failure or {'ok': True, 'n_wps': N, 'n_sub': M}
        on success. Legacy callers that unpacked (idx, contacts) continue
        to work via the 2-tuple properties wrapper below.

        Phase 9: includes gripper_joint in the checked state (OMPL carries
        it through the trajectory; defaulting to 0 made post-validation
        disagree with OMPL's internal checker on jaw-world collisions).

        Post-Phase-9 fix: the original implementation only probed waypoints.
        A 37-wp 106° pan sweep has ~2.9° joint motion per segment, which
        is larger than the angular cross-section a cup subtends at arm
        reach. Straight-line joint interp between two clear waypoints
        routinely swept through a cup. We now subsample each segment.
        """
        if not joint_trajectory.points:
            return _TrajCheckResult(ok=False, bad_wp=0, bad_subidx=None,
                                    sub_t=0.0, contacts=[], n_wps=0, n_sub=0)
        if not MOVEIT_AVAILABLE or not hasattr(self, 'validity_client') \
                or not self.validity_client.service_is_ready():
            # No checker available — can't prove anything; treat as "ok"
            # so we don't block execution on missing infra (same as before).
            return _TrajCheckResult(ok=True, bad_wp=None, bad_subidx=None,
                                    sub_t=0.0, contacts=[],
                                    n_wps=len(joint_trajectory.points),
                                    n_sub=0)
        if subsamples is None:
            # Use the module-level default (hot-reloadable).
            subsamples = TRAJ_SUBSEGMENT_SAMPLES
        with self.joint_lock:
            gj_live = self._actual_positions.get(
                GRIPPER_JOINT_NAME,
                self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0))
        names = list(joint_trajectory.joint_names)

        def _probe(joints_map):
            gj = joints_map.get(GRIPPER_JOINT_NAME, gj_live)
            req = GetStateValidity.Request()
            rs = RobotState()
            rs.joint_state.name = list(ALL_JOINT_NAMES)
            rs.joint_state.position = [
                joints_map.get(n, 0.0) for n in ARM_JOINT_NAMES] + [float(gj)]
            rs.is_diff = True
            req.robot_state = rs
            req.group_name = 'arm'
            future = self.validity_client.call_async(req)
            self._wait_future(future, timeout_sec=1.0)
            return future.result()

        pts = joint_trajectory.points
        total_sub = 0
        for i, pt in enumerate(pts):
            wp_map = dict(zip(names, pt.positions))
            res = _probe(wp_map)
            if res is None:
                continue
            if not res.valid:
                return _TrajCheckResult(
                    ok=False, bad_wp=i, bad_subidx=None, sub_t=0.0,
                    contacts=list(res.contacts), n_wps=len(pts),
                    n_sub=total_sub)
            # Sub-segment check between this waypoint and the next one.
            if subsamples > 0 and i + 1 < len(pts):
                next_map = dict(zip(names, pts[i + 1].positions))
                for k in range(1, subsamples):
                    t = k / float(subsamples)
                    mid = {n: (1.0 - t) * wp_map.get(n, 0.0)
                              + t * next_map.get(n, 0.0)
                           for n in ARM_JOINT_NAMES}
                    mid[GRIPPER_JOINT_NAME] = (
                        (1.0 - t) * wp_map.get(GRIPPER_JOINT_NAME, gj_live)
                        + t * next_map.get(GRIPPER_JOINT_NAME, gj_live))
                    sub_res = _probe(mid)
                    total_sub += 1
                    if sub_res is None:
                        continue
                    if not sub_res.valid:
                        return _TrajCheckResult(
                            ok=False, bad_wp=i, bad_subidx=k, sub_t=t,
                            contacts=list(sub_res.contacts),
                            n_wps=len(pts), n_sub=total_sub)
        return _TrajCheckResult(ok=True, bad_wp=None, bad_subidx=None,
                                sub_t=0.0, contacts=[], n_wps=len(pts),
                                n_sub=total_sub)

    def _cmd_validate_last_trajectory(self):
        """Re-run the sub-segment post-check on the most recently executed
        trajectory (/tmp/last_arm_trajectory.json) against the CURRENT
        /planning_scene and log detailed results.

        Purpose: close the blind spot where the pre-execution post-check
        said "all clear" but the physical motion still contacted a cup.
        This service replays the same check so we can (a) reproduce the
        verdict after the fact, (b) test with different subsample counts,
        (c) confirm the scene contains the cups we expect before blaming
        the checker.

        Reports to GUI log:
          - trajectory file path, wps, n_sub checked
          - first invalid state (if any): waypoint index + sub-t + contacts
          - full contact list written to /tmp/trajectory_validation_report.json
        """
        import json as _json
        import os as _os
        path = getattr(self, '_last_trajectory_path',
                       '/tmp/last_arm_trajectory.json')
        if not _os.path.isfile(path):
            self._append_log(f'validate_last_trajectory: no file at {path}', 'warn')
            self._cmd_error = 'no last trajectory'
            return
        try:
            dump = _json.load(open(path))
        except Exception as e:
            self._append_log(f'validate_last_trajectory: load failed: {e}', 'err')
            self._cmd_error = f'load failed: {e}'
            return

        # Rebuild a minimal JointTrajectory for the existing checker.
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        jt = JointTrajectory()
        jt.joint_names = list(dump.get('joint_names', []))
        for pt in dump.get('points', []):
            jp = JointTrajectoryPoint()
            jp.positions = [float(x) for x in pt['positions']]
            jt.points.append(jp)

        chk = self._trajectory_first_invalid_with_contacts(jt)
        report = {
            'file': path,
            'n_wps': chk.n_wps,
            'n_sub_checked': chk.n_sub,
            'subsamples_per_segment': TRAJ_SUBSEGMENT_SAMPLES,
            'ok': chk.ok,
        }
        if chk.ok:
            self._append_log(
                f'validate_last_trajectory OK: {chk.n_wps} wps + {chk.n_sub} '
                f'sub-states clear (file={_os.path.basename(path)})')
        else:
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            contacts = [{'a': c.contact_body_1, 'b': c.contact_body_2,
                         'depth_mm': float(c.depth) * 1000.0}
                        for c in chk.contacts]
            summary = '; '.join(
                f'{c["a"]}↔{c["b"]}(d={c["depth_mm"]:.1f}mm)'
                for c in contacts[:5]) or 'no contact info'
            report['first_invalid'] = {
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t, 'contacts': contacts}
            self._append_log(
                f'validate_last_trajectory FAIL: {where} '
                f'(checked {chk.n_wps} wps + {chk.n_sub} sub-states): '
                f'{summary} (file={_os.path.basename(path)})', 'warn')
        try:
            with open('/tmp/trajectory_validation_report.json', 'w') as f:
                _json.dump(report, f, indent=2)
        except Exception:
            pass

    def _execute_full_trajectory(self, joint_trajectory, on_complete_event):
        """Send a pre-validated multi-waypoint JointTrajectory to arm_controller.

        Sets on_complete_event when controller reports goal result (or on any
        failure/timeout).
        """
        n_wps = len(joint_trajectory.points) if joint_trajectory.points else 0
        tracer.event('execute_start', n_wps=n_wps, source='full_trajectory')
        # DIAG: dump every executed trajectory to disk for post-hoc replay.
        # Writes to both /tmp/arm_traj/{timestamp}.json (history, never
        # overwritten) and /tmp/last_arm_trajectory.json (convenience pointer
        # for the /validate_last_trajectory debug service). Previously the
        # history wasn't kept — a post-drop return-home overwrote the drop-
        # point trajectory before we could inspect why cups were knocked.
        try:
            import json as _json
            import os as _os
            from datetime import datetime as _dt
            # Snapshot planning-scene context so replay can reconstruct
            # EXACTLY what the post-check saw. Without this, when we replay
            # a trajectory later we're validating it against whatever the
            # scene looks like NOW, not what it looked like at plan time —
            # which is how we got stuck guessing whether attach was pending
            # or cups had been displaced prior.
            _attached = list(getattr(self, '_attached_lego_name', None) or [])
            if isinstance(_attached, str) or _attached == []:
                _attached = [self._attached_lego_name] if self._attached_lego_name else []
            with getattr(self, '_drop_lock', __import__('threading').Lock()):
                _drops = dict(getattr(self, '_drop_data', {}))
            with getattr(self, 'joint_lock', __import__('threading').Lock()):
                _joints = {n: float(self._actual_positions.get(
                    n, self.joint_positions.get(n, 0.0)))
                           for n in ALL_JOINT_NAMES}
            _dump = {
                'joint_names': list(joint_trajectory.joint_names),
                'points': [
                    {'t': p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                     'positions': list(p.positions)}
                    for p in joint_trajectory.points],
                # Reconstruction context — snapshot at execute time (the
                # post-check has just approved the plan against this scene).
                'scene_at_plan_time': {
                    'wall_clock': _dt.now().isoformat(timespec='milliseconds'),
                    'attached_lego_name': self._attached_lego_name,
                    'drop_data': _drops,
                    'actual_joints': _joints,
                    'motion_tag': getattr(self, '_last_motion_tag', 'motion'),
                },
            }
            _hist_dir = '/tmp/arm_traj'
            _os.makedirs(_hist_dir, exist_ok=True)
            _stamp = _dt.now().strftime('%Y%m%dT%H%M%S_%f')[:-3]
            _tag = getattr(self, '_last_motion_tag', 'motion')
            _hist_path = _os.path.join(_hist_dir, f'{_stamp}_{_tag}.json')
            with open(_hist_path, 'w') as _f:
                _json.dump(_dump, _f, default=str)
            with open('/tmp/last_arm_trajectory.json', 'w') as _f:
                _json.dump(_dump, _f, default=str)
            self._last_trajectory_path = _hist_path
            # One compact line to rosout so post-run grep can locate this
            # trajectory instantly by tag + timestamp.
            try:
                self.get_logger().info(
                    f'TRAJ_DUMP path={_hist_path} tag={_tag} '
                    f'attached={self._attached_lego_name or "none"} '
                    f'n_wps={len(joint_trajectory.points)}')
            except Exception:
                pass
        except Exception:
            pass
        if not self.arm_action_client.server_is_ready():
            self._append_log(
                'arm_controller action server not ready', 'warn')
            tracer.event('execute_done', outcome='action_server_not_ready')
            self._last_motion_status = {
                'ok': False, 'outcome': 'action_server_not_ready',
                'msg': 'arm_controller action server not ready'}
            on_complete_event.set()
            return

        # Cancel any previous goal
        with self._arm_goal_lock:
            if self._arm_goal_handle is not None:
                try:
                    self._arm_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._arm_goal_handle = None

        # Update the UI joint state to the final target so sliders reflect
        # reality post-motion (controller handles interpolation).
        final = joint_trajectory.points[-1]
        final_positions = {n: p for n, p in
                           zip(joint_trajectory.joint_names, final.positions)}
        with self.joint_lock:
            for n in ARM_JOINT_NAMES:
                if n in final_positions:
                    self.joint_positions[n] = final_positions[n]
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_arm_sliders, dict(final_positions))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory

        def _on_result(result_future):
            try:
                res = result_future.result()
                ec = getattr(getattr(res, 'result', None), 'error_code', None)
                tracer.event('execute_done',
                             outcome='completed',
                             error_code=(ec if ec is not None else None))
                # FollowJointTrajectory.Result.SUCCESSFUL = 0
                ok = (ec is None) or (ec == 0)
                self._last_motion_status = {
                    'ok': ok,
                    'outcome': 'completed' if ok else 'trajectory_error',
                    'msg': (f'trajectory complete ({n_wps} wps)' if ok
                            else f'trajectory error_code={ec}'),
                    'error_code': ec}
            except Exception as e:
                tracer.event('execute_done', outcome='result_exception',
                             error=repr(e))
                self._last_motion_status = {
                    'ok': False, 'outcome': 'result_exception',
                    'msg': f'result callback exception: {e!r}'}
            on_complete_event.set()

        def _on_accept(send_future):
            try:
                gh = send_future.result()
            except Exception as e:
                tracer.event('execute_done', outcome='send_exception',
                             error=repr(e))
                self._last_motion_status = {
                    'ok': False, 'outcome': 'send_exception',
                    'msg': f'action send_goal exception: {e!r}'}
                on_complete_event.set()
                return
            if not gh.accepted:
                tracer.event('execute_done', outcome='goal_rejected')
                self._last_motion_status = {
                    'ok': False, 'outcome': 'goal_rejected',
                    'msg': 'arm action server rejected the goal'}
                on_complete_event.set()
                return
            with self._arm_goal_lock:
                self._arm_goal_handle = gh
            rf = gh.get_result_async()
            rf.add_done_callback(_on_result)

        sf = self.arm_action_client.send_goal_async(goal)
        sf.add_done_callback(_on_accept)

    # ------------------------------------------------------------------
    # Hot-reload services (so agents can trigger reload without GUI focus)
    # ------------------------------------------------------------------

    def _cmd_hot_reload(self):
        """Hot-reload logic only (same as Ctrl+R). Updates methods + constants."""
        self._hot_reload_logic()

    def _cmd_hot_reload_gui(self):
        """Hot-reload logic + rebuild GUI (same as Ctrl+Shift+R)."""
        self._hot_reload_gui()

    def _build_dump_services_output(self):
        """Build the markdown button↔service mapping + service inventory.

        Used by both:
          - _srv_dump_services (Trigger service — returns the string in response.message)
          - scripts/regen_agent_debug_guide.sh (pastes the string into docs/AGENT_DEBUG_GUIDE.md)
        """
        lines = []
        lines.append('# Button ↔ Service Mapping (auto-generated from dump_services)')
        lines.append('')
        lines.append('> Regenerated from control_gui.py via '
                     '`ros2 service call /so_arm101_control_gui/dump_services '
                     'std_srvs/srv/Trigger {}`. Do NOT hand-edit — rerun the '
                     'service after code changes.')
        lines.append('')

        cmd_methods = {n for n in dir(self) if n.startswith('_cmd_')}
        wrapper_methods = {n for n in dir(self)
                           if '_btn_' in n and n.startswith('_')}

        # Table 1 — Button map
        lines.append('## Buttons')
        lines.append('')
        lines.append('| Tab | Section | Button | Bound Method | Registered Service | Mapping OK |')
        lines.append('|-----|---------|--------|--------------|--------------------|------------|')
        bound_cmds = set()
        for entry in self._button_registry:
            cmd_name = entry.get('command_name', '')
            tab = entry.get('tab') or '-'
            section = entry.get('section') or '-'
            text = entry.get('text', '')
            if cmd_name.startswith('_cmd_') and cmd_name in cmd_methods:
                svc = '~/' + cmd_name[5:]
                ok = '✓'
                bound_cmds.add(cmd_name)
            elif cmd_name in wrapper_methods:
                svc = '(wrapper)'
                ok = 'WRAPPER'
            else:
                svc = '-'
                ok = '❌ INLINE LAMBDA'
            lines.append(f'| {tab} | {section} | {text} | `{cmd_name}` | {svc} | {ok} |')

        # Table 2 — Services without buttons (agent-only)
        lines.append('')
        lines.append('## Agent-only services (no button binding)')
        lines.append('')
        lines.append('| Service | Method |')
        lines.append('|---------|--------|')
        for name in sorted(cmd_methods):
            if name in bound_cmds:
                continue
            lines.append(f'| `~/{name[5:]}` | `{name}` |')

        return '\n'.join(lines)

    def _srv_dump_services(self, request, response):
        """Manual service — returns the button↔service mapping as markdown."""
        try:
            response.message = self._build_dump_services_output()
            response.success = True
        except Exception as e:
            response.message = f'dump_services error: {e}'
            response.success = False
        return response

    # ------------------------------------------------------------------
    # Drop motion commands (auto-registered as ~/drop_refresh, ~/drop_select,
    # ~/drop_point, ~/drop_sweep, ~/drop_release Trigger services)
    # ------------------------------------------------------------------

    def _drop_btn_update_topic(self):
        """GUI-only wrapper: read the drop topic entry and switch subscription.

        This is a thin _*_btn_* wrapper (not auto-registered as a service)
        because the agent-callable path for changing topics is the parameter
        + _cmd_drop_refresh pattern, not a Trigger service.
        """
        topic = self._drop_topic_var.get().strip()
        self._update_drop_topic(topic)

    def _cmd_drop_refresh(self):
        """Resync cup state from /drop_poses. Mirrors _cmd_grasp_refresh:
        remove current collision objects, schedule repopulate + re-add +
        marker republish at fixed delays. Never clears _drop_data — the
        /drop_poses subscription refills it at ~12 Hz so ≥6 messages have
        arrived by the 500 ms scheduled read.

        Forces sub topics back to sim before pulling so a prior Real Test
        session can't leak /drop_poses_real data into the sim cup
        collision scene.
        """
        if not hasattr(self, '_drop_listbox'):
            return
        if tracer.is_active():
            tracer.close_cycle('user_canceled',
                               note='drop_refresh while cycle open')
        self._sim_ensure_sim_topics()
        self._remove_cup_collision_objects()
        if getattr(self, '_gui_ready', False):
            self.root.after(500, self._populate_drop_list)
            self.root.after(700, self._add_cup_collision_objects)
            self.root.after(900, self._refresh_display_markers)

    def _cmd_drop_select(self):
        """Select a drop target by name (via ik_target param).
        Usage: ros2 param set ... ik_target "drop_1" then call this service.
        Errors if no name provided (matches GUI behavior — must explicitly choose).
        """
        if not hasattr(self, '_drop_listbox') or self._drop_listbox.size() == 0:
            self._append_log('No drop targets to select', 'warn')
            return
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        if not name_hint or not name_hint.startswith('drop_'):
            self._append_log(
                'No drop target specified. Set param first: '
                'ros2 param set /so_arm101_control_gui ik_target "drop_1"',
                'warn')
            return
        for i in range(self._drop_listbox.size()):
            entry = self._drop_listbox.get(i)
            entry_name = entry.split(' [')[0]
            if entry_name == name_hint:
                self._drop_listbox.selection_clear(0, tk.END)
                self._drop_listbox.selection_set(i)
                self._append_log(f'Selected drop target: {self._drop_listbox.get(i)}')
                return
        self._append_log(f'Drop target {name_hint!r} not found in list', 'warn')

    def _cmd_drop_point(self):
        """Rotate shoulder_pan to face the selected drop target. ARM-02.

        Phase 9: now routes through the validated motion primitive (tier1
        linear-interp + OMPL fallback) instead of raw _execute_trajectory.
        The previous implementation dispatched joint targets directly to
        the controller with ZERO collision validation — which allowed the
        usb_camera link to plow through the cup rim at large wrist_flex
        angles. With the held-lego world-pose tracker in place, this now
        also validates the carried block's swept volume against cups/
        other legos/ground during the pan rotation.
        """
        result = self._get_selected_drop_pose()
        if result is None:
            return
        name, x, y, z = result

        evt = threading.Event()
        self._motion_event = evt


        from so_arm101_control.compute_workspace import X_PAN
        pan = math.atan2(-y, x - X_PAN)
        with self.joint_lock:
            current = dict(self.joint_positions)
        target = dict(current)
        target['shoulder_pan'] = pan
        # Drop Point holds wrist_roll at grasp_home's default (-\u03c0/2 + URDF_PITCH).
        # Previously this read self._drop_wrist_roll_sign and flipped to \u00b1\u03c0/2
        # for Real Test's ArUco scan orientation, but Drop Scan (Real Test) now
        # has its own dedicated function (_cmd_real_drop_scan) with a hardcoded
        # SCAN_WRIST_ROLL_RAD. So Quickstart's Drop Point can stay at the
        # canonical home wrist_roll \u2014 saves a ~177\u00b0 wrist swing per cycle
        # that was pure scan-artifact in sim mode.
        from so_arm101_control.compute_workspace import WRIST_ROLL_URDF_PITCH
        wrist_roll_target = -math.pi / 2 + WRIST_ROLL_URDF_PITCH
        target['wrist_roll'] = wrist_roll_target
        self._append_log(
            f'Drop Point: pan={math.degrees(pan):.1f}\u00b0 '
            f'wrist_roll={math.degrees(wrist_roll_target):+.1f}\u00b0 toward {name}')
        # tier1 linear-interp + OMPL fallback. Validates every waypoint
        # against cups, world legos, ground, AND the held-lego world copy.
        # duration 3.0 s: matches grasp_home / grasp_move; the prior 1.0 s
        # default put pan rotation at ~107°/s (1.87 rad/s) — beyond the PD
        # drive's tracking envelope per the forum-findings note, and a key
        # cause of lego-vs-cup contact during the pan-to-cup sweep with a
        # carried block.
        self._joint_space_collision_free_execute(
            target, on_complete_event=evt, duration_s=3.0)

    def _cmd_drop_sweep(self):
        """IK-planned drop sweep: geometric IK → collision check → MoveIt path.

        Phase 9: the IK target represents the **jaw-gap center** (where the
        held block actually sits), not tcp_link. Previously we passed the
        cup's (x, y, z+127mm) directly as tcp_link target, which placed the
        fixed jaw tip above the cup — but the block was offset by ~half the
        jaw gap from tcp. The dropped block missed the cup center.

        Now: we derive the current jaw gap from gripper_joint (the same
        linear model _gripper_angle_for_object uses), compute the offset
        from gap-center to tcp (fixed-jaw direction, same convention as
        _compute_jaw_offset), and shift the IK target accordingly. Net
        effect: the gap center (block center) ends up at the cup target.
        """
        result = self._get_selected_drop_pose()
        if result is None:
            return
        name, x, y, z = result

        evt = threading.Event()
        self._motion_event = evt


        # Drop target for the BLOCK CENTER at (cup_rim_height + hover).
        # /drop_poses publishes cup BODY-CENTER (z = cup_base + half_height),
        # so rim is at z + half_height and the block target is
        # z + half_height + HOVER_ABOVE_RIM. Default raised 30 → 50 mm so the
        # attached block clears the cup wall during the wrist_flex sweep
        # (previously OMPL papered over 1-3 mm mid-trajectory cup clips via
        # tortuous paths; deterministic planner refuses those correctly, so
        # we need actual geometric clearance).
        hover = getattr(self, '_drop_hover_above_rim_var', None)
        hover_m = hover.get() if hover is not None else 0.050
        gap_x, gap_y = float(x), float(y)
        gap_z = float(z) + CUP_BODY_HEIGHT_M / 2.0 + hover_m

        # Convert block-center target → tcp_link target by offsetting along
        # the fixed-jaw direction. At grip_angle=π/4 and wrist_roll=-π/2, FK
        # across the reachable pan range (-110° to +124°) shows the gripper's
        # local jaw-opening axis projects to (+sin(pan), +cos(pan)) in world
        # with dot = +1.000 everywhere — block→fixed_jaw direction.
        #
        # Shift magnitude: use the MEASURED |ax| from the physical attach
        # (block_in_tcp.x, set by _attach_lego_to_gripper). Fallback to the
        # theoretical half_gap only when no attach offset is available (e.g.
        # dry-run drop_sweep with empty gripper). The measured offset is what
        # the planning scene sees for collision, so matching it eliminates
        # the 1-2 mm target-vs-reality drift that surfaced as cup-wall
        # penetration under the deterministic planner.
        from so_arm101_control.compute_workspace import X_PAN
        pan = math.atan2(-gap_y, gap_x - X_PAN)
        with self.joint_lock:
            gj = self._actual_positions.get(
                GRIPPER_JOINT_NAME,
                self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0))
        jaw_gap = BASELINE_JAW_GAP + JAW_GAP_RATE * gj
        half_gap = jaw_gap / 2.0
        attach_offset = getattr(self, '_attached_lego_tcp_offset', None)
        if attach_offset is not None:
            # ax is negative (block sits at -tcp_x from tcp_link origin).
            # -ax is the positive distance block→tcp along jaw-opening axis.
            shift_mag = -float(attach_offset[0])
            shift_source = 'measured'
        else:
            shift_mag = half_gap
            shift_source = 'theoretical'
        dx = shift_mag * math.sin(pan)
        dy = shift_mag * math.cos(pan)
        target_x = gap_x + dx
        target_y = gap_y + dy
        target_z = gap_z  # Z unchanged — gap axis is horizontal

        tracer.event('drop_sweep_start',
                     drop_name=name,
                     gap_target=(gap_x, gap_y, gap_z),
                     tcp_target=(target_x, target_y, target_z),
                     jaw_gap_mm=jaw_gap * 1000,
                     gripper_joint=gj)
        tracer.snapshot_scene('drop_sweep_start', self)
        def _close_on_drop_done():
            evt.wait(timeout=30.0)
            tracer.event('drop_sweep_done')
        threading.Thread(target=_close_on_drop_done, daemon=True).start()
        self._append_log(
            f'Drop Sweep: gap=({gap_x:.3f},{gap_y:.3f},{gap_z:.3f})m '
            f'→ tcp=({target_x:.3f},{target_y:.3f},{target_z:.3f}) '
            f'(shift={shift_mag*1000:.1f}mm [{shift_source}], '
            f'hover={hover_m*1000:.0f}mm, jaw_gap={jaw_gap*1000:.1f}mm)')

        grip_deg = getattr(self, '_drop_grip_angle_var', None)
        grip_angle = math.radians(grip_deg.get() if grip_deg else 45)
        # Lock shoulder_pan to the post-drop_point physical pan so the
        # sweep is a pure wrist-tilt, not "tilt + 1° base yaw drift".
        # See _plan_collision_free_execute(lock_pan=...) docstring.
        with self.joint_lock:
            locked_pan = self._actual_positions.get(
                'shoulder_pan',
                self.joint_positions.get('shoulder_pan', 0.0))
        # Sweep duration from the GUI spinbox (Grasp tab → Drop section).
        # 3.0 s default matches drop_point / grasp_home / grasp_move so
        # the whole pick-drop cycle has consistent joint velocities. User
        # can raise via the spinbox if a particular cup/lego combo needs
        # gentler placement.
        sweep_dur_var = getattr(self, '_drop_duration_var', None)
        sweep_dur = float(sweep_dur_var.get()) if sweep_dur_var else 3.0
        self._plan_collision_free_execute(
            target_x, target_y, target_z,
            grip_angle=grip_angle,
            wrist_roll=-math.pi / 2,
            on_complete=evt,
            lock_pan=locked_pan,
            duration_s=sweep_dur)

    def _cmd_drop_release(self):
        """Open gripper to release held object into cup. ARM-04.

        SYNCHRONOUS: does not return until (a) gripper-open motion has
        completed, and (b) the attached-collision-object has been removed
        from the planning scene. This is required so the subsequent
        grasp_home plans against a scene without the phantom attached
        block — the async version raced with grasp_home's OMPL call and
        caused deterministic Mode A failures (ec=-2) on the return leg.
        """
        last = getattr(self, '_last_grasped_object', None)
        tracer.event('drop_release_start', last_grasped=last)
        if last and hasattr(self, 'obj_listbox'):
            for i in range(self.obj_listbox.size()):
                if self.obj_listbox.get(i).split('  ')[0] == last:
                    self.obj_listbox.selection_clear(0, tk.END)
                    self.obj_listbox.selection_set(i)
                    break
        self._cmd_gripper_open_for_object()
        # Wait for the gripper-open motion to complete. _cmd_gripper_open_for_object
        # assigned a fresh Event to self._motion_event — capture it before waiting
        # so a later trigger-service callback that clears the attribute can't
        # swap it out from under us.
        evt = getattr(self, '_motion_event', None)
        if evt is not None:
            evt.wait(timeout=10.0)
        # Synchronously detach the AttachedCollisionObject. Returning from
        # this service guarantees the next grasp_home sees a clean scene.
        self._ensure_lego_state()
        if self._attached_lego_name:
            try:
                self._detach_lego_sync()
            except Exception as exc:
                self._append_log(f'Detach step errored: {exc}', 'warn')

    def _cmd_mtc_pick_place(self):
        """MTC pick-and-place spike. Runs full cycle for hardcoded target pair
        (green_2x3_1 -> drop_0) via the so_arm101_mtc C++ node.
        Requires control.launch.py mtc:=true. See /home/aaugus11/.claude/plans/
        abstract-swinging-sundae.md for the hypothesis being validated.
        """
        self._ensure_lego_state()
        if self._attached_lego_name:
            self._append_log(
                f'MTC refusing to run: lego "{self._attached_lego_name}" is already attached. '
                'Call /drop_release first.', 'warn')
            return
        # Refresh scene so MTC sees current lego + cup collision objects.
        try:
            self._add_lego_collision_objects()
        except Exception as exc:
            self._append_log(f'MTC pre-step (lego scene refresh) errored: {exc}', 'warn')
        try:
            self._add_cup_collision_objects()
        except Exception as exc:
            self._append_log(f'MTC pre-step (cup scene refresh) errored: {exc}', 'warn')

        client = getattr(self, '_mtc_run_client', None)
        if client is None:
            self._append_log('MTC client not initialised (launch with mtc:=true).', 'err')
            return
        if not client.wait_for_service(timeout_sec=2.0):
            self._append_log(
                'MTC service /so_arm101_mtc/run unavailable — is the node running '
                '(launch with mtc:=true)?', 'err')
            return

        self._append_log('MTC: calling /so_arm101_mtc/run ...')
        future = client.call_async(Trigger.Request())
        deadline = time.time() + 60.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            self._append_log('MTC: timeout waiting for /so_arm101_mtc/run', 'err')
            return
        try:
            resp = future.result()
        except Exception as exc:
            self._append_log(f'MTC call failed: {exc}', 'err')
            return
        level = 'info' if resp.success else 'err'
        self._append_log(f'MTC result: success={resp.success} msg="{resp.message}"', level)

    def _cmd_grasp_select(self):
        """Select an object in the listbox by name (via ik_target param) or first item.
        Usage: ros2 param set ... ik_target "green_1" then call this service.
        """
        if not hasattr(self, 'obj_listbox') or self.obj_listbox.size() == 0:
            self._append_log('No objects to select', 'warn')
            return
        # Check if a name was specified via the ik_target parameter
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        target_idx = 0
        if name_hint and '=' not in name_hint:
            for i in range(self.obj_listbox.size()):
                if self.obj_listbox.get(i).split('  ')[0] == name_hint:
                    target_idx = i
                    break
        self.obj_listbox.selection_clear(0, tk.END)
        self.obj_listbox.selection_set(target_idx)
        text = self.obj_listbox.get(target_idx)
        self._append_log(f'Selected: {text.split("  ")[0]}')

    @classmethod
    def _grasp_orientation(cls, obj_x, obj_y, obj_qz, obj_qw):
        """Compute a top-down grasp quaternion for a table object.
        The gripper approaches from above (pitch=90°), with yaw aligned
        to the pan angle toward the object plus the object's z-rotation.
        Returns (qx, qy, qz, qw) for the EE in the base frame.
        """
        pan = math.atan2(obj_y, obj_x) if abs(obj_x) + abs(obj_y) > 0.001 else 0.0
        obj_yaw = math.atan2(2.0 * obj_qw * obj_qz, 1.0 - 2.0 * obj_qz * obj_qz)
        yaw_deg = math.degrees(pan + obj_yaw)
        return cls._rpy_deg_to_quat(0.0, 90.0, yaw_deg)

    def _get_selected_object_name(self):
        """Return the object name selected in the grasp listbox, or None."""
        sel = self.obj_listbox.curselection()
        if not sel:
            return None
        return self.obj_listbox.get(sel[0]).split('  ')[0]

    def _get_grip_width(self, obj_name):
        """Return the grip width for the object, respecting cross-axis checkbox."""
        bbox = self._lookup_bbox(obj_name)
        if not bbox:
            return None
        cross = self._grasp_cross_var.get() if hasattr(self, '_grasp_cross_var') else False
        return max(bbox['sx'], bbox['sy']) if cross else min(bbox['sx'], bbox['sy'])

    def _compute_jaw_offset(self, obj_name, obj_yaw):
        """Compute (dx, dy) to shift TCP so the object center sits between jaws.

        TCP is at the fixed jaw tip. The moving jaw opens along the object's
        local +Y axis: (-sin(obj_yaw), cos(obj_yaw)) in world frame.
        (wrist_roll compensates for pan, so only obj_yaw matters.)
        We shift TCP from object center toward the fixed jaw by grip_width/2.
        Returns (0, 0) when TCP offset is disabled via the checkbox.
        """
        if hasattr(self, '_bbox_enabled_var') and not self._bbox_enabled_var.get():
            return 0.0, 0.0
        grip_width = self._get_grip_width(obj_name)
        if grip_width is None:
            return 0.0, 0.0
        tcp_clear = (self._tcp_clearance_var.get() / 1000
                     if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M)
        half_offset = grip_width / 2 + tcp_clear
        # fixed_jaw_dir = (-sin(obj_yaw), +cos(obj_yaw))
        dx = -half_offset * math.sin(obj_yaw)
        dy = half_offset * math.cos(obj_yaw)
        return dx, dy

    def _gripper_angle_for_object(self, obj_name):
        """Return (open_angle, close_angle) in radians for the selected object.

        Uses jaw_gap = BASELINE_JAW_GAP + JAW_GAP_RATE * angle model derived
        from STL mesh analysis. Falls back to full range if no bbox data.
        """
        grip_width = self._get_grip_width(obj_name) if obj_name else None
        if grip_width is None:
            return JOINT_LIMITS['gripper_joint'][1], JOINT_LIMITS['gripper_joint'][0]
        # Read tunable values from UI (mm → m), fall back to module constants
        tcp_clear = (self._tcp_clearance_var.get() / 1000
                     if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M)
        open_cl = (self._jaw_open_clearance_var.get() / 1000
                   if hasattr(self, '_jaw_open_clearance_var') else JAW_OPEN_CLEARANCE_M)
        close_cl = (self._jaw_close_clearance_var.get() / 1000
                    if hasattr(self, '_jaw_close_clearance_var') else JAW_CLOSE_CLEARANCE_M)
        # Symmetric baseline = grip_width + 2*tcp_clearance (tcp_clear gap each side)
        # Open/close clearances are extra gap on top of that baseline
        # angle = (desired_gap - baseline_jaw_gap) / rate
        open_gap = grip_width + 2 * tcp_clear + open_cl
        open_angle = (open_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        open_angle = max(JOINT_LIMITS['gripper_joint'][0],
                         min(open_angle, JOINT_LIMITS['gripper_joint'][1]))
        close_gap = grip_width + 2 * tcp_clear + close_cl
        close_angle = (close_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        close_angle = max(JOINT_LIMITS['gripper_joint'][0],
                          min(close_angle, JOINT_LIMITS['gripper_joint'][1]))
        return open_angle, close_angle

    def _cmd_gripper_open_for_object(self):
        """Open gripper to the angle matching the selected object's width.
        After motion completes, switches the planning group back to 'arm'
        so the next arm command renders correctly in RViz."""
        obj_name = self._get_selected_object_name()
        bbox = self._lookup_bbox(obj_name) if obj_name else None
        if not bbox:
            self._append_log('Grasp Open: no object selected or no bbox data')
            return
        open_angle, _ = self._gripper_angle_for_object(obj_name)
        duration = self._grasp_grip_duration_var.get()
        self._append_log(
            f'Grasp Open: {math.degrees(open_angle):.1f}° for {obj_name}')
        # Update slider/UI on tkinter thread, send goal on background thread
        self._gripper_command(open_angle, execute=False, duration_s=duration)
        evt = threading.Event()
        self._motion_event = evt
        def _send():
            self._send_gripper_goal(open_angle, duration_s=duration, blocking=True)
            evt.set()
            self._select_planning_group('arm')
        threading.Thread(target=_send, daemon=True).start()

    def _cmd_gripper_close_for_object(self):
        """Close gripper to the object's width minus threshold.
        On success, attach the object to tcp_link as an AttachedCollisionObject
        so MoveIt plans subsequent motions aware of the carried-block envelope.
        """
        obj_name = self._get_selected_object_name()
        bbox = self._lookup_bbox(obj_name) if obj_name else None
        if not bbox:
            self._append_log('Grasp Close: no object selected or no bbox data')
            return
        _, close_angle = self._gripper_angle_for_object(obj_name)
        duration = self._grasp_grip_duration_var.get()
        self._append_log(f'Grasp Close: {math.degrees(close_angle):.1f}° for {obj_name}')
        self._gripper_command(close_angle, execute=False, duration_s=duration)
        evt = threading.Event()
        self._motion_event = evt
        def _send():
            ok = self._send_gripper_goal(close_angle, duration_s=duration, blocking=True)
            # Phase 9: attach to tcp_link after physical closure succeeds
            if ok is not False:
                try:
                    self._attach_lego_to_gripper(obj_name)
                except Exception as exc:
                    self._append_log(f'Attach step errored: {exc}', 'warn')
            evt.set()
        threading.Thread(target=_send, daemon=True).start()

    def _cmd_grasp_move(self):
        sel = self.obj_listbox.curselection()
        if not sel:
            self._append_log('No object selected', 'warn')
            return
        text = self.obj_listbox.get(sel[0])
        obj_name = text.split('  ')[0]
        # Remember the grasped object so drop_release can open for it later
        self._last_grasped_object = obj_name
        # Forensic trace: open a new cycle scoped grasp_move -> grasp_home.
        # If a prior cycle is still open, it's force-closed with outcome='abandoned'.
        with self.joint_lock:
            _entry_joints = {n: self._actual_positions.get(
                n, self.joint_positions.get(n, 0.0)) for n in ARM_JOINT_NAMES}
        tracer.open_cycle(
            obj_name=obj_name,
            grasp_topic=self._grasp_topic_var.get().strip(),
            cross=bool(self._grasp_cross_var.get()),
            entry_joints=_entry_joints,
        )
        # Reset detach-seen flag for the new cycle. Cycle closes on the
        # first grasp_home_done AFTER detach (post-drop return).
        self._cycle_detach_seen = False
        tracer.snapshot_scene('grasp_move_start', self)
        # Phase 9: remove the target-being-grasped from the world collision
        # scene so MoveIt's FC-1 reachability check doesn't flag 'gripper vs
        # target-lego' as a collision (the gripper is SUPPOSED to touch it).
        # If grasp_move ultimately fails, the next _objects_callback (>3mm
        # movement in sim) will trigger _add_lego_collision_objects and
        # restore it; for same-scene failures we also trigger a resync in
        # the failure paths below.
        self._ensure_lego_state()
        self._remove_single_lego_from_world(obj_name)
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return

        topic = self._grasp_topic_var.get().strip()
        z_offset = 0.05 if topic == '/drop_poses' else 0.0

        # Grasp TCP target Z = block center + half_height (→ block top)
        # + _GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M (gripper-geometry clearance
        # so the gripper link's downward-facing collision mesh does not
        # tangent-penetrate the block's world collision object at grasp
        # pose). See _GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M definition for
        # rationale. pos['z'] is the block bbox-center published by
        # /objects_poses_sim (post 2026-04-22 lego USD origin fix).
        obj_z_override = self._grasp_obj_z_var.get()
        if abs(obj_z_override) > 1e-4:
            base_z = obj_z_override
        else:
            bbox_sz = (self._lookup_bbox(obj_name) or {}).get('sz', 0.0)
            base_z = (pos['z']
                      + float(bbox_sz) / 2.0
                      + _GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M)
        target_z = base_z + z_offset
        action = 'drop' if topic == '/drop_poses' else 'grab'

        # Compute object yaw for wrist_roll alignment (two-stage IK)
        obj_qz = pos.get('qz', 0.0)
        obj_qw = pos.get('qw', 1.0)
        obj_yaw = math.atan2(2.0 * obj_qw * obj_qz, 1.0 - 2.0 * obj_qz * obj_qz)

        # Minor-axis (cross) grasp: rotate 90° to close across the short axis
        cross = self._grasp_cross_var.get()
        if cross:
            obj_yaw += math.pi / 2

        # Normalize yaw so wrist_roll stays within joint limits
        # (gripper jaws are symmetric: yaw ≡ yaw+π for grasping)
        pan = math.atan2(pos['y'], pos['x']) if abs(pos['x']) + abs(pos['y']) > 0.001 else 0.0
        obj_yaw = _normalize_grasp_yaw(obj_yaw, pan)

        # Jaw offset: shift target so object sits between both jaws
        jaw_dx, jaw_dy = self._compute_jaw_offset(obj_name, obj_yaw)
        tx, ty = pos['x'] + jaw_dx, pos['y'] + jaw_dy
        if abs(jaw_dx) > 0.001 or abs(jaw_dy) > 0.001:
            gw = (self._get_grip_width(obj_name) or 0) * 1000
            self._append_log(
                f'  Jaw offset: ({jaw_dx*1000:+.1f}, {jaw_dy*1000:+.1f})mm '
                f'for {gw:.0f}mm grip{"[cross]" if cross else ""}')

        self._append_log(
            f'Grasp: {action} "{obj_name}" at '
            f'({tx:.3f}, {ty:.3f}, {target_z:.3f})'
            f'{" [cross]" if cross else ""}')

        self._grasp_arm_duration = self._grasp_arm_duration_var.get()
        # Compute top-down grasp orientation (pitch=90°, yaw aligned to object)
        gqx, gqy, gqz, gqw = self._grasp_orientation(
            tx, ty, obj_qz, obj_qw)

        approach_h = self._grasp_approach_height_var.get()

        # Create motion event BEFORE spawning thread so trigger callback can see it
        evt = threading.Event()
        self._motion_event = evt

        # Pre-validate ALL stages before moving the arm
        def _prevalidate_and_execute():
            ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None

            poses_to_check = []
            if approach_h > 0:
                approach_z = target_z + approach_h
                poses_to_check.append(('approach', tx, ty, approach_z))
                poses_to_check.append(('final', tx, ty, target_z))
            else:
                poses_to_check.append(('final', tx, ty, target_z))

            # Workspace bbox check first — fast reject for clearly-out-of-reach.
            gate_a_results = []
            for stage, px, py, pz in poses_to_check:
                ok, reason = check_grasp_reachable(px, py, pz, ground_z=ground_z)
                gate_a_results.append({
                    'stage': stage, 'x': px, 'y': py, 'z': pz,
                    'ok': ok, 'reason': reason,
                })
                if not ok:
                    self._append_log(
                        f'Grasp rejected ({stage}): {reason} '
                        f'({px:.3f}, {py:.3f}, {pz:.3f})', 'warn')
                    # Phase 9: grasp failed — restore lego to world scene
                    self._add_lego_collision_objects()
                    tracer.event('gate_a_done', per_stage=gate_a_results)
                    tracer.close_cycle('grasp_unreachable_workspace',
                                       stage=stage, reason=reason,
                                       target=(px, py, pz))
                    self._last_motion_status = {
                        'ok': False, 'outcome': 'grasp_unreachable_workspace',
                        'msg': (f'Gate A reject @ {stage} '
                                f'({px:.3f}, {py:.3f}, {pz:.3f}): {reason}')}
                    evt.set()
                    return
            tracer.event('gate_a_done', per_stage=gate_a_results)

            # Find a yaw where geometric_ik returns solutions for BOTH stages.
            # Using a single yaw across stages avoids mid-grasp wrist twists.
            yaw_used, stage_sols, dbg_lines = find_reachable_grasp_yaw(
                poses_to_check, obj_yaw)
            tracer.event('gate_b_done',
                         yaw_requested=obj_yaw,
                         yaw_used=yaw_used,
                         stage_sols={k: [dict(s) for s in v]
                                     for k, v in (stage_sols or {}).items()},
                         dbg_lines=list(dbg_lines or []))
            if yaw_used is None:
                self._append_log(
                    f'Grasp unreachable: no geometric IK at any yaw for '
                    f'{[s[0] for s in poses_to_check]} stages', 'warn')
                for line in dbg_lines:
                    self._append_log(line, 'warn')
                self._add_lego_collision_objects()
                tracer.close_cycle('grasp_unreachable_ik',
                                   dbg_lines=list(dbg_lines or []))
                self._last_motion_status = {
                    'ok': False, 'outcome': 'grasp_unreachable_ik',
                    'msg': (f'Gate B reject: no geometric IK at any yaw for '
                            f'stages {[s[0] for s in poses_to_check]}')}
                evt.set()
                return
            if yaw_used != obj_yaw:
                self._append_log(
                    f'Grasp yaw fallback: '
                    f'{math.degrees(obj_yaw):.1f}° → '
                    f'{math.degrees(yaw_used):.1f}° '
                    f'(Δ{math.degrees(yaw_used - obj_yaw):+.1f}°)')

            # Collision-check each stage's solutions, stop at first valid.
            validated = {}
            gate_c_attempts = []
            for stage, px, py, pz in poses_to_check:
                found = False
                for i, sol in enumerate(stage_sols[stage]):
                    config = 'elbow-up' if i == 0 else 'elbow-down'
                    valid = self._check_state_valid(sol)
                    gate_c_attempts.append({
                        'stage': stage, 'config': config,
                        'sol': dict(sol), 'valid': bool(valid),
                    })
                    if valid:
                        validated[stage] = sol
                        self._append_log(
                            f'  {stage}: {config}, '
                            f'wrist_roll={math.degrees(sol["wrist_roll"]):.1f}°')
                        found = True
                        break
                    self._append_log(
                        f'  {stage}: {config} collides', 'warn')
                if not found:
                    self._append_log(
                        f'Grasp unreachable ({stage}): all solutions collide '
                        f'({px:.3f}, {py:.3f}, {pz:.3f})', 'warn')
                    self._add_lego_collision_objects()
                    tracer.event('gate_c_done',
                                 attempts=gate_c_attempts,
                                 validated={k: dict(v) for k, v in validated.items()})
                    tracer.close_cycle('gate_c_all_collide',
                                       stage=stage, target=(px, py, pz))
                    self._last_motion_status = {
                        'ok': False, 'outcome': 'gate_c_all_collide',
                        'msg': (f'Gate C reject @ {stage} '
                                f'({px:.3f}, {py:.3f}, {pz:.3f}): '
                                f'all IK solutions collide')}
                    evt.set()
                    return
            tracer.event('gate_c_done',
                         attempts=gate_c_attempts,
                         validated={k: dict(v) for k, v in validated.items()})

            # All stages validated — execute
            duration = self._grasp_arm_duration
            if approach_h > 0 and 'approach' in validated and 'final' in validated:
                self._append_log(f'  Both stages validated, executing approach')
                self._grasp_final_joints = validated['final']
                self._grasp_motion_event = evt
                def _apply_approach():
                    self._ik_apply_and_act(validated['approach'], 'grasp_approach')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, _apply_approach)
            elif 'final' in validated:
                self._grasp_motion_event = evt
                def _apply_final():
                    self._ik_apply_and_act(validated['final'], 'grasp_execute')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, _apply_final)
            else:
                # Validated dict has no 'final' stage — this shouldn't happen
                # if gate C passed, but guard anyway.
                self._last_motion_status = {
                    'ok': False, 'outcome': 'no_final_stage',
                    'msg': ('Gate C passed but no final-stage solution '
                            'recorded — logic bug')}
                evt.set()  # nothing to execute

        threading.Thread(target=_prevalidate_and_execute, daemon=True).start()

    # ------------------------------------------------------------------
    # Tab 3: Gripper Control
    # ------------------------------------------------------------------

    def _cmd_gripper_close(self):
        self._gripper_command(JOINT_LIMITS['gripper_joint'][0], execute=True)

    def _cmd_gripper_open(self):
        """Open gripper to JOINT_LIMITS max. Mirrors `_cmd_gripper_open_for_object`'s
        motion-event pattern (background thread, blocking goal, evt.set()) so
        callers can wait on completion. After the motion lands, switches the
        planning group back to 'arm' so RViz's MotionPlanning panel shows the
        orange arm goal-state ghost on the next arm command instead of leaving
        it stuck on the gripper group."""
        open_angle = JOINT_LIMITS['gripper_joint'][1]
        duration = 1.0
        self._append_log(f'Open Gripper: {math.degrees(open_angle):.1f}°')
        self._gripper_command(open_angle, execute=False, duration_s=duration)
        evt = threading.Event()
        self._motion_event = evt
        def _send():
            self._send_gripper_goal(open_angle, duration_s=duration, blocking=True)
            evt.set()
            self._select_planning_group('arm')
        threading.Thread(target=_send, daemon=True).start()

    def _cmd_gripper_open_range(self):
        """Open gripper to the range spinbox upper value (grasp tab)."""
        angle = math.radians(self._grasp_grip_open_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    def _cmd_gripper_close_range(self):
        """Close gripper to the range spinbox lower value (grasp tab)."""
        angle = math.radians(self._grasp_grip_close_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    def _cmd_set_jaw_open_clearance(self):
        """Set jaw open clearance: ros2 param set ... jaw_open_clearance_mm 5.0"""
        val = self.get_parameter('jaw_open_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_open_clearance_var'):
            self._jaw_open_clearance_var.set(val)
        self._append_log(f'Jaw open clearance set to {val:.1f}mm')

    def _cmd_set_jaw_close_clearance(self):
        """Set jaw close clearance: ros2 param set ... jaw_close_clearance_mm 0.0
        +ve = more gap, -ve = tighter"""
        val = self.get_parameter('jaw_close_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_close_clearance_var'):
            self._jaw_close_clearance_var.set(val)
        self._append_log(f'Jaw close clearance set to {val:+.1f}mm')

    def _cmd_set_tcp_clearance(self):
        """Set TCP IK clearance from param: ros2 param set ... tcp_clearance_mm 1.0"""
        val = self.get_parameter('tcp_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_tcp_clearance_var'):
            self._tcp_clearance_var.set(val)
        self._append_log(f'TCP clearance set to {val:.1f}mm')

    def _cmd_check_grasp_reachable(self):
        """Check if the selected object is within the top-down grasp workspace."""
        obj_name = self._get_selected_object_name()
        if not obj_name:
            self._append_log('No object selected for reachability check', 'warn')
            return
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return
        obj_z_override = self._grasp_obj_z_var.get()
        z = obj_z_override if abs(obj_z_override) > 1e-4 else pos['z']
        ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None
        ok, reason = check_grasp_reachable(pos['x'], pos['y'], z, ground_z=ground_z)
        r = math.sqrt(pos['x']**2 + pos['y']**2)
        if ok:
            self._append_log(
                f'Grasp reachable: "{obj_name}" r={r:.3f}m z={z:.3f}m '
                f'[R: {GRASP_WORKSPACE_BOUNDS["R_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["R_MAX"]:.3f}m, '
                f'Z: {GRASP_WORKSPACE_BOUNDS["Z_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["Z_MAX"]:.3f}m]')
        else:
            self._append_log(
                f'Grasp unreachable: "{obj_name}" — {reason}', 'warn')

    def _gripper_command(self, jaw_target, execute=False, duration_s=1.0):
        """Set gripper goal. If execute=True, also send to controller."""
        self._slider_driven = True
        self._select_planning_group('gripper')
        with self.joint_lock:
            self.joint_positions['gripper_joint'] = jaw_target
        if 'gripper_joint' in self.sliders:
            self.sliders['gripper_joint'].set(jaw_target)
            self.slider_labels['gripper_joint'].config(text=f'{jaw_target:.3f}')
        if hasattr(self, '_ik_jaw_label'):
            self._ik_jaw_label.config(text=f'{jaw_target:.3f}')
        self._publish_goal_state()
        if execute:
            self._send_gripper_goal(jaw_target, duration_s=duration_s)

    # ------------------------------------------------------------------
    # Trajectory execution (arm joints via action interface)
    # ------------------------------------------------------------------

    def _execute_trajectory(self, target_positions, duration_s=2.0, on_complete=None,
                            blocking=False):
        """Send trajectory to arm_controller via action interface.
        Source: trajectory logic adapted from JETANK_description/jetank_control_gui.py

        If blocking=True, sends with blocking and waits for both the
        controller result and the UI animation to finish before returning.
        Must be called from a background thread when blocking.
        """
        if not self._traj_lock.acquire(blocking=False):
            self._append_log('Trajectory already in progress', 'warn')
            return

        self._slider_driven = True

        # Send the full trajectory via action (controller handles interpolation)
        self._send_arm_goal(target_positions, duration_s=duration_s, blocking=blocking)

        # Animate the UI sliders to show progress
        with self.joint_lock:
            start = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}

        def _run():
            try:
                steps = int(duration_s * 50)
                for i in range(steps + 1):
                    if not self.running:
                        break
                    t = i / max(steps, 1)
                    t = t * t * (3.0 - 2.0 * t)

                    positions = {}
                    for name in ARM_JOINT_NAMES:
                        s = start.get(name, 0.0)
                        e = target_positions.get(name, s)
                        lo, hi = JOINT_LIMITS[name]
                        positions[name] = max(lo, min(hi, s + (e - s) * t))

                    with self.joint_lock:
                        self.joint_positions.update(positions)

                    if getattr(self, '_gui_ready', False):
                        self.root.after(0, self._sync_arm_sliders, dict(positions))

                    self._publish_goal_state()

                    if self.use_real_hardware:
                        self._send_hw_command(positions)

                    time.sleep(1.0 / 50)

                self._append_log('Trajectory complete')
                self._slider_driven = False
            finally:
                self._traj_lock.release()
            if on_complete is not None:
                on_complete()

        threading.Thread(target=_run, daemon=True).start()

    def _sync_arm_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')

    # ------------------------------------------------------------------
    # Hardware mode
    # ------------------------------------------------------------------

    def _toggle_hardware(self):
        use_real = self.hw_var.get() == 'real'
        with self.hw_lock:
            if use_real == self.use_real_hardware:
                return
            self.use_real_hardware = use_real

        if use_real:
            self._append_log('Switching to REAL hardware mode', 'warn')
            self.status_var.set('Mode: Real Hardware')
            self.set_joints_btn.config(state=tk.DISABLED)
        else:
            self._append_log('Switching to SIMULATION mode')
            self.status_var.set('Mode: Simulation')
            self.set_joints_btn.config(state=tk.NORMAL)

        # Update grasp topic to match hardware mode
        if hasattr(self, '_grasp_topic_var'):
            new_topic = '/objects_poses_real' if use_real else '/objects_poses_sim'
            self._grasp_topic_var.set(new_topic)
            self._cmd_grasp_update_topic()

    def _cmd_toggle_ground_plane(self):
        """Add/remove a ground plane collision object in MoveIt's planning scene."""
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            self._append_log('MoveIt not available — cannot update planning scene', 'warn')
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log('apply_planning_scene service not available', 'warn')
                return

            # Step 1: Add or remove the collision object
            scene = PlanningSceneMsg()
            scene.is_diff = True

            co = CollisionObject()
            co.header.frame_id = 'base'
            co.id = 'ground_plane'

            adding = self._ground_plane_var.get()
            if adding:
                co.operation = CollisionObject.ADD
                box = SolidPrimitive()
                box.type = SolidPrimitive.BOX
                box.dimensions = [2.0, 2.0, 0.01]
                co.primitives.append(box)
                pose = Pose()
                z = self._ground_z_var.get()
                pose.position.z = z - 0.005  # center of 0.01-thick box
                pose.orientation.w = 1.0
                co.primitive_poses.append(pose)
            else:
                co.operation = CollisionObject.REMOVE

            scene.world.collision_objects.append(co)
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            if future.result() is None or not future.result().success:
                action = 'add' if adding else 'remove'
                self._append_log(f'Failed to {action} ground plane', 'warn')
                return

            if not adding:
                self._append_log('Ground plane removed')
                return

            # Step 2: Allow base <-> ground_plane collision in the ACM
            # (base sits on the ground — contact is expected)
            if not self._get_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log(f'Ground plane added at z={z:.3f} (ACM not updated)', 'warn')
                return

            get_req = GetPlanningSceneSrv.Request()
            get_req.components.components = 128  # ALLOWED_COLLISION_MATRIX
            future = self._get_scene_client.call_async(get_req)
            self._wait_future(future, timeout_sec=5.0)
            if future.result() is None:
                self._append_log(f'Ground plane added at z={z:.3f} (ACM not updated)', 'warn')
                return

            acm = future.result().scene.allowed_collision_matrix
            gp_name = 'ground_plane'
            if gp_name not in acm.entry_names:
                # Add new column (False) to every existing row
                for entry in acm.entry_values:
                    entry.enabled.append(False)
                # Add new row for ground_plane
                gp_row = AllowedCollisionEntry()
                gp_row.enabled = [False] * len(acm.entry_names) + [True]  # self=True
                # Allow contact with base
                if 'base' in acm.entry_names:
                    base_idx = acm.entry_names.index('base')
                    gp_row.enabled[base_idx] = True
                    acm.entry_values[base_idx].enabled[-1] = True
                acm.entry_names.append(gp_name)
                acm.entry_values.append(gp_row)

            acm_scene = PlanningSceneMsg()
            acm_scene.is_diff = True
            acm_scene.allowed_collision_matrix = acm
            req2 = ApplyPlanningScene.Request()
            req2.scene = acm_scene
            future2 = self._apply_scene_client.call_async(req2)
            self._wait_future(future2, timeout_sec=5.0)
            if future2.result() is not None and future2.result().success:
                self._append_log(f'Ground plane added at z={z:.3f}')
            else:
                self._append_log(f'Ground plane added at z={z:.3f} (ACM update failed)', 'warn')

        threading.Thread(target=_apply, daemon=True).start()

    def _real_js_callback(self, msg):
        if not self.use_real_hardware:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_all_sliders, positions)

    def _sync_all_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                if name in self.slider_labels:
                    self.slider_labels[name].config(text=f'{val:.3f}')

    def _send_hw_command(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = list(positions.values())
        self.joint_cmd_pub.publish(msg)

    def _ext_cmd_callback(self, msg):
        """Handle external joint commands — update sliders and send to controllers."""
        self._slider_driven = True
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    lo, hi = JOINT_LIMITS.get(name, (-math.pi, math.pi))
                    self.joint_positions[name] = max(lo, min(hi, msg.position[i]))
            positions = dict(self.joint_positions)
        # External commands directly move the robot
        self._send_arm_goal(positions, duration_s=0.2)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.2)
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    node = SOArm101ControlGUI()
    # Explicit thread count — default is multiprocessing.cpu_count() which can
    # leave too few free threads when service handlers block on .wait()
    # (Phase 9 sync-refresh pattern). 4 threads is ample headroom.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # Tracks whether a prior signal already armed the force-exit timer, so
    # a repeated SIGTERM doesn't double-arm and so we can distinguish
    # "handler never fired" (cosmetic script race) from "handler fired but
    # Tk deadlocked" (real bug — force-exit wins).
    _shutdown_state = {'armed': False}

    def _shutdown_handler(signum, frame):
        node.running = False
        # Tell tkinter to quit from its own event loop. root.after(0, ...)
        # is the only thread-safe way to drive the Tk interpreter from a
        # signal handler (which runs on the main thread, not the Tk thread).
        if hasattr(node, 'root'):
            try:
                node.root.after(0, node._on_close)
            except Exception:
                pass

        # BELT-AND-SUSPENDERS: historically the restart script would send
        # SIGTERM, this handler would fire, Tk's event loop would NOT pick
        # up the after() callback in time (blocked on an X server round-trip
        # or a slow widget repaint), the script would time out, warn "X11
        # processes still alive (NOT force-killing)," and the next launch's
        # control_gui would SIGSEGV trying to take over the occupied display
        # + duplicate ROS2 node name. We were leaking processes on every
        # restart and producing the "RViz without GUI / GUI without RViz"
        # failure modes. This timer forces the process to die within ~2.5s
        # of the first signal even if Tk is wedged — the restart script
        # can then trust that SIGTERM actually killed the process.
        if _shutdown_state['armed']:
            return
        _shutdown_state['armed'] = True

        def _force_exit():
            # Give the Tk path ~2.5s to land cleanly. After that, os._exit
            # bypasses atexit/destructors but that's ACCEPTABLE here: the
            # process is already being torn down by SIGTERM and we own all
            # its external state (no uncommitted files, sockets handed off
            # to the OS). Clean-exit would be nicer but deadlocked-forever
            # is strictly worse.
            import os
            os._exit(0)

        import threading as _t
        _t.Timer(2.5, _force_exit).start()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Spin ROS2 executor in background thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Run tkinter on main thread (required by macOS Cocoa, safe on Linux)
    try:
        node._run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
