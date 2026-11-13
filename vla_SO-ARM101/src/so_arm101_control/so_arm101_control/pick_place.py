#!/usr/bin/env python3
"""
Shared SO-ARM101 pick-and-place ROS node: MoveIt IK, trajectory execution,
object detection, and dual hardware/simulation modes.

Source: adapted from RoboSort/JETANK_description/jetank_control_gui.py (3133 lines)
        IK delegated to MoveIt compute_ik service (KDL solver) instead of custom scipy.
        Joint mapping from MuammerBay/SO-ARM_ROS2_URDF and SO-ARM101_MoveIt_IsaacSim.
"""

import datetime
from contextlib import contextmanager
import json
import math
import os
import random
import signal
import subprocess
import sys
import shutil
import threading
import time
import weakref
from so_arm101_control.scene_transactions import SceneMutationClient, SceneMutationIncomplete
from so_arm101_control.motion_operation import MotionOperation
from pathlib import Path

from so_arm101_control.grasp_trace import tracer

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
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
from so_arm101_control.grasp_geometry import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    ALL_JOINT_NAMES,
    object_grip_width, jaw_offset, gripper_angles, make_grasp_plan,
    JOINT_LIMITS,
    _grasp_home_data_target,
    BASELINE_JAW_GAP,
    JAW_GAP_RATE,
    JAW_OPEN_CLEARANCE_M,
    JAW_CLOSE_CLEARANCE_M,
    TCP_CLEARANCE_M,
    _WRIST_ROLL_URDF_PITCH,
    _CHAIN_BASE_ROTATION,
    _CUP_STL_SCALE,
    CUP_BODY_HEIGHT_M,
    _CUP_COLLISION_PADDING,
    _normalize_grasp_yaw,
    _load_workspace_yaml,
    _WS,
    WORKSPACE_BOUNDS,
    _GWS,
    GRASP_WORKSPACE_BOUNDS,
    check_grasp_reachable,
    _GRASP_YAW_FALLBACK_OFFSETS,
    find_reachable_grasp_yaw,
)

# --- Record Sim tab constants -----------------------------------------------
# Module-level (not class-level) so hot_reload picks them up on the running
# instance — class attributes only attach to a fresh class object on import.

def _resolve_repo_root():
    """Locate the Exploring-VLAs repo root.

    Resolution order:
      1. ``SOARM_REPO_ROOT`` env var (highest precedence — colleague override).
      2. Walk up from this file's realpath. With ``colcon build --symlink-install``
         the install path's symlink resolves into the source tree, so 4 levels
         up gives the repo root regardless of where the workspace was built.
      3. Legacy fallback: ``~/Projects/Exploring-VLAs`` (matches the maintainer
         layout that was hardcoded before the cleanup).
    Returns an absolute path (not necessarily existing on disk).
    """
    env = os.environ.get("SOARM_REPO_ROOT", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.dirname(os.path.realpath(__file__))
    # Expected layout: <root>/vla_SO-ARM101/src/so_arm101_control/so_arm101_control/control_gui.py
    candidate = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    if (os.path.isdir(os.path.join(candidate, "linux-env")) and
            os.path.isdir(os.path.join(candidate, "vla_SO-ARM101"))):
        return candidate
    return os.path.expanduser("~/Projects/Exploring-VLAs")


_REPO_ROOT = _resolve_repo_root()
REC_LEROBOT_SCRIPT = os.path.join(_REPO_ROOT, "linux-env/scripts/record_sim_isaac.sh")
REC_MIRROR_SCRIPT = os.path.join(_REPO_ROOT, "linux-env/scripts/joint_states_to_commands.py")
REC_SIM_RESET_SCRIPT = os.path.join(_REPO_ROOT, "vla_SO-ARM101/scripts/sim_reset.sh")
# Datasets land in HuggingFace's cache by convention — env-overridable for CI / tmpfs.
REC_DATASET_ROOT = os.environ.get(
    "LEROBOT_LOCAL_DATASET_ROOT",
    os.path.expanduser("~/.cache/huggingface/lerobot/local"),
)
# Isaac Sim digital twin — the soarm101-dt scene tools are served on the
# omni-kit-mcp bridge as `soarm101.<name>` (registered by isaac/soarm101_tools).
# We talk to it with omni-kit-mcp's stdlib-only reference client
# (kit_mcp.client), imported from the in-repo submodule so this works under
# system Humble Python 3.10 without any pip install.
REC_KIT_MCP_HOST = os.environ.get("SOARM_KIT_MCP_HOST", "127.0.0.1")
REC_KIT_MCP_PORT = int(os.environ.get("SOARM_KIT_MCP_PORT", "8767"))
REC_KIT_MCP_NAMESPACE = "soarm101"
REC_KIT_MCP_DIR = os.environ.get(
    "SOARM_KIT_MCP_DIR", os.path.join(_REPO_ROOT, "omni-kit-mcp"))


def _import_kit_mcp_client():
    """Return the ``kit_mcp.client`` module from the omni-kit-mcp submodule.

    ``SOARM_KIT_MCP_DIR`` overrides the location (default:
    ``<repo>/omni-kit-mcp``). Raises ImportError with a pointer at the
    submodule when it isn't checked out.
    """
    if os.path.isdir(REC_KIT_MCP_DIR) and REC_KIT_MCP_DIR not in sys.path:
        sys.path.insert(0, REC_KIT_MCP_DIR)
    try:
        from kit_mcp import client as kit_client
    except ImportError as exc:
        raise ImportError(
            f"kit_mcp.client not importable from {REC_KIT_MCP_DIR} ({exc}); "
            "run `git submodule update --init omni-kit-mcp` or set "
            "SOARM_KIT_MCP_DIR") from exc
    return kit_client
# Hardcoded scene contract — MUST stay in sync with extension.py
# LEGO_USDS block_names. 1 episode = 1 lego pick-place. Lego rotates
# through this list per episode for uniform dataset coverage.
# Current scene contract: 2 of each color × 2x3 (suffixed _0 / _1),
# spawned via Cloner(replicate_physics=True).
REC_LEGOS_BY_COLOR = {
    "red":   ["red_2x3_0", "red_2x3_1"],
    "green": ["green_2x3_0", "green_2x3_1"],
    "blue":  ["blue_2x3_0", "blue_2x3_1"],
}
REC_MAX_RETRIES_PER_LEGO = 5
# Per-episode lerobot timing (seconds). Caller can also pass these flags
# directly to record_sim_isaac.sh for ad-hoc runs; the Record Sim tab uses
# these constants so hot-reload picks up changes without a process restart.
REC_EPISODE_TIME_S = int(os.environ.get("SOARM_REC_EPISODE_TIME_S", "120"))
REC_RESET_TIME_S = int(os.environ.get("SOARM_REC_RESET_TIME_S", "15"))


# --- Inference tab constants ----------------------------------------------
# Same module-level rationale as REC_*: hot-reload picks up changes on the
# running instance.

INF_SCRIPT = os.path.join(_REPO_ROOT, "linux-env/scripts/inference_smolvla.sh")
INF_ASYNC_SCRIPT = os.path.join(_REPO_ROOT, "linux-env/scripts/inference_async_lerobot.sh")

# Two inference methods, selected via the Method radio in the Inference tab:
#   "custom"   — our chunk-replan node (linux-env/scripts/smolvla_inference.py).
#                Three-process orchestration with mirror + lerobot-record when
#                "Record this run" is on. Maximum control over chunk replay.
#   "async_rtc"— LeRobot's async inference + Real-Time Action Chunking.
#                Two-process (policy server + robot client). Threshold-based
#                replan with weighted-average aggregation between consecutive
#                chunks (no queue starvation, smoother transitions). Same code
#                path that ships for real-hardware deploy → easier transition.
INF_METHOD_LABELS = {
    "custom":    "Custom (chunk replan @ 1 Hz)",
    "async_rtc": "Async (LeRobot + real RTC)",
}

# Shared with deploy_sim.sh; each preset is one (path, checkpoint, rename map) row.
from so_arm101_control.model_presets import MODEL_PRESETS
INF_MODEL_PRESETS = {**MODEL_PRESETS, "Custom": ("", "", "{}")}
INF_DEFAULT_TASK = "Pick a blue lego and place it in blue cup"
# Client action replay matches the Phase 1 wall cadence; cameras publish rgb8.
INF_DEFAULT_INFERENCE_RATE = 1.0
INF_DEFAULT_PUBLISH_RATE = 17.4


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
from so_arm101_control.grasp_geometry import TCP_ABOVE_BLOCK_CLEARANCE_M as _GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M


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


# ArUco marker ID -> human-readable color label for drop targets.
# New convention (mirrors soarm101-dt CUP_ARUCO_CONFIG and
# aruco_camera_localizer aruco_config.json): blue=1, green=2, red=3.
DROP_ID_LABELS = {
    "drop_1": "blue",
    "drop_2": "green",
    "drop_3": "red",
}

# Visual marker colors for cups in RViz (RGBA, alpha<1.0 avoids ros2/rviz#875)
_CUP_VISUAL_COLORS = {
    "drop_1": (0.15, 0.3, 0.9, 0.99),   # blue
    "drop_2": (0.1, 0.75, 0.2, 0.99),   # green
    "drop_3": (0.9, 0.15, 0.1, 0.99),   # red
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
REAL_COLOR_TO_CUP = {'red': 'drop_3', 'green': 'drop_2', 'blue': 'drop_1'}

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


class MotionSetting:
    """Mutable motion setting; GUI adapters may bind the same get/set interface."""
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class PickPlaceNode(Node):
    """Shared motion planner and cycle runner used by GUI and unattended collection."""

    def __init__(self, node_name="so_arm101_pick_place", use_sim_time=None,
                 joint_states_topic='/joint_states'):
        overrides = [] if use_sim_time is None else [Parameter('use_sim_time', value=use_sim_time)]
        super().__init__(node_name, parameter_overrides=overrides)

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
        self._motion_context = threading.local()
        self._set_motion_status(None)

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
            JointState, joint_states_topic, self._joint_states_callback, 10)
        self._joint_states_topic = joint_states_topic
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
        # child_frame_id ('drop_1'/'drop_2'/'drop_3') →
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
            self._apply_scene_client = SceneMutationClient(self._apply_scene_client,
                                                       self._motion_guard, self._scene_unresolved)
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

        self.running = True
        self._gui_ready = False
        self.sliders, self.slider_labels = {}, {}
        self._qs_abort_evt, self._qs_resume_evt = threading.Event(), threading.Event()
        self._qs_resume_evt.set()
        self._qs_state = 'idle'
        self._selected_object = None
        self._selected_drop = None
        self._trajectory_executor = None
        self._gripper_executor = None
        self._cycle_ok = False
        self.approach_offset = (0.0, 0.0)
        self.speed_scale = 1.0
        defaults = {
            'velocity_scale_var': 0.5, '_grasp_topic_var': '/objects_poses_sim',
            '_grasp_arm_duration_var': 2.5, '_grasp_grip_duration_var': 3.0,
            '_grasp_approach_height_var': 0.020, '_grasp_obj_z_var': 0.0,
            '_grasp_cross_var': False, '_bbox_enabled_var': True,
            '_tcp_clearance_var': TCP_CLEARANCE_M * 1000,
            '_jaw_open_clearance_var': JAW_OPEN_CLEARANCE_M * 1000,
            '_jaw_close_clearance_var': JAW_CLOSE_CLEARANCE_M * 1000,
            '_drop_duration_var': 3.0, '_drop_hover_above_rim_var': 0.050,
            '_drop_grip_angle_var': 45, '_ground_z_var': 0.0,
            '_ground_plane_var': True, '_planning_attempts_var': 50,
        }
        for name, value in defaults.items():
            setattr(self, name, MotionSetting(value))
        if node_name != 'so_arm101_control_gui':
            self.objects_sub = self.create_subscription(
                TFMessage, '/objects_poses_sim', self._objects_callback, 10,
                callback_group=self._sub_cb_group)
            self._drop_sub = self.create_subscription(
                TFMessage, '/drop_poses', self._drop_callback, 10,
                callback_group=self._sub_cb_group)

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
        if getattr(self, '_gripper_executor', None) is not None:
            return self._gripper_executor(jaw_position, duration_s, blocking)
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


    def _joint_states_callback(self, msg):
        """Update internal state and GUI sliders from the selected feedback topic."""
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
                f'Initial joint sync from {self._joint_states_topic}: '
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


    def _select_planning_group(self, group_name):
        """Switch the active planning group in RViz.

        On first call, cycles through gripper→arm to clear startup markers.
        """
        with self._motion_guard():
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


    def _publish_goal_state(self):
        """Publish current slider positions (arm + gripper) as goal state to RViz."""
        with self._motion_guard():
            if not MOVEIT_AVAILABLE or not hasattr(self, '_goal_state_pub'):
                return
            goal_state = RobotState()
            goal_state.is_diff = True  # Required by MotionPlanning plugin
            goal_state.joint_state.name = list(ALL_JOINT_NAMES)
            with self.joint_lock:
                goal_state.joint_state.position = [
                    self.joint_positions[n] for n in ALL_JOINT_NAMES]
            self._goal_state_pub.publish(goal_state)


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
            self._set_motion_status({
                'ok': False, 'outcome': 'ik_no_solution',
                'msg': (f'geometric IK returned no solutions for '
                        f'({x:.3f}, {y:.3f}, {z:.3f})')}, event=on_complete)
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
            self._set_motion_status({
                'ok': False, 'outcome': 'ik_all_collide',
                'msg': 'all geometric IK solutions collided with scene'}, event=on_complete)
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
                self._set_motion_status({
                    'ok': False, 'outcome': 'ik_pan_lock_collide',
                    'msg': (f'pan-locked IK solution at '
                            f'{math.degrees(lock_pan):.1f}\u00b0 collides')}, event=on_complete)
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
            on_complete = self._new_motion_event()
        self._joint_space_collision_free_execute(
            chosen, on_complete_event=on_complete, duration_s=duration_s)


    _QS_SEQUENCE = [
        ('move to home',        '_cmd_grasp_home_data',       {'skip_if_home': True}),
        ('grasp open',          '_cmd_gripper_open_for_object', {}),
        ('grasp move',          '_cmd_grasp_move',            {}),
        ('grasp close',         '_cmd_gripper_close_for_object', {}),
        ('return home (carry)', '_cmd_grasp_home_data',       {}),
        ('drop sweep',          '_cmd_drop_sweep',            {}),
        ('release',             '_cmd_drop_release',          {}),
        ('return home',         '_cmd_grasp_home_data',       {}),
    ]


    def _qs_auto_drop_for_lego(self, lego_name):
        """Return the /drop_poses child_frame_id of the cup matching this
        lego's color, or None if it can't be inferred.

        Per soarm101-dt CUP_ARUCO_CONFIG the ArUco marker ID mapping is
        blue → 1, green → 2, red → 3.
        """
        if not lego_name:
            return None
        color = lego_name.split('_', 1)[0].lower()
        return {'red': 'drop_3', 'green': 'drop_2',
                'blue': 'drop_1'}.get(color)


    def _lego_is_in_cup(self, pose):
        """True iff a lego's pose puts it inside any cup's footprint.

        Reuses the same XY+Z test as the MoveIt collision-scene path
        (where in-cup legos are skipped to avoid phantom collisions —
        see `_inside_any_cup` in the collision-diff method). A block
        that's been dropped into a cup is "consumed" and shouldn't be
        offered as a Quickstart pick target — top-down geometric IK
        often can't reach the cup rim anyway (cups are placed within
        the drop-IK envelope, not the grasp-IK envelope).
        """
        if pose is None:
            return False
        with self._drop_lock:
            cups = dict(self._drop_data) if self._drop_data else {}
        if not cups:
            return False
        cup_r = 0.039  # cup opening radius (matches CUP_R in collision-diff)
        lx = float(pose.get('x', 0.0))
        ly = float(pose.get('y', 0.0))
        lz = float(pose.get('z', 0.0))
        for cpos in cups.values():
            cx = float(cpos['x'])
            cy = float(cpos['y'])
            cz = float(cpos['z'])
            cup_base_z = cz - CUP_BODY_HEIGHT_M / 2.0
            if (lx - cx) ** 2 + (ly - cy) ** 2 <= (cup_r + 0.005) ** 2:
                if lz >= cup_base_z - 0.005:
                    return True
        return False


    def _qs_is_at_home(self, tol_deg=2.0):
        """Check whether the arm joints are already at the QS data-collection
        home pose. Compares against _grasp_home_data_target() since
        _QS_SEQUENCE's 'move to home' step routes through
        _cmd_grasp_home_data. Manual _cmd_grasp_home callers don't use
        this skip-if-home check."""
        target = _grasp_home_data_target()
        tol = math.radians(tol_deg)
        with self.joint_lock:
            for n in ARM_JOINT_NAMES:
                live = self._actual_positions.get(n, self.joint_positions.get(n, 0.0))
                if abs(live - target[n]) > tol:
                    return False
        return True


    def _qs_wait_for_step(self, timeout_s=60.0):
        """Wait for the most recent _cmd_* to complete, gated on pause.

        Completion events carry their operation's verdict. The collector
        requires that verdict and validates its generation after waits. GUI
        action callbacks without an adapter retain their display-status fallback.
        """
        evt = self._current_motion_event()
        if evt is None:
            return True  # nothing to wait on
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._qs_abort_evt.is_set():
                return False
            # Soft pause: block here until resume
            if not self._qs_resume_evt.wait(timeout=0.2):
                continue
            self._require_motion_generation()
            if evt.wait(timeout=0.2):
                self._require_motion_generation()
                status = self._motion_verdict(evt)
                if status is None and getattr(self, '_motion_effect_guard', None):
                    return False
                if status is not None and status.get('ok') is False:
                    # Surface WHY we halted so the log trail reads end-to-end.
                    outcome = status.get('outcome', 'unknown')
                    msg = status.get('msg', '')
                    self._append_log(
                        f'Quickstart: step failed ({outcome}): {msg}', 'err')
                    return False
                return True
        return False


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
        self._begin_motion(self._new_motion_event())
        self._set_motion_status(None)
        self._cmd_error = None
        # Tag the next trajectory dump with this step's cmd name.
        if method_name.startswith('_cmd_'):
            with self._motion_guard():
                self._last_motion_tag = method_name[5:]

        handler = getattr(self, method_name, None)
        if handler is None:
            self._append_log(f'Quickstart: no such handler {method_name}', 'err')
            return False
        try:
            handler()
        except Exception as exc:
            self._set_motion_status({'ok': False, 'outcome': 'step_exception', 'msg': str(exc)})
            self._append_log(f'Quickstart step {method_name} raised: {exc}', 'err')
            return False

        # A rejected planner tier can warn before a fallback succeeds. The
        # motion status is the verdict shared by the GUI and headless adapter.
        status = self._motion_verdict(self._current_motion_event())
        if status and status.get('ok') is False:
            return False
        ok = self._qs_wait_for_step(timeout_s=60.0)
        if not ok and self._qs_abort_evt.is_set():
            return False
        return ok


    def _qs_run(self):
        """Lifecycle runner — sequential dispatch of _QS_SEQUENCE steps."""
        try:
            self._cycle_ok = False
            self._prepare_cycle()
            self._qs_set_status('Running', step='(starting)')
            for i, (label, method_name, kwargs) in enumerate(self._QS_SEQUENCE, start=1):
                if self._qs_abort_evt.is_set():
                    self._append_log('Quickstart aborted')
                    return
                self._qs_resume_evt.wait()  # honor pause
                self._require_motion_generation()
                self._qs_set_status('Running', step=f'{i}/{len(self._QS_SEQUENCE)}: {label}')
                ok = self._qs_execute_step(method_name, kwargs)
                if not ok:
                    if self._qs_abort_evt.is_set():
                        self._qs_set_status('Aborted', step=label)
                    else:
                        self._qs_set_status('ERROR', step=label)
                        self._append_log(f'Quickstart halted at: {label}', 'err')
                    return
            self._cycle_ok = True
            self._qs_set_status('Complete', step='✓ done')
            self._append_log('Quickstart: pick-and-drop cycle complete')
        finally:
            self._qs_state = 'idle'


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


    def _ik_apply_and_act(self, target, mode, *, operation=None, descent=None):
        """Commit UI state briefly, then plan using immutable operation inputs.

        Approach/descent continuations carry their targets and events directly;
        no operation can read another worker's mutable target/event slots.
        """
        if operation is None:
            operation = self._operation(target, duration_s=2.)
        self._require_operation(operation)
        if mode == 'grasp_approach' and descent is None:
            raise ValueError('A grasp approach requires its operation-owned descent')
        if descent is not None:
            self._require_operation(descent)
            if descent.epoch != operation.epoch:
                raise ValueError('Approach/descent origins differ')

        def _apply():
            with self._motion_guard(operation):
                joints = operation.joints()
                self._slider_driven = True
                self._select_planning_group('arm')
                with self.joint_lock:
                    self.joint_positions.update(joints)
                for name, value in joints.items():
                    if name in self.sliders:
                        self.sliders[name].set(value)
                        self.slider_labels[name].config(text=f'{value:.3f}')
                self._publish_goal_state()
                self._ik_planned_target = dict(joints)
                self._ik_valid = True
            # Planning/validation may take many wall seconds. Only the UI
            # mutation above and final command commit own the generation.
            if mode == 'set_joints':
                self._cmd_set_joints()
            elif mode == 'plan_execute':
                self._cmd_plan_execute()
            elif mode == 'grasp_approach':
                def _wait_then_descend():
                    finished = operation.event.wait(timeout=60.)
                    self._require_operation(operation)
                    self._require_operation(descent)
                    verdict = self._motion_verdict(operation.event)
                    if not finished or not verdict or not verdict.get('ok'):
                        with self._motion_guard(descent):
                            self._set_motion_status(dict(ok=False, outcome='approach_failed'), descent.event)
                            descent.event.set()
                        return
                    self._append_log('Approach complete, descending to grasp')
                    self._ik_apply_and_act(descent.joints(), 'grasp_execute', operation=descent)
                self._motion_thread(target=_wait_then_descend, daemon=True).start()
                self._joint_space_collision_free_execute(operation.joints(), operation.event,
                    duration_s=operation.duration_s, operation=operation)
            elif mode == 'grasp_execute':
                self._joint_space_collision_free_execute(operation.joints(), operation.event,
                    duration_s=operation.duration_s, operation=operation)
        self._dispatch_motion(_apply)


    def _ee_pose_callback(self, msg):
        if getattr(self, '_gui_ready', False):
            p = msg.pose.position
            o = msg.pose.orientation
            self.root.after(0, self._update_ee_display,
                           p.x, p.y, p.z, o.x, o.y, o.z, o.w)


    def _objects_callback(self, msg):
        new_any_big_move = False
        with self.objects_lock:
            if getattr(self, '_physical_pose_cache', False):
                return
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
                self._trace('event', 'objects_callback_readd_scheduled',
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


    def _drop_callback(self, msg):
        """Populate _drop_data from /drop_poses TFMessage.

        Cache-only. _cmd_drop_refresh republishes visuals and collisions
        on a fixed time delay (matching _cmd_grasp_refresh); no event
        signaling happens here.
        """
        with self._drop_lock:
            if getattr(self, '_physical_pose_cache', False):
                return
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
        with self._motion_guard():
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
                    # Mesh origin is the base in the cup's oriented local frame.
                    qx,qy,qz,qw = (p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w)
                    norm = qx*qx+qy*qy+qz*qz+qw*qw
                    if norm < 1e-15:
                        raise ValueError('Invalid cup orientation')
                    h = CUP_BODY_HEIGHT_M / 2.0
                    p.position.x -= 2*(qx*qz+qw*qy)*h/norm
                    p.position.y -= 2*(qy*qz-qw*qx)*h/norm
                    p.position.z = pos['z'] - (1-2*(qx*qx+qy*qy)/norm)*h
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

        self._dispatch_scene_update(_apply)


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

        self._dispatch_scene_update(_apply)


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

        self._dispatch_scene_update(_apply)


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

        self._dispatch_scene_update(_apply)


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

        self._dispatch_scene_update(_apply)


    def _wait_arm_at_rest(self, timeout_sec=1.5, vel_threshold_rad_s=0.02,
                           dwell_sec=0.15):
        """Block until all arm joint velocities are below threshold for
        dwell_sec continuous time. Returns True if settled, False on timeout.

        Purpose: synchronize the TF and /objects_poses_sim streams before
        reading them in _attach_lego_to_gripper / _refresh_attached_pose.
        When the arm is moving, Layer-B lag (documented in
        upstream isaac-sim-mcp repo, docs/DEBUG-GUIDE.md § 4) makes /joint_states-derived
        TF disagree with Isaac physics' object poses — baking a 10-50 mm
        offset into the attached lego's mesh_poses[0] that then looks like
        the block hovering offset from the gripper in RViz. Waiting for
        zero-velocity eliminates the mismatch.
        """
        # A step-driven transport can prove that physical time is held.
        # Waiting for new velocity samples cannot settle a paused simulation.
        held = getattr(self, '_simulation_held', None)
        if held and held():
            return True
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
        with self._motion_guard():
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
            except SceneMutationIncomplete:
                raise
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
                except SceneMutationIncomplete:
                    raise
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
                except SceneMutationIncomplete:
                    raise
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
                self._trace('event', 'attach_applied',
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


    def _detach_lego_sync(self):
        """Blocking release of the currently-held lego.

        Sends AttachedCollisionObject op=REMOVE and a matching world
        CollisionObject REMOVE in one scene diff. MoveIt detaches the
        lego; the next /objects_poses_sim tick refreshes its world pose
        from Isaac Sim truth (wherever it actually landed after release).
        """
        with self._motion_guard():
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
                response = self._wait_future(fut, timeout_sec=3.0)
                if response is None:
                    return False
                # Humble may return false for an already-missing world REMOVE. Verify
                # the attached-object postcondition before allowing the next motion.
                check = GetPlanningSceneSrv.Request()
                check.components.components = 4  # ROBOT_STATE_ATTACHED_OBJECTS
                observed = self._wait_future(self._get_scene_client.call_async(check), timeout_sec=3.0)
                if observed is None or any(a.object.id == obj_id for a in observed.scene.robot_state.attached_collision_objects):
                    return False
            except SceneMutationIncomplete:
                raise
            except Exception as exc:
                self._append_log(f'Detach apply exception: {exc}', 'warn')
                return False
            self._attached_lego_name = None
            self._attached_lego_tcp_offset = None
            self._cycle_detach_seen = True
            self._trace('event', 'release_applied', obj_name=name)
            self._append_log(f'Detached {obj_id}')
            return True


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
        self._trace('event', 'grasp_home_start', target=dict(target))
        evt = self._new_motion_event()
        self._begin_motion(evt)
        # Emit grasp_home_done when motion settles. Close the trace cycle
        # iff a detach was seen in this cycle — that signals this is the
        # post-drop return-to-home, the final leg of the pick-place.
        # Pre-grasp and post-pick grasp_home calls leave _cycle_detach_seen
        # at False, so they just event and don't close.
        def _emit_home_done():
            evt.wait(timeout=30.0)
            with self._motion_guard():
                self._trace('event', 'grasp_home_done')
                if getattr(self, '_cycle_detach_seen', False) and self._trace('is_active'):
                    self._trace('close_cycle', 'completed')
                    self._cycle_detach_seen = False
        self._motion_thread(target=_emit_home_done, daemon=True).start()
        # Speed override: if home_velocity_scale param > 0, temporarily
        # swap velocity_scale_var for this motion's plan, then restore.
        # Lets the speed-vs-lag experiment sweep grasp_home speed live
        # via `ros2 param set` without affecting other motions.
        hvs = float(self.get_parameter('home_velocity_scale').value)
        # Snapshot the home override in this request, never in shared settings.
        operation = self._operation(target, evt, duration_s=3.)
        if hvs > 0.:
            operation = operation.with_planning(velocity_scale=hvs)
        self._joint_space_collision_free_execute(
            target, on_complete_event=evt, duration_s=3.0,
            allow_ompl_fallback=False, operation=operation)

        return evt


    def _cmd_grasp_home_data(self):
        """Move arm to the calibrated data-collection home pose.

        Used by _QS_SEQUENCE (Quickstart tab cycle) and the Record Sim
        tab's per-episode arm reset. Differs from _cmd_grasp_home in
        the target dict only — same deterministic-tier execution and
        same trace events (so existing telemetry / verify pipelines
        keep working).

        Target sourced from _grasp_home_data_target() so any future
        recalibration touches a single helper. Manual home button,
        Real Test tab, and IK tab keep using _cmd_grasp_home with the
        original all-zeros target.
        """
        target = _grasp_home_data_target()
        self._append_log(
            f'Grasp Home (data): pan={math.degrees(target["shoulder_pan"]):+.2f}° '
            f'lift={math.degrees(target["shoulder_lift"]):+.2f}° '
            f'elbow={math.degrees(target["elbow_flex"]):+.2f}° '
            f'wrist_flex={math.degrees(target["wrist_flex"]):.1f}° '
            f'wrist_roll={math.degrees(target["wrist_roll"]):.1f}°')
        self._trace('event', 'grasp_home_start', target=dict(target),
                     calibration='data')
        evt = self._new_motion_event()
        self._begin_motion(evt)
        def _emit_home_done():
            evt.wait(timeout=30.0)
            with self._motion_guard():
                self._trace('event', 'grasp_home_done')
                if getattr(self, '_cycle_detach_seen', False) and self._trace('is_active'):
                    self._trace('close_cycle', 'completed')
                    self._cycle_detach_seen = False
        self._motion_thread(target=_emit_home_done, daemon=True).start()
        hvs = float(self.get_parameter('home_velocity_scale').value)
        # Snapshot the home override in this request, never in shared settings.
        operation = self._operation(target, evt, duration_s=3.)
        if hvs > 0.:
            operation = operation.with_planning(velocity_scale=hvs)
        self._joint_space_collision_free_execute(
            target, on_complete_event=evt, duration_s=3.0,
            allow_ompl_fallback=False, operation=operation)
        return evt


    _NEUTRAL_NON_PAN_JOINTS = {
        'shoulder_lift': -0.2,
        'elbow_flex': 0.0,
        'wrist_flex': math.pi / 2,
        'wrist_roll': -math.pi / 2,
    }


    def _joint_space_collision_free_execute(self, target, on_complete_event,
                                             duration_s=3.0, waypoints=50,
                                             allow_ompl_fallback=True, operation=None):
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
        if operation is None:
            operation = self._operation(target, on_complete_event, duration_s=duration_s)
        self._require_operation(operation)
        target = operation.joints()
        on_complete_event = operation.event
        duration_s = operation.duration_s / operation.speed_scale
        with self._motion_guard(operation):
            self._select_planning_group('arm')
            with self.joint_lock:
                current = {n:self._actual_positions.get(n, self.joint_positions.get(n,0.))
                           for n in ARM_JOINT_NAMES}
                self.joint_positions.update(target)
            for n,value in target.items():
                if n in self.sliders:
                    self.sliders[n].set(value)
                    if n in self.slider_labels:
                        self.slider_labels[n].config(text=f'{value:.3f}')
            self._publish_goal_state()
        target_full = {n:target.get(n,current[n]) for n in ARM_JOINT_NAMES}
        operation = operation.retarget(target_full)

        # Already-at-target short-circuit (0.5° tol on every joint).
        ALREADY_AT_TARGET_TOL = math.radians(0.5)
        max_delta = max(abs(target_full[n] - current[n]) for n in ARM_JOINT_NAMES)
        if max_delta < ALREADY_AT_TARGET_TOL:
            self._append_log(
                f'  already at target (max Δ={math.degrees(max_delta):.2f}° '
                f'< 0.5°) — no motion dispatched')
            self._trace('event', 'tier1_noop',
                         max_delta_deg=math.degrees(max_delta),
                         target=dict(target_full), current=dict(current))
            self._set_motion_status({
                'ok': True, 'outcome': 'already_at_target',
                'msg': (f'already at target '
                        f'(max Δ={math.degrees(max_delta):.2f}°)')}, event=on_complete_event)
            on_complete_event.set()
            return True

        # Tier 1: linear joint-space interp + per-waypoint check.
        traj = self._plan_linear_joint_path(
            current, target_full, duration_s, n_samples=waypoints)
        if traj is not None:
            self._trace('event', 'planner_used', which='linear',
                         n_wps=len(traj.points))
            self._append_log(
                f'  tier1 linear: {len(traj.points)} wps clean, executing')
            self._execute_full_trajectory(traj, on_complete_event, operation=operation)
            return True

        # Tier 2: retract-pan-settle decomposition.
        traj2 = self._plan_retract_pan_settle(
            current, target_full, duration_s, n_samples=waypoints)
        if traj2 is not None:
            self._trace('event', 'planner_used', which='retract_pan_settle',
                         n_wps=len(traj2.points))
            self._append_log(
                f'  tier2 retract-pan-settle: {len(traj2.points)} wps clean, '
                f'executing')
            self._execute_full_trajectory(traj2, on_complete_event, operation=operation)
            return True

        # Fallback: OMPL (opt-in).
        if not allow_ompl_fallback:
            self._trace('event', 'planner_used', which='none')
            self._append_log(
                '  REFUSED: tier1 + tier2 both collided; OMPL fallback '
                'disabled (deterministic-only mode)', 'err')
            self._set_motion_status({
                'ok': False, 'outcome': 'tiered_planner_exhausted',
                'msg': ('tier1 linear and tier2 retract-pan-settle both '
                        'collided; OMPL fallback disabled')}, event=on_complete_event)
            on_complete_event.set()
            return False

        self._trace('event', 'planner_used', which='ompl_fallback')
        self._append_log(
            '  tier1 + tier2 both collided; falling back to OMPL')
        return self._ompl_plan_validate_execute(target_full, on_complete_event, operation=operation)


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


    def _ompl_plan_validate_execute(self, target, on_complete_event, operation=None):
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
        if operation is None:
            operation = self._operation(target, on_complete_event)
        self._require_operation(operation)
        target = operation.joints()
        on_complete_event = operation.event
        if not MOVEIT_AVAILABLE or self.plan_client is None \
                or not self.plan_client.service_is_ready():
            self._append_log('OMPL: service not available', 'warn')
            self._set_motion_status({
                'ok': False, 'outcome': 'ompl_service_unavailable',
                'msg': 'OMPL plan service not available'}, event=on_complete_event)
            on_complete_event.set()
            return False

        with self.joint_lock:
            current = {n: self._actual_positions.get(
                n, self.joint_positions.get(n, 0.0))
                       for n in ALL_JOINT_NAMES}

        self._trace('snapshot_scene', 'before_ompl_plan', self)
        traj, ec_info = self._ompl_plan_sync(current, target, tolerance=0.01, operation=operation)

        if traj is None or not traj.joint_trajectory.points:
            self._trace('snapshot_scene', 'after_ompl_returns_modeA', self)
            self._trace('event', 'ompl_attempt', attempt=0, perturb='none',
                         mode='A', ec_info=repr(ec_info))
            self._trace('event', 'ompl_exhausted', n_attempts=1, mode='A',
                         ec_info=repr(ec_info))
            self._append_log(
                f'  OMPL Mode A: planner returned no plan ({ec_info})', 'warn')
            self._set_motion_status({
                'ok': False, 'outcome': 'ompl_mode_a',
                'msg': f'OMPL Mode A: no plan ({ec_info})'}, event=on_complete_event)
            on_complete_event.set()
            return False

        self._trace('snapshot_scene', 'after_ompl_returns', self)
        chk = self._trajectory_first_invalid_with_contacts(
            traj.joint_trajectory)
        if not chk.ok:
            contact_summary = '; '.join(
                f'{c.contact_body_1}↔{c.contact_body_2}(d={c.depth*1000:.1f}mm)'
                for c in chk.contacts[:5]) or 'no contact info'
            where = (f'wp[{chk.bad_wp}]' if chk.bad_subidx is None
                     else f'wp[{chk.bad_wp}→{chk.bad_wp + 1}] sub-t={chk.sub_t:.2f}')
            self._trace('snapshot_scene', 'after_waypoint_check_modeB', self)
            self._trace('event', 'ompl_attempt', attempt=0, perturb='none',
                         mode='B',
                         bad_wp=chk.bad_wp, bad_subidx=chk.bad_subidx,
                         sub_t=chk.sub_t,
                         n_wps=chk.n_wps, n_sub=chk.n_sub,
                         contacts=[{
                             'a': c.contact_body_1,
                             'b': c.contact_body_2,
                             'depth_mm': c.depth * 1000,
                         } for c in chk.contacts[:10]])
            self._trace('event', 'ompl_exhausted', n_attempts=1, mode='B',
                         bad_wp=chk.bad_wp,
                         n_wps=chk.n_wps)
            self._append_log(
                f'  OMPL post-check: {where} invalid '
                f'(checked {chk.n_wps} wps + {chk.n_sub} sub-states): '
                f'{contact_summary}', 'warn')
            self._set_motion_status({
                'ok': False, 'outcome': 'ompl_mode_b',
                'msg': (f'post-check rejected {where}: {contact_summary}'),
                'bad_wp': chk.bad_wp, 'bad_subidx': chk.bad_subidx,
                'sub_t': chk.sub_t}, event=on_complete_event)
            on_complete_event.set()
            return False

        self._trace('event', 'ompl_validated', attempt=0, perturb='none',
                     n_wps=chk.n_wps, n_sub=chk.n_sub)
        # Honest log: we checked every waypoint AND N intermediate states
        # per segment. "Validated" here means collision-free at discrete
        # sample points under the current /planning_scene, NOT proof of
        # clearance at infinitesimal resolution — see TRAJ_SUBSEGMENT_SAMPLES.
        self._append_log(
            f'  OMPL post-check: {chk.n_wps} wps + {chk.n_sub} sub-states '
            f'clear ({TRAJ_SUBSEGMENT_SAMPLES}/seg)')
        self._execute_full_trajectory(
            traj.joint_trajectory, on_complete_event, operation=operation)
        return True


    def _ompl_plan_sync(self, start_joint_dict, target_joint_dict, tolerance=0.01, operation=None):
        """Blocking OMPL plan call. Returns (RobotTrajectory, error detail).

        tolerance: joint goal tolerance (rad). Default 0.01 matches the
        _cmd_plan_execute interface.
        """
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.pipeline_id = 'ompl'
        if operation is None:
            operation = self._operation(target_joint_dict)
        self._require_operation(operation)
        mpr.num_planning_attempts = operation.planning_attempts
        # 10s (was 5s). Phase 9 data: tier1 check sometimes rejects the
        # linear-interp path at a 0.03mm cup-rim clip, and the 5s OMPL
        # budget was not always enough for RRTConnect to find an alternative.
        # One-shot plan → 10s keeps worst-case latency ≤10s (vs the old
        # retry loop's 50s) while drastically reducing sampling variance.
        mpr.allowed_planning_time = 10.0
        vel_scale = operation.velocity_scale
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
            return None, ('no_response', None)
        r = future.result()
        ec = r.motion_plan_response.error_code.val
        if ec != 1:
            return None, (f'ec={ec}', r.motion_plan_response.planning_time)
        return r.motion_plan_response.trajectory, None


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


    def _execute_full_trajectory(self, joint_trajectory, on_complete_event, operation=None):
        """Send a pre-validated multi-waypoint JointTrajectory to arm_controller.

        Sets on_complete_event when controller reports goal result (or on any
        failure/timeout).
        """
        if operation is None:
            duration = max(.001, joint_trajectory.points[-1].time_from_start.sec +
                           joint_trajectory.points[-1].time_from_start.nanosec/1e9)
            operation = self._operation(dict(zip(joint_trajectory.joint_names,
                joint_trajectory.points[-1].positions)), on_complete_event, duration_s=duration)
        self._require_operation(operation)
        if getattr(self, '_trajectory_executor', None) is not None:
            return self._trajectory_executor(joint_trajectory, operation.event, operation=operation)
        n_wps = len(joint_trajectory.points) if joint_trajectory.points else 0
        self._trace('event', 'execute_start', n_wps=n_wps, source='full_trajectory')
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
                    'motion_tag': operation.tag,
                },
            }
            _hist_dir = '/tmp/arm_traj'
            _os.makedirs(_hist_dir, exist_ok=True)
            _stamp = _dt.now().strftime('%Y%m%dT%H%M%S_%f')[:-3]
            _tag = operation.tag
            _hist_path = _os.path.join(_hist_dir, f'{_stamp}_{_tag}.json')
            with open(_hist_path, 'w') as _f:
                _json.dump(_dump, _f, default=str)
            with open('/tmp/last_arm_trajectory.json', 'w') as _f:
                _json.dump(_dump, _f, default=str)
            with self._motion_guard(operation):
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
            self._trace('event', 'execute_done', outcome='action_server_not_ready')
            self._set_motion_status({
                'ok': False, 'outcome': 'action_server_not_ready',
                'msg': 'arm_controller action server not ready'}, event=on_complete_event)
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
            with self._motion_guard(operation):
                try:
                    res = result_future.result()
                    ec = getattr(getattr(res, 'result', None), 'error_code', None)
                    self._trace('event', 'execute_done',
                                 outcome='completed',
                                 error_code=(ec if ec is not None else None))
                    # FollowJointTrajectory.Result.SUCCESSFUL = 0
                    ok = (ec is None) or (ec == 0)
                    self._set_motion_status({
                        'ok': ok,
                        'outcome': 'completed' if ok else 'trajectory_error',
                        'msg': (f'trajectory complete ({n_wps} wps)' if ok
                                else f'trajectory error_code={ec}'),
                        'error_code': ec}, event=on_complete_event)
                except Exception as e:
                    self._trace('event', 'execute_done', outcome='result_exception',
                                 error=repr(e))
                    self._set_motion_status({
                        'ok': False, 'outcome': 'result_exception',
                        'msg': f'result callback exception: {e!r}'}, event=on_complete_event)
                on_complete_event.set()

        def _on_accept(send_future):
            with self._motion_guard(operation):
                try:
                    gh = send_future.result()
                except Exception as e:
                    self._trace('event', 'execute_done', outcome='send_exception',
                                 error=repr(e))
                    self._set_motion_status({
                        'ok': False, 'outcome': 'send_exception',
                        'msg': f'action send_goal exception: {e!r}'}, event=on_complete_event)
                    on_complete_event.set()
                    return
                if not gh.accepted:
                    self._trace('event', 'execute_done', outcome='goal_rejected')
                    self._set_motion_status({
                        'ok': False, 'outcome': 'goal_rejected',
                        'msg': 'arm action server rejected the goal'}, event=on_complete_event)
                    on_complete_event.set()
                    return
                with self._arm_goal_lock:
                    self._arm_goal_handle = gh
                rf = gh.get_result_async()
                rf.add_done_callback(_on_result)

        sf = self.arm_action_client.send_goal_async(goal)
        sf.add_done_callback(_on_accept)


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

        evt = self._new_motion_event()
        self._begin_motion(evt)


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
        """IK-planned drop sweep: combined pan + wrist tilt to deliver
        the held block over the selected cup.

        Self-contained as of Phase 10.2 — no longer requires
        _cmd_drop_point to have run first. Geometric IK solves for all
        five arm joints (pan, lift, elbow, wrist_flex, wrist_roll) at
        the over-cup target, and _joint_space_collision_free_execute
        interpolates the full joint vector in one trajectory. Net
        effect: the wrist lifts the block as it tilts forward, while
        the base pans across — block clears the cup rims during the
        sweep (vs. a pan-only-then-tilt sequence that flatlined the
        block at carry height during the cross).

        The _QS_SEQUENCE no longer dispatches a separate "point to drop
        cup" step; drop_sweep owns the full motion. The Drop Point
        button still exists for manual use.

        Phase 9 invariant preserved (block-center target, not tcp_link):
        derives the current jaw gap from gripper_joint, computes the
        offset from gap-center to tcp along the fixed-jaw direction, and
        shifts the IK target accordingly. The MEASURED |ax| from the
        physical attach is preferred over the theoretical half_gap so
        off-center grasps land cleanly.
        """
        result = self._get_selected_drop_pose()
        if result is None:
            return
        name, x, y, z = result

        evt = self._new_motion_event()
        self._begin_motion(evt)
        self._set_motion_status(None)

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

        self._trace('event', 'drop_sweep_start',
                     drop_name=name,
                     gap_target=(gap_x, gap_y, gap_z),
                     tcp_target=(target_x, target_y, target_z),
                     jaw_gap_mm=jaw_gap * 1000,
                     gripper_joint=gj)
        self._trace('snapshot_scene', 'drop_sweep_start', self)
        def _close_on_drop_done():
            evt.wait(timeout=30.0)
            with self._motion_guard():
                self._trace('event', 'drop_sweep_done')
        self._motion_thread(target=_close_on_drop_done, daemon=True).start()
        self._append_log(
            f'Drop Sweep: gap=({gap_x:.3f},{gap_y:.3f},{gap_z:.3f})m '
            f'→ tcp=({target_x:.3f},{target_y:.3f},{target_z:.3f}) '
            f'(shift={shift_mag*1000:.1f}mm [{shift_source}], '
            f'hover={hover_m*1000:.0f}mm, jaw_gap={jaw_gap*1000:.1f}mm)')

        grip_deg = getattr(self, '_drop_grip_angle_var', None)
        grip_angle = math.radians(grip_deg.get() if grip_deg else 45)
        # Lock shoulder_pan to the cup-derived pan (atan2(-y, x-X_PAN)).
        # Phase 10.2: was previously read from the live measured pan, which
        # required _cmd_drop_point to have run first. Cup-derived pan makes
        # drop_sweep self-sufficient — the IK-planned trajectory drives
        # pan from current → cup_pan as part of the combined motion (the
        # wrist tilts up at the same time, lifting the held block over
        # the cup rims during the cross). See _plan_collision_free_execute
        # (lock_pan=...) docstring for the post-IK overwrite semantics.
        locked_pan = pan
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
        self._trace('event', 'drop_release_start', last_grasped=last)
        self._select_grasp_object(last)
        self._cmd_gripper_open_for_object()
        # Wait for the gripper-open motion to complete. _cmd_gripper_open_for_object
        # assigned a fresh Event to self._motion_event — capture it before waiting
        # so a later trigger-service callback that clears the attribute can't
        # swap it out from under us.
        evt = self._current_motion_event()
        if evt is not None:
            if not evt.wait(timeout=10.0):
                raise TimeoutError('Gripper release timed out')
            self._require_motion_generation()
            verdict = self._motion_verdict(evt)
            if verdict is not None and not verdict.get('ok'):
                raise RuntimeError('Gripper release operation failed')
        # Synchronously detach the AttachedCollisionObject. Returning from
        # this service guarantees the next grasp_home sees a clean scene.
        self._ensure_lego_state()
        if self._attached_lego_name:
            try:
                if not self._detach_lego_sync():
                    raise RuntimeError('Planning scene did not acknowledge detach')
            except SceneMutationIncomplete:
                raise
            except Exception as exc:
                raise RuntimeError(f'Detach failed: {exc}') from exc


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


    def _get_grip_width(self, obj_name):
        cross = self._grasp_cross_var.get() if hasattr(self, '_grasp_cross_var') else False
        return object_grip_width(self._lookup_bbox(obj_name), cross)


    def _compute_jaw_offset(self, obj_name, obj_yaw):
        enabled = self._bbox_enabled_var.get() if hasattr(self, '_bbox_enabled_var') else True
        clearance = self._tcp_clearance_var.get() / 1000 if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M
        return jaw_offset(self._get_grip_width(obj_name), obj_yaw, clearance, enabled)


    def _gripper_angle_for_object(self, obj_name):
        width = self._get_grip_width(obj_name) if obj_name else None
        tcp = self._tcp_clearance_var.get() / 1000 if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M
        opening = self._jaw_open_clearance_var.get() / 1000 if hasattr(self, '_jaw_open_clearance_var') else JAW_OPEN_CLEARANCE_M
        closing = self._jaw_close_clearance_var.get() / 1000 if hasattr(self, '_jaw_close_clearance_var') else JAW_CLOSE_CLEARANCE_M
        return gripper_angles(width, tcp, opening, closing)


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
        evt = self._new_motion_event()
        self._begin_motion(evt)
        def _send():
            ok = self._send_gripper_goal(open_angle, duration_s=duration, blocking=True)
            with self._motion_guard():
                self._set_motion_status({'ok': bool(ok), 'outcome': 'gripper_open'}, event=evt)
                evt.set()
                self._select_planning_group('arm')
        self._motion_thread(target=_send, daemon=True).start()


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
        evt = self._new_motion_event()
        self._begin_motion(evt)
        def _send():
            ok = self._send_gripper_goal(close_angle, duration_s=duration, blocking=True)
            # Phase 9: attach to tcp_link after physical closure succeeds
            with self._motion_guard():
                if ok is not False:
                    try:
                        ok = self._attach_lego_to_gripper(obj_name) and self._attached_lego_name == obj_name
                    except SceneMutationIncomplete:
                        raise
                    except Exception as exc:
                        ok = False
                        self._append_log(f'Attach step errored: {exc}', 'warn')
                self._set_motion_status({'ok': bool(ok), 'outcome': 'gripper_close_attach'}, event=evt)
                evt.set()
        self._motion_thread(target=_send, daemon=True).start()


    def _cmd_grasp_move(self):
        obj_name = self._get_selected_object_name()
        if not obj_name:
            raise ValueError('No object selected')
        # Remember the grasped object so drop_release can open for it later
        with self._motion_guard():
            self._last_grasped_object = obj_name
        # Forensic trace: open a new cycle scoped grasp_move -> grasp_home.
        # If a prior cycle is still open, it's force-closed with outcome='abandoned'.
        with self.joint_lock:
            _entry_joints = {n: self._actual_positions.get(
                n, self.joint_positions.get(n, 0.0)) for n in ARM_JOINT_NAMES}
        self._trace('open_cycle',
            obj_name=obj_name,
            grasp_topic=self._grasp_topic_var.get().strip(),
            cross=bool(self._grasp_cross_var.get()),
            entry_joints=_entry_joints,
        )
        # Reset detach-seen flag for the new cycle. Cycle closes on the
        # first grasp_home_done AFTER detach (post-drop return).
        with self._motion_guard():
            self._cycle_detach_seen = False
        self._trace('snapshot_scene', 'grasp_move_start', self)
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
            pos = dict(self.objects_data[obj_name]) if obj_name in self.objects_data else None
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return

        topic = self._grasp_topic_var.get().strip()
        cross = self._grasp_cross_var.get()
        approach_h = self._grasp_approach_height_var.get()
        with self.joint_lock:
            current_wr = self._actual_positions.get('wrist_roll')
        geometry = make_grasp_plan(
            pos, self._lookup_bbox(obj_name), cross=cross,
            current_wrist_roll=current_wr,
            tcp_clearance=self._tcp_clearance_var.get() / 1000,
            bbox_enabled=self._bbox_enabled_var.get(),
            object_z_override=self._grasp_obj_z_var.get(),
            approach_height=approach_h,
            approach_offset=getattr(self, 'approach_offset', (0.0, 0.0)),
            drop=topic == '/drop_poses')
        tx, ty, target_z = geometry['tcp']
        obj_yaw = geometry['yaw']
        jaw_dx, jaw_dy = geometry['jaw_offset']
        obj_qz, obj_qw = pos.get('qz', 0.0), pos.get('qw', 1.0)
        action = 'drop' if topic == '/drop_poses' else 'grab'
        if abs(jaw_dx) > 0.001 or abs(jaw_dy) > 0.001:
            gw = (self._get_grip_width(obj_name) or 0) * 1000
            self._append_log(
                f'  Jaw offset: ({jaw_dx*1000:+.1f}, {jaw_dy*1000:+.1f})mm '
                f'for {gw:.0f}mm grip{"[cross]" if cross else ""}')

        self._append_log(
            f'Grasp: {action} "{obj_name}" at '
            f'({tx:.3f}, {ty:.3f}, {target_z:.3f})'
            f'{" [cross]" if cross else ""}')

        duration = self._grasp_arm_duration_var.get()
        # Compute top-down grasp orientation (pitch=90°, yaw aligned to object)
        gqx, gqy, gqz, gqw = self._grasp_orientation(
            tx, ty, obj_qz, obj_qw)

        approach_h = self._grasp_approach_height_var.get()

        # Create motion event BEFORE spawning thread so trigger callback can see it
        evt = self._new_motion_event()
        self._begin_motion(evt)

        operation = self._operation({}, evt, duration_s=duration)
        ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None
        poses_to_check = tuple(tuple(p) for p in geometry['poses'])
        # Pre-validate ALL stages before moving the arm
        def _prevalidate_and_execute():
            self._require_operation(operation)

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
                    self._trace('event', 'gate_a_done', per_stage=gate_a_results)
                    self._trace('close_cycle', 'grasp_unreachable_workspace',
                                       stage=stage, reason=reason,
                                       target=(px, py, pz))
                    self._set_motion_status({
                        'ok': False, 'outcome': 'grasp_unreachable_workspace',
                        'msg': (f'Gate A reject @ {stage} '
                                f'({px:.3f}, {py:.3f}, {pz:.3f}): {reason}')}, event=evt)
                    evt.set()
                    return
            self._trace('event', 'gate_a_done', per_stage=gate_a_results)

            # Find a yaw where geometric_ik returns solutions for BOTH stages.
            # Using a single yaw across stages avoids mid-grasp wrist twists.
            yaw_used, stage_sols, dbg_lines = find_reachable_grasp_yaw(
                poses_to_check, obj_yaw)
            self._trace('event', 'gate_b_done',
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
                self._trace('close_cycle', 'grasp_unreachable_ik',
                                   dbg_lines=list(dbg_lines or []))
                self._set_motion_status({
                    'ok': False, 'outcome': 'grasp_unreachable_ik',
                    'msg': (f'Gate B reject: no geometric IK at any yaw for '
                            f'stages {[s[0] for s in poses_to_check]}')}, event=evt)
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
                    self._trace('event', 'gate_c_done',
                                 attempts=gate_c_attempts,
                                 validated={k: dict(v) for k, v in validated.items()})
                    self._trace('close_cycle', 'gate_c_all_collide',
                                       stage=stage, target=(px, py, pz))
                    self._set_motion_status({
                        'ok': False, 'outcome': 'gate_c_all_collide',
                        'msg': (f'Gate C reject @ {stage} '
                                f'({px:.3f}, {py:.3f}, {pz:.3f}): '
                                f'all IK solutions collide')}, event=evt)
                    evt.set()
                    return
            self._trace('event', 'gate_c_done',
                         attempts=gate_c_attempts,
                         validated={k: dict(v) for k, v in validated.items()})

            # Publish only immutable operation-owned results after planning.
            self._require_operation(operation)
            if approach_h > 0 and 'approach' in validated and 'final' in validated:
                final = operation.retarget(validated['final'])
                approach = operation.retarget(validated['approach'],
                    self._new_motion_event(origin=operation.epoch))
                self._ik_apply_and_act(approach.joints(), 'grasp_approach',
                                       operation=approach, descent=final)
            elif 'final' in validated:
                final = operation.retarget(validated['final'])
                self._ik_apply_and_act(final.joints(), 'grasp_execute', operation=final)
            else:
                # Validated dict has no 'final' stage — this shouldn't happen
                # if gate C passed, but guard anyway.
                self._set_motion_status({
                    'ok': False, 'outcome': 'no_final_stage',
                    'msg': ('Gate C passed but no final-stage solution '
                            'recorded — logic bug')}, event=evt)
                evt.set()  # nothing to execute

        self._motion_thread(target=_prevalidate_and_execute, daemon=True).start()


    def _gripper_command(self, jaw_target, execute=False, duration_s=1.0):
        """Set gripper goal. If execute=True, also send to controller."""
        with self._motion_guard():
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

        self._dispatch_scene_update(_apply)


    def _scene_unresolved(self, exc):
        self._scene_terminal_error = str(exc)
        handler = getattr(self, '_scene_failure_handler', None)
        if handler is not None:
            handler(exc)
        raise SceneMutationIncomplete(str(exc))

    @contextmanager
    def _motion_guard(self, operation=None):
        if getattr(self,'_scene_terminal_error',None):
            raise SceneMutationIncomplete(self._scene_terminal_error)
        epoch = self._motion_token() if operation is None else operation.epoch
        guard = getattr(self, '_motion_effect_guard', None)
        if guard is None:
            yield
        else:
            with guard(epoch):
                if epoch != self._motion_token():
                    raise ValueError('Motion operation and worker have different origins')
                yield

    def _trace(self, method, *args, **kwargs):
        with self._motion_guard():
            return getattr(tracer, method)(*args, **kwargs)

    def _operation(self, target, event=None, duration_s=3.):
        event = event if event is not None else self._new_motion_event()
        return MotionOperation.create(self._motion_token(), target, event, duration_s,
                                      getattr(self,'speed_scale',1.), getattr(event,'motion_tag','motion'),
                                      self.velocity_scale_var.get(), self._planning_attempts_var.get())

    def _require_operation(self, operation):
        if operation.event.motion_epoch != operation.epoch:
            raise ValueError('Motion target/event origin mismatch')
        with self._motion_guard(operation):
            pass

    def _begin_motion(self, event):
        with self._motion_guard():
            if event.motion_epoch != self._motion_token():
                raise ValueError("Cannot publish another generation's motion event")
            self._motion_context.event = event
            self._motion_event = event  # UI/service inventory only, never continuation input.
        return event

    def _current_motion_event(self):
        event = getattr(self._motion_context, 'event', None)
        if event is None and not getattr(self,'_motion_effect_guard',None):
            return getattr(self,'_motion_event',None)
        return event

    def _require_motion_generation(self):
        with self._motion_guard():
            pass

    def _motion_verdict(self, event):
        with self._motion_guard():
            if event is not None and getattr(self,'_motion_effect_guard',None) and event.motion_epoch != self._motion_token():
                raise ValueError('Verdict event and worker origins differ')
            verdict = getattr(event, 'verdict', None)
            if verdict is None and not getattr(self, '_motion_effect_guard', None):
                return getattr(self, '_last_motion_status', None)
            return verdict

    def _set_motion_status(self, status, event=None):
        with self._motion_guard():
            event = event if event is not None else self._current_motion_event()
            if event is not None:
                if getattr(self,'_motion_effect_guard',None) and event.motion_epoch != self._motion_token():
                    raise ValueError('Status event and worker origins differ')
                event.verdict = dict(status) if status is not None else None
            self._last_motion_status = status

    def _motion_token(self):
        context = getattr(self, '_motion_context', None)
        if context is not None and hasattr(context, 'epoch'):
            return context.epoch
        provider = getattr(self, '_motion_generation_provider', None)
        return provider() if provider else None

    def _new_motion_event(self, origin=Ellipsis):
        event = threading.Event()
        event.motion_epoch = self._motion_token() if origin is Ellipsis else origin
        event.motion_tag = getattr(self,'_last_motion_tag','motion')
        event.verdict = None
        return event

    def _motion_thread(self, target, **kwargs):
        # Capture before starting the worker, never after its planning/waits.
        epoch = self._motion_token()
        event = self._current_motion_event()
        def run():
            self._motion_context.epoch = epoch
            self._motion_context.event = event
            try:
                target()
            except Exception:
                provider = getattr(self, '_motion_generation_provider', None)
                if provider is None or provider() == epoch:
                    raise
                # A retired worker may leave a wait, but cannot claim success
                # or mutate the scene after its generation was invalidated.
        return threading.Thread(target=run, **kwargs)

    def _dispatch_scene_update(self, callback):
        def apply():
            with self._motion_guard():
                callback()
        if self._gui_ready:
            self._motion_thread(target=apply, daemon=True).start()
        else:
            apply()

    def _dispatch_motion(self, callback):
        # The callback owns only its short state/dispatch commits. Planning
        # and validation inside it must run without generation ownership.
        callback()

    def _prepare_cycle(self):
        self._cycle_ok = False

    def _append_log(self, text, level='info'):
        log = self.get_logger()
        if level in ('err', 'error'):
            log.error(text)
        elif level in ('warn', 'warning'):
            log.warning(text)
        else:
            log.info(text)

    def _qs_set_status(self, status, step=None):
        self._append_log(f'{status}: {step or ""}')

    def _get_selected_object_name(self):
        return self._selected_object

    def _qs_get_selected_object(self):
        return self._get_selected_object_name()

    def _select_grasp_object(self, name):
        with self._motion_guard():
            self._selected_object = name

    def _qs_sync_grasp_listbox(self, name):
        with self._motion_guard():
            self._selected_object = name
            return name in self.objects_data

    def _qs_sync_drop_listbox(self, name):
        with self._motion_guard():
            self._selected_drop = name
            return name in self._drop_data

    def _get_selected_drop_pose(self):
        with self._drop_lock:
            pose = self._drop_data.get(self._selected_drop)
        if pose is None:
            return None
        return self._selected_drop, pose['x'], pose['y'], pose['z']

    def run_cycle(self, object_name):
        """Run a complete cycle synchronously from a non-executor thread."""
        if self._qs_state != 'idle':
            raise RuntimeError('A pick-and-place cycle is already running')
        self._selected_object = object_name
        self._qs_abort_evt.clear()
        self._qs_resume_evt.set()
        self._qs_state = 'running'
        self._qs_run()
        return self._cycle_ok

__all__ = [name for name in ['ALL_JOINT_NAMES', 'ARM_JOINT_NAMES', 'ActionClient', 'AllowedCollisionEntry', 'ApplyPlanningScene', 'AttachedCollisionObject', 'BASELINE_JAW_GAP', 'CUP_BODY_HEIGHT_M', 'CollisionObject', 'Constraints', 'DROP_ID_LABELS', 'DROP_SCAN_AVG_DURATION_S', 'DROP_SCAN_FOV_HALF_RAD', 'DROP_SCAN_INITIAL_DURATION_S', 'DROP_SCAN_MAX_XY_VARIANCE_M2', 'DROP_SCAN_MIN_AVG_SAMPLES', 'DROP_SCAN_PAN_END_RAD', 'DROP_SCAN_PAN_MAX_RAD', 'DROP_SCAN_PAN_MIN_RAD', 'DROP_SCAN_PAN_START_RAD', 'DROP_SCAN_PAN_STEP_RAD', 'DROP_SCAN_PAN_STEP_SIGNED_RAD', 'DROP_SCAN_SAMPLE_PER_STEP_S', 'DROP_SCAN_SETTLE_AFTER_STEP_S', 'DROP_SCAN_STEP_DURATION_S', 'Duration', 'ExecuteKnownTrajectory', 'FollowJointTrajectory', 'GRASP_WORKSPACE_BOUNDS', 'GRIPPER_JOINT_NAME', 'GetMotionPlan', 'GetPlanningSceneSrv', 'GetPositionFK', 'GetPositionIK', 'GetStateValidity', 'INF_ASYNC_SCRIPT', 'INF_DEFAULT_INFERENCE_RATE', 'INF_DEFAULT_PUBLISH_RATE', 'INF_DEFAULT_TASK', 'INF_METHOD_LABELS', 'INF_MODEL_PRESETS', 'INF_SCRIPT', 'JAW_CLOSE_CLEARANCE_M', 'JAW_GAP_RATE', 'JAW_OPEN_CLEARANCE_M', 'JOINT_LIMITS', 'JointConstraint', 'JointState', 'JointTrajectory', 'JointTrajectoryPoint', 'MODEL_PRESETS', 'MarkerArray', 'MeshTriangle', 'MotionPlanRequest', 'MultiThreadedExecutor', 'Node', 'Path', 'PickPlaceNode', 'PlanningSceneMsg', 'Point', 'Pose', 'PoseStamped', 'PositionIKRequest', 'REAL_COLOR_TO_CUP', 'REC_DATASET_ROOT', 'REC_EPISODE_TIME_S', 'REC_KIT_MCP_DIR', 'REC_KIT_MCP_HOST', 'REC_KIT_MCP_NAMESPACE', 'REC_KIT_MCP_PORT', 'REC_LEGOS_BY_COLOR', 'REC_LEROBOT_SCRIPT', 'REC_MAX_RETRIES_PER_LEGO', 'REC_MIRROR_SCRIPT', 'REC_RESET_TIME_S', 'REC_SIM_RESET_SCRIPT', 'ReentrantCallbackGroup', 'RobotState', 'SetBool', 'ShapeMesh', 'SolidPrimitive', 'String', 'TCP_CLEARANCE_M', 'TFMessage', 'TRAJ_SUBSEGMENT_SAMPLES', 'TfBuffer', 'TransformListener', 'Trigger', 'Vector3', 'VisMarker', 'WORKSPACE_BOUNDS', 'WRIST_ROLL_URDF_PITCH', '_CHAIN_BASE_ROTATION', '_CUP_COLLISION_PADDING', '_CUP_STL_SCALE', '_CUP_VISUAL_COLORS', '_GRASP_YAW_FALLBACK_OFFSETS', '_GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M', '_GWS', '_LEGO_SIZES', '_REPO_ROOT', '_TrajCheckResult', '_WRIST_ROLL_URDF_PITCH', '_WS', '_get_cup_stl_uri', '_grasp_home_data_target', '_import_kit_mcp_client', '_lego_size_from_name', '_load_cup_mesh', '_load_lego_mesh', '_load_workspace_yaml', '_normalize_grasp_yaw', '_patch_methods', '_resolve_repo_root', 'check_grasp_reachable', 'datetime', 'find_reachable_grasp_yaw', 'geometric_ik', 'get_package_share_directory', 'json', 'kit_client', 'math', 'namedtuple', 'os', 'random', 'rclpy', 'shutil', 'signal', 'subprocess', 'sys', 'threading', 'time', 'tracer', 'trimesh', 'weakref'] if name in globals()]
