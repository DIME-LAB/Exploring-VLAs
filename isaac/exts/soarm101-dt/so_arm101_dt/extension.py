import omni.ext
import omni.ui as ui
import asyncio
import numpy as np
import os
import sys
import time
import threading
import glob
import shutil
import omni.client
import omni.kit.commands
import carb
import math
import json
import traceback
from typing import Dict, Any

from isaacsim.core.api.world import World
from isaacsim.gui.components.element_wrappers import ScrollingWindow

# rclpy: Isaac Sim 6.1 bundles a Python 3.12 build of ROS 2 Humble's rclpy and message
# packages (isaacsim.ros2.core/humble), loaded by the ROS 2 bridge. Nothing to add to sys.path.

# Reference: ported from exts/ur5e-dt/ur5e_dt/extension.py (Phase 13)
# Video recording output directory — goes next to the extension unless overridden
_VIDEOS_DIR_ENV = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if _VIDEOS_DIR_ENV:
    VIDEOS_DIR = os.path.abspath(os.path.join(_VIDEOS_DIR_ENV, "videos"))
else:
    _ext_dir_vid = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEOS_DIR = os.path.join(_ext_dir_vid, "videos")


class _RecordingState:
    """Holds ffmpeg process and recording state outside the extension class.
    Module-level singleton so recording survives hot-reloads."""
    def __init__(self):
        self.ffmpeg_proc = None
        self.frame_sub = None
        self.recording = False
        self.output_path = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 30
        self.last_print = 0
        self.state_dir = "/tmp/isaacsim_recording"
        self.state_file = os.path.join(self.state_dir, "recorder_state.json")

    def save_state(self):
        os.makedirs(self.state_dir, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"recording": True, "output_path": self.output_path, "fps": self.fps}, f)

    def clear_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def is_recording(self):
        if self.recording and self.ffmpeg_proc is not None:
            return True
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                return state.get("recording", False)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return False


if "_recording_mgr" not in globals():
    _recording_mgr = _RecordingState()


class _ViewportCameraPublisher:
    """Publishes viewport frames to ROS2 topics using capture_viewport_to_buffer.
    Each topic owns its own (optionally hidden) viewport + camera so multiple
    cameras can publish simultaneously without the "all publishers share the
    active viewport" pitfall. Uses a single persistent rclpy node to avoid
    stale DDS participants on hot-reload.
    """

    def __init__(self):
        self._publishers = {}  # {topic_name: entry}
        self._rclpy_initialized = False
        self._node = None

    def _ensure_rclpy(self):
        if not self._rclpy_initialized:
            import rclpy
            try:
                if not rclpy.utilities.get_default_context().ok():
                    rclpy.init()
            except Exception:
                rclpy.init()
            self._rclpy_initialized = True

    def _ensure_node(self):
        import rclpy
        if self._node is None:
            self._ensure_rclpy()
            self._node = rclpy.create_node("soarm101_viewport_publisher")
            print(f"[ViewportPub] Created shared node: {self._node.get_name()}")

    def is_rclpy_available(self):
        try:
            import rclpy  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, camera_prim_path, topic_name, frame_id=None):
        """Publish the MAIN user-facing viewport (pointed at camera_prim_path) to a ROS2 Image topic.

        Reuses the already-rendered viewport frame — zero extra GPU render cost,
        physics-friendly. Forces the main viewport's camera to camera_prim_path so the
        UI viewport AND the topic both reflect that camera's POV. Dedicated-viewport
        mode was removed after empirical measurement showed it costs ~0.3 RTF per
        extra viewport, which defeats the purpose of this class.

        Important: targets the main viewport (named "Viewport", first instance)
        rather than `get_active_viewport()`. The latter can return a secondary
        programmatic viewport (e.g. "Viewport 2") whose camera is updated, but
        whose render frames are NOT what the user sees in the UI — leading to a
        silent mismatch where the topic published Persp content while metadata
        claimed the camera was correctly assigned. (See 2026-04-25 debugging.)
        """
        import rclpy  # noqa: F401
        import omni.kit.viewport.utility as vp_utils
        from sensor_msgs.msg import Image
        import ctypes, array

        if frame_id is None:
            frame_id = topic_name

        if topic_name in self._publishers:
            self.stop(topic_name)

        self._ensure_node()
        pub = self._node.create_publisher(Image, f"/{topic_name}", 10)

        # Pick the MAIN UI viewport — prefer the one named "Viewport", fall back
        # to the first instance, and only use get_active_viewport() as last resort.
        viewport = None
        try:
            from omni.kit.viewport.window import get_viewport_window_instances
            instances = list(get_viewport_window_instances())
            for w in instances:
                name = getattr(w, "name", None) or str(w)
                if name == "Viewport":
                    viewport = w.viewport_api
                    break
            if viewport is None and instances:
                viewport = instances[0].viewport_api
        except Exception as e:
            print(f"[ViewportPub] window enumeration failed ({e}), falling back to active viewport")
        if viewport is None:
            viewport = vp_utils.get_active_viewport()
        viewport.camera_path = camera_prim_path

        entry = {
            "pub": pub, "sub": None, "viewport": viewport,
            "camera_path": camera_prim_path, "frame_id": frame_id,
            "active": True, "count": 0,
        }

        def on_capture(buffer, buffer_size, width, height, fmt):
            if not entry["active"]:
                return
            try:
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                c_arr = (ctypes.c_ubyte * buffer_size).from_address(ptr)
                a = array.array('B'); a.frombytes(c_arr)
                msg = Image()
                msg.header.frame_id = entry["frame_id"]
                msg.height = height; msg.width = width
                msg.encoding = "rgba8"; msg.is_bigendian = False
                msg.step = width * 4
                msg.data = a
                entry["pub"].publish(msg)
                entry["count"] += 1
            except Exception:
                pass

        def on_frame(event):
            if not entry["active"]:
                return
            vp_utils.capture_viewport_to_buffer(entry["viewport"], on_capture)

        entry["sub"] = viewport.subscribe_to_frame_change(on_frame)
        self._publishers[topic_name] = entry
        print(f"[ViewportPub] Started /{topic_name} from {camera_prim_path} (active viewport reuse)")

    def stop(self, topic_name):
        if topic_name not in self._publishers:
            return
        entry = self._publishers[topic_name]
        entry["active"] = False
        entry["sub"] = None
        if self._node is not None:
            try:
                self._node.destroy_publisher(entry["pub"])
            except Exception as e:
                print(f"[ViewportPub] destroy_publisher warning: {e}")
        del self._publishers[topic_name]
        print(f"[ViewportPub] Stopped /{topic_name}")

    def stop_all(self):
        for topic in list(self._publishers.keys()):
            self.stop(topic)

    def is_active(self, topic_name=None):
        if topic_name:
            return topic_name in self._publishers and self._publishers[topic_name]["active"]
        return any(e["active"] for e in self._publishers.values())

    def get_status(self):
        return {t: {"camera": e["camera_path"], "active": e["active"], "count": e["count"]}
                for t, e in self._publishers.items()}


# Module-level singleton — survives hot-reloads via globals() guard
if "_viewport_pub" not in globals():
    _viewport_pub = _ViewportCameraPublisher()


class _ReplicatorCameraPublisher:
    """Publishes a camera prim's rendered frames to a ROS2 Image topic via
    a dedicated hidden viewport. Bypasses the Kit ROS2CameraHelper
    action-graph node, which silently fails to register DDS publishers on
    Isaac Sim 5.0 / Kit 107.3 / isaacsim.ros2.bridge 4.9.3 — graph nodes
    compute, but no publisher ever appears on the wire.

    Why a hidden viewport (not replicator annotators or isaacsim Camera):
    both of those rely on rep.orchestrator stepping to populate frames,
    which only runs in standalone mode (not during Kit play). Viewports
    render every frame regardless. Each hidden viewport costs ~0.3 RTF
    — acceptable for recording use.

    Use this for cameras that aren't the active UI viewport. For the one
    camera you also want to see in the UI, _ViewportCameraPublisher
    (active-viewport reuse) is free (~0 RTF).
    """

    def __init__(self):
        self._publishers = {}  # {topic_name: entry}

    def is_rclpy_available(self):
        try:
            import rclpy  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, camera_prim_path, topic_name, width=640, height=480, frame_id=None):
        """Create a hidden viewport pointed at the camera prim and publish
        each rendered frame as sensor_msgs/Image on the topic."""
        import rclpy  # noqa: F401
        import omni.kit.viewport.utility as vp_utils
        from omni.kit.viewport.window import ViewportWindow
        from sensor_msgs.msg import Image
        import ctypes, array

        if frame_id is None:
            frame_id = topic_name

        if topic_name in self._publishers:
            self.stop(topic_name)

        # Reuse _ViewportCameraPublisher's persistent rclpy node to keep all
        # extension publishers on a single DDS participant.
        global _viewport_pub
        _viewport_pub._ensure_node()
        node = _viewport_pub._node
        pub = node.create_publisher(Image, f"/{topic_name}", 10)

        # Create a dedicated viewport window pointed at the target camera.
        # We hide the window from the UI but force updates_enabled=True so
        # the viewport keeps rendering — the kit default disables updates
        # on hidden windows to save GPU, which is the wrong tradeoff for us.
        window_name = f"HiddenViewport_{topic_name}"
        viewport_window = ViewportWindow(window_name, width=width, height=height)
        viewport_window.visible = False
        viewport = viewport_window.viewport_api
        # Set camera_path eagerly — and re-set it defensively in on_frame for
        # the first few frames. Kit silently drops the assignment if the
        # viewport_api isn't fully constructed yet, leaving the viewport
        # rendering /OmniverseKit_Persp (verified 2026-04-27). Late-binding the
        # assignment in the frame callback is the only reliable fix found.
        viewport.camera_path = camera_prim_path
        viewport.resolution = (width, height)
        try:
            viewport.updates_enabled = True
        except Exception:  # noqa: BLE001
            pass  # API may be read-only on some Kit versions; subscribe still works

        entry = {
            "pub": pub, "sub": None,
            "viewport": viewport, "window": viewport_window,
            "camera_path": camera_prim_path, "frame_id": frame_id,
            "width": width, "height": height,
            "active": True, "count": 0,
            "camera_set_attempts": 0,  # tracks defensive re-binding in on_frame
        }

        def on_capture(buffer, buffer_size, vp_width, vp_height, fmt):
            if not entry["active"]:
                return
            try:
                import array
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                c_arr = (ctypes.c_ubyte * buffer_size).from_address(ptr)
                a = array.array('B'); a.frombytes(c_arr)
                msg = Image()
                msg.header.frame_id = entry["frame_id"]
                msg.height = vp_height; msg.width = vp_width
                msg.encoding = "rgba8"; msg.is_bigendian = False
                msg.step = vp_width * 4
                msg.data = a
                entry["pub"].publish(msg)
                entry["count"] += 1
            except Exception:
                pass

        def on_frame(_event):
            if not entry["active"]:
                return
            # Defensive re-bind for first ~10 frames: Kit drops camera_path
            # assignment when the viewport_api isn't fully constructed at start()
            # time. Once the assignment sticks, stop trying.
            if entry["camera_set_attempts"] < 10:
                vp = entry["viewport"]
                if str(vp.camera_path) != str(entry["camera_path"]):
                    vp.camera_path = entry["camera_path"]
                entry["camera_set_attempts"] += 1
            vp_utils.capture_viewport_to_buffer(entry["viewport"], on_capture)

        entry["sub"] = viewport.subscribe_to_frame_change(on_frame)
        self._publishers[topic_name] = entry
        print(f"[ReplicatorPub] Started /{topic_name} from {camera_prim_path} ({width}x{height}) via hidden viewport")

    def stop(self, topic_name):
        if topic_name not in self._publishers:
            return
        entry = self._publishers[topic_name]
        entry["active"] = False
        entry["sub"] = None  # drops the frame_change subscription
        # Destroy the hidden viewport window
        try:
            if entry.get("window"):
                entry["window"].destroy()
        except Exception as e:  # noqa: BLE001
            print(f"[ReplicatorPub] window destroy warning: {e}")
        global _viewport_pub
        if _viewport_pub._node is not None:
            try:
                _viewport_pub._node.destroy_publisher(entry["pub"])
            except Exception as e:  # noqa: BLE001
                print(f"[ReplicatorPub] destroy_publisher warning: {e}")
        del self._publishers[topic_name]
        print(f"[ReplicatorPub] Stopped /{topic_name}")

    def stop_all(self):
        for topic in list(self._publishers.keys()):
            self.stop(topic)

    def is_active(self, topic_name=None):
        if topic_name:
            return topic_name in self._publishers and self._publishers[topic_name]["active"]
        return any(e["active"] for e in self._publishers.values())

    def get_status(self):
        return {t: {"camera": e["camera_path"], "active": e["active"], "count": e["count"]}
                for t, e in self._publishers.items()}


# Module-level singleton — survives hot-reloads via globals() guard
if "_replicator_pub" not in globals():
    _replicator_pub = _ReplicatorCameraPublisher()

# MCP output directory configuration
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    BASE_OUTPUT_DIR = os.path.abspath(BASE_OUTPUT_DIR)
    RESOURCES_DIR = os.path.join(BASE_OUTPUT_DIR, "resources")
else:
    RESOURCES_DIR = "resources"

# Object loading configuration (local assets only)
def _get_assets_folder():
    _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return "file://" + os.path.abspath(os.path.join(_ext_dir, "assets")) + "/"


OBJECTS_CONFIG = {
    "folder": None,  # Resolved at runtime via _objects_folder_path
    "skip_patterns": ["ARM101"],  # skip robot USDs when loading objects
}

# Shared color palette — cups and legos use the same colors so they visually match.
# Values are in LINEAR RGB (what USD materials expect).
OBJECT_COLORS = {
    "red":    (1.000, 0.036, 0.000),   # bright red-orange
    "green":  (0.182, 0.404, 0.014),   # deep grass green
    "blue":   (0.000, 0.138, 0.487),
    # "yellow": (0.863, 0.765, 0.110),
}
BLOCK_COLORS = OBJECT_COLORS

def _linear_to_srgb(c):
    """Convert a single linear RGB channel to sRGB."""
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055

def _srgb_to_linear(c):
    """Convert a single sRGB channel to linear RGB."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

# Lego assets folder and per-color USD files (local only)
def _get_lego_folder():
    _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local = os.path.join(_ext_dir, "assets", "legos")
    if not os.path.isdir(_local):
        raise FileNotFoundError(
            f"Lego assets folder not found at {_local}. "
            "Place lego USD files in exts/soarm101-dt/assets/legos/"
        )
    return "file://" + os.path.abspath(_local) + "/"


def _get_camera_mount_usd_path():
    """Local assets/wrist_mounts/ only."""
    _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local = os.path.join(_ext_dir, "assets", "wrist_mounts", "camera_wrist_mount.usd")
    if not os.path.isfile(_local):
        raise FileNotFoundError(
            f"Camera mount USD not found at {_local}. "
            "Place camera_wrist_mount.usd in exts/soarm101-dt/assets/wrist_mounts/"
        )
    return "file://" + os.path.abspath(_local)

def _get_usb_camera_usd_path():
    """Local assets/cameras/ only."""
    _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local = os.path.join(_ext_dir, "assets", "cameras", "usb_cam_elp.usd")
    if not os.path.isfile(_local):
        raise FileNotFoundError(
            f"USB camera USD not found at {_local}. "
            "Place usb_cam_elp.usd in exts/soarm101-dt/assets/cameras/"
        )
    return "file://" + os.path.abspath(_local)


# Each color maps to a list of unique USD filenames — one per block instance.
# Each USD has a uniquely-named body prim matching its filename (e.g. red_2x2).
#
# We tried duplicating USD references for "2 of each color × 2x3 only" but
# PhysX hangs in friction-patch generation when two rigid bodies share an
# inner prim name (verified 2026-04-27 — quick_start hung after the 32-patch
# limit warning). Going back to one-USD-per-instance with unique inner prim
# names avoids the issue. Until we either rename inner prims post-reference
# or generate per-instance USDs, the count is 3 sizes × 3 colors = 9 legos.
# Each color maps to a list of (basename, usd_filename, count) tuples.
# add_objects spawns `count` instances of each USD, named "{basename}_0",
# "{basename}_1", ... etc. The first instance is created via AddReference
# and the rest via isaacsim.core.cloner.Cloner(replicate_physics=True),
# which registers per-instance PhysX views — avoiding the friction-patch
# hang from naive multi-reference of the same source USD.
#
# Index-suffix naming is required because Cloner's PhysX replicator uses
# `root_path + str(index)` for per-instance physics view registration.
# The wrapper prim name and the USD's inner body prim name no longer
# match for index>0, so _get_prim_path falls back to structural lookup
# (find first child with UsdPhysics.RigidBodyAPI).
#
# Current scene contract: 2 of each color × 2x3 = 6 legos. Recording-side
# REC_LEGOS_BY_COLOR (control_gui.py) MUST mirror the generated names.
LEGO_USDS = {
    "red":    [("red_2x3", "lego_red_2x3.usd", 2)],
    "green":  [("green_2x3", "lego_green_2x3.usd", 2)],
    "blue":   [("blue_2x3", "lego_blue_2x3.usd", 2)],
}

# =============================================================================
# ROBOT COORDINATE FRAME
# =============================================================================
# All positions below are defined in robot-relative coordinates:
#   forward  = along the robot's facing direction (toward workspace)
#   lateral  = perpendicular to forward (positive = left of robot)
#   up       = +Z always
#
# ROBOT_FORWARD_AXIS maps robot-relative → world coordinates:
#   "X" → forward=+X, lateral=+Y   (SO-ARM101 default)
#   "Y" → forward=+Y, lateral=-X
# =============================================================================
ROBOT_FORWARD_AXIS = "X"

def _get_pan_axis_xy():
    """Get the shoulder_pan axis world XY position from the robot USD.

    Returns (x, y) of the shoulder_link prim, which is the pivot point
    about which the arm rotates.  Falls back to (0, 0) if robot not loaded.
    """
    stage = omni.usd.get_context().get_stage()
    for name in ["shoulder_link", "Shoulder_Link"]:
        prim = stage.GetPrimAtPath(f"/World/SO_ARM101/{name}")
        if prim and prim.IsValid():
            from pxr import UsdGeom
            tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            pos = tf.ExtractTranslation()
            return float(pos[0]), float(pos[1])
    return 0.0, 0.0


def _to_world(forward, lateral, anchor=None):
    """Convert (forward, lateral) robot-relative coords to (world_x, world_y).

    anchor: (x, y) world offset to add (e.g. pan axis position).
             If None, uses world origin.
    """
    ax, ay = anchor if anchor else (0.0, 0.0)
    if ROBOT_FORWARD_AXIS == "X":
        return forward + ax, lateral + ay
    else:  # "Y"
        return -lateral + ax, forward + ay

# Block workspace (robot-relative)
BLOCK_SPAWN_FORWARD = 0.3     # forward distance for initial row layout
# Phase 9: bounds tightened from (0.10, 0.25) → (0.135, 0.285) after the
# self-collision-aware reachability sweep (`compute_workspace --with-gate-c`)
# showed that the original Gate-B-only sweep over-reports reachability by
# including configurations inside the self-collision envelope (camera_mount
# ↔ shoulder, shoulder ↔ gripper, usb_camera ↔ upper_arm) at low r.
# Authoritative workspace bounds live in
# src/so_arm101_control/so_arm101_control/workspace_bounds.yaml →
# grasp_workspace_bounds: r_min=0.1205 r_max=0.3095 z_min=0.014 z_max=0.086
# Spawn range accounts for the jaw-offset that _compute_jaw_offset applies
# (shifts grasp target ~10 mm from block center toward the fixed jaw tip).
# So block r must stay at least 10 mm INSIDE the workspace r bounds:
#   r_min = workspace_r_min (0.1205) + 10 mm jaw + 4.5 mm safety = 0.135
#   r_max = workspace_r_max (0.3095) − 10 mm jaw − 14.5 mm safety = 0.285
# Verified via cycle 1 trace: block at r=0.126 → grasp target at r=0.116
# → 'too close: r=0.116 < 0.120'. Moving r_min from 0.125 to 0.135 fixes
# this by construction.
BLOCK_RANDOM_FORWARD = (0.135, 0.285)
BLOCK_RANDOM_LATERAL = (-0.10, 0.10)

# Main ground plane (collision + visual). Covers the entire scene so any
# lego or cup placed outside the workspace still rests on the floor.
# Color + roughness control the "outdoors" look beyond the workspace
# tile defined below.
GROUND_PLANE_SIZE_M = 5000               # square edge length in meters
GROUND_PLANE_COLOR = (0.647, 0.648, 0.497)  # warm tan wall — picker-tuned via sim-color-matcher (ratio apply)
GROUND_PLANE_ROUGHNESS = 1.0             # fully diffuse

# Workspace ground tile — visual-only black quad that sits on top of
# /World/defaultGroundPlane in the active pick-place area. Gives
# camera frames a clean black backdrop for dataset capture without
# turning the whole scene pitch black. Per-side extents in world
# frame; the UI panel edits these values and re-authors the tile.
# Set WORKSPACE_GROUND_X_RANGE to None to disable the tile entirely.
WORKSPACE_GROUND_X_RANGE = (-0.04, 0.70)    # (x_min, x_max) world-frame X (forward), meters
WORKSPACE_GROUND_Y_RANGE = (-0.40, 0.30)    # (y_min, y_max) world-frame Y (lateral), meters — propagates to WORKSPACE_EXT_TILE_Y_RANGE below
WORKSPACE_GROUND_Z_OFFSET_M = 0.002         # lift above main ground (z-fight avoidance)
WORKSPACE_GROUND_COLOR = (0.035, 0.058, 0.077) # dark slate table — picker-tuned via sim-color-matcher (ratio apply)
WORKSPACE_GROUND_ROUGHNESS = 1.0            # fully diffuse

# Extension tile — visual-only floor patch joining the main workspace tile on its
# -X edge. Independent material so it can be color-matched separately. The tile
# joins at x=WORKSPACE_GROUND_X_RANGE[0] and extends 1.0m further in -X.
WORKSPACE_EXT_TILE_X_RANGE = (-1.04, -0.04)
WORKSPACE_EXT_TILE_Y_RANGE = WORKSPACE_GROUND_Y_RANGE   # same Y span (visually continuous)
WORKSPACE_EXT_TILE_COLOR = (0.18941, 0.08416, 0.06668)   # warm brown — picker-tuned (re-take after restart) via sim-color-matcher
WORKSPACE_EXT_TILE_ROUGHNESS = 1.0

# "Interior daylight — robot on a table near two windows"
# Designed to match a real-world reference (warm interior, cream wall, soft window
# light from upper-left, no direct sun visible on subject). Verified against UsdLux:
#   - inputs:intensity is luminance in nits (cd/m²) per LightAPI spec
#   - RectLight width/height are scene units (meters; metersPerUnit=1)
#   - RectLight emits along local -Z; quaternions rotate -Z to face into the room
#   - normalize=False (default) means larger area = more total flux at same intensity
INTERIOR_DAYLIGHT_PRESET = (
    # Key window: positioned high + far on -Y so its falloff doesn't streak the floor.
    # Size widened to 4x3m for soft coverage. Intensity tuned with broad overhead fill.
    {"kind": "RectLight", "name": "Window_Key", "intensity": 2500.0,
     "width": 4.0, "height": 3.0, "color": (1.00, 0.97, 0.92),
     "translate": (0.4, -2.5, 2.5),
     "quat_xyzw": (0.70711, 0.0, 0.0, 0.70711)},   # face +Y (into room)
    # Fill: opposite side, smaller + warmer (mimics warm interior bounce)
    {"kind": "RectLight", "name": "Fill_Right", "intensity": 1500.0,
     "width": 1.5, "height": 1.5, "color": (1.00, 0.92, 0.82),
     "translate": (0.4,  1.2, 1.0),
     "quat_xyzw": (-0.70711, 0.0, 0.0, 0.70711)},  # face -Y (into room)
    # Top fill: large overhead RectLight that flattens floor color front-to-back.
    # Eliminates the streak gradient the side-key would otherwise leave on the floor.
    {"kind": "RectLight", "name": "Top_Fill", "intensity": 1000.0,
     "width": 6.0, "height": 6.0, "color": (1.00, 0.98, 0.95),
     "translate": (0.5, 0.0, 4.0),
     "quat_xyzw": (0.0, 0.0, 0.0, 1.0)},   # identity — RectLight emits along -Z (down)
    # Heavy ambient bounce dome — warm cream (sim of white walls + warm furniture).
    # Provides the "everything is visible" effect of a real interior.
    {"kind": "DomeLight", "name": "Bounce", "intensity": 1000.0,
     "color": (1.00, 0.96, 0.90),
     "translate": (0.0, 0.0, 0.0),
     "quat_xyzw": (0.0, 0.0, 0.0, 1.0)},
)

# Cup container assets (one per block color)
CUP_USDS = {
    "red":   "cup_red.usd",
    "green": "cup_green.usd",
    "blue":  "cup_blue.usd",
}

# Cup placement config (robot-relative)
#
# mode: "arc"  — cups spread along a circular arc (each cup at its own angle)
#        "line" — cups in a straight row; center cup at the angle, others extend along lateral axis
#
# radius:      distance from robot base to center cup
# angle_deg:   direction of center cup (0°=straight ahead, negative=right, positive=left)
# gap:         spacing between cup edges in "line" mode (meters). Computed from bbox at runtime.
# color_order: left-to-right order when facing the cups
#
# In "arc" mode, angles_deg overrides angle_deg — one angle per cup.
CUP_LAYOUT_DEFAULTS = {
    "mode": "line",
    "radius": 0.28,
    "angle_deg": -90,
    "angles_deg": [-35, 0, 35],
    "gap": 0.01,
    "face_origin": True,       # rotate cups to face the robot base (perpendicular to radial)
    "color_order": ["red", "green", "blue"],
    # Cluster-level offset in WORLD frame. Applied AFTER _to_world() so
    # cups still face the pan axis via face_origin (the cluster
    # translates without re-aiming). Honored by add_cups,
    # _add_cups_from_ui, and randomize_cups.
    "cluster_offset_x": 0.0,   # meters; +X = forward of pan axis
    "cluster_offset_y": 0.0,   # meters; +Y = lateral (left of robot for ROBOT_FORWARD_AXIS="X")
}
CUP_LAYOUT = dict(CUP_LAYOUT_DEFAULTS)

# Real-world calibration overrides — applied by the "Match Real World"
# UI button and automatically during quick_start. Only the keys listed
# here are overwritten on CUP_LAYOUT; everything else (mode, radius,
# angle_deg, color_order, face_origin) keeps whatever value the user has
# set in CUP_LAYOUT_DEFAULTS or the live UI.
CUP_LAYOUT_REAL_WORLD = {
    "cluster_offset_x": 0.12,
    "cluster_offset_y": 0.04,   # was 0.06 — re-calibrated against real rig
    "gap": 0.02,
}

# ArUco markers on cup surfaces — one marker per cup, placed on the side facing the robot.
# dictionary: ArUco dictionary name
# marker_size_m: physical width of the full printed marker in meters,
#   matching the standard ArUco convention (the outer black square,
#   black border bit included). With the white quiet zone disabled in
#   generate_aruco_marker.py, the PNG is the bare 6x6-cell ArUco marker
#   (4x4 data bits + 1 black border bit per side). The pure-data-bit
#   area inside is 4/6 = 66.7% of marker_size_m.
# height_fraction: how far up the cup side (0=bottom, 1=top rim)
CUP_ARUCO_CONFIG = {
    "dictionary": "DICT_4X4_50",
    "marker_size_m": 0.035,       # 35mm full marker (incl. black border)
    "marker_png_pixels": 200,     # resolution of generated PNG
    "height_fraction": 0.5855,    # marker center 40mm below the top rim
                                  # (cup body height ~96.5mm × 0.5855 ≈ 56.5mm
                                  # from bottom = 40mm from top)
    "ids": {                      # ArUco ID per cup color (blue=1, green=2, red=3).
        "red": 3,                 # /drop_poses publishes drop_1 (blue),
        "green": 2,               # drop_2 (green), drop_3 (red).
        "blue": 1,                # Mirrored in vla_SO-ARM101 control_gui.py
    },                            # and aruco_camera_localizer aruco_config.json.
}

def _cup_positions_arc():
    """Compute cup (forward, lateral) from arc config.

    angle_deg: center direction of the arc
    gap: angular spacing between cups (converted from meters to degrees via arc length)
    """
    colors = CUP_LAYOUT["color_order"]
    radius = CUP_LAYOUT["radius"]
    center_deg = CUP_LAYOUT["angle_deg"]
    gap_m = CUP_LAYOUT["gap"]

    # Convert gap (meters) to angular spacing (degrees) along the arc
    # arc_length = radius * angle_rad  →  angle_deg = gap / radius * 180/pi
    # Add cup width (~0.078m) to gap for edge-to-edge spacing
    cup_width = 0.078
    spacing_deg = math.degrees((cup_width + gap_m) / radius)

    # Center the spread around angle_deg
    n = len(colors)
    total_spread = spacing_deg * (n - 1)
    start_deg = center_deg - total_spread / 2.0

    positions = {}
    for i, color in enumerate(colors):
        angle_rad = math.radians(start_deg + spacing_deg * i)
        fwd = radius * math.cos(angle_rad)
        lat = radius * math.sin(angle_rad)
        positions[color] = (fwd, lat)
    return positions

def _cup_positions_line(cup_widths):
    """Compute cup (forward, lateral) in a straight row.

    Center cup sits at (radius * cos(angle), radius * sin(angle)).
    Other cups extend **perpendicular** to the direction from the robot
    base to the center cup, spaced by bbox widths + gap.

    At angle=0 (front):  row extends along lateral axis
    At angle=-90 (right): row extends along forward axis
    At angle=45 (front-left): row extends diagonally

    Args:
        cup_widths: dict {color: width_meters} — extent of each cup bbox.
    """
    colors = CUP_LAYOUT["color_order"]
    gap = CUP_LAYOUT["gap"]
    angle_rad = math.radians(CUP_LAYOUT["angle_deg"])

    # Center cup position in robot-relative (forward, lateral)
    center_fwd = CUP_LAYOUT["radius"] * math.cos(angle_rad)
    center_lat = CUP_LAYOUT["radius"] * math.sin(angle_rad)

    # Perpendicular direction to the radial line (rotated 90° CCW)
    # If radial direction is (cos(a), sin(a)), perpendicular is (-sin(a), cos(a))
    perp_fwd = -math.sin(angle_rad)
    perp_lat = math.cos(angle_rad)

    # Compute offsets along the perpendicular
    widths = [cup_widths.get(c, 0.08) for c in colors]
    total = sum(widths) + gap * (len(widths) - 1)
    cursor = -total / 2.0

    positions = {}
    for i, color in enumerate(colors):
        w = widths[i]
        offset = cursor + w / 2.0
        positions[color] = (
            center_fwd + offset * perp_fwd,
            center_lat + offset * perp_lat,
        )
        cursor += w + gap
    return positions

def _apply_cup_cluster_offset(positions):
    """Shift a dict of cup positions by CUP_LAYOUT's cluster_offset_x/y.

    Operates in robot-relative (forward, lateral) space. Offset values
    are authored as WORLD-frame XY for intuitive UX ("shift the cluster
    by 5 cm in world X"), so we convert to robot-relative based on
    ROBOT_FORWARD_AXIS:

      "X" axis: world (dx, dy) ≡ robot-relative (dfwd, dlat)
      "Y" axis: world (dx, dy) ≡ robot-relative (-dlat, dfwd) → invert

    Applied BEFORE _to_world() so the offset participates in any
    downstream validation (e.g. randomize_cups' lego/robot-AABB checks).
    Returns a fresh dict; caller's input dict is untouched.
    """
    ox = CUP_LAYOUT.get("cluster_offset_x", 0.0)
    oy = CUP_LAYOUT.get("cluster_offset_y", 0.0)
    if not ox and not oy:
        return positions  # no-op fast path
    if ROBOT_FORWARD_AXIS == "X":
        fwd_off, lat_off = ox, oy
    else:  # "Y"
        fwd_off, lat_off = oy, -ox
    return {c: (f + fwd_off, l + lat_off) for c, (f, l) in positions.items()}


def _get_cups_folder():
    """Local assets/cups/ only."""
    _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _local = os.path.join(_ext_dir, "assets", "cups")
    if not os.path.isdir(_local):
        raise FileNotFoundError(
            f"Cup assets folder not found at {_local}. "
            "Place cup USD files in exts/soarm101-dt/assets/cups/"
        )
    return "file://" + os.path.abspath(_local) + "/"


# =============================================================================
# MCP TOOL REGISTRY - Single source of truth for all MCP tools
# =============================================================================
# Each tool definition includes:
#   - description: Tool description shown to MCP clients
#   - parameters: Dict of parameter definitions (name -> {type, description, required, default})
#
# The MCP server queries this registry on startup and dynamically registers tools.
# To add a new tool:
#   1. Add entry here with description and parameters
#   2. Add handler method _cmd_<tool_name>() in this extension
#   3. That's it! The server will automatically discover and expose it.
# =============================================================================

# Module-level state for direct PhysX contact-event subscription.
# Populated by DigitalTwin._setup_contact_sensors(). Exposed at module
# scope so motion_logger's snapshot script (which runs inline via
# execute_python_code) can drain events in the same tick as DOF state.
#
# We bypass isaacsim.sensors.physics.ContactSensor entirely — that
# wrapper depends on `omni.physx.contact`, which is absent in Isaac Sim
# 5.0 under that name, so every sensor came back is_valid=False. The
# core PhysX simulation interface (`omni.physx.get_physx_simulation_interface()`)
# exposes subscribe_contact_report_events natively and fires for any
# prim that has PhysxContactReportAPI applied. That's what we use here.
#
# _CONTACT_EVENTS is a bounded deque of dict events appended by the
# physics-thread callback; the MCP tool / motion_logger drain it.
from collections import deque as _deque
_CONTACT_EVENTS = _deque(maxlen=2048)
_CONTACT_SUB = None               # carb.Subscription (keep alive)
_CONTACT_WATCHED_PATHS = set()    # path prefixes we care about (cups + robot links)
# Back-compat alias — older snapshot scripts check this for "sensors installed".
_CONTACT_SENSORS = {}


# Tool table (description + params + handler method per command) lives in a
# pure-data sibling module so the omni-kit-mcp tool package can read it
# without importing this Kit-dependent module.
from .tool_table import MCP_TOOL_REGISTRY, MCP_HANDLERS  # noqa: E402,F401

# The running DigitalTwin, so the omni-kit-mcp tool package can forward soarm101.<name>
# calls to its _cmd_* methods. Set at the end of on_startup, cleared in on_shutdown.
_INSTANCE = None


def get_instance():
    """The running DigitalTwin extension instance, or None when it is not started."""
    return _INSTANCE

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation as ArticulationView
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.prims import SingleArticulation as Articulation
import omni.graph.core as og
import omni.usd
from pxr import UsdGeom, Gf, Sdf, Usd, UsdPhysics, UsdShade, PhysxSchema
from scipy.spatial.transform import Rotation as R


class LogRedirector:
    """Redirects stdout/stderr to both the original stream and a UI callback."""
    def __init__(self, original, callback, throttle_seconds=0.0):
        self._original = original
        self._callback = callback
        self._throttle = throttle_seconds
        self._last_msgs = {}  # dedup cache: msg -> timestamp

    def write(self, text):
        self._original.write(text)
        stripped = text.strip()
        if not stripped:
            return
        if self._throttle > 0:
            import time as _t
            now = _t.monotonic()
            if stripped in self._last_msgs and now - self._last_msgs[stripped] < self._throttle:
                return
            self._last_msgs[stripped] = now
            if len(self._last_msgs) > 200:
                cutoff = now - self._throttle * 3
                self._last_msgs = {k: v for k, v in self._last_msgs.items() if v > cutoff}
        self._callback(text.rstrip("\n"))

    def flush(self):
        self._original.flush()


class DigitalTwin(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[SO-ARM101-DT] Digital Twin startup")

        # Ported from isaac-sim-mcp 109363a (ur5e-dt, "fix PhysX cooking wedge — disable UJITSO
        # multiprocess collision cooking"). UJITSO, the default collision cooker, forks
        # subprocesses; forking from the live multithreaded Kit app deadlocks in carb.tasking ->
        # omni.physx.cooking. With a cold DerivedDataCache that froze quick_start inside
        # load_robot's reset_async (reproduced on 6.1: main thread in futex_wait, bridge silent).
        # In-process cooking produces the same collider. Session-wide, set before any cook.
        try:
            carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
        except Exception as e:
            carb.log_warn(f"[SO-ARM101-DT] could not disable UJITSO collision cooking: {e}")

        self._main_loop = asyncio.get_event_loop()
        self._timeline = omni.timeline.get_timeline_interface()
        self._robot_view = None
        self._articulation = None
        self._gripper_view = None
        self._viewport_rendering_enabled = True
        self._viewport_toggle_btn = None

        # Force publisher physics callback state
        self._force_physx_sub = None
        self._effort_articulation = None
        self._force_graph_path = None
        self._force_pub_node = None
        self._force_warmup = 0

        # BBox publisher state
        self._bbox_graph_path = None
        self._bbox_pub_attr_path = None
        self._bbox_physx_sub = None

        # Add Objects UI state
        self._objects_config = OBJECTS_CONFIG
        self._object_gap = 0.02  # Gap (meters) between object bounding boxes
        self._y_offset = 0.20  # Y offset for all objects
        self._ground_plane_z = 0.0  # Ground plane Z position
        self._robot_base_z_offset = 0.0
        self._robot_base_rpy_deg = np.array([0.0, 0.0, 0.0])  # Robot base orientation (roll, pitch, yaw)
        self._hidden_objects = {}  # {name: {ref_path, body_pose, category}}

        # Output directory pushed by MCP server on connect (None = use default)
        self._output_dir = None

        # Physics scene settings
        self._min_frame_rate = 60
        self._time_steps_per_second = 120

        # SO-ARM101 joint drive parameters - hardware defaults (ST3215: 30 kg·cm ≈ 3 Nm)
        # UR5e stiffness/max_force=0.5, damping/max_force=0.05 ratios
        self._robot_max_force = 3.0        # STS3215 stall torque (30 kg·cm)
        # self._robot_stiffness = 1.5       # 0.5 × max_force (hardware default)
        self._robot_stiffness = 15.0       # real2sim: tuned for sim behavior
        self._robot_damping = 0.15         # 0.05 × max_force

        # SO-ARM101 gripper - same ST3215 servo
        self._gripper_max_force = 3.0      # 30 kg·cm
        # self._gripper_stiffness = 1.5     # 0.5 × max_force (hardware default)
        self._gripper_stiffness = 15.0     # real2sim: tuned for sim behavior
        self._gripper_damping = 0.15       # 0.05 × max_force

        # Joint name mapping: real robot ROS2 names → USD names (same kinematic order)
        self._joint_name_map = {
            "Rotation": "shoulder_pan",
            "Pitch": "shoulder_lift",
            "Elbow": "elbow_flex",
            "Wrist_Pitch": "wrist_flex",
            "Wrist_Roll": "wrist_roll",
            "Jaw": "gripper_joint",
        }
        # USD joint names in kinematic chain order (matches URDF order)
        self._usd_joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper_joint",
        ]

        # Gripper physics material (TODO: tune for SO-ARM101 gripper)
        self._gripper_dynamic_friction = 0.6
        self._gripper_static_friction = 0.5
        self._gripper_restitution = 0.0
        self._gripper_friction_combine_mode = "max"
        self._gripper_restitution_combine_mode = "min"

        # Common physics material (applied to all objects)
        self._object_dynamic_friction = 0.1
        self._object_static_friction = 0.1
        self._object_restitution = 0.0
        self._object_friction_combine_mode = "max"
        self._object_restitution_combine_mode = "min"

        # Shared collision settings
        self._sdf_resolution = 300
        self._contact_offset = 0.003

        # Per-category prim settings (collision approximation, rest offset, angular damping)
        # Board (type: "board", e.g. base1, base2, base3)
        self._board_collision_approximation = "sdf"  # "sdf" or "convexDecomposition"
        self._board_rest_offset = 0.0
        self._board_angular_damping = 30.0

        # Block (subtype: "block", e.g. u_brown, fork_orange, line_green)
        self._block_collision_approximation = "sdf"
        self._block_rest_offset = -0.0005
        self._block_angular_damping = 30.0

        # Socket (subtype: "socket", e.g. inverted_u_brown)
        self._socket_collision_approximation = "sdf"
        self._socket_rest_offset = -0.0005
        self._socket_angular_damping = 30.0

        # Peg (subtype: "peg", e.g. hex_blue, hex_red)
        self._peg_collision_approximation = "sdf"
        self._peg_rest_offset = -0.0005
        self._peg_angular_damping = 50.0

        # Isaac Sim handles ROS2 initialization automatically through its bridge
        print("ROS2 bridge will be initialized by Isaac Sim when needed")

        # Create the window UI
        self._window = ScrollingWindow(title="SO-ARM101 Digital Twin", width=300, height=800)
        with self._window.frame:
            with ui.VStack(spacing=5):
                self.create_ui()

        global _INSTANCE
        _INSTANCE = self

    def create_ui(self):
        with ui.VStack(spacing=5):
            with ui.CollapsableFrame(title="Simulation Setup", collapsed=False, height=0):
                with ui.VStack(spacing=5, height=0):
                    with ui.HStack(spacing=5):
                        ui.Button("Load Scene", width=100, height=35, clicked_fn=lambda: asyncio.ensure_future(self.load_scene()))
                        self._viewport_toggle_btn = ui.Button("Disable Viewport", width=140, height=35, clicked_fn=self._toggle_viewport_rendering)
                        ui.Button("Quick Start", width=120, height=35, clicked_fn=lambda: asyncio.ensure_future(self.quick_start()))

            with ui.CollapsableFrame(title="Workspace Ground", collapsed=True, height=0):
                with ui.VStack(spacing=5, height=0):
                    # 4 per-side extent fields + Apply / Reset buttons.
                    # Edits the visual workspace tile in place — the
                    # main /World/defaultGroundPlane (collision) is
                    # untouched.
                    self._wsg_x_min_model = ui.SimpleFloatModel(WORKSPACE_GROUND_X_RANGE[0])
                    self._wsg_x_max_model = ui.SimpleFloatModel(WORKSPACE_GROUND_X_RANGE[1])
                    self._wsg_y_min_model = ui.SimpleFloatModel(WORKSPACE_GROUND_Y_RANGE[0])
                    self._wsg_y_max_model = ui.SimpleFloatModel(WORKSPACE_GROUND_Y_RANGE[1])
                    with ui.HStack(spacing=5, height=0):
                        ui.Label("X (forward) min:", width=120)
                        ui.FloatField(model=self._wsg_x_min_model, width=70)
                        ui.Label("max:", width=40)
                        ui.FloatField(model=self._wsg_x_max_model, width=70)
                    with ui.HStack(spacing=5, height=0):
                        ui.Label("Y (lateral) min:", width=120)
                        ui.FloatField(model=self._wsg_y_min_model, width=70)
                        ui.Label("max:", width=40)
                        ui.FloatField(model=self._wsg_y_max_model, width=70)
                    with ui.HStack(spacing=5, height=0):
                        ui.Button("Apply", width=80, height=30,
                                  clicked_fn=self._cmd_apply_workspace_ground)
                        ui.Button("Reset to Defaults", width=140, height=30,
                                  clicked_fn=self._cmd_reset_workspace_ground)

            with ui.CollapsableFrame(title="Workspace Ground Extension (-X)", collapsed=True, height=0):
                with ui.VStack(spacing=5, height=0):
                    # Mirror layout of Workspace Ground panel — independent
                    # tile joining the main one on its -X edge.
                    self._wsg_ext_x_min_model = ui.SimpleFloatModel(WORKSPACE_EXT_TILE_X_RANGE[0])
                    self._wsg_ext_x_max_model = ui.SimpleFloatModel(WORKSPACE_EXT_TILE_X_RANGE[1])
                    self._wsg_ext_y_min_model = ui.SimpleFloatModel(WORKSPACE_EXT_TILE_Y_RANGE[0])
                    self._wsg_ext_y_max_model = ui.SimpleFloatModel(WORKSPACE_EXT_TILE_Y_RANGE[1])
                    with ui.HStack(spacing=5, height=0):
                        ui.Label("X (forward) min:", width=120)
                        ui.FloatField(model=self._wsg_ext_x_min_model, width=70)
                        ui.Label("max:", width=40)
                        ui.FloatField(model=self._wsg_ext_x_max_model, width=70)
                    with ui.HStack(spacing=5, height=0):
                        ui.Label("Y (lateral) min:", width=120)
                        ui.FloatField(model=self._wsg_ext_y_min_model, width=70)
                        ui.Label("max:", width=40)
                        ui.FloatField(model=self._wsg_ext_y_max_model, width=70)
                    with ui.HStack(spacing=5, height=0):
                        ui.Button("Apply", width=80, height=30,
                                  clicked_fn=self._cmd_apply_workspace_ext_tile)
                        ui.Button("Reset to Defaults", width=140, height=30,
                                  clicked_fn=self._cmd_reset_workspace_ext_tile)

            with ui.CollapsableFrame(title="SO-ARM101 Control", collapsed=False, height=0):
                with ui.VStack(spacing=5, height=0):
                    with ui.HStack(spacing=5):
                        ui.Button("Import SO-ARM101", width=150, height=35, clicked_fn=lambda: asyncio.ensure_future(self.load_robot()))
                        ui.Button("Setup Action Graph", width=200, height=35, clicked_fn=lambda: asyncio.ensure_future(self.setup_action_graph()))
                    with ui.HStack(spacing=5):
                        ui.Button("Setup Force Publisher", width=200, height=35, clicked_fn=self.setup_force_publish_action_graph)

            # Intel RealSense Camera
            with ui.CollapsableFrame(title="Intel RealSense Camera", collapsed=False, height=0):
                with ui.VStack(spacing=5, height=0):
                    with ui.HStack(spacing=5):
                        ui.Button("Import RealSense Camera", width=170, height=35, clicked_fn=self.import_realsense_camera)
                        ui.Button("Attach Camera to Robot", width=160, height=35, clicked_fn=self.attach_camera_to_robot)
                        ui.Button("Setup Camera Action Graph", width=200, height=35, clicked_fn=self.setup_camera_action_graph)

            # Wrist Camera (OV9732)
            with ui.CollapsableFrame(title="Wrist Camera (OV9732)", collapsed=False, height=0):
                with ui.VStack(spacing=5, height=0):
                    with ui.HStack(spacing=5):
                        ui.Button("Attach Camera Mount", width=170, height=35, clicked_fn=self.attach_camera_mount_mesh)
                        ui.Button("Attach USB Camera", width=160, height=35, clicked_fn=self.attach_usb_camera)
                        ui.Button("Setup Camera Action Graph", width=200, height=35, clicked_fn=self.setup_wrist_camera_action_graph)

            # NEW SECTION: Additional Camera
            with ui.CollapsableFrame(title="Additional Camera", collapsed=True, height=0):
                with ui.VStack(spacing=5, height=0):
                    ui.Label("Camera Configuration", alignment=ui.Alignment.LEFT)
                    
                    # Camera type selection with checkboxes and labels
                    with ui.VStack(spacing=5):
                        with ui.HStack(spacing=5):
                            self._workspace_checkbox = ui.CheckBox(width=20)
                            self._workspace_checkbox.model.set_value(False)  # Default unchecked
                            ui.Label("Workspace Camera", alignment=ui.Alignment.LEFT, width=120)

                        with ui.HStack(spacing=5):
                            self._custom_checkbox = ui.CheckBox(width=20)
                            ui.Label("Custom Camera", alignment=ui.Alignment.LEFT, width=120)

                    # Custom camera prim path input
                    with ui.HStack(spacing=5):
                        ui.Label("Camera Prim Path:", alignment=ui.Alignment.LEFT, width=120)
                        self._custom_camera_prim_field = ui.StringField(width=200)
                        self._custom_camera_prim_field.model.set_value("/World/custom_camera")
                    
                    # Custom camera ROS2 topic name input
                    with ui.HStack(spacing=5):
                        ui.Label("ROS2 Topic Name:", alignment=ui.Alignment.LEFT, width=120)
                        self._custom_camera_topic_field = ui.StringField(width=200)
                        self._custom_camera_topic_field.model.set_value("custom_camera")

                    # Resolution selection
                    with ui.HStack(spacing=5):
                        ui.Label("Resolution:", alignment=ui.Alignment.LEFT, width=120)
                        self._resolution_combo = ui.ComboBox(0, "640x480", "1280x720", "1920x1080", width=150)

                    # Camera control buttons
                    with ui.HStack(spacing=5):
                        ui.Button("Create Camera", width=150, height=35, clicked_fn=self.create_additional_camera)
                        ui.Button("Create Action Graph", width=180, height=35, clicked_fn=self.create_additional_camera_actiongraph)

            with ui.CollapsableFrame(title="Objects", collapsed=False, height=0):
                with ui.VStack(spacing=5, height=0):
                    with ui.CollapsableFrame(title="Colors", name="subFrame", collapsed=False, height=0):
                        with ui.VStack(spacing=3, height=0):
                            self._color_pickers = {}
                            for cname in OBJECT_COLORS:
                                # Convert linear → sRGB for display (ColorWidget shows raw values)
                                sr, sg, sb = [_linear_to_srgb(c) for c in OBJECT_COLORS[cname]]
                                with ui.HStack(spacing=5):
                                    ui.Label(cname.capitalize(), width=50, alignment=ui.Alignment.LEFT_CENTER)
                                    cw = ui.ColorWidget(sr, sg, sb, 1.0, width=80, height=20)
                                    self._color_pickers[cname] = cw
                            ui.Button("Apply", width=60, height=25, clicked_fn=self._apply_color_pickers)

                    with ui.CollapsableFrame(title="Objects", name="subFrame", collapsed=False, height=0):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(spacing=5):
                                ui.Button("Add Objects", width=150, height=35, clicked_fn=self.add_objects)
                                ui.Button("Delete Objects", width=150, height=35, clicked_fn=self.delete_objects)
                            # Randomize row + visibility constraint controls.
                            # The combobox lets the user demand that at least
                            # `Min visible` legos of the chosen color land in
                            # the workspace camera's FOV after Randomize. Used
                            # for VLA-style recordings where the target object
                            # MUST be observable in the camera feed.
                            with ui.HStack(spacing=5):
                                ui.Button("Randomize", width=150, height=35,
                                          clicked_fn=self._on_randomize_clicked)
                                ui.Button("Sort", width=150, height=35, clicked_fn=self.sort_objects)
                            with ui.HStack(spacing=5):
                                ui.Label("Lock color in view:", width=130,
                                         alignment=ui.Alignment.LEFT_CENTER)
                                self._visible_color_combo = ui.ComboBox(
                                    0, "(none)", "red", "green", "blue", width=110)
                                ui.Label("Min visible:", width=80,
                                         alignment=ui.Alignment.LEFT_CENTER)
                                self._visible_count_model = ui.SimpleIntModel(1)
                                ui.IntField(model=self._visible_count_model, width=40)
                            with ui.HStack(spacing=5):
                                ui.Button("Add ArUco Markers", width=180, height=35, clicked_fn=self.add_aruco_markers)
                                ui.Button("Setup Pose Publisher", width=180, height=35, clicked_fn=self.create_pose_publisher)
                            with ui.HStack(spacing=5):
                                ui.Button("Setup BBox Publisher", width=180, height=35, clicked_fn=self.setup_bbox_publisher)
                    with ui.CollapsableFrame(title="Containers", name="subFrame", collapsed=False, height=0):
                        with ui.VStack(spacing=5, height=0):
                            # Layout mode dropdown + reset
                            with ui.HStack(spacing=5):
                                ui.Label("Mode:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_mode_combo = ui.ComboBox(
                                    0 if CUP_LAYOUT["mode"] == "line" else 1,
                                    "Line", "Arc", width=100)
                                self._cup_mode_combo.model.add_item_changed_fn(self._on_cup_layout_changed)
                                ui.Button("Reset", width=45, height=25, clicked_fn=self._reset_cup_defaults,
                                          tooltip="Reset to defaults")

                            # Angle slider
                            with ui.HStack(spacing=5):
                                ui.Label("Angle:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_angle_slider = ui.FloatSlider(
                                    min=-180, max=180, step=5, width=180)
                                self._cup_angle_slider.model.set_value(float(CUP_LAYOUT["angle_deg"]))
                                self._cup_angle_slider.model.add_value_changed_fn(self._on_cup_layout_changed)
                                self._cup_angle_label = ui.Label(f"{CUP_LAYOUT['angle_deg']}°", width=40)

                            # Radius slider
                            with ui.HStack(spacing=5):
                                ui.Label("Radius:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_radius_slider = ui.FloatSlider(
                                    min=0.15, max=0.40, step=0.01, width=180)
                                self._cup_radius_slider.model.set_value(CUP_LAYOUT["radius"])
                                self._cup_radius_slider.model.add_value_changed_fn(self._on_cup_layout_changed)
                                self._cup_radius_label = ui.Label(f"{CUP_LAYOUT['radius']:.2f}m", width=45)

                            # Gap slider (line mode)
                            with ui.HStack(spacing=5):
                                ui.Label("Gap:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_gap_slider = ui.FloatSlider(
                                    min=0.0, max=0.05, step=0.005, width=180)
                                self._cup_gap_slider.model.set_value(CUP_LAYOUT["gap"])
                                self._cup_gap_slider.model.add_value_changed_fn(self._on_cup_layout_changed)
                                self._cup_gap_label = ui.Label(f"{CUP_LAYOUT['gap']*100:.0f}cm", width=45)

                            # Face origin checkbox
                            with ui.HStack(spacing=5):
                                self._cup_face_origin_cb = ui.CheckBox(width=20)
                                self._cup_face_origin_cb.model.set_value(CUP_LAYOUT.get("face_origin", True))
                                self._cup_face_origin_cb.model.add_value_changed_fn(self._on_cup_layout_changed)
                                ui.Label("Face origin", alignment=ui.Alignment.LEFT)

                            # Cluster X offset (world frame; shifts ALL cups)
                            with ui.HStack(spacing=5):
                                ui.Label("Offset X:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_offset_x_slider = ui.FloatSlider(
                                    min=-0.20, max=0.20, step=0.01, width=180)
                                self._cup_offset_x_slider.model.set_value(
                                    float(CUP_LAYOUT.get("cluster_offset_x", 0.0)))
                                self._cup_offset_x_slider.model.add_value_changed_fn(
                                    self._on_cup_layout_changed)
                                self._cup_offset_x_label = ui.Label(
                                    f"{CUP_LAYOUT.get('cluster_offset_x', 0.0)*100:.0f}cm",
                                    width=45)

                            # Cluster Y offset (world frame)
                            with ui.HStack(spacing=5):
                                ui.Label("Offset Y:", width=60, alignment=ui.Alignment.LEFT)
                                self._cup_offset_y_slider = ui.FloatSlider(
                                    min=-0.20, max=0.20, step=0.01, width=180)
                                self._cup_offset_y_slider.model.set_value(
                                    float(CUP_LAYOUT.get("cluster_offset_y", 0.0)))
                                self._cup_offset_y_slider.model.add_value_changed_fn(
                                    self._on_cup_layout_changed)
                                self._cup_offset_y_label = ui.Label(
                                    f"{CUP_LAYOUT.get('cluster_offset_y', 0.0)*100:.0f}cm",
                                    width=45)

                            # Buttons
                            with ui.HStack(spacing=5):
                                ui.Button("Add Cups", width=80, height=35, clicked_fn=self.add_cups)
                                ui.Button("Update", width=70, height=35, clicked_fn=self._add_cups_from_ui)
                                ui.Button("Delete", width=70, height=35, clicked_fn=self.delete_cups)
                            with ui.HStack(spacing=5):
                                ui.Button("Match Real World", width=160, height=30,
                                          clicked_fn=lambda: self._cmd_match_real_world(reposition=True),
                                          tooltip="Apply CUP_LAYOUT_REAL_WORLD overrides (offset + gap) and re-place cups. Auto-called during quick_start.")
                            with ui.HStack(spacing=5):
                                ui.Button("Randomize", width=100, height=35,
                                          clicked_fn=lambda: self._cmd_randomize_cups())
                                ui.Button("Sort Into Cups", width=150, height=35, clicked_fn=self.sort_into_cups)
                                ui.Button("Unsort", width=100, height=35, clicked_fn=self.disassemble_objects)
                            with ui.HStack(spacing=5):
                                ui.Button("Publish Drop Poses", width=220, height=35,
                                          clicked_fn=lambda: self._cmd_publish_drop_poses())

                    with ui.CollapsableFrame(title="Scene State", name="subFrame", collapsed=True, height=0):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(spacing=5):
                                ui.Button("Sync Real Poses", width=180, height=35, clicked_fn=self.sync_real_poses)
                            with ui.HStack(spacing=5):
                                ui.Button("Save State", width=120, height=35,
                                    clicked_fn=lambda: self._cmd_save_scene_state())
                                ui.Button("Restore State", width=120, height=35,
                                    clicked_fn=lambda: self._cmd_restore_scene_state())
                            with ui.CollapsableFrame(title="Exclude / Include Objects", collapsed=True, height=0):
                                self._visibility_frame = ui.Frame(build_fn=self._build_visibility_ui)

            with ui.CollapsableFrame(title="Logs", collapsed=True, height=0):
                with ui.VStack(spacing=3, height=0):
                    self._log_tab_style = {
                        "RadioButton": {
                            "margin": 2, "border_radius": 2, "font_size": 16,
                            "background_color": 0xFF212121, "color": 0xFF444444,
                        },
                        "RadioButton.Label": {"font_size": 16, "color": 0xFF777777},
                        "RadioButton:checked": {"background_color": 0xFF3A3A3A, "color": 0xFF222222},
                        "RadioButton.Label:checked": {"color": 0xFFCCCCCC},
                    }
                    self._log_tab_collection = ui.RadioCollection()
                    with ui.HStack(spacing=3, height=30, style=self._log_tab_style):
                        ui.RadioButton(text="Extension", width=80,
                            radio_collection=self._log_tab_collection,
                            clicked_fn=lambda: self._switch_log_tab("extension"))
                        ui.RadioButton(text="System", width=80,
                            radio_collection=self._log_tab_collection,
                            clicked_fn=lambda: self._switch_log_tab("system"))
                    self._log_stack = ui.ZStack(height=100)
                    with self._log_stack:
                        self._ext_log_scroll = ui.ScrollingFrame(
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            style={"background_color": 0xFF24211F},
                        )
                        with self._ext_log_scroll:
                            self._ext_log_vstack = ui.VStack(spacing=0)
                        self._sys_log_scroll = ui.ScrollingFrame(
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                            visible=False,
                            style={"background_color": 0xFF24211F},
                        )
                        with self._sys_log_scroll:
                            self._sys_log_vstack = ui.VStack(spacing=0)
                    with ui.HStack(spacing=5):
                        ui.Button("Copy Logs", width=120, height=30, clicked_fn=self._copy_logs)
                        ui.Button("Clear Logs", width=120, height=30, clicked_fn=self._clear_logs)

        self._active_log_tab = "extension"
        self._ext_log_lines = []
        self._sys_log_lines = []
        self._setup_log_redirect()

        # Re-subscribe physics callbacks if action graphs already exist (e.g. after hot-reload)
        self._resubscribe_physics_callbacks()

    def _setup_log_redirect(self):
        # Unwrap any leftover redirector from a previous failed startup
        if isinstance(sys.stdout, LogRedirector):
            sys.stdout = sys.stdout._original
        if isinstance(sys.stderr, LogRedirector):
            sys.stderr = sys.stderr._original
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = LogRedirector(self._orig_stdout, self._append_ext_log)
        sys.stderr = LogRedirector(self._orig_stderr, lambda t: self._append_ext_log(f"[ERROR] {t}"), throttle_seconds=2.0)
        logging = carb.logging.acquire_logging()
        self._carb_log_sub = logging.add_logger(self._on_carb_log)

    def _on_carb_log(self, source, level, filename, linenum, msg):
        # carb levels: VERBOSE=-2, INFO=-1, WARN=0, ERROR=1, FATAL=2
        import time as _t
        now = _t.monotonic()
        # Throttle: skip duplicate messages within 2 seconds
        key = (source, level, msg)
        last = getattr(self, '_carb_log_last', {})
        if key in last and now - last[key] < 2.0:
            return
        last[key] = now
        # Evict stale entries periodically
        if len(last) > 200:
            cutoff = now - 5.0
            last = {k: v for k, v in last.items() if v > cutoff}
        self._carb_log_last = last

        if level >= carb.logging.LEVEL_ERROR:
            self._append_sys_log(f"[ERROR] [{source}] {msg}")
        elif level == carb.logging.LEVEL_WARN:
            self._append_sys_log(f"[WARN] [{source}] {msg}")
        # elif level == carb.logging.LEVEL_INFO:
        #     self._append_sys_log(f"[INFO] [{source}] {msg}")
        # else:
        #     self._append_sys_log(f"[VERBOSE] [{source}] {msg}")

    def _teardown_log_redirect(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if hasattr(self, "_carb_log_sub") and self._carb_log_sub is not None:
            logging = carb.logging.acquire_logging()
            logging.remove_logger(self._carb_log_sub)
            self._carb_log_sub = None

    def _log_color(self, text):
        # Matches Isaac Sim script editor colors (0xAABBGGRR format)
        if "[ERROR]" in text or "[FATAL]" in text:
            return 0xFFAAB1F6  # #F6B1AA - light coral
        elif "[WARN]" in text:
            return 0xFF4ACBDF  # #DFCB4A - yellow/gold
        # elif "[INFO]" in text:
        #     return 0xFFEBBA79  # #79BAEB - light blue
        # elif "[VERBOSE]" in text:
        #     return 0xFFBABABA  # #BABABA - gray
        return 0xFFFFFFFF  # white

    def _add_log_label(self, vstack, scroll, text):
        with vstack:
            ui.Label(text, alignment=ui.Alignment.LEFT_TOP, word_wrap=True,
                     style={"color": self._log_color(text), "font_size": 13, "margin": 0},
                     height=0)
        scroll.scroll_y = scroll.scroll_y_max + 100

    def _rebuild_log_vstack(self, vstack, lines):
        vstack.clear()
        with vstack:
            for line in lines:
                ui.Label(line, alignment=ui.Alignment.LEFT_TOP, word_wrap=True,
                         style={"color": self._log_color(line), "font_size": 13, "margin": 0},
                         height=0)

    def _run_on_main_thread(self, fn):
        """Schedule fn to run on the main thread, safe from any thread."""
        if threading.current_thread() is threading.main_thread():
            fn()
        else:
            self._main_loop.call_soon_threadsafe(fn)

    def _append_ext_log(self, text):
        self._ext_log_lines.append(text)
        if len(self._ext_log_lines) > 500:
            self._ext_log_lines = self._ext_log_lines[-500:]
            self._run_on_main_thread(lambda: self._rebuild_log_vstack(self._ext_log_vstack, self._ext_log_lines))
            return
        if hasattr(self, "_ext_log_vstack") and self._ext_log_vstack:
            self._run_on_main_thread(lambda: self._add_log_label(self._ext_log_vstack, self._ext_log_scroll, text))

    def _append_sys_log(self, text):
        self._sys_log_lines.append(text)
        if len(self._sys_log_lines) > 500:
            self._sys_log_lines = self._sys_log_lines[-500:]
            self._run_on_main_thread(lambda: self._rebuild_log_vstack(self._sys_log_vstack, self._sys_log_lines))
            return
        if hasattr(self, "_sys_log_vstack") and self._sys_log_vstack:
            self._run_on_main_thread(lambda: self._add_log_label(self._sys_log_vstack, self._sys_log_scroll, text))

    def _switch_log_tab(self, tab):
        self._active_log_tab = tab
        is_ext = tab == "extension"
        self._ext_log_scroll.visible = is_ext
        self._sys_log_scroll.visible = not is_ext

    def _copy_logs(self):
        lines = self._ext_log_lines if self._active_log_tab == "extension" else self._sys_log_lines
        try:
            import subprocess
            text = "\n".join(lines)
            proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            proc.communicate(text.encode())
        except Exception:
            import omni.kit.clipboard
            omni.kit.clipboard.copy("\n".join(lines))

    def _clear_logs(self):
        if self._active_log_tab == "extension":
            self._ext_log_lines.clear()
            self._ext_log_vstack.clear()
        else:
            self._sys_log_lines.clear()
            self._sys_log_vstack.clear()

    def _resubscribe_physics_callbacks(self):
        """Re-subscribe physics callbacks for graphs that survived a hot-reload."""
        import omni.physx

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # Force publisher action graph
        force_graph = "/Graph/ActionGraph_SO_ARM101_ForcePublish"
        if stage.GetPrimAtPath(force_graph):
            self._force_graph_path = force_graph
            # Recover the publisher OG node handle from the existing graph
            pub_prim = stage.GetPrimAtPath(f"{force_graph}/publisher")
            if pub_prim:
                self._force_pub_node = og.get_node_by_path(f"{force_graph}/publisher")
            self._force_physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step_force
            )
            print(f"[HotReload] Re-subscribed force publisher physics callback for {force_graph}")

        # BBox publisher action graph
        bbox_graph = "/Graph/ActionGraph_objects_bbox"
        if stage.GetPrimAtPath(bbox_graph):
            self._bbox_graph_path = bbox_graph
            self._bbox_pub_attr_path = f"{bbox_graph}/publisher.inputs:data"
            self._bbox_publish_counter = 0
            self._bbox_physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step_bbox
            )
            print(f"[HotReload] Re-subscribed bbox publisher physics callback for {bbox_graph}")

    @property
    def _objects_folder_path(self):
        folder = self._objects_config.get("folder")
        return folder if folder else _get_assets_folder()

    @property
    def _skip_patterns(self):
        return self._objects_config.get("skip_patterns", [])

    def _apply_sim_rate(self, hz=None):
        """Apply Kit wall pacing after World initialization; physics stays at 60 Hz."""
        if World.instance() is None:
            raise RuntimeError("Create the World before applying sim pacing")
        settings = carb.settings.get_settings()
        if hz is None:
            hz = settings.get("/soarm101/rateHz")
            if hz is None:
                raise RuntimeError("Launch with isaacsim_launch.sh to configure sim pacing")
        hz = float(hz)
        if not math.isfinite(hz) or hz < 0:
            raise ValueError("Sim rate must be finite and nonnegative")
        settings.set_float("/soarm101/rateHz", hz)
        settings.set_bool("/app/runLoops/main/rateLimitEnabled", hz > 0)
        if hz > 0:
            settings.set_float("/app/runLoops/main/rateLimitFrequency", hz)
        actual = {"enabled": settings.get("/app/runLoops/main/rateLimitEnabled"),
                  "hz": settings.get("/app/runLoops/main/rateLimitFrequency")}
        if actual["enabled"] != (hz > 0) or (hz > 0 and abs(actual["hz"] - hz) > 1e-4):
            raise RuntimeError(f"Kit pacing readback mismatch: {actual}")
        print(f"[SimRate] requested={hz:g} Hz, readback={actual}")
        return actual

    async def load_scene(self):
        stage = omni.usd.get_context().get_stage()

        # Only initialize simulation context if physics scene doesn't exist yet
        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        if not physics_scene_prim or not physics_scene_prim.IsValid():
            world = World()
            await world.initialize_simulation_context_async()
            print("Initialized simulation context and physics scene")
        else:
            world = World()
            print("Physics scene already exists, skipping initialization")

        self._apply_sim_rate()

        # Check for ground plane on stage and add if missing.
        # Two layers: a large default-gray collision plane covers the
        # whole scene; a smaller configurable WORKSPACE_GROUND tile
        # (visual only, no collision) sits on top inside the active
        # pick-place region for high-contrast dataset frames. See
        # GROUND_PLANE_* and WORKSPACE_GROUND_* module constants.
        ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if ground_prim and ground_prim.IsValid():
            print("Ground plane already exists, skipping")
        else:
            import numpy as np
            from isaacsim.core.api.objects import GroundPlane
            from pxr import Sdf
            GroundPlane(
                prim_path="/World/defaultGroundPlane",
                z_position=0.0,
                size=GROUND_PLANE_SIZE_M,
                color=np.array(GROUND_PLANE_COLOR),
            )
            # GroundPlane authors a UsdPreviewSurface shader at
            # /World/Looks/visual_material/shader with only diffuseColor
            # set. Default roughness=0.5 leaves the surface visibly
            # shiny; force fully matte for clean camera frames.
            shader = stage.GetPrimAtPath(
                "/World/Looks/visual_material/shader")
            if shader and shader.IsValid():
                shader.CreateAttribute(
                    "inputs:roughness", Sdf.ValueTypeNames.Float
                ).Set(GROUND_PLANE_ROUGHNESS)
                shader.CreateAttribute(
                    "inputs:metallic", Sdf.ValueTypeNames.Float
                ).Set(0.0)
            print(
                f"Added ground plane (color={GROUND_PLANE_COLOR}, "
                f"matte)")

        # Workspace ground tile (visual overlay, no collision).
        self._create_workspace_ground_tile(stage)
        # Extension tile joining the main one on its -X edge (independent material).
        self._create_workspace_ext_tile(stage)

        # Default lighting: interior daylight (key window + warm fill + bounce dome).
        # Matches the real-world reference scene; hides any pre-existing default lights
        # non-destructively. Falls back to a plain dome if the preset apply fails.
        try:
            self._apply_interior_daylight()
        except Exception as e:
            print(f"[load_scene] interior-daylight auto-apply failed: {e} — falling back to default dome")
            from pxr import UsdLux
            has_light = any(prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight)
                            for prim in stage.Traverse())
            if not has_light:
                dome = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
                dome.CreateIntensityAttr().Set(1000.0)
                print("Added default dome light at 1000 lux (fallback)")

        # Set minimum frame rate to 60
        settings = carb.settings.get_settings()
        settings.set("/persistent/simulation/minFrameRate", self._min_frame_rate)
        print(f"Set persistent/simulation/minFrameRate to {self._min_frame_rate}")

        # Set the persistent viewport flag that suppresses Kit's auto-generated camera
        # icons. The actual prim-deactivation runs at the END of quick_start (after cameras
        # are created — the load_scene step happens BEFORE camera creation, so a SetActive
        # loop here would find nothing). See _suppress_camera_mesh_decorations.
        for vid in range(4):
            settings.set(f"/persistent/app/viewport/Viewport/Viewport{vid}/scene/cameras/visible", False)

        # Configure physics scene
        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim and physics_scene_prim.IsValid():
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
            physx_scene_api.CreateEnableGPUDynamicsAttr().Set(False)
            physx_scene_api.CreateTimeStepsPerSecondAttr().Set(self._time_steps_per_second)
            physx_scene_api.CreateEnableCCDAttr().Set(False)
            print(f"GPU dynamics disabled, timeStepsPerSecond={self._time_steps_per_second}, CCD disabled on /physicsScene")
        else:
            print("Warning: /physicsScene not found")

        print("Scene loaded successfully.")

    def _toggle_viewport_rendering(self):
        """Toggle viewport rendering on/off to free GPU resources."""
        import omni.kit.viewport.utility as vp_utils
        viewport = vp_utils.get_active_viewport()
        if viewport is None:
            print("Warning: No active viewport found")
            return
        self._viewport_rendering_enabled = not self._viewport_rendering_enabled
        viewport.updates_enabled = self._viewport_rendering_enabled
        state = "enabled" if self._viewport_rendering_enabled else "disabled"
        print(f"Viewport rendering {state}")
        if self._viewport_toggle_btn:
            self._viewport_toggle_btn.text = "Disable Viewport" if self._viewport_rendering_enabled else "Enable Viewport"

    def _cmd_apply_workspace_ground(self):
        """Read the 4 extent fields from the Workspace Ground UI panel
        and re-author the workspace ground tile. Validates min < max
        on each axis; complains in the log and returns on bad input.
        """
        x_min = self._wsg_x_min_model.as_float
        x_max = self._wsg_x_max_model.as_float
        y_min = self._wsg_y_min_model.as_float
        y_max = self._wsg_y_max_model.as_float
        if not (x_min < x_max and y_min < y_max):
            print(
                f"[workspace_ground] Invalid extents: "
                f"X=({x_min}, {x_max}), Y=({y_min}, {y_max}) — "
                f"min must be < max on each axis"
            )
            return
        stage = omni.usd.get_context().get_stage()
        self._create_workspace_ground_tile(
            stage,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

    def _cmd_reset_workspace_ground(self):
        """Snap the 4 extent fields back to WORKSPACE_GROUND_X_RANGE /
        WORKSPACE_GROUND_Y_RANGE module defaults and re-author the tile.
        """
        self._wsg_x_min_model.set_value(WORKSPACE_GROUND_X_RANGE[0])
        self._wsg_x_max_model.set_value(WORKSPACE_GROUND_X_RANGE[1])
        self._wsg_y_min_model.set_value(WORKSPACE_GROUND_Y_RANGE[0])
        self._wsg_y_max_model.set_value(WORKSPACE_GROUND_Y_RANGE[1])
        self._cmd_apply_workspace_ground()

    def _cmd_apply_workspace_ext_tile(self):
        """Re-author the -X extension tile with the values from the UI fields."""
        x_min = self._wsg_ext_x_min_model.as_float
        x_max = self._wsg_ext_x_max_model.as_float
        y_min = self._wsg_ext_y_min_model.as_float
        y_max = self._wsg_ext_y_max_model.as_float
        if not (x_min < x_max and y_min < y_max):
            print(f"[ext_tile] Invalid extents: X=({x_min}, {x_max}), Y=({y_min}, {y_max})")
            return
        stage = omni.usd.get_context().get_stage()
        self._create_workspace_ext_tile(stage, x_range=(x_min, x_max), y_range=(y_min, y_max))

    def _cmd_reset_workspace_ext_tile(self):
        """Reset ext-tile fields to module defaults and re-author."""
        self._wsg_ext_x_min_model.set_value(WORKSPACE_EXT_TILE_X_RANGE[0])
        self._wsg_ext_x_max_model.set_value(WORKSPACE_EXT_TILE_X_RANGE[1])
        self._wsg_ext_y_min_model.set_value(WORKSPACE_EXT_TILE_Y_RANGE[0])
        self._wsg_ext_y_max_model.set_value(WORKSPACE_EXT_TILE_Y_RANGE[1])
        self._cmd_apply_workspace_ext_tile()

    _INTERIOR_DAYLIGHT_PREFIX = "/World/Lights/Interior_"

    def _apply_interior_daylight(self):
        """Apply the INTERIOR_DAYLIGHT_PRESET (key window + warm fill + soft dome bounce).
        Hides any pre-existing lights non-destructively (they remain on stage as
        invisible prims). See the constant's docstring for the design notes."""
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()

        # Ensure /World/Lights scope
        scope = stage.GetPrimAtPath("/World/Lights")
        if not scope or not scope.IsValid():
            UsdGeom.Scope.Define(stage, "/World/Lights")

        # Hide every non-preset light (recoverable: just MakeVisible() to restore)
        hidden = 0
        for prim in stage.Traverse():
            if not str(prim.GetTypeName()).endswith("Light"):
                continue
            if str(prim.GetPath()).startswith(self._INTERIOR_DAYLIGHT_PREFIX):
                continue
            img = UsdGeom.Imageable(prim)
            if img and img.ComputeVisibility() != UsdGeom.Tokens.invisible:
                img.MakeInvisible()
                hidden += 1

        # Author preset lights (idempotent — Define is no-op on existing prims)
        for spec in INTERIOR_DAYLIGHT_PRESET:
            path = f"{self._INTERIOR_DAYLIGHT_PREFIX}{spec['name']}"
            kind = spec["kind"]
            if kind == "RectLight":
                light = UsdLux.RectLight.Define(stage, path)
                light.CreateWidthAttr().Set(spec["width"])
                light.CreateHeightAttr().Set(spec["height"])
            elif kind == "DomeLight":
                light = UsdLux.DomeLight.Define(stage, path)
            else:
                continue
            light.CreateIntensityAttr().Set(spec["intensity"])
            if "color" in spec:
                light.CreateColorAttr().Set(Gf.Vec3f(*spec["color"]))
            UsdGeom.Imageable(light.GetPrim()).MakeVisible()

            xform = UsdGeom.Xform(light.GetPrim())
            xform.ClearXformOpOrder()
            tx, ty, tz = spec["translate"]
            xform.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
            qx, qy, qz, qw = spec["quat_xyzw"]
            xform.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
        print(f"Applied interior daylight: {len(INTERIOR_DAYLIGHT_PRESET)} lights "
              f"(hid {hidden} pre-existing).")

    async def quick_start(self):
        """Quick start: load scene, SO-ARM101, joint action graph, camera mount, lego objects, publishers.

        Use await app.next_update_async() after steps that add/modify prims or action graphs,
        so the Omniverse update loop applies changes before the next step runs.
        """
        print("=== Quick Start ===")
        app = omni.kit.app.get_app()

        # 1. Load scene
        print("--- Loading scene ---")
        await self.load_scene()

        # 2. Import SO-ARM101
        print("--- Importing SO-ARM101 ---")
        await self.load_robot()
        await app.next_update_async()

        # 3. Setup SO-ARM101 action graph
        print("--- Setting up SO-ARM101 Action Graph ---")
        await self.setup_action_graph()
        await app.next_update_async()

        # 4. Attach camera mount + USB camera (uses local assets/wrist_mounts/)
        print("--- Attaching camera mount ---")
        self.attach_camera_mount_mesh()
        await app.next_update_async()
        print("--- Attaching USB camera ---")
        self.attach_usb_camera()
        await app.next_update_async()

        # 4.5. Override 3D-printed-part materials to matte white plastic
        # (servos keep their default urdf-imported material). Runs after
        # the camera mount is attached so it can also be styled white.
        print("--- Applying plastic-white material to robot prints ---")
        self._apply_robot_plastic_material()
        await app.next_update_async()

        # 5. Setup wrist camera action graph
        print("--- Setting up wrist camera action graph ---")
        self.setup_wrist_camera_action_graph()
        await app.next_update_async()

        # 6. Setup force publisher
        print("--- Setting up force publisher ---")
        self.setup_force_publish_action_graph()
        await app.next_update_async()

        # 7. Add objects and setup publishers
        print("--- Adding lego objects ---")
        self.add_objects()
        await app.next_update_async()

        print("--- Randomizing object poses ---")
        self.randomize_object_poses()
        await app.next_update_async()

        print("--- Setting up pose publisher ---")
        self.create_pose_publisher()
        await app.next_update_async()

        print("--- Setting up bbox publisher ---")
        self.setup_bbox_publisher()
        await app.next_update_async()

        # 8. Add cups with ArUco markers. Apply real-world calibration
        # (CUP_LAYOUT_REAL_WORLD overrides) BEFORE add_cups so the
        # placement uses calibrated offset + gap on first creation.
        # reposition=False because cups don't exist yet — add_cups
        # below reads CUP_LAYOUT directly.
        print("--- Matching real-world cup calibration ---")
        self._cmd_match_real_world(reposition=False)
        print("--- Adding cups ---")
        self.add_cups()
        await app.next_update_async()

        # 9. Publish drop poses (ArUco marker poses on cups)
        print("--- Publishing drop poses ---")
        self._cmd_publish_drop_poses()
        await app.next_update_async()

        # 10. Play the scene
        print("--- Playing scene ---")
        self._timeline.play()
        # Ported from isaac-sim-mcp b6485d4 (ur5e-dt quick_start crash fix): let PhysX finish its
        # warmup before the camera/viewport/OmniGraph mutations below.
        await self._wait_physx_warmup()

        # 10b. Create workspace camera (skip if already exists) and start publisher
        # Prim path is /World/workspace_camera (no _sim suffix — the prim *is* the camera).
        # Topic name + frame_id keep the _sim suffix for sim/real disambiguation.
        print("--- Creating workspace camera ---")
        ws_prim_path = "/World/workspace_camera"
        ws_stage = omni.usd.get_context().get_stage()
        ws_existing = ws_stage.GetPrimAtPath(ws_prim_path)
        if ws_existing and ws_existing.IsValid():
            print(f"Workspace camera already exists at {ws_prim_path}, skipping creation.")
        else:
            ws_width, ws_height = 640, 480
            # Hand-tuned for SO-ARM101 — captured from the live workspace
            # camera in a positioned session. To re-capture: orbit the
            # Persp (or workspace) view to the desired pose, then read
            # translate + orient via
            # UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0).
            ws_position = (0.36, 0.355, 0.34973)
            ws_quat_xyzw = (0.095726, 0.481244, 0.861477, 0.130760)  # rotateXYZ(-57.187°, 16.905°, 172.0°)

            ws_camera_prim = UsdGeom.Camera.Define(ws_stage, ws_prim_path)
            if ws_camera_prim:
                # Configure intrinsics. RealSense's authored values are HFOV 69.4°,
                # VFOV 42.5° at 16:9 (1280x720). At 4:3 (640x480) the same VFOV
                # crops the frustum vertically — apparent zoom. Derive VFOV from
                # HFOV and the actual aspect so the rendered frame matches the
                # OpenCVPinhole calibration. Same square-pixel formula as the
                # wrist camera (extension.py:2417, 2520).
                hfov_deg = 69.4
                vfov_deg = hfov_deg * ws_height / ws_width
                fx = ws_width / (2 * np.tan(np.deg2rad(hfov_deg / 2)))
                fy = ws_height / (2 * np.tan(np.deg2rad(vfov_deg / 2)))
                cx, cy = ws_width * 0.5, ws_height * 0.5
                horizontal_aperture_mm = 36.0
                focal_length_mm = fx * horizontal_aperture_mm / ws_width
                vertical_aperture_mm = ws_height * focal_length_mm / fy

                cam = UsdGeom.Camera(ws_camera_prim.GetPrim())
                cam.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
                cam.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
                cam.CreateFocalLengthAttr().Set(focal_length_mm)
                cam.CreateProjectionAttr().Set("perspective")
                cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

                cp = ws_camera_prim.GetPrim()
                cp.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.String).Set("opencvPinhole")
                cp.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(ws_width, ws_height))
                cp.CreateAttribute("omni:lensdistortion:opencvPinhole:fx", Sdf.ValueTypeNames.Float).Set(fx)
                cp.CreateAttribute("omni:lensdistortion:opencvPinhole:fy", Sdf.ValueTypeNames.Float).Set(fy)
                cp.CreateAttribute("omni:lensdistortion:opencvPinhole:cx", Sdf.ValueTypeNames.Float).Set(cx)
                cp.CreateAttribute("omni:lensdistortion:opencvPinhole:cy", Sdf.ValueTypeNames.Float).Set(cy)
                for attr_name in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]:
                    cp.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{attr_name}", Sdf.ValueTypeNames.Float).Set(0.0)

                # Set camera pose
                xform = UsdGeom.Xform(ws_camera_prim.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*ws_position))
                quat_ws = Gf.Quatd(ws_quat_xyzw[3], ws_quat_xyzw[0], ws_quat_xyzw[1], ws_quat_xyzw[2])
                xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat_ws)
                print(f"Workspace camera created at {ws_prim_path} with resolution {ws_width}x{ws_height}")
            else:
                print(f"Warning: Failed to create workspace camera at {ws_prim_path}")

        # Start viewport publisher for workspace camera. Force the viewport's
        # render resolution to match the camera's intrinsic resolution so the
        # ROS2 frames match the OpenCVPinhole calibration on the prim. Without
        # this the active UI viewport renders at its window size (e.g.
        # 1280x720) regardless of camera intrinsics.
        if not _viewport_pub.is_active("workspace_camera_sim"):
            _viewport_pub.start(ws_prim_path, "workspace_camera_sim", frame_id="workspace_camera_sim")
            try:
                ws_entry = _viewport_pub._publishers.get("workspace_camera_sim")
                if ws_entry and ws_entry.get("viewport"):
                    ws_entry["viewport"].resolution = (ws_width, ws_height)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to force workspace viewport resolution: {e}")
            print(f"Workspace camera viewport publisher started @ {ws_width}x{ws_height}.")
        await app.next_update_async()

        # Switch active viewport to workspace camera
        self._set_active_viewport_camera(ws_prim_path)

        # 10c. Start replicator-based wrist camera publisher.
        # The action graph at /Graph/ActionGraph_WristCamera is left in place
        # (forward-compatible with the eventual Kit fix), but doesn't publish
        # on Isaac Sim 5.0 / Kit 107.3 / isaacsim.ros2.bridge 4.9.3 — the
        # IsaacCreateRenderProduct → ROS2CameraHelper chain silently fails to
        # register a DDS publisher. _ReplicatorCameraPublisher bypasses that
        # broken chain by using omni.replicator.core directly + manual rclpy
        # publish. Costs ~0.05 RTF (one extra GPU render pass for the wrist
        # camera, no UI viewport overhead).
        wrist_cam_prim = "/World/SO_ARM101/camera_mount_link/usb_camera/wrist_camera"
        wrist_stage = omni.usd.get_context().get_stage()
        if wrist_stage.GetPrimAtPath(wrist_cam_prim).IsValid():
            if not _replicator_pub.is_active("wrist_camera_rgb_sim"):
                _replicator_pub.start(
                    wrist_cam_prim, "wrist_camera_rgb_sim",
                    width=640, height=480, frame_id="camera_link",
                )
                print("Wrist camera replicator publisher started.")
            await app.next_update_async()
        else:
            print(f"Warning: wrist camera prim {wrist_cam_prim} not found, skipping replicator publisher")

        # 11. Create contact sensors on cups (for motion_logger at 50 Hz).
        # Must be AFTER play — sensor backend initializes on Play per
        # isaacsim.sensors.physics ContactSensor docs. One physics tick is
        # needed before the sensor is queryable.
        await app.next_update_async()
        await app.next_update_async()
        self._setup_contact_sensors()

        # Suppress Kit's auto-authored camera icon meshes now that cameras exist.
        # SetActive(False) beats MakeInvisible here — Kit re-asserts visibility on
        # decoration prims, but it leaves deactivated prims alone (they're not
        # visited by stage.Traverse()).
        self._suppress_camera_mesh_decorations()

        self._apply_sim_rate()
        print("=== Quick Start Complete ===")

    def _suppress_camera_mesh_decorations(self):
        """Deactivate every OmniverseKitViewportCameraMesh on the live stage. Called
        at the end of quick_start once all cameras (workspace + wrist) have been
        authored and Kit has decorated them. The persistent viewport flag set in
        load_scene only suppresses FUTURE gizmos; this handles the existing ones.
        Snapshots paths first to avoid expired-prim errors on a mid-flux stage."""
        stage = omni.usd.get_context().get_stage()
        paths = []
        for prim in stage.Traverse():
            try:
                if prim.GetName() == "OmniverseKitViewportCameraMesh":
                    paths.append(str(prim.GetPath()))
            except Exception:
                pass
        deactivated = 0
        for path in paths:
            p = stage.GetPrimAtPath(path)
            if p and p.IsValid():
                p.SetActive(False)
                deactivated += 1
        if deactivated:
            print(f"Deactivated {deactivated} OmniverseKitViewportCameraMesh prim(s)")

    async def load_robot(self):
        # Local fixed USD only (base_delta and camera mount reference baked in)
        _ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _local_usd = os.path.join(_ext_dir, "assets", "SO-ARM101-USD.usd")
        if not os.path.isfile(_local_usd):
            raise FileNotFoundError(
                f"SO-ARM101 USD not found at {_local_usd}. "
                "Run: python3 scripts/fix_soarm101_usd.py"
            )
        asset_path = "file://" + os.path.abspath(_local_usd)
        _apply_base_delta = False  # Pre-fixed USD, skip runtime correction
        prim_path = "/World/SO_ARM101"

        # 1. Ensure World exists - create and initialize if needed (won't clear existing stage)
        world = World.instance()
        if world is None:
            print("World not initialized. Creating simulation context for existing stage...")
            world = World()
            await world.initialize_simulation_context_async()
            # Note: Not adding ground plane here - assumes existing scene has one or user doesn't want it
            print("Simulation context initialized.")

        # 2. Add the USD asset
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)

        # 3. Wait for prim to exist
        stage = omni.usd.get_context().get_stage()
        for _ in range(10):
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Failed to load prim at {prim_path}")

        # 4. Apply translation and orientation
        xform = UsdGeom.Xform(prim)
        xform.ClearXformOpOrder()
        position = Gf.Vec3d(0.0, 0.0, self._robot_base_z_offset)
        rpy_rad = np.deg2rad(self._robot_base_rpy_deg)
        quat_xyzw = euler_angles_to_quats(rpy_rad)
        quat = Gf.Quatd(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])
        xform.AddTranslateOp().Set(position)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)

        print(f"Applied translation and rotation to {prim_path}")

        # 4b. base_delta is baked into local USD; no runtime correction needed.
        # Local pre-fixed USD (assets/SO-ARM101-USD.usd) has base_delta already baked in.
        if _apply_base_delta:
            # The original USD was exported with a ~17mm base-link offset vs the URDF.
            base_delta = Gf.Vec3d(0.016826307, 8.404913e-8, -0.00240026)
            sp_prim = stage.GetPrimAtPath(f"{prim_path}/joints/shoulder_pan")
            if sp_prim.IsValid():
                old_lp0 = sp_prim.GetAttribute("physics:localPos0").Get()
                sp_prim.GetAttribute("physics:localPos0").Set(
                    Gf.Vec3f(old_lp0[0] + base_delta[0], old_lp0[1] + base_delta[1], old_lp0[2] + base_delta[2])
                )
                print(f"Fixed shoulder_pan localPos0: {old_lp0} -> {sp_prim.GetAttribute('physics:localPos0').Get()}")
            link_names = ["shoulder_link", "upper_arm_link", "lower_arm_link",
                         "wrist_link", "gripper_link", "moving_jaw_so101_v1_link",
                         "camera_mount_link", "camera_link", "camera_optical_frame"]
            for lname in link_names:
                lp = stage.GetPrimAtPath(f"{prim_path}/{lname}")
                if lp.IsValid():
                    old_t = lp.GetAttribute("xformOp:translate").Get()
                    lp.GetAttribute("xformOp:translate").Set(
                        Gf.Vec3d(old_t[0] + base_delta[0], old_t[1] + base_delta[1], old_t[2] + base_delta[2])
                    )
            print(f"Applied URDF base-offset correction (delta={base_delta})")

        # 5. Setup Articulation
        self._robot_view = ArticulationView(prim_paths_expr=prim_path, name="so_arm101_view")
        World.instance().scene.add(self._robot_view)
        await World.instance().reset_async()
        self._timeline.stop()

        self._articulation = Articulation(prim_path)

        # SO-ARM101 USD joint names (from USD prim hierarchy):
        #   Located at: /World/SO_ARM101/joints/<name>
        #   All PhysicsRevoluteJoint types
        #   Arm joints (5-DOF): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        #   Gripper joint: gripper_joint (integrated, not separate USD)
        #   All servos: STS3215 (effort=10, velocity=10)
        arm_joint_names = [
            "shoulder_pan",    # base rotation (±110°)
            "shoulder_lift",   # shoulder pitch (±100°)
            "elbow_flex",      # elbow (−100° to +90°)
            "wrist_flex",      # wrist pitch (±95°)
            "wrist_roll",      # wrist roll (±160°)
        ]

        # Joints are under /World/SO_ARM101/joints/ (USD default prim maps to prim_path)
        for joint_name in arm_joint_names:
            joint_path = f"{prim_path}/joints/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_path)

            if not joint_prim.IsValid():
                print(f"Warning: Joint not found at {joint_path}")
                continue

            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive_api.GetMaxForceAttr().Set(self._robot_max_force)
            drive_api.GetStiffnessAttr().Set(self._robot_stiffness)
            drive_api.GetDampingAttr().Set(self._robot_damping)
            print(f"Set {joint_name}: maxForce={self._robot_max_force}, stiffness={self._robot_stiffness}, damping={self._robot_damping}")

        # Also configure the integrated gripper joint
        gripper_joint_path = f"{prim_path}/joints/gripper_joint"
        gripper_joint_prim = stage.GetPrimAtPath(gripper_joint_path)
        if gripper_joint_prim.IsValid():
            drive_api = UsdPhysics.DriveAPI.Apply(gripper_joint_prim, "angular")
            drive_api.GetMaxForceAttr().Set(self._gripper_max_force)
            drive_api.GetStiffnessAttr().Set(self._gripper_stiffness)
            drive_api.GetDampingAttr().Set(self._gripper_damping)
            print(f"Set gripper: maxForce={self._gripper_max_force}, stiffness={self._gripper_stiffness}, damping={self._gripper_damping}")
        else:
            print(f"Warning: Gripper joint not found at {gripper_joint_path}")

        # Configure gripper physics material for better grasping
        self._configure_gripper_material(prim_path, stage)

        # Camera links are part of the articulation but have no colliders,
        # causing invalid inertia {1,1,1} warnings and physics instability.
        # Can't remove RigidBodyAPI from articulation links, so set proper
        # mass and inertia instead.
        from pxr import PhysxSchema
        for cam_link in ["camera_mount_link", "camera_link", "camera_optical_frame"]:
            cam_prim = stage.GetPrimAtPath(f"{prim_path}/{cam_link}")
            if cam_prim.IsValid() and cam_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass_api = UsdPhysics.MassAPI.Apply(cam_prim)
                mass_api.CreateMassAttr().Set(0.01)
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
                print(f"Set mass/inertia on {cam_link}")

        print("SO-ARM101 robot loaded successfully (arm + integrated gripper)!")


    async def setup_action_graph(self):
        import omni.graph.core as og
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.utils.extensions import enable_extension

        print("Setting up ROS 2 Action Graph...")

        # Check that SO-ARM101 exists before creating graph
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/SO_ARM101"):
            print("Error: SO-ARM101 not found at /World/SO_ARM101. Load robot first.")
            return

        # Ensure extensions are enabled
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.core.nodes")
        enable_extension("omni.graph.action")

        graph_path = "/Graph/ActionGraph_SO_ARM101"

        # Check if graph already exists
        if stage.GetPrimAtPath(graph_path):
            print(f"Action Graph already exists at {graph_path}, skipping creation.")
            return

        (graph, nodes, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution"
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    # Use OnPlaybackTick (render rate, ~60 Hz) for the ROS2 bridge
                    # nodes. OnPhysicsStep was evaluated and rejected: ROS2 callbacks
                    # run on the main thread and don't process new messages from a
                    # physics-step tick, so the entire command path stopped delivering
                    # joint targets to the articulation (verified empirically on
                    # 2026-04-23 — wrist_flex held at 0 throughout a commanded sweep).
                    # Latency from this render-rate tick is ~240 ms but correctness-
                    # preserving; higher-rate alternatives are out of scope here.
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("isaac_read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ros2_subscribe_joint_state", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ros2_publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("articulation_controller", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ros2_context.inputs:useDomainIDEnvVar", True),
                    ("ros2_context.inputs:domain_id", 0),
                    ("ros2_subscribe_joint_state.inputs:topicName", "/joint_states"),
                    ("ros2_subscribe_joint_state.inputs:nodeNamespace", ""),
                    ("ros2_publish_clock.inputs:topicName", "/clock"),
                    ("ros2_publish_clock.inputs:nodeNamespace", ""),
                    ("isaac_read_simulation_time.inputs:resetOnStop", False),
                    ("articulation_controller.inputs:robotPath", "/World/SO_ARM101"),
                    # Use joint indices [0..5] instead of joint names because
                    # the real robot publishes URDF names (Rotation, Pitch, ...)
                    # which don't match the USD names (shoulder_pan, shoulder_lift, ...).
                    # Both kinematic chains have the same ordering so index mapping is 1:1.
                    ("articulation_controller.inputs:jointIndices", [0, 1, 2, 3, 4, 5]),
                ],
                og.Controller.Keys.CONNECT: [
                    ("on_playback_tick.outputs:tick", "ros2_subscribe_joint_state.inputs:execIn"),
                    ("on_playback_tick.outputs:tick", "ros2_publish_clock.inputs:execIn"),
                    ("on_playback_tick.outputs:tick", "articulation_controller.inputs:execIn"),
                    ("ros2_context.outputs:context", "ros2_subscribe_joint_state.inputs:context"),
                    ("ros2_context.outputs:context", "ros2_publish_clock.inputs:context"),
                    ("isaac_read_simulation_time.outputs:simulationTime", "ros2_publish_clock.inputs:timeStamp"),
                    ("ros2_subscribe_joint_state.outputs:positionCommand", "articulation_controller.inputs:positionCommand"),
                    ("ros2_subscribe_joint_state.outputs:velocityCommand", "articulation_controller.inputs:velocityCommand"),
                    ("ros2_subscribe_joint_state.outputs:effortCommand", "articulation_controller.inputs:effortCommand"),
                    # NOTE: jointNames NOT connected - using jointIndices instead
                    # to avoid name mismatch between URDF and USD joint names
                ],
            }
        )

        print("ROS 2 Action Graph setup complete.")

    def _stop_physics_callback(self, sub_attr, artic_attr, active_attr, warmup_attr):
        """Stop a physics callback and clean up its resources."""
        if hasattr(self, sub_attr) and getattr(self, sub_attr) is not None:
            setattr(self, sub_attr, None)
        if hasattr(self, artic_attr):
            setattr(self, artic_attr, None)
        setattr(self, active_attr, False)
        setattr(self, warmup_attr, 0)

    def _lazy_init_articulation(self, artic_attr, prim_path, name_prefix, warmup_attr):
        """Lazy-init an ArticulationView, returning it or None if not ready.

        Also handles warmup countdown and stale handle detection.
        Returns the articulation if ready, None otherwise.
        """
        warmup = getattr(self, warmup_attr)
        if warmup > 0:
            setattr(self, warmup_attr, warmup - 1)
            return None

        artic = getattr(self, artic_attr)
        if artic is None:
            try:
                import time as _t
                from isaacsim.core.prims import Articulation as _AV
                artic = _AV(prim_paths_expr=prim_path, name=f"{name_prefix}_{int(_t.time()*1000)}")
                artic.initialize()
                setattr(self, artic_attr, artic)
            except Exception:
                setattr(self, artic_attr, None)
                return None

        if not artic.is_physics_handle_valid():
            setattr(self, artic_attr, None)
            setattr(self, warmup_attr, 30)
            return None

        return artic

    def setup_force_publish_action_graph(self):
        """Setup force publishing as geometry_msgs/WrenchStamped on /force_torque_sensor_broadcaster/wrench_sim.

        Uses a physics step callback to read measured joint forces at physics rate
        (~60 Hz). Uses get_measured_joint_forces() which returns 6-DOF spatial forces
        [Fx, Fy, Fz, Tx, Ty, Tz] per joint. Publishes full wrench from the end-effector joint,
        matching the real robot's force_torque_sensor_broadcaster message format.
        """
        import omni.physx

        print("Setting up SO-ARM101 Force Publisher (WrenchStamped)...")

        # Clean up any previous physics callback
        self._stop_force_publish()

        # Check that SO-ARM101 prim exists in the scene
        stage_check = omni.usd.get_context().get_stage()
        if not stage_check.GetPrimAtPath("/World/SO_ARM101"):
            print("Error: SO-ARM101 prim not found at /World/SO_ARM101. Load robot first.")
            return

        # Reuse existing ArticulationView if available; otherwise physics callback will lazy-init
        if self._robot_view is not None:
            self._effort_articulation = self._robot_view
            print(f"Using existing ArticulationView. Joints: {self._effort_articulation.joint_names}")
        else:
            self._effort_articulation = None
            print("ArticulationView not cached (hot-reload?). Will lazy-init in physics callback.")

        # Delete existing graph if it exists
        graph_path = "/Graph/ActionGraph_SO_ARM101_ForcePublish"
        keys = og.Controller.Keys
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(graph_path):
            stage.RemovePrim(graph_path)

        # Create OmniGraph with OnPlaybackTick for ROS2 publisher execution
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("tick", "omni.graph.action.OnPlaybackTick"),
                    ("context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("publisher", "isaacsim.ros2.bridge.ROS2Publisher")
                ],
                keys.SET_VALUES: [
                    ("publisher.inputs:messageName", "WrenchStamped"),
                    ("publisher.inputs:messagePackage", "geometry_msgs"),
                    ("publisher.inputs:topicName", "force_torque_sensor_broadcaster/wrench_sim"),
                ],
                keys.CONNECT: [
                    ("tick.outputs:tick", "publisher.inputs:execIn"),
                    ("context.outputs:context", "publisher.inputs:context"),
                ]
            }
        )

        # Store graph path and publisher node reference for the physics callback.
        # The ROS2Publisher C++ node auto-creates dynamic attributes (wrench:force:x, etc.)
        # based on the WrenchStamped message definition. We access them via og.Controller.attribute()
        # using colon-separated field paths (the convention for nested ROS2 message fields in OG).
        self._force_graph_path = graph_path
        self._force_pub_node = nodes[2]  # publisher node

        # Set header frame_id
        og.Controller.attribute("inputs:header:frame_id", self._force_pub_node).set("tool0")

        # Physics callback reads joint forces at physics rate and updates the attributes
        self._force_physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step_force
        )

        self._force_publish_active = True

        print(f"SO-ARM101 Force Publisher created at {graph_path}")
        print("Publishing geometry_msgs/WrenchStamped to topic: /force_torque_sensor_broadcaster/wrench_sim")
        print("Full 6-DOF wrench (Fx, Fy, Fz, Tx, Ty, Tz) from SO-ARM101 end-effector joint")
        print("Joint forces read at physics rate (~60 Hz)")

    def _on_physics_step_force(self, dt):
        """Physics step callback - read joint forces and update OmniGraph WrenchStamped attributes.

        Uses get_measured_joint_forces() which returns 6-DOF spatial forces
        [Fx, Fy, Fz, Tx, Ty, Tz] per joint. Publishes full wrench from the last joint
        (wrist joint) as geometry_msgs/WrenchStamped.
        """
        try:
            artic = self._lazy_init_articulation(
                '_effort_articulation', '/World/SO_ARM101',
                'force_pub_ctrl', '_force_warmup')
            if artic is None:
                return
            self._effort_articulation = artic

            # get_measured_joint_forces returns shape (num_articulations, num_joints, 6)
            # where 6 = [Fx, Fy, Fz, Tx, Ty, Tz]
            forces = self._effort_articulation.get_measured_joint_forces()
            if forces is not None and len(forces) > 0:
                # Use last joint (wrist joint) - full 6-DOF wrench
                wrist = forces[0][-1]
                node = self._force_pub_node
                og.Controller.attribute("inputs:wrench:force:x", node).set(float(wrist[0]))
                og.Controller.attribute("inputs:wrench:force:y", node).set(float(wrist[1]))
                og.Controller.attribute("inputs:wrench:force:z", node).set(float(wrist[2]))
                og.Controller.attribute("inputs:wrench:torque:x", node).set(float(wrist[3]))
                og.Controller.attribute("inputs:wrench:torque:y", node).set(float(wrist[4]))
                og.Controller.attribute("inputs:wrench:torque:z", node).set(float(wrist[5]))
        except Exception as e:
            print(f"[Force callback] {e}")

    def _stop_force_publish(self):
        """Stop the physics-rate force publisher and clean up resources."""
        self._stop_physics_callback('_force_physx_sub', '_effort_articulation',
                                    '_force_publish_active', '_force_warmup')

    def _configure_gripper_material(self, robot_prim_path, stage):
        """Configure gripper physics material on the jaw and gripper collision meshes."""
        jaw_link = f"{robot_prim_path}/moving_jaw_so101_v1_link"
        gripper_link = f"{robot_prim_path}/gripper_link"

        gripper_mat_path = f"{robot_prim_path}/GripperPhysicsMaterial"
        gripper_material = UsdShade.Material.Define(stage, gripper_mat_path)
        gripper_mat_api = UsdPhysics.MaterialAPI.Apply(gripper_material.GetPrim())
        gripper_mat_api.CreateDynamicFrictionAttr().Set(self._gripper_dynamic_friction)
        gripper_mat_api.CreateRestitutionAttr().Set(self._gripper_restitution)
        gripper_mat_api.CreateStaticFrictionAttr().Set(self._gripper_static_friction)
        gripper_physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(gripper_material.GetPrim())
        gripper_physx_mat_api.CreateFrictionCombineModeAttr().Set(self._gripper_friction_combine_mode)
        gripper_physx_mat_api.CreateRestitutionCombineModeAttr().Set(self._gripper_restitution_combine_mode)
        print(f"Created gripper physics material at {gripper_mat_path}")

        gripper_mat_sdf_path = Sdf.Path(gripper_mat_path)
        for link_path in [f"{jaw_link}/collisions", f"{gripper_link}/collisions"]:
            link_prim = stage.GetPrimAtPath(link_path)
            if link_prim and link_prim.IsValid():
                from omni.physx.scripts import physicsUtils
                physicsUtils.add_physics_material_to_prim(stage, link_prim, gripper_mat_sdf_path)
                print(f"Bound gripper physics material to {link_path}")
            else:
                print(f"Warning: Collision prim not found at {link_path}")

    def import_realsense_camera(self):
        """Import Intel RealSense D455 camera"""
        usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/5.0/Isaac/Sensors/Intel/RealSense/rsd455.usd"
        filename = os.path.splitext(os.path.basename(usd_path))[0]
        prim_path = f"/World/{filename}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        print(f"Prim imported at {prim_path}")

    def attach_camera_to_robot(self):
        """Attach RealSense camera to SO-ARM101 gripper frame link."""
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()

        # Check camera prim exists
        if not stage.GetPrimAtPath("/World/rsd455").IsValid():
            print("[ERROR] Camera prim /World/rsd455 not found — import the RealSense camera first")
            return

        # SO-ARM101 end effector frame (inside gripper_link)
        target_link = "/World/SO_ARM101/gripper_link"
        if not stage.GetPrimAtPath(target_link).IsValid():
            print(f"[ERROR] {target_link} not found — import the SO-ARM101 first")
            return

        # Move the prim
        camera_dest = f"{target_link}/rsd455"
        result = omni.kit.commands.execute('MovePrim',
                                 path_from="/World/rsd455",
                                 path_to=camera_dest)
        if not result:
            print(f"[ERROR] Failed to move camera prim to {target_link}")
            return

        omni.kit.commands.execute('ChangeProperty',
                                 prop_path=f"{camera_dest}.xformOp:translate",
                                 value=Gf.Vec3d(0.0, 0.0, 0.05),
                                 prev=None)
        omni.kit.commands.execute('ChangeProperty',
                                 prop_path=f"{camera_dest}.xformOp:rotateZYX",
                                 value=Gf.Vec3d(0, 0, 0),
                                 prev=None)

        # Remove RigidBodyAPI from nested camera prim
        rsd455_prim = stage.GetPrimAtPath(f"{camera_dest}/RSD455")
        if rsd455_prim.IsValid() and rsd455_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(rsd455_prim).GetRigidBodyEnabledAttr().Set(False)
            rsd455_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print("Removed RigidBodyAPI from RSD455")

        print(f"RealSense camera attached to SO-ARM101 gripper_link at {target_link}")

    def setup_camera_action_graph(self):
        """Create ActionGraph for camera ROS2 publishing"""
        import yaml

        CAMERA_PRIM = "/World/SO_ARM101/gripper_link/rsd455/RSD455/Camera_OmniVision_OV9782_Color"

        # Check that camera prim exists
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(CAMERA_PRIM):
            print(f"Error: Camera prim not found at {CAMERA_PRIM}. Attach camera to robot first.")
            return

        # Load camera config from robot_config.yaml
        config_path = os.path.expanduser(
            "~/Desktop/ros2_ws/src/aruco_camera_localizer/config/robot_config.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cam_cfg = yaml.safe_load(f)
            print(f"Loaded camera config from {config_path}")
        else:
            cam_cfg = {}
            print(f"Warning: {config_path} not found, using defaults")

        # Configuration from yaml (with fallbacks)
        IMAGE_WIDTH = cam_cfg.get("camera_width", 1280)
        IMAGE_HEIGHT = cam_cfg.get("camera_height", 720)
        ROS2_TOPIC = "intel_camera_rgb_sim"

        graph_path = "/Graph/ActionGraph_Camera"

        # Skip if already created
        if stage.GetPrimAtPath(graph_path):
            print(f"ActionGraph already exists at {graph_path}, skipping creation.")
            return

        print(f"Creating ActionGraph: {graph_path}")
        print(f"Camera: {CAMERA_PRIM}")
        print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        print(f"ROS2 Topic: {ROS2_TOPIC}")

        keys = og.Controller.Keys

        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("TFPublish", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ],
                keys.SET_VALUES: [
                    ("RenderProduct.inputs:cameraPrim", CAMERA_PRIM),
                    ("RenderProduct.inputs:width", IMAGE_WIDTH),
                    ("RenderProduct.inputs:height", IMAGE_HEIGHT),
                    ("CameraInfoPublish.inputs:topicName", "camera_info"),
                    ("CameraInfoPublish.inputs:frameId", "camera_link"),
                    ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                    ("RGBPublish.inputs:topicName", ROS2_TOPIC),
                    ("RGBPublish.inputs:type", "rgb"),
                    ("RGBPublish.inputs:frameId", "camera_link"),
                    ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                    ("TFPublish.inputs:topicName", "tf_cam"),
                    ("TFPublish.inputs:targetPrims", CAMERA_PRIM),
                    ("TFPublish.inputs:parentPrim", "/World"),
                    ("ReadSimTime.inputs:resetOnStop", False),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "TFPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                    ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
                    ("Context.outputs:context", "RGBPublish.inputs:context"),
                    ("Context.outputs:context", "TFPublish.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "TFPublish.inputs:timeStamp"),
                ],
            }
        )

        # Override the USD camera's baked-in intrinsics to match the real Intel RealSense
        camera_prim = stage.GetPrimAtPath(CAMERA_PRIM)
        if camera_prim and camera_prim.IsValid():
            from pxr import Sdf
            import numpy as np

            # Use calibrated camera_matrix [fx, fy, cx, cy] from yaml if available,
            # otherwise fall back to HFOV/VFOV computation
            camera_matrix = cam_cfg.get("camera_matrix")
            if camera_matrix and len(camera_matrix) == 4:
                fx, fy, cx, cy = camera_matrix
                print(f"Using calibrated camera_matrix from config: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            else:
                hfov_deg = cam_cfg.get("camera_hfov", 69.4)
                vfov_deg = cam_cfg.get("camera_vfov", 42.5)
                fx = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(hfov_deg / 2)))
                fy = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(vfov_deg / 2)))
                cx = IMAGE_WIDTH * 0.5
                cy = IMAGE_HEIGHT * 0.5
                print(f"Using HFOV/VFOV from config: hfov={hfov_deg}, vfov={vfov_deg}")

            # Distortion coefficients from yaml
            dist_coeffs = cam_cfg.get("distortion_coeffs", [0, 0, 0, 0, 0])

            # Set USD aperture/focal-length so the renderer matches the FOV
            horizontal_aperture_mm = 36.0
            focal_length_mm = fx * horizontal_aperture_mm / IMAGE_WIDTH
            vertical_aperture_mm = IMAGE_HEIGHT * focal_length_mm / fy

            camera = UsdGeom.Camera(camera_prim)
            camera.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
            camera.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
            camera.CreateFocalLengthAttr().Set(focal_length_mm)
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

            # Set opencvPinhole lens distortion model for correct CameraInfo publishing
            camera_prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.String).Set("opencvPinhole")
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(IMAGE_WIDTH, IMAGE_HEIGHT))
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fx", Sdf.ValueTypeNames.Float).Set(fx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fy", Sdf.ValueTypeNames.Float).Set(fy)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cx", Sdf.ValueTypeNames.Float).Set(cx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cy", Sdf.ValueTypeNames.Float).Set(cy)
            # Set distortion coefficients (k1, k2, p1, p2, k3, ..., s4)
            dist_attr_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
            for i, attr_name in enumerate(dist_attr_names):
                val = dist_coeffs[i] if i < len(dist_coeffs) else 0.0
                camera_prim.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{attr_name}", Sdf.ValueTypeNames.Float).Set(val)

            print(f"Overrode camera intrinsics on {CAMERA_PRIM}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"Distortion coefficients: {dist_coeffs}")

        print("\nCamera ActionGraph created successfully!")
        print(f"Test with: ros2 topic echo /{ROS2_TOPIC}")

    def attach_camera_mount_mesh(self):
        """Attach camera wrist mount mesh to camera_mount_link (visual only)."""
        stage = omni.usd.get_context().get_stage()

        robot_path = "/World/SO_ARM101"
        camera_mount_link = f"{robot_path}/camera_mount_link"

        if not stage.GetPrimAtPath(camera_mount_link).IsValid():
            print(f"[ERROR] {camera_mount_link} not found — load the robot first")
            return

        mesh_prim_path = f"{camera_mount_link}/camera_wrist_mount"
        if stage.GetPrimAtPath(mesh_prim_path).IsValid():
            print(f"Camera mount mesh already attached at {mesh_prim_path}")
        else:
            prim = stage.DefinePrim(mesh_prim_path, "Xform")
            prim.GetReferences().AddReference(_get_camera_mount_usd_path())
            # URDF visual origin: xyz="-0.763660 -0.207080 -0.025950" rpy="0 0 0"
            # STL is in mm, scale 0.001 for meters
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(-0.763660, -0.207080, -0.025950))
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            xformable.AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
            print(f"Attached camera mount mesh at {mesh_prim_path}")

    def attach_usb_camera(self):
        """Attach USB camera (ELP OV9732) mesh to camera_mount_link and create camera prim.

        URDF chain: camera_mount -> usb_camera (usb_camera_joint) -> camera_link (camera_link_joint)
        The USB camera mesh is placed using usb_camera_joint transform.
        The camera prim is placed at the lens center using camera_link_joint transform,
        with Ry(90°) applied to convert ROS +X-forward to Isaac Sim -Z-forward.

        Structure under camera_mount_link:
          usb_camera/            <- Xform with usb_camera_joint pos + rot (NO scale)
            usb_cam_mesh/        <- Xform with mesh reference + visual origin + scale 0.001
            wrist_camera         <- Camera at lens center with Isaac Sim orientation
        """
        stage = omni.usd.get_context().get_stage()

        robot_path = "/World/SO_ARM101"
        camera_mount_link = f"{robot_path}/camera_mount_link"

        if not stage.GetPrimAtPath(camera_mount_link).IsValid():
            print(f"[ERROR] {camera_mount_link} not found — attach camera mount first")
            return

        # 1. Parent Xform with usb_camera_joint transform (no scale)
        usb_cam_path = f"{camera_mount_link}/usb_camera"
        if not stage.GetPrimAtPath(usb_cam_path).IsValid():
            prim = stage.DefinePrim(usb_cam_path, "Xform")
            xformable = UsdGeom.Xformable(prim)
            # usb_camera_joint: xyz=(-0.010340, 0.004827, -0.019687)
            #                   rpy=(1.570625, 0.000515, 2.705234)
            xformable.AddTranslateOp().Set(Gf.Vec3d(-0.010340, 0.004827, -0.019687))
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(0.15324586, 0.15286412, 0.69032376, 0.69036321))
            print(f"Created usb_camera Xform at {usb_cam_path}")

        # 2. Mesh child with visual origin + scale 0.001
        mesh_path = f"{usb_cam_path}/usb_cam_mesh"
        if not stage.GetPrimAtPath(mesh_path).IsValid():
            mesh_prim = stage.DefinePrim(mesh_path, "Xform")
            mesh_prim.GetReferences().AddReference(_get_usb_camera_usd_path())
            xformable = UsdGeom.Xformable(mesh_prim)
            # URDF visual origin: xyz=(0.0, -0.007299, 0.0) — in meters, but parent
            # is unscaled so translate is in meters. Scale converts STL mm to meters.
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.007299, 0.0))
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            xformable.AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
            print(f"Attached USB camera mesh at {mesh_path}")
        else:
            print(f"USB camera mesh already at {mesh_path}")

        # 3. Camera prim as sibling of mesh (not inside scaled mesh)
        camera_prim_path = f"{usb_cam_path}/wrist_camera"
        if stage.GetPrimAtPath(camera_prim_path).IsValid():
            print(f"Wrist camera already exists at {camera_prim_path}")
            return

        stage.DefinePrim(camera_prim_path, "Camera")
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        camera = UsdGeom.Camera(camera_prim)

        # OV9732: 100° HFOV, rendered at 640x480
        IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
        HFOV_DEG = 100.0
        VFOV_DEG = HFOV_DEG * IMAGE_HEIGHT / IMAGE_WIDTH

        fx = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(HFOV_DEG / 2)))
        fy = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(VFOV_DEG / 2)))

        h_aperture = 36.0
        focal_length = fx * h_aperture / IMAGE_WIDTH
        v_aperture = IMAGE_HEIGHT * focal_length / fy

        camera.CreateHorizontalApertureAttr().Set(h_aperture)
        camera.CreateVerticalApertureAttr().Set(v_aperture)
        camera.CreateFocalLengthAttr().Set(focal_length)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

        # Lens center: camera_link_joint xyz=(0, 0.0139, 0) in usb_camera frame
        # Orientation: Rx(+90°) * Rz(-90°) — points camera -Z along lens direction
        # and aligns image up-vector so gripper appears at bottom of frame.
        # Combined quaternion (wxyz) = (0.5, 0.5, 0.5, -0.5)
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0.0139, 0))
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(0.5, 0.5, 0.5, -0.5))

        print(f"Created OV9732 wrist camera at {camera_prim_path}")
        print(f"  Lens at usb_camera (0, 13.9mm, 0), looking along usb_camera -Y")
        print(f"  HFOV={HFOV_DEG}, {IMAGE_WIDTH}x{IMAGE_HEIGHT}, fl={focal_length:.2f}mm")

    def setup_wrist_camera_action_graph(self):
        """Create ActionGraph for wrist camera (OV9732) ROS2 publishing.
        Also attaches USB camera and creates camera prim if not present."""
        CAMERA_PRIM = "/World/SO_ARM101/camera_mount_link/usb_camera/wrist_camera"
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480
        ROS2_TOPIC = "wrist_camera_rgb_sim"

        stage = omni.usd.get_context().get_stage()

        # Auto-attach USB camera module if camera prim doesn't exist
        if not stage.GetPrimAtPath(CAMERA_PRIM).IsValid():
            self.attach_camera_mount_mesh()
            self.attach_usb_camera()
        if not stage.GetPrimAtPath(CAMERA_PRIM).IsValid():
            print(f"[ERROR] Wrist camera not found at {CAMERA_PRIM} after attach attempt.")
            return

        graph_path = "/Graph/ActionGraph_WristCamera"
        if stage.GetPrimAtPath(graph_path) and stage.GetPrimAtPath(graph_path).IsValid():
            print(f"ActionGraph already exists at {graph_path}, skipping.")
            return

        print(f"Creating wrist camera ActionGraph: {graph_path}")

        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    # No ROS2CameraHelper RGB node here. On Isaac Sim 5.0 it never published, so the
                    # hidden-viewport _ReplicatorCameraPublisher (rgba8) became the wrist image source,
                    # and the training datasets were recorded from it. Isaac Sim 6.1 fixed the helper,
                    # so keeping it made a second publisher (rgb8) on the same topic, interleaving two
                    # encodings and breaking lerobot's ROS2Camera (reshape to 4 channels failed).
                ],
                keys.SET_VALUES: [
                    ("RenderProduct.inputs:cameraPrim", CAMERA_PRIM),
                    ("RenderProduct.inputs:width", IMAGE_WIDTH),
                    ("RenderProduct.inputs:height", IMAGE_HEIGHT),
                    ("CameraInfoPublish.inputs:topicName", "wrist_camera_info"),
                    ("CameraInfoPublish.inputs:frameId", "camera_link"),
                    ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                    ("ReadSimTime.inputs:resetOnStop", False),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
                ],
            }
        )

        # Re-author USD camera intrinsics + opencvPinhole AFTER the action graph is
        # created. This mirrors the exact ordering in setup_camera_action_graph
        # (Intel cam, ur5e-port pattern) and exts/ur5e-dt/.../setup_camera_action_graph.
        # The post-graph re-authoring appears to be load-bearing: ROS2CameraHelper
        # silently fails to register a DDS publisher if intrinsics are only set
        # before the graph references the camera prim. Re-authoring after the
        # render-product node is wired triggers the binding properly.
        camera_prim = stage.GetPrimAtPath(CAMERA_PRIM)
        if camera_prim and camera_prim.IsValid():
            import numpy as np
            # OV9732: 100° HFOV, square pixels (matches sim render product + real lens)
            HFOV_DEG = 100.0
            VFOV_DEG = HFOV_DEG * IMAGE_HEIGHT / IMAGE_WIDTH
            fx = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(HFOV_DEG / 2)))
            fy = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(VFOV_DEG / 2)))
            cx = IMAGE_WIDTH * 0.5
            cy = IMAGE_HEIGHT * 0.5

            # Re-author USD aperture/focal/clipping (already set in attach_usb_camera,
            # but re-setting after graph creation is what makes ROS2CameraHelper bind).
            horizontal_aperture_mm = 36.0
            focal_length_mm = fx * horizontal_aperture_mm / IMAGE_WIDTH
            vertical_aperture_mm = IMAGE_HEIGHT * focal_length_mm / fy
            camera = UsdGeom.Camera(camera_prim)
            camera.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
            camera.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
            camera.CreateFocalLengthAttr().Set(focal_length_mm)
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

            # Set opencvPinhole lens distortion model.
            camera_prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.String).Set("opencvPinhole")
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(IMAGE_WIDTH, IMAGE_HEIGHT))
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fx", Sdf.ValueTypeNames.Float).Set(fx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fy", Sdf.ValueTypeNames.Float).Set(fy)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cx", Sdf.ValueTypeNames.Float).Set(cx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cy", Sdf.ValueTypeNames.Float).Set(cy)
            for attr_name in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]:
                camera_prim.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{attr_name}", Sdf.ValueTypeNames.Float).Set(0.0)
            print(f"Wrist camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        print(f"Wrist camera ActionGraph created!")
        print(f"Test with: ros2 topic echo /{ROS2_TOPIC}")

    def create_additional_camera(self):
        """Create additional camera based on selected view type"""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom
        import numpy as np

        def set_camera_pose(prim_path: str, position_xyz, quat_xyzw):
            """Apply translation and quaternion orientation (x, y, z, w) to a camera prim."""
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Camera prim '{prim_path}' not found.")
            xform = UsdGeom.Xform(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*position_xyz))
            quat = Gf.Quatd(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)

        def configure_camera_properties(camera_prim, width, height):
            """Configure camera properties to match Intel RealSense camera intrinsics."""
            # Intel RealSense specs: HFOV=69.4°, VFOV=42.5° at 1280x720
            hfov_deg = 69.4
            vfov_deg = 42.5

            # Compute intrinsics: fx = width / (2 * tan(hfov/2))
            fx = width / (2 * np.tan(np.deg2rad(hfov_deg / 2)))
            fy = height / (2 * np.tan(np.deg2rad(vfov_deg / 2)))
            cx = width * 0.5
            cy = height * 0.5

            # Derive USD aperture/focal-length so the renderer matches the FOV
            # Keep horizontal_aperture fixed, compute focal_length from fx
            horizontal_aperture_mm = 36.0
            focal_length_mm = fx * horizontal_aperture_mm / width
            vertical_aperture_mm = height * focal_length_mm / fy

            camera = UsdGeom.Camera(camera_prim)
            camera.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
            camera.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
            camera.CreateFocalLengthAttr().Set(focal_length_mm)
            camera.CreateProjectionAttr().Set("perspective")
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

            # Set opencvPinhole lens distortion model so ROS2 CameraInfo
            # publishes fx/fy/cx/cy directly
            camera_prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.String).Set("opencvPinhole")
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fx", Sdf.ValueTypeNames.Float).Set(fx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:fy", Sdf.ValueTypeNames.Float).Set(fy)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cx", Sdf.ValueTypeNames.Float).Set(cx)
            camera_prim.CreateAttribute("omni:lensdistortion:opencvPinhole:cy", Sdf.ValueTypeNames.Float).Set(cy)
            # Zero distortion coefficients (k1-k6, p1, p2, s1-s4)
            for attr_name in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]:
                camera_prim.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{attr_name}", Sdf.ValueTypeNames.Float).Set(0.0)

        # Check which camera type is selected
        is_workspace = self._workspace_checkbox.model.get_value_as_bool()
        is_custom = self._custom_checkbox.model.get_value_as_bool()

        # Get selected resolution from combo box
        resolution_index = self._resolution_combo.model.get_item_value_model().get_value_as_int()
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        resolution = resolutions[resolution_index]

        stage = omni.usd.get_context().get_stage()
        if not stage:
            print("Error: No stage found")
            return

        if is_workspace:
            prim_path = "/World/workspace_camera"
            position = (0.6980338662434342, -0.3958419458255837, 0.45249457626357603)
            # Hand-tuned for SO-ARM101 — overhead-front view of robot + cup arc
            quat_xyzw = (0.4306700623322567, 0.2612220733325859, 0.4480130626995852, 0.7386275255262917)  # x, y, z, w

            # Create camera prim using UsdGeom.Camera
            camera_prim = UsdGeom.Camera.Define(stage, prim_path)
            if not camera_prim:
                print(f"Error: Failed to create camera at {prim_path}")
                return

            # Configure camera properties
            configure_camera_properties(camera_prim.GetPrim(), resolution[0], resolution[1])

            # Set camera pose
            set_camera_pose(prim_path, position, quat_xyzw)
            print(f"Workspace camera created at {prim_path} with resolution {resolution[0]}x{resolution[1]}")

        if is_custom:
            # Get custom prim path from text field
            custom_prim_path = self._custom_camera_prim_field.model.get_value_as_string()
            if not custom_prim_path or custom_prim_path.strip() == "":
                print("Error: Please enter a valid camera prim path")
                return
            
            # Check if the prim already exists
            existing_prim = stage.GetPrimAtPath(custom_prim_path)
            if existing_prim and existing_prim.IsValid():
                print(f"Camera prim already exists at {custom_prim_path}. Using existing prim.")
                camera_prim = UsdGeom.Camera(existing_prim)
                if not camera_prim:
                    # If it exists but is not a camera, convert it
                    print(f"Prim at {custom_prim_path} exists but is not a camera. Creating camera...")
                    camera_prim = UsdGeom.Camera.Define(stage, custom_prim_path)
            else:
                # Create new camera prim at the specified path
                camera_prim = UsdGeom.Camera.Define(stage, custom_prim_path)
                if not camera_prim:
                    print(f"Error: Failed to create camera at {custom_prim_path}")
                    return
            
            # Configure camera properties
            configure_camera_properties(camera_prim.GetPrim(), resolution[0], resolution[1])
            
            # Note: Pose is not set for custom prim - user should position it manually or it uses existing pose
            print(f"Custom camera created/updated at {custom_prim_path} with resolution {resolution[0]}x{resolution[1]}")

    def create_additional_camera_actiongraph(self):
        """Create ActionGraph for additional camera ROS2 publishing"""
        # Check which cameras exist and create action graphs accordingly
        is_workspace = self._workspace_checkbox.model.get_value_as_bool()
        is_custom = self._custom_checkbox.model.get_value_as_bool()

        # Get selected resolution from combo box
        resolution_index = self._resolution_combo.model.get_item_value_model().get_value_as_int()
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        width, height = resolutions[resolution_index]

        stage = omni.usd.get_context().get_stage()

        if is_workspace:
            if not stage.GetPrimAtPath("/World/workspace_camera"):
                print("Error: Workspace camera not found at /World/workspace_camera. Create it first.")
            else:
                self._create_camera_actiongraph(
                    "/World/workspace_camera",
                    width, height,
                    "workspace_camera_sim",
                    "WorkspaceCameraSim"
                )

        if is_custom:
            # Get custom prim path from text field
            custom_prim_path = self._custom_camera_prim_field.model.get_value_as_string()
            if not custom_prim_path or custom_prim_path.strip() == "":
                print("Error: Please enter a valid camera prim path")
                return

            # Get ROS2 topic name from text field, or generate from prim path if empty
            topic_name = self._custom_camera_topic_field.model.get_value_as_string()
            if not topic_name or topic_name.strip() == "":
                # Extract a clean name for the topic from the prim path if topic is empty
                prim_name = custom_prim_path.split("/")[-1] if "/" in custom_prim_path else custom_prim_path
                topic_name = prim_name.lower().replace(" ", "_").replace("-", "_")
                print(f"Topic name not specified, using generated name: {topic_name}")

            # Use topic name for graph suffix (sanitize for graph name - remove special chars, capitalize)
            # Convert topic name to a valid graph suffix: remove underscores, capitalize words
            graph_suffix = topic_name.replace("_", " ").replace("-", " ").title().replace(" ", "")

            if not stage.GetPrimAtPath(custom_prim_path):
                print(f"Error: Camera prim not found at {custom_prim_path}. Create it first.")
            else:
                self._create_camera_actiongraph(
                    custom_prim_path,
                    width, height,
                    topic_name,
                    graph_suffix
                )

    def _project_world_to_camera(self, world_xyz, camera_prim_path="/World/workspace_camera",
                                  width=640, height=480, hfov_deg=69.4):
        """Project a world-space point through the workspace camera to image pixels.

        USD camera convention: camera looks down -Z in its local frame, +X right,
        +Y up. Image convention is +X right, +Y down — we flip the v sign to match.

        Args:
            world_xyz: (x, y, z) in world meters.
            camera_prim_path: defaults to the workspace camera the recorder uses.
            width / height / hfov_deg: must match the publisher's intrinsics.
                vfov is derived (square-pixel formula) — same as
                _ViewportCameraPublisher uses. Don't change unless the recording
                pipeline changes too.

        Returns:
            (u_px, v_px, depth_m) on success, or None if the point is behind the
            camera or the camera prim is missing.
        """
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(camera_prim_path)
        if not cam_prim or not cam_prim.IsValid():
            return None
        cam_xform = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        view = cam_xform.GetInverse()
        p_cam = view.Transform(Gf.Vec3d(float(world_xyz[0]),
                                         float(world_xyz[1]),
                                         float(world_xyz[2])))
        # Camera looks down -Z. Visible only if z is negative.
        if p_cam[2] >= 0:
            return None
        depth = -p_cam[2]
        # Square-pixel intrinsics derived from HFOV (matches publisher path).
        fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg / 2.0)))
        vfov_deg = hfov_deg * height / width
        fy = height / (2.0 * np.tan(np.deg2rad(vfov_deg / 2.0)))
        u = fx * p_cam[0] / depth + width / 2.0
        # Flip v: USD Y-up → image Y-down.
        v = -fy * p_cam[1] / depth + height / 2.0
        return (u, v, depth)

    def _is_in_camera_view(self, world_xyz, margin_px=30,
                           camera_prim_path="/World/workspace_camera",
                           width=640, height=480, hfov_deg=69.4):
        """True iff `world_xyz` projects to within (margin_px) of the camera frame.

        Margin is for both u and v. With the default 30 px on a 640×480 frame,
        an in-view point sits ≥30 px from any edge — enough headroom that a
        20 mm lego (≈25-40 px wide at the workspace camera's typical depth)
        is fully visible, not clipped at the rim.
        """
        proj = self._project_world_to_camera(
            world_xyz, camera_prim_path, width, height, hfov_deg)
        if proj is None:
            return False
        u, v, _ = proj
        return (margin_px <= u <= width - margin_px and
                margin_px <= v <= height - margin_px)

    def _set_active_viewport_camera(self, camera_path: str) -> bool:
        """Switch the active Isaac Sim viewport to render through the given camera prim path.
        Records the previous camera path in self._prev_viewport_cam for potential undo.
        # TODO: Add ComboBox UI picker (option 3a) in a future task — deferred to keep this change atomic.
        """
        try:
            import omni.kit.viewport.utility as vp_utils
            from pxr import Sdf
            viewport = vp_utils.get_active_viewport()
            if viewport is None:
                print("No active viewport found — skipping camera switch.")
                return False
            prev = str(viewport.camera_path)
            if prev != camera_path:
                self._prev_viewport_cam = prev
            viewport.camera_path = Sdf.Path(camera_path)
            print(f"Active viewport → {camera_path}")
            return True
        except Exception as ex:
            print(f"Error setting active viewport camera: {ex}")
            return False

    def _create_camera_actiongraph(self, camera_prim, width, height, topic, graph_suffix):
        """Helper method to create camera ActionGraph using og.Controller.edit().

        Follows the same pattern as isaacsim.ros2.bridge's built-in camera graph
        shortcut (og_rtx_sensors.py). Creates all nodes, sets values, and wires
        connections in a single batch call. If the graph already exists it is
        deleted and recreated with the current settings.
        """
        graph_path = f"/Graph/ActionGraph_{graph_suffix}"

        # Delete existing graph so it gets recreated with current settings
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(graph_path):
            stage.RemovePrim(graph_path)
            print(f"Removed existing ActionGraph at {graph_path}")

        print(f"Creating ActionGraph: {graph_path}")
        print(f"Camera: {camera_prim}, Resolution: {width}x{height}, Topic: /{topic}")

        keys = og.Controller.Keys

        try:
            (graph_handle, nodes, _, _) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                        ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    keys.SET_VALUES: [
                        ("RenderProduct.inputs:cameraPrim", camera_prim),
                        ("RenderProduct.inputs:width", width),
                        ("RenderProduct.inputs:height", height),
                        ("CameraInfoPublish.inputs:topicName", "camera_info"),
                        ("CameraInfoPublish.inputs:frameId", "camera_link"),
                        ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                        ("RGBPublish.inputs:topicName", topic),
                        ("RGBPublish.inputs:type", "rgb"),
                        ("RGBPublish.inputs:frameId", "camera_link"),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                        ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                        ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                        ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                        ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
                        ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                        ("Context.outputs:context", "RGBPublish.inputs:context"),
                    ],
                },
            )
            print(f"{graph_suffix} ActionGraph created successfully!")
            print(f"Test with: ros2 topic echo /{topic}")
        except Exception as e:
            print(f"Error creating ActionGraph: {e}")
            traceback.print_exc()

    def _get_object_names(self, folder_path="/World/Objects"):
        """Return set of object names currently in the scene."""
        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath(folder_path)
        if not objects_root or not objects_root.IsValid():
            return set()
        return {child.GetName() for child in objects_root.GetChildren()
                if child.GetName() != "PhysicsMaterial"}

    def _get_block_params(self):
        """Return physics settings for lego block objects."""
        return {
            "collision_approximation": self._block_collision_approximation,
            "rest_offset": self._block_rest_offset,
            "angular_damping": self._block_angular_damping,
        }

    def _get_prim_bbox_size(self, prim_path):
        """Return (sx, sy, sz) world-aligned bounding box size for a prim."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return (0.0, 0.0, 0.0)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        size = bbox.ComputeAlignedRange().GetSize()
        return (size[0], size[1], size[2])

    def _get_ground_z(self, prim_path):
        """Return the Z translation that places a prim's bbox bottom on the ground plane."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return self._ground_plane_z
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_min_z = bbox.ComputeAlignedRange().GetMin()[2]
        # Current prim Z translation
        xform = UsdGeom.Xformable(prim)
        current_translate = Gf.Vec3d(0, 0, 0)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                current_translate = op.Get()
                break
        current_z = float(current_translate[2])
        # The bbox bottom in world = current_z + local_bbox_bottom
        # We want world bbox bottom = ground_plane_z
        # So new_z = current_z + (ground_plane_z - bbox_min_z)
        return current_z + (self._ground_plane_z - bbox_min_z)

    def _get_block_color(self, block_name):
        """Get the color key for a block from its name (e.g. 'red_1' -> 'red')."""
        for color in BLOCK_COLORS:
            if block_name.startswith(color):
                return color
        return None

    def _set_block_material_color(self, prim_path, color_rgb):
        """Override color + plastic look on every shader under a prim. All edits are
        runtime overrides on the live stage — nothing is baked back into source USDs.

        Tuned against the WhatsApp real-world reference (matte plastic, mild sheen,
        no waxy/wet look). Doc range for 'glossy molded plastic' was clearcoat 0.1-0.3
        but our ambient-dominant lighting (DomeLight @ 1000 nits) amplifies it ~2×, so
        we sit at the low end. Values:
            roughness         0.45    (matches casual everyday-LEGO look in the photo)
            specular          0.20    (low — matte injection-molded ABS)
            clearcoat         0.05    (just enough to read as plastic, no sheen)
            clearcoatRoughness 0.10   (sharp coat reflection if it does show)
            metallic          0       (always for plastic)
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        rgb = Gf.Vec3f(*color_rgb)
        for desc in Usd.PrimRange(prim):
            if not desc.IsA(UsdShade.Shader):
                continue
            # Diffuse — both UsdPreviewSurface and OmniPBR naming
            for attr_name in ("inputs:diffuseColor", "inputs:diffuse_color_constant"):
                attr = desc.GetAttribute(attr_name)
                if attr:
                    attr.Set(rgb)
            # UsdPreviewSurface plastic look (no-ops on OmniPBR shaders)
            for attr_name, value in (
                ("inputs:roughness", 0.45),
                ("inputs:specular", 0.20),
                ("inputs:clearcoat", 0.05),
                ("inputs:clearcoatRoughness", 0.10),
                ("inputs:metallic", 0.0),
            ):
                a = desc.GetAttribute(attr_name)
                if a and a.IsValid():
                    a.Set(value)

    def _create_workspace_ground_tile(self, stage, x_range=None, y_range=None):
        """Author (or re-author) the workspace ground tile — a
        visual-only quad that overlays /World/defaultGroundPlane in
        the pick-place area.

        Authored as a UsdGeom.Mesh with 4 vertices + 1 quad face (no
        collision applied — the main ground plane handles physics).
        Sits at z = WORKSPACE_GROUND_Z_OFFSET_M to avoid z-fighting.
        Material is a UsdPreviewSurface bound to color/roughness from
        the WORKSPACE_GROUND_* constants.

        x_range / y_range: optional (min, max) tuples in world frame.
        When omitted, falls back to the WORKSPACE_GROUND_X_RANGE /
        WORKSPACE_GROUND_Y_RANGE module constants. The UI's Apply
        callback passes new ranges here to re-author the tile in place.

        No-op when the resolved x_range is None (lets the UI / a
        constant flip disable the tile entirely).
        """
        x_range = x_range if x_range is not None else WORKSPACE_GROUND_X_RANGE
        y_range = y_range if y_range is not None else WORKSPACE_GROUND_Y_RANGE
        if x_range is None or y_range is None:
            # Disabled: ensure no stale tile remains
            existing = stage.GetPrimAtPath("/World/workspace_ground")
            if existing and existing.IsValid():
                stage.RemovePrim("/World/workspace_ground")
            return

        plane_path = "/World/workspace_ground"
        # Always re-author so the UI can resize an existing tile in place.
        existing = stage.GetPrimAtPath(plane_path)
        if existing and existing.IsValid():
            stage.RemovePrim(plane_path)

        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        z = WORKSPACE_GROUND_Z_OFFSET_M

        mesh = UsdGeom.Mesh.Define(stage, plane_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(x_min, y_min, z),
            Gf.Vec3f(x_max, y_min, z),
            Gf.Vec3f(x_max, y_max, z),
            Gf.Vec3f(x_min, y_max, z),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateExtentAttr([
            Gf.Vec3f(x_min, y_min, z),
            Gf.Vec3f(x_max, y_max, z),
        ])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        # Material — UsdPreviewSurface matte. Created once and re-asserted on every
        # call so changes to WORKSPACE_GROUND_COLOR / WORKSPACE_GROUND_ROUGHNESS
        # propagate without needing a fresh stage.
        looks_path = "/World/Looks"
        UsdGeom.Scope.Define(stage, looks_path)
        mat_path = f"{looks_path}/workspace_ground_matte"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
        # Re-assert (or initially set) the inputs every call
        shader_prim = stage.GetPrimAtPath(mat_path + "/shader")
        if shader_prim and shader_prim.IsValid():
            shader_prim.GetAttribute("inputs:diffuseColor").Set(
                Gf.Vec3f(*WORKSPACE_GROUND_COLOR))
            shader_prim.GetAttribute("inputs:roughness").Set(
                WORKSPACE_GROUND_ROUGHNESS)
            shader_prim.GetAttribute("inputs:metallic").Set(0.0)
        material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        print(
            f"Authored workspace ground tile: "
            f"X=({x_min:.3f}, {x_max:.3f}) m, "
            f"Y=({y_min:.3f}, {y_max:.3f}) m"
        )

    def _create_workspace_ext_tile(self, stage, x_range=None, y_range=None):
        """Author (or re-author) the -X extension tile joining the main workspace
        tile at its -X edge. Visual-only quad with its own UsdPreviewSurface
        material so it can be color-tuned independently.

        Reads WORKSPACE_EXT_TILE_* module constants by default; pass
        x_range / y_range to override (used by the UI Apply button).
        Material inputs are re-asserted on every call so constants propagate."""
        plane_path = "/World/workspace_ground_ext"
        existing = stage.GetPrimAtPath(plane_path)
        if existing and existing.IsValid():
            stage.RemovePrim(plane_path)

        xr = x_range if x_range is not None else WORKSPACE_EXT_TILE_X_RANGE
        yr = y_range if y_range is not None else WORKSPACE_EXT_TILE_Y_RANGE
        x_min, x_max = float(xr[0]), float(xr[1])
        y_min, y_max = float(yr[0]), float(yr[1])
        z = WORKSPACE_GROUND_Z_OFFSET_M

        mesh = UsdGeom.Mesh.Define(stage, plane_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(x_min, y_min, z),
            Gf.Vec3f(x_max, y_min, z),
            Gf.Vec3f(x_max, y_max, z),
            Gf.Vec3f(x_min, y_max, z),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateExtentAttr([
            Gf.Vec3f(x_min, y_min, z),
            Gf.Vec3f(x_max, y_max, z),
        ])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        # Independent material — separate from workspace_ground_matte
        looks_path = "/World/Looks"
        UsdGeom.Scope.Define(stage, looks_path)
        mat_path = f"{looks_path}/workspace_ground_ext_matte"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
        # Re-assert inputs every call so constants changes propagate
        sh = stage.GetPrimAtPath(mat_path + "/shader")
        if sh and sh.IsValid():
            sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*WORKSPACE_EXT_TILE_COLOR))
            sh.GetAttribute("inputs:roughness").Set(WORKSPACE_EXT_TILE_ROUGHNESS)
            sh.GetAttribute("inputs:metallic").Set(0.0)
        material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        print(
            f"Authored workspace ext tile: "
            f"X=({x_min:.3f}, {x_max:.3f}) m, "
            f"Y=({y_min:.3f}, {y_max:.3f}) m (joins main tile at -X edge)"
        )

    def _create_omnipbr_matte(self, mat_path, diffuse_rgb, roughness=0.8):
        """Create (or return existing) OmniPBR matte material at mat_path.

        Uses CreateAndBindMdlMaterialFromLibrary then MovePrim to
        relocate to the chosen Looks scope, then sets diffuse color +
        roughness + metallic=0 for a matte plastic appearance.

        If the material already exists, re-asserts diffuse/roughness/metallic
        on its shader so callers can re-tune by changing arguments without
        having to delete-and-recreate.
        """
        stage = omni.usd.get_context().get_stage()
        existing = stage.GetPrimAtPath(mat_path)
        if existing and existing.IsValid():
            sh = stage.GetPrimAtPath(mat_path + "/Shader")
            if sh and sh.IsValid():
                d = sh.GetAttribute("inputs:diffuse_color_constant")
                if d and d.IsValid():
                    d.Set(Gf.Vec3f(*diffuse_rgb))
                r = sh.GetAttribute("inputs:reflection_roughness_constant")
                if r and r.IsValid():
                    r.Set(roughness)
                m = sh.GetAttribute("inputs:metallic_constant")
                if m and m.IsValid():
                    m.Set(0.0)
            return UsdShade.Material(existing)

        looks_path = mat_path.rsplit("/", 1)[0]
        UsdGeom.Scope.Define(stage, looks_path)
        out = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=out,
            select_new_prim=False,
        )
        if not out:
            return None
        omni.kit.commands.execute(
            "MovePrim", path_from=out[0], path_to=mat_path
        )
        shader = stage.GetPrimAtPath(mat_path + "/Shader")
        if shader and shader.IsValid():
            shader.CreateAttribute(
                "inputs:diffuse_color_constant",
                Sdf.ValueTypeNames.Color3f,
            ).Set(Gf.Vec3f(*diffuse_rgb))
            shader.CreateAttribute(
                "inputs:reflection_roughness_constant",
                Sdf.ValueTypeNames.Float,
            ).Set(roughness)
            shader.CreateAttribute(
                "inputs:metallic_constant",
                Sdf.ValueTypeNames.Float,
            ).Set(0.0)
        return UsdShade.Material(stage.GetPrimAtPath(mat_path))

    def _apply_robot_plastic_material(self):
        """Override visuals of SO-ARM101 with material-targeted matte plastics.

        Per OpenUSD spec_usdpreviewsurface.html best practices (verified docs):
          * 3D-printed white plastic (arm structural):  diffuse ~(0.92,0.92,0.92), roughness 0.7, metallic 0
          * Light-grey matte plastic (wrist mount):     diffuse ~(0.55,0.55,0.55), roughness 0.7
          * Matte black plastic (Feetech STS-3215 servos): diffuse (0.04,0.04,0.04), roughness 0.75, metallic 0

        Each <link>/visuals Xform is a USD instance from the URDF importer;
        authoring on instance-proxy children is forbidden, so we de-instance
        per link first (essentially free with one robot per scene).
        """
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/SO_ARM101").IsValid():
            return

        looks_path = "/World/SO_ARM101/Looks"
        white = self._create_omnipbr_matte(
            f"{looks_path}/plastic_white_matte",
            diffuse_rgb=(0.92, 0.92, 0.92),
            roughness=0.7,  # 0.6-0.8 doc range, mid-low for less-aggressive matte
        )
        grey = self._create_omnipbr_matte(
            f"{looks_path}/plastic_grey_light_matte",
            diffuse_rgb=(0.55, 0.55, 0.55),
            roughness=0.7,
        )
        black = self._create_omnipbr_matte(
            f"{looks_path}/plastic_black_matte",
            diffuse_rgb=(0.04, 0.04, 0.04),
            roughness=0.75,  # 0.7-0.85 doc range for matte black plastic
        )
        if not (white and black):
            return

        # IMPORTANT: URDF importer authors direct material:binding on each Mesh child
        # (e.g. .../sts3215_xx/mesh → Looks/DefaultMaterial). USD picks the most-specific
        # binding, so a binding on the parent Xform is silently overridden by the mesh's
        # own. Bind at BOTH levels: the Xform (for any future-added child geometry) AND
        # every Mesh descendant (to override the URDF default).
        white_bound = 0
        black_bound = 0
        for link in stage.GetPrimAtPath("/World/SO_ARM101").GetChildren():
            visuals = stage.GetPrimAtPath(f"{link.GetPath()}/visuals")
            if not visuals or not visuals.IsValid():
                continue
            if visuals.IsInstanceable():
                visuals.SetInstanceable(False)
            for p in visuals.GetChildren():
                target = black if p.GetName().startswith("sts3215_") else white
                # Bind on the part Xform + every Mesh descendant
                for d in Usd.PrimRange(p):
                    if d == p or d.IsA(UsdGeom.Mesh):
                        UsdShade.MaterialBindingAPI.Apply(d)
                        UsdShade.MaterialBindingAPI(d).Bind(target)
                if p.GetName().startswith("sts3215_"):
                    black_bound += 1
                else:
                    white_bound += 1

        # Wrist camera mount: distinct light-grey so it pops against the
        # white arm in camera frames. USB camera body next to it stays
        # at its imported default (it's electronics, not a 3D print).
        # Same Mesh-level rebind as above (URDF default binding override).
        cam_mount = stage.GetPrimAtPath(
            "/World/SO_ARM101/camera_mount_link/camera_wrist_mount")
        grey_bound = 0
        if cam_mount and cam_mount.IsValid() and grey:
            for d in Usd.PrimRange(cam_mount):
                if d == cam_mount or d.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI.Apply(d)
                    UsdShade.MaterialBindingAPI(d).Bind(grey)
            grey_bound = 1

        print(
            f"[plastic] Bound matte-white to {white_bound} parts, "
            f"matte-black to {black_bound} servos, "
            f"matte-grey to {grey_bound} mount(s)"
        )

    def _get_blocks_by_color(self, color=None, folder_path="/World/Objects"):
        """Get block names filtered by color. If color is None, return all blocks."""
        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath(folder_path)
        if not objects_root or not objects_root.IsValid():
            return []
        blocks = []
        for child in objects_root.GetChildren():
            name = child.GetName()
            if name == "PhysicsMaterial":
                continue
            if color is None or self._get_block_color(name) == color:
                blocks.append(name)
        return blocks

    @staticmethod
    def _default_block_xy_ranges(x_range=None, y_range=None):
        """Fill a missing world-frame X/Y sampling range from the module-level block workspace
        (BLOCK_RANDOM_FORWARD / BLOCK_RANDOM_LATERAL, mapped through ROBOT_FORWARD_AXIS)."""
        if x_range is None or y_range is None:
            fwd_range = BLOCK_RANDOM_FORWARD
            lat_range = BLOCK_RANDOM_LATERAL
            if ROBOT_FORWARD_AXIS == "X":
                x_range = x_range or fwd_range
                y_range = y_range or lat_range
            else:
                x_range = x_range or (lat_range[0], lat_range[1])  # lateral maps to X when forward=Y
                y_range = y_range or fwd_range
        return x_range, y_range

    def _sample_non_overlapping_objects(
        self,
        num_objects,
        x_range=None,
        y_range=None,
        min_sep=0.05,
        yaw_range=(-180.0, 180.0),
        z_values=None,
        fixed_positions=None,
        max_attempts=10_000,
        max_retries=100,
        radii=None,
        exclusion_zones=None,
        must_be_visible_indices=None,
        camera_fov_margin_px=30,
    ):
        """
        Sample non-overlapping object poses in world frame.

        Places largest objects first (when radii provided) then shuffles
        on retries to explore different configurations.

        Args:
            num_objects: Number of objects to place.
            x_range: X position range in world frame.
            y_range: Y position range in world frame.
            min_sep: Uniform minimum separation (used when radii is None).
            yaw_range: Yaw rotation range in degrees.
            z_values: List of Z values for each object (preserves current Z).
            fixed_positions: List of fixed positions to avoid.
            max_attempts: Maximum placement attempts per retry.
            max_retries: Maximum number of full restarts.
            radii: Per-object half-extents (half diagonal). When provided,
                   the required separation between objects i and j is
                   radii[i] + radii[j] + gap instead of a flat min_sep.
            must_be_visible_indices: Optional list of orig_idx values whose
                   sampled positions must project into the workspace camera's
                   image (with `camera_fov_margin_px` of headroom from the
                   frame edges). Used by VLA-style recordings where the
                   target object has to be visible in the observation feed
                   for the policy to learn — an out-of-view lego is dataset
                   noise. Other indices are unconstrained.
            camera_fov_margin_px: minimum pixel distance from any image edge
                   that a must-be-visible point must satisfy.
        """
        x_range, y_range = self._default_block_xy_ranges(x_range, y_range)

        gap = self._object_gap

        # Convert fixed positions to numpy arrays for distance checking
        fixed_xy = []
        if fixed_positions:
            for pos in fixed_positions:
                if isinstance(pos, (list, tuple, np.ndarray)):
                    fixed_xy.append(np.array(pos[:2]))
                elif hasattr(pos, '__getitem__'):
                    fixed_xy.append(np.array([pos[0], pos[1]]))

        # Build placement order: largest first on attempt 0, shuffle on retries
        indices = list(range(num_objects))
        if radii is not None:
            indices.sort(key=lambda i: radii[i], reverse=True)

        for retry in range(max_retries):
            if retry > 0:
                np.random.shuffle(indices)

            # placed[order_pos] = pose for the order_pos-th object placed
            placed = []
            attempts = 0

            while len(placed) < num_objects and attempts < max_attempts:
                attempts += 1
                candidate_xy = np.array([
                    np.random.uniform(*x_range),
                    np.random.uniform(*y_range)
                ])
                orig_idx = indices[len(placed)]
                valid = True

                # Check against already-placed objects
                for order_j, p in enumerate(placed):
                    orig_j = indices[order_j]
                    if radii is not None:
                        req = radii[orig_idx] + radii[orig_j] + gap
                    else:
                        req = min_sep
                    if np.linalg.norm(candidate_xy - p["position"][:2]) < req:
                        valid = False
                        break

                # Check against fixed positions
                if valid and fixed_xy:
                    req_fixed = (radii[orig_idx] + min_sep / 2) if radii is not None else min_sep
                    for fixed in fixed_xy:
                        if np.linalg.norm(candidate_xy - fixed) < req_fixed:
                            valid = False
                            break

                # Check against AABB exclusion zones (e.g. cup footprints)
                if valid and exclusion_zones:
                    obj_r = radii[orig_idx] if radii is not None else min_sep / 2
                    margin = obj_r + gap
                    for cx, cy, hw, hh in exclusion_zones:
                        if (abs(candidate_xy[0] - cx) < hw + margin and
                            abs(candidate_xy[1] - cy) < hh + margin):
                            valid = False
                            break

                # Pre-compute z so the FOV check below has a 3D point to project.
                candidate_z = (z_values[orig_idx]
                               if z_values and orig_idx < len(z_values)
                               else self._ground_plane_z)

                # Camera-FOV check for the visibility-constrained subset.
                if valid and must_be_visible_indices and orig_idx in must_be_visible_indices:
                    if not self._is_in_camera_view(
                            (candidate_xy[0], candidate_xy[1], candidate_z),
                            margin_px=camera_fov_margin_px):
                        valid = False

                if valid:
                    yaw_deg = np.random.uniform(*yaw_range)
                    placed.append({
                        "position": np.array([candidate_xy[0], candidate_xy[1], candidate_z]),
                        "yaw_deg": yaw_deg
                    })

            if len(placed) == num_objects:
                if retry > 0:
                    print(f"Placement succeeded after {retry + 1} retries")
                # Reorder results back to original indices
                result_poses = [None] * num_objects
                for order_pos, pose in enumerate(placed):
                    result_poses[indices[order_pos]] = pose
                return result_poses

        # Density-tolerant fallback: after max_retries exhausted, return
        # whatever was placed plus stacked extras just above the placed
        # region. The grasp pipeline is robust to close-packed blocks (it
        # removes the target lego from the world scene at grasp_move start),
        # so this is safer than raising — the caller always gets num_objects
        # back.
        print(f"[_sample_non_overlapping_objects] density-tolerant fallback: "
              f"placed {len(placed)}/{num_objects} before exhausting "
              f"{max_retries} retries; stacking the remaining {num_objects - len(placed)} "
              f"offstage")
        offstage_y = (y_range[1] + 0.05) if y_range else 0.45
        offstage_z_base = (z_values[0] if z_values else self._ground_plane_z) + 0.05
        while len(placed) < num_objects:
            orig_idx = indices[len(placed)]
            # Stack extras in a tight column offstage at y > y_range[1]
            placed.append({
                "position": np.array([
                    (x_range[0] + x_range[1]) / 2.0 if x_range else 0.18,
                    offstage_y,
                    offstage_z_base + 0.02 * (len(placed) - len(placed)),
                ]),
                "yaw_deg": 0.0,
            })
        # Reorder results back to original indices
        result_poses = [None] * num_objects
        for order_pos, pose in enumerate(placed):
            result_poses[indices[order_pos]] = pose
        return result_poses

    def _set_obj_prim_pose(self, prim_path, position, rotation_xyz_deg=None):
        """Set object position and orientation on its body prim.

        Precision-aware: introspects the existing xformOp:orient and
        xformOp:translate attribute types and builds matching Gf values.
        This lets the helper serve both legos (Quatf/Vec3d created by the
        block-spawn path) and cups (Quatd/Vec3d created with
        ``AddOrientOp(PrecisionDouble)`` at add_cups:3029). Passing a
        Quatf to a Quatd-typed property raises
        ``<pxr.Tf.ErrorException> Type mismatch`` and the ChangeProperty
        command fails silently inside Kit's undo stack.

        Args:
            prim_path: USD prim path (body prim with RigidBodyAPI).
            position: Gf.Vec3d/Gf.Vec3f or (x, y, z) tuple.
            rotation_xyz_deg: (rx, ry, rz) euler angles in degrees, or None for identity.
        """
        import omni.kit.commands
        import math

        # Convert euler XYZ degrees to quaternion (w, x, y, z) in f64 math
        rot = rotation_xyz_deg if rotation_xyz_deg is not None else (0.0, 0.0, 0.0)
        rx, ry, rz = [math.radians(a) for a in rot]
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz

        # Introspect existing attribute types (fall back to Gf.Vec3d / Gf.Quatf
        # if attributes don't exist yet — matches prior lego behavior).
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        orient_attr = prim.GetAttribute("xformOp:orient") if prim else None
        translate_attr = prim.GetAttribute("xformOp:translate") if prim else None

        # Orient: match precision of existing attribute.
        if orient_attr and orient_attr.IsValid():
            type_name = str(orient_attr.GetTypeName()).lower()
            if "quatd" in type_name:
                orient_value = Gf.Quatd(qw, qx, qy, qz)
            else:
                orient_value = Gf.Quatf(qw, qx, qy, qz)
        else:
            orient_value = Gf.Quatf(qw, qx, qy, qz)

        # Position: convert to matching vector precision if possible.
        if isinstance(position, (Gf.Vec3d, Gf.Vec3f)):
            pos_tuple = (position[0], position[1], position[2])
        else:
            pos_tuple = tuple(position)
        if translate_attr and translate_attr.IsValid():
            type_name = str(translate_attr.GetTypeName()).lower()
            if "vector3f" in type_name or "float3" in type_name:
                pos_value = Gf.Vec3f(*pos_tuple)
            else:
                pos_value = Gf.Vec3d(*pos_tuple)
        else:
            pos_value = Gf.Vec3d(*pos_tuple)

        if translate_attr and translate_attr.IsValid():
            translate_attr.Set(pos_value)
        if orient_attr and orient_attr.IsValid():
            orient_attr.Set(orient_value)

    def sort_objects(self, color=None, folder_path="/World/Objects"):
        """Sort blocks by placing same-color blocks in tight clusters at X=0.3, spread along Y.

        Clusters are laid out along Y with bbox-aware spacing so they never overlap.
        Within each cluster, blocks are placed in a row along X centered at 0.3.

        Args:
            color: Optional color filter ('red', 'green', 'blue'). If None, sorts all.
            folder_path: Parent prim path for objects.
        Returns:
            (sorted_count, 0)
        """
        stage = omni.usd.get_context().get_stage()

        colors_to_sort = [color] if color else list(BLOCK_COLORS.keys())
        sorted_count = 0

        # Build cluster info: for each color, collect blocks and measure extents
        clusters = []  # [(color, [(name, prim_path, sx, sy)], cluster_x_span, max_sy)]
        for clr in colors_to_sort:
            blocks = self._get_blocks_by_color(clr, folder_path)
            if not blocks:
                continue
            block_info = []
            for name in sorted(blocks):
                prim_path = self._get_prim_path(name, folder_path)
                sx, sy, _ = self._get_prim_bbox_size(prim_path)
                block_info.append((name, prim_path, sx, sy))
            # Cluster X span = sum of block widths + gaps between them
            cluster_x_span = sum(b[2] for b in block_info) + self._object_gap * (len(block_info) - 1)
            # Cluster Y extent = max Y extent of any block in this cluster
            max_sy = max(b[3] for b in block_info)
            clusters.append((clr, block_info, cluster_x_span, max_sy))

        if not clusters:
            print(f"[sort] No blocks found for color={color or 'all'}")
            return 0, 0

        # Layout clusters along lateral axis, centered at 0
        cluster_gap = self._object_gap * 3  # wider gap between color groups
        total_lat_span = sum(c[3] for c in clusters) + cluster_gap * (len(clusters) - 1)
        lat_cursor = -total_lat_span / 2.0

        for clr, block_info, cluster_fwd_span, max_lat in clusters:
            cluster_lat = lat_cursor + max_lat / 2.0

            # Place blocks in a row along forward, centered at BLOCK_SPAWN_FORWARD
            fwd_cursor = BLOCK_SPAWN_FORWARD - cluster_fwd_span / 2.0
            for name, prim_path, sx, sy in block_info:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                fwd_position = fwd_cursor + sx / 2.0
                z = self._get_ground_z(prim_path)
                wx, wy = _to_world(fwd_position, cluster_lat)
                pos = Gf.Vec3d(wx, wy, z)
                self._set_obj_prim_pose(prim_path, pos)
                fwd_cursor += sx + self._object_gap
                sorted_count += 1

            lat_cursor += max_lat + cluster_gap

        print(f"[sort] Sorted {sorted_count} blocks (color={color or 'all'})")
        return sorted_count, 0

    # ==================== Color Pickers ====================

    def _on_randomize_clicked(self):
        """UI Randomize button handler. Reads the visibility-constraint widgets
        (Lock color in view + Min visible) and forwards to randomize_object_poses
        with the corresponding params."""
        # ComboBox index → color string. 0 = (none), 1 = red, 2 = green, 3 = blue.
        idx = 0
        try:
            model = self._visible_color_combo.model
            item = model.get_item_value_model(model.get_item_children()[0])
            idx = int(item.as_int)
        except Exception:
            idx = 0
        required_visible_color = (None, "red", "green", "blue")[idx] if 0 <= idx <= 3 else None
        try:
            count = max(1, int(self._visible_count_model.as_int))
        except Exception:
            count = 1
        self.randomize_object_poses(
            required_visible_color=required_visible_color,
            required_visible_count=count,
        )

    def _apply_color_pickers(self):
        """Read colors from UI pickers and defer actual application to next frame.

        Deferring ensures the ColorWidget model values are fully committed
        before we read them (fixes the double-click issue).
        """
        async def _deferred_apply():
            await omni.kit.app.get_app().next_update_async()
            for cname, cw in self._color_pickers.items():
                model = cw.model
                items = model.get_item_children()
                # ColorWidget values are in sRGB — convert to linear for USD materials
                r = _srgb_to_linear(model.get_item_value_model(items[0]).as_float)
                g = _srgb_to_linear(model.get_item_value_model(items[1]).as_float)
                b = _srgb_to_linear(model.get_item_value_model(items[2]).as_float)
                OBJECT_COLORS[cname] = (r, g, b)

            # Apply to all legos in scene
            stage = omni.usd.get_context().get_stage()
            objects_prim = stage.GetPrimAtPath("/World/Objects")
            if objects_prim and objects_prim.IsValid():
                for child in objects_prim.GetChildren():
                    name = child.GetName()
                    for cname in OBJECT_COLORS:
                        if name.startswith(cname):
                            self._set_block_material_color(str(child.GetPath()), OBJECT_COLORS[cname])
                            break

            # Apply to all cups in scene
            containers_prim = stage.GetPrimAtPath("/World/Containers")
            if containers_prim and containers_prim.IsValid():
                for child in containers_prim.GetChildren():
                    cname = child.GetName().replace("cup_", "")
                    if cname in OBJECT_COLORS:
                        self._set_block_material_color(str(child.GetPath()), OBJECT_COLORS[cname])

            print(f"[colors] Applied: { {k: tuple(round(c,3) for c in v) for k,v in OBJECT_COLORS.items()} }")

        asyncio.ensure_future(_deferred_apply())

    # ==================== Containers (Cups) ====================

    def _on_cup_layout_changed(self, *args):
        """Update CUP_LAYOUT from UI sliders/combo and refresh labels."""
        if getattr(self, '_cup_updating', False):
            return
        mode_idx = self._cup_mode_combo.model.get_item_value_model().as_int
        CUP_LAYOUT["mode"] = "line" if mode_idx == 0 else "arc"
        CUP_LAYOUT["angle_deg"] = self._cup_angle_slider.model.as_float
        CUP_LAYOUT["radius"] = self._cup_radius_slider.model.as_float
        CUP_LAYOUT["gap"] = self._cup_gap_slider.model.as_float
        CUP_LAYOUT["face_origin"] = self._cup_face_origin_cb.model.get_value_as_bool()
        # Cluster offsets only present once the UI panel is built — guard
        # so the function survives early calls from other layout edits.
        if hasattr(self, "_cup_offset_x_slider"):
            CUP_LAYOUT["cluster_offset_x"] = self._cup_offset_x_slider.model.as_float
            CUP_LAYOUT["cluster_offset_y"] = self._cup_offset_y_slider.model.as_float

        self._cup_angle_label.text = f"{CUP_LAYOUT['angle_deg']:.0f}°"
        self._cup_radius_label.text = f"{CUP_LAYOUT['radius']:.2f}m"
        self._cup_gap_label.text = f"{CUP_LAYOUT['gap']*100:.0f}cm"
        if hasattr(self, "_cup_offset_x_label"):
            self._cup_offset_x_label.text = (
                f"{CUP_LAYOUT.get('cluster_offset_x', 0.0)*100:.0f}cm")
            self._cup_offset_y_label.text = (
                f"{CUP_LAYOUT.get('cluster_offset_y', 0.0)*100:.0f}cm")

    def _reset_cup_defaults(self):
        """Reset cup layout sliders to defaults (deferred to next frame)."""
        async def _do_reset():
            await omni.kit.app.get_app().next_update_async()
            self._cup_updating = True
            CUP_LAYOUT.update(CUP_LAYOUT_DEFAULTS)
            self._cup_mode_combo.model.get_item_value_model().set_value(0 if CUP_LAYOUT["mode"] == "line" else 1)
            self._cup_angle_slider.model.set_value(float(CUP_LAYOUT["angle_deg"]))
            self._cup_radius_slider.model.set_value(CUP_LAYOUT["radius"])
            self._cup_gap_slider.model.set_value(CUP_LAYOUT["gap"])
            self._cup_angle_label.text = f"{CUP_LAYOUT['angle_deg']:.0f}°"
            self._cup_radius_label.text = f"{CUP_LAYOUT['radius']:.2f}m"
            self._cup_gap_label.text = f"{CUP_LAYOUT['gap']*100:.0f}cm"
            self._cup_face_origin_cb.model.set_value(CUP_LAYOUT.get("face_origin", True))
            if hasattr(self, "_cup_offset_x_slider"):
                self._cup_offset_x_slider.model.set_value(
                    float(CUP_LAYOUT.get("cluster_offset_x", 0.0)))
                self._cup_offset_y_slider.model.set_value(
                    float(CUP_LAYOUT.get("cluster_offset_y", 0.0)))
                self._cup_offset_x_label.text = (
                    f"{CUP_LAYOUT.get('cluster_offset_x', 0.0)*100:.0f}cm")
                self._cup_offset_y_label.text = (
                    f"{CUP_LAYOUT.get('cluster_offset_y', 0.0)*100:.0f}cm")
            self._cup_updating = False
            print("[cups] Reset to defaults")
        asyncio.ensure_future(_do_reset())

    def _cmd_match_real_world(self, reposition=True):
        """Apply CUP_LAYOUT_REAL_WORLD overrides to CUP_LAYOUT and the
        Containers UI sliders.

        Only the keys present in CUP_LAYOUT_REAL_WORLD are touched
        (currently cluster_offset_x, cluster_offset_y, gap) — the rest
        of CUP_LAYOUT (mode, radius, angle_deg, color_order, etc.) is
        preserved. Auto-called from quick_start, also bound to the
        "Match Real World" button.

        reposition: if True and cups already exist, re-place them via
        _add_cups_from_ui so the real-world values take immediate
        effect. Quick_start passes False because add_cups will run
        AFTER this and pick up the live CUP_LAYOUT itself.
        """
        for key, value in CUP_LAYOUT_REAL_WORLD.items():
            CUP_LAYOUT[key] = value
        # Mirror to UI sliders (guard for early calls before UI exists).
        self._cup_updating = True
        try:
            if hasattr(self, "_cup_offset_x_slider"):
                self._cup_offset_x_slider.model.set_value(
                    float(CUP_LAYOUT["cluster_offset_x"]))
                self._cup_offset_y_slider.model.set_value(
                    float(CUP_LAYOUT["cluster_offset_y"]))
                self._cup_offset_x_label.text = (
                    f"{CUP_LAYOUT['cluster_offset_x']*100:.0f}cm")
                self._cup_offset_y_label.text = (
                    f"{CUP_LAYOUT['cluster_offset_y']*100:.0f}cm")
            if hasattr(self, "_cup_gap_slider"):
                self._cup_gap_slider.model.set_value(
                    float(CUP_LAYOUT["gap"]))
                self._cup_gap_label.text = (
                    f"{CUP_LAYOUT['gap']*100:.0f}cm")
        finally:
            self._cup_updating = False
        print(
            f"[cups] Matched real world: "
            f"offset=({CUP_LAYOUT['cluster_offset_x']:.3f}, "
            f"{CUP_LAYOUT['cluster_offset_y']:.3f}) m, "
            f"gap={CUP_LAYOUT['gap']:.3f} m"
        )
        repositioned = False
        if reposition:
            # If cups already exist, re-snap them via the standard path.
            stage = omni.usd.get_context().get_stage()
            if stage and stage.GetPrimAtPath("/World/Containers/cup_red").IsValid():
                self._add_cups_from_ui()
                repositioned = True
        return {
            "status": "success",
            "message": (
                f"CUP_LAYOUT matched real-world: "
                f"offset=({CUP_LAYOUT['cluster_offset_x']:.3f},"
                f"{CUP_LAYOUT['cluster_offset_y']:.3f})m, "
                f"gap={CUP_LAYOUT['gap']:.3f}m, "
                f"repositioned={repositioned}"
            ),
        }

    def _add_cups_from_ui(self):
        """Reposition existing cups with current UI settings.

        Phase 8: action-graph-preserving path. Cups are teleported in place
        via RigidPrim.set_world_pose() — the /Graph/ActionGraph_drop_poses
        publisher keeps running throughout and /drop_poses never blips.

        If any cup prim is missing (first-time creation, or user hit Delete
        earlier), falls back to the heavyweight delete+add+republish path.
        """
        self._on_cup_layout_changed()  # sync UI sliders → CUP_LAYOUT

        stage = omni.usd.get_context().get_stage()
        container_path = "/World/Containers"
        colors = CUP_LAYOUT["color_order"]

        # First-time / recovery path: any cup prim missing → rebuild.
        missing = [c for c in colors
                   if not stage.GetPrimAtPath(
                       f"{container_path}/cup_{c}").IsValid()]
        if missing:
            print(f"[update_cups] {missing} missing — rebuilding via delete+add")
            self.delete_cups()
            self.add_cups()
            self._cmd_publish_drop_poses()
            return

        # Compute target positions using the same logic as add_cups.
        mode = CUP_LAYOUT["mode"]
        if mode == "line":
            # Rotation-invariant width: measure the Mesh descendant, not the
            # cup root. Bounding the root leaks the root's authored
            # xformOp:orient into the AABB (verified empirically: a 74° yaw
            # inflates the root's X/Y extent from 0.078 → 0.120). The mesh
            # child has no xformOps of its own, so its untransformed bound
            # is the raw vertex extent — truly orientation-invariant and
            # independent of prior arc/line state. Same canonical pattern
            # _on_physics_step_bbox already uses for the lego bbox topic.
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            cup_widths = {}
            for c in colors:
                cup_root = stage.GetPrimAtPath(f"{container_path}/cup_{c}")
                mesh_prim = None
                for p in Usd.PrimRange(cup_root):
                    if p.GetTypeName() == "Mesh" and "aruco" not in p.GetName():
                        mesh_prim = p
                        break
                if mesh_prim is None:
                    cup_widths[c] = 0.08  # fallback — matches _cup_positions_line default
                    continue
                r = bbox_cache.ComputeUntransformedBound(
                    mesh_prim).ComputeAlignedRange()
                mn, mx = r.GetMin(), r.GetMax()
                cup_widths[c] = (mx[1] - mn[1] if ROBOT_FORWARD_AXIS == "X"
                                 else mx[0] - mn[0])
            positions = _cup_positions_line(cup_widths)
        else:
            positions = _cup_positions_arc()
        positions = _apply_cup_cluster_offset(positions)

        pan_xy = _get_pan_axis_xy()
        ground_z = getattr(self, "_ground_plane_z", 0.0)
        n = 0
        for c in colors:
            cup_path = f"{container_path}/cup_{c}"
            fwd, lat = positions.get(c, (0, 0))
            x, y = _to_world(fwd, lat, anchor=pan_xy)
            # ArUco-preserving orient (same math as add_cups).
            dx_pan, dy_pan = x - pan_xy[0], y - pan_xy[1]
            if (CUP_LAYOUT.get("face_origin", False)
                    and (abs(dx_pan) > 1e-4 or abs(dy_pan) > 1e-4)):
                if mode == "arc":
                    yaw_rad = math.atan2(-dy_pan, -dx_pan)
                else:
                    yaw_rad = math.radians(CUP_LAYOUT["angle_deg"]) + math.pi
                quat_wxyz = (math.cos(yaw_rad / 2), 0.0, 0.0,
                             math.sin(yaw_rad / 2))
            else:
                quat_wxyz = (1.0, 0.0, 0.0, 0.0)

            # Teleport via the same Kit-command path lego randomize uses
            # (_set_obj_prim_pose) so ChangeProperty propagates through Kit's
            # event system — PhysX, RTX/Hydra, and USD listeners all receive
            # authoritative teleport notifications regardless of simulation
            # state. Replaces the prior RigidPrim-with-direct-.Set()-fallback
            # pattern, whose fallback branch was silently overridden by PhysX
            # when the primary branch failed at runtime.
            #
            # quat_wxyz is yaw-only (z-axis rotation), so extract yaw by
            # qw + qz components: yaw = 2·atan2(qz, qw). Works for both
            # face_origin=True (yaw from pan direction) and identity quat
            # (yaw=0).
            qw, _qx, _qy, qz = quat_wxyz
            yaw_deg = math.degrees(2.0 * math.atan2(qz, qw))
            self._set_obj_prim_pose(
                cup_path,
                position=Gf.Vec3d(x, y, ground_z),
                rotation_xyz_deg=(0.0, 0.0, yaw_deg),
            )
            n += 1
        print(f"[update_cups] Teleported {n} cups in place "
              f"(action graph preserved)")

    def randomize_cups(self, container_path="/World/Containers",
                        radius_min=0.22, radius_max=0.30,
                        angle_min_deg=-50.0, angle_max_deg=50.0,
                        gap_min=0.005, gap_max=0.030,
                        modes=("arc", "line"),
                        randomize_order=True,
                        yaw_jitter_deg=15.0,
                        objects_path="/World/Objects",
                        cup_lego_clearance=0.04,
                        robot_path="/World/SO_ARM101",
                        cup_robot_clearance=0.015,
                        max_attempts=200, seed=None):
        """Randomize cup layout via the existing arc/line layout functions.

        Rather than sampling per-cup positions independently (which often
        violates spacing or robot-corridor constraints and converges
        slowly), this samples the LAYOUT PARAMETERS that
        `_cup_positions_arc` and `_cup_positions_line` already use:
          - mode: "arc" or "line"
          - radius: distance from pan axis to layout center
          - angle_deg: heading of the layout's center axis (0 = forward)
          - gap: spacing parameter (meters along the arc/line)

        With `randomize_order=True` (default), the cup-color → slot
        mapping is also shuffled each sample, so red/green/blue can
        end up in any of the 6 possible orderings along the arc/line.

        Each random sample produces a coherent 3-cup layout with
        proper spacing built in, then is validated against:
          - Lego footprints (mirrors `randomize_object_poses` exclusion)
          - Robot-link AABBs at current pose (queried live, so the
            exclusion adapts — call after grasp_home for the home-pose
            corridor; only links with z-min ≤ 0.15 m count, since
            higher links can't intersect a table-level cup)
        On violation, retry with fresh params. Convergence is fast
        because every sample is a complete valid layout, not 3
        independently-sampled points that have to satisfy mutual
        exclusion plus all obstacle clearances.

        Yaw is anchored on the face-origin direction (same math as
        `_add_cups_from_ui` lines ~2996-3007). For "arc" mode each cup
        independently faces the pan axis (yaw = atan2(-dy, -dx)); for
        "line" mode all cups share angle_deg + π. A `yaw_jitter_deg`
        window adds per-cup variation while keeping ArUco markers in
        the camera-side cone.

        Poses applied through `_set_obj_prim_pose` (same teleport path
        as `update_cups`). After teleporting, calls
        `_cmd_publish_drop_poses` to refresh action graph wrappers —
        otherwise /drop_poses keeps broadcasting stale cached transforms.

        Defaults:
          radius ∈ [0.22, 0.30] m  — inside grasp_workspace r_max=0.31.
          angle_deg ∈ [-50°, +50°] — heading of layout center.
          gap ∈ [0.005, 0.030] m   — tight to wide cup spacing.
          modes = ("arc", "line")  — both layout topologies sampled.

        Args:
            container_path: USD parent path containing cup_red/green/blue.
            radius_min, radius_max: layout radius bounds (m).
            angle_min_deg, angle_max_deg: layout heading bounds (deg).
            gap_min, gap_max: cup-to-cup gap bounds (m).
            modes: tuple of layout modes to sample from. Use a single-
                element tuple to force one topology (e.g. ("arc",)).
            randomize_order: if True, shuffle CUP_LAYOUT["color_order"]
                each sample so cup-color → slot assignment varies
                across runs (6 permutations for 3 cups). False keeps
                the canonical [red, green, blue] order.
            yaw_jitter_deg: half-width of yaw jitter around the face-
                origin base direction. 0 = always face origin exactly.
            objects_path: USD parent path of lego blocks.
            cup_lego_clearance: extra margin (m) for cup-lego check.
            robot_path: USD root of the robot articulation. Set to ""
                or None to skip robot-collision checks.
            cup_robot_clearance: extra margin (m) for cup-robot check.
            max_attempts: how many full-layout samples to try before
                giving up.
            seed: RNG seed for reproducibility (None = nondeterministic).

        Returns:
            {"status": "success" | "error", "message": ...,
             "params": {"mode", "radius", "angle_deg", "gap"},
             "placements": [{"color", "fwd", "lat", "yaw_deg",
                             "base_yaw_deg"}, ...]}.
        """
        import random as _rnd
        rng = _rnd.Random(seed) if seed is not None else _rnd

        stage = omni.usd.get_context().get_stage()
        colors = CUP_LAYOUT["color_order"]
        pan_xy = _get_pan_axis_xy()
        ground_z = getattr(self, "_ground_plane_z", 0.0)

        # First-time / recovery: any cup missing → build defaults first.
        missing = [c for c in colors
                   if not stage.GetPrimAtPath(
                       f"{container_path}/cup_{c}").IsValid()]
        if missing:
            print(f"[randomize_cups] {missing} missing — adding default "
                  "cups first via delete+add")
            self.delete_cups()
            self.add_cups()
            self._cmd_publish_drop_poses()

        # Cup opening radius — used for cup ↔ obstacle exclusion check.
        cup_info = self._get_cup_positions(container_path)
        cup_radius = max(
            (info["opening_radius"] for info in cup_info.values()),
            default=0.05)

        # Helper: world (x, y) → robot-frame (fwd, lat).
        def _world_to_robot(wx, wy):
            if ROBOT_FORWARD_AXIS == "X":
                return wx - pan_xy[0], wy - pan_xy[1]
            return wy - pan_xy[1], -(wx - pan_xy[0])

        # Lego-exclusion list in robot-relative (fwd, lat) frame.
        lego_exclusions = []
        objects_root = stage.GetPrimAtPath(objects_path)
        if objects_root and objects_root.IsValid():
            for child in objects_root.GetChildren():
                if child.GetName() == "PhysicsMaterial":
                    continue
                try:
                    xform = UsdGeom.Xformable(child)
                    tf = xform.ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
                    pos = tf.ExtractTranslation()
                    fwd_lego, lat_lego = _world_to_robot(
                        float(pos[0]), float(pos[1]))
                    sx, sy, _ = self._get_prim_bbox_size(
                        str(child.GetPath()))
                    lego_radius = math.hypot(sx, sy) / 2.0
                    lego_exclusions.append((fwd_lego, lat_lego, lego_radius))
                except Exception:
                    pass

        # Robot-link exclusions — direct children of robot root, with
        # z-min filter so only table-level links count. Skip prims with
        # ±inf bounds (frames/joints with no geometry).
        FINITE_BBOX_LIMIT = 1.0e10
        CUP_INTERSECT_Z_MAX = 0.15  # m
        robot_exclusions = []
        if robot_path:
            robot_root = stage.GetPrimAtPath(robot_path)
            if robot_root and robot_root.IsValid():
                bbox_cache_robot = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                for child in robot_root.GetChildren():
                    if not child.IsValid():
                        continue
                    try:
                        wb = bbox_cache_robot.ComputeWorldBound(child)
                        rng_aabb = wb.ComputeAlignedRange()
                        mn, mx = rng_aabb.GetMin(), rng_aabb.GetMax()
                        if (abs(mn[0]) > FINITE_BBOX_LIMIT
                                or abs(mx[0]) > FINITE_BBOX_LIMIT):
                            continue
                        if mn[2] > CUP_INTERSECT_Z_MAX:
                            continue
                        cx_w = (mn[0] + mx[0]) * 0.5
                        cy_w = (mn[1] + mx[1]) * 0.5
                        half_x = (mx[0] - mn[0]) * 0.5
                        half_y = (mx[1] - mn[1]) * 0.5
                        rad = math.hypot(half_x, half_y)
                        fwd_link, lat_link = _world_to_robot(cx_w, cy_w)
                        robot_exclusions.append(
                            (fwd_link, lat_link, rad))
                    except Exception:
                        pass

        # Layout-validation helper: given a positions dict, check every
        # cup against legos + robot exclusions.
        def _layout_valid(positions):
            for fwd, lat in positions.values():
                # vs. legos
                for (lf, ll, lr) in lego_exclusions:
                    if (math.hypot(fwd - lf, lat - ll)
                            < cup_radius + lr + cup_lego_clearance):
                        return False
                # vs. robot links
                for (rf, rl, rr) in robot_exclusions:
                    if (math.hypot(fwd - rf, lat - rl)
                            < cup_radius + rr + cup_robot_clearance):
                        return False
            return True

        # Sample layout params + delegate position generation to the
        # existing _cup_positions_arc / _cup_positions_line functions.
        # Those read CUP_LAYOUT module-level state, so we temporarily
        # override it inside the loop and restore on exit.
        saved_layout = dict(CUP_LAYOUT)
        canonical_order = list(saved_layout.get("color_order", colors))
        chosen_params = None
        chosen_positions = None
        try:
            for attempt in range(max_attempts):
                mode = rng.choice(list(modes))
                radius = rng.uniform(radius_min, radius_max)
                angle_deg = rng.uniform(angle_min_deg, angle_max_deg)
                gap = rng.uniform(gap_min, gap_max)
                if randomize_order:
                    sample_order = list(canonical_order)
                    rng.shuffle(sample_order)
                else:
                    sample_order = list(canonical_order)

                CUP_LAYOUT["mode"] = mode
                CUP_LAYOUT["radius"] = radius
                CUP_LAYOUT["angle_deg"] = angle_deg
                CUP_LAYOUT["gap"] = gap
                CUP_LAYOUT["face_origin"] = True
                CUP_LAYOUT["color_order"] = sample_order

                if mode == "arc":
                    positions = _cup_positions_arc()
                else:
                    cup_widths = {c: 0.078 for c in sample_order}
                    positions = _cup_positions_line(cup_widths)
                # Apply cluster offset BEFORE validation so the
                # lego/robot-AABB checks see the actual placement.
                positions = _apply_cup_cluster_offset(positions)

                if _layout_valid(positions):
                    chosen_params = {
                        "mode": mode, "radius": radius,
                        "angle_deg": angle_deg, "gap": gap,
                        "color_order": sample_order,
                    }
                    chosen_positions = positions
                    break
        finally:
            # If we found a valid layout, KEEP CUP_LAYOUT updated to the
            # new params (so a subsequent "Update" click reproduces this
            # random layout). On failure, restore to the original.
            if chosen_params is None:
                CUP_LAYOUT.update(saved_layout)

        if chosen_params is None:
            msg = (f"could not find a valid cup layout after "
                   f"{max_attempts} samples (radius ∈ "
                   f"[{radius_min:.2f}, {radius_max:.2f}], "
                   f"angle ∈ [{angle_min_deg:.0f}°, {angle_max_deg:.0f}°], "
                   f"gap ∈ [{gap_min:.3f}, {gap_max:.3f}] m, "
                   f"modes={list(modes)}, "
                   f"{len(lego_exclusions)} legos + "
                   f"{len(robot_exclusions)} robot links to avoid). "
                   "Try widening param bounds, lowering clearances, "
                   "tightening lego placement zone first, or calling "
                   "from a different arm pose.")
            print(f"[randomize_cups] ERROR: {msg}")
            return {"status": "error", "message": msg}

        # Apply: teleport cups via _set_obj_prim_pose with face-origin
        # yaw + jitter (mirrors _add_cups_from_ui's apply loop, but with
        # the validated random layout instead of UI-slider values).
        mode = chosen_params["mode"]
        angle_deg = chosen_params["angle_deg"]
        results = []
        for color, (fwd, lat) in chosen_positions.items():
            x, y = _to_world(fwd, lat, anchor=pan_xy)
            dx_pan, dy_pan = x - pan_xy[0], y - pan_xy[1]
            if abs(dx_pan) > 1e-4 or abs(dy_pan) > 1e-4:
                if mode == "arc":
                    base_yaw_rad = math.atan2(-dy_pan, -dx_pan)
                else:  # "line": all cups share the layout heading
                    base_yaw_rad = math.radians(angle_deg) + math.pi
            else:
                base_yaw_rad = 0.0
            jitter_rad = math.radians(
                rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
            yaw_deg = math.degrees(base_yaw_rad + jitter_rad)
            cup_path = f"{container_path}/cup_{color}"
            self._set_obj_prim_pose(
                cup_path,
                position=Gf.Vec3d(x, y, ground_z),
                rotation_xyz_deg=(0.0, 0.0, yaw_deg),
            )
            results.append({"color": color, "fwd": fwd, "lat": lat,
                            "yaw_deg": yaw_deg,
                            "base_yaw_deg": math.degrees(base_yaw_rad)})

        # Refresh /drop_poses action graph wrappers.
        try:
            self._cmd_publish_drop_poses()
        except Exception as exc:
            print(f"[randomize_cups] WARN: publish_drop_poses refresh "
                  f"failed: {exc}")

        print(f"[randomize_cups] Randomized layout: mode={mode} "
              f"radius={chosen_params['radius']:.3f}m "
              f"angle={chosen_params['angle_deg']:+.1f}° "
              f"gap={chosen_params['gap']:.3f}m "
              f"order={chosen_params['color_order']} "
              f"(yaw_jitter ±{yaw_jitter_deg:.0f}° around face-origin, "
              f"avoiding {len(lego_exclusions)} legos + "
              f"{len(robot_exclusions)} robot links)")
        return {"status": "success",
                "message": (f"Randomized cups: mode={mode} "
                            f"radius={chosen_params['radius']:.3f} "
                            f"angle={chosen_params['angle_deg']:+.1f}°"),
                "params": chosen_params,
                "placements": results}

    def add_cups(self, container_path="/World/Containers"):
        """Add colored cups using CUP_LAYOUT config (arc or line mode).

        In "arc" mode each cup gets its own angle around the forward axis.
        In "line" mode the center cup sits at the configured angle/radius
        and others extend in a straight row along the lateral axis,
        spaced by their measured bounding box width + gap.
        """
        stage = omni.usd.get_context().get_stage()

        # Skip if containers already exist
        if stage.GetPrimAtPath(container_path).IsValid():
            existing = [c.GetName() for c in stage.GetPrimAtPath(container_path).GetChildren()
                        if c.GetName() != "PhysicsMaterial"]
            if len(existing) >= len(CUP_USDS):
                print(f"[add_cups] Cups already exist: {existing}")
                return False

        # Create container folder
        if not stage.GetPrimAtPath(container_path):
            UsdGeom.Xform.Define(stage, container_path)

        cups_folder = _get_cups_folder()
        mode = CUP_LAYOUT["mode"]
        colors = CUP_LAYOUT["color_order"]

        # --- First pass: create USD references (needed to measure bbox in line mode) ---
        for color_name in colors:
            usd_file = CUP_USDS.get(color_name)
            if not usd_file:
                continue
            prim_path = f"{container_path}/cup_{color_name}"
            if stage.GetPrimAtPath(prim_path).IsValid():
                continue
            prim = stage.DefinePrim(prim_path)
            prim.GetReferences().AddReference(cups_folder + usd_file)

        # --- Compute positions ---
        if mode == "line":
            # Measure lateral bbox width of each cup
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            cup_widths = {}
            for color_name in colors:
                prim_path = f"{container_path}/cup_{color_name}"
                cup_prim = stage.GetPrimAtPath(prim_path)
                if cup_prim.IsValid():
                    bbox = bbox_cache.ComputeWorldBound(cup_prim)
                    r = bbox.ComputeAlignedRange()
                    mn, mx = r.GetMin(), r.GetMax()
                    # Lateral width depends on forward axis
                    if ROBOT_FORWARD_AXIS == "X":
                        cup_widths[color_name] = mx[1] - mn[1]
                    else:
                        cup_widths[color_name] = mx[0] - mn[0]
            positions = _cup_positions_line(cup_widths)
        else:
            positions = _cup_positions_arc()
        positions = _apply_cup_cluster_offset(positions)

        # --- Second pass: apply transforms, collision, physics ---
        # Anchor cups around the shoulder pan axis, not world origin
        pan_xy = _get_pan_axis_xy()
        placed = 0
        for color_name in colors:
            prim_path = f"{container_path}/cup_{color_name}"
            cup_prim = stage.GetPrimAtPath(prim_path)
            if not cup_prim or not cup_prim.IsValid():
                continue

            fwd, lat = positions.get(color_name, (0, 0))
            x, y = _to_world(fwd, lat, anchor=pan_xy)
            z = self._ground_plane_z

            # Compute orientation: face_origin rotates cup to face the pan axis
            dx_pan, dy_pan = x - pan_xy[0], y - pan_xy[1]
            if CUP_LAYOUT.get("face_origin", False) and (abs(dx_pan) > 1e-4 or abs(dy_pan) > 1e-4):
                if mode == "arc":
                    # Arc mode: each cup faces pan axis individually
                    yaw_rad = math.atan2(-dy_pan, -dx_pan)
                else:
                    # Line mode: all cups face the same direction (along the line's radial)
                    angle_rad = math.radians(CUP_LAYOUT["angle_deg"])
                    yaw_rad = angle_rad + math.pi
                # Quaternion for Z-axis rotation: (cos(θ/2), 0, 0, sin(θ/2))
                orient = Gf.Quatd(math.cos(yaw_rad / 2), 0, 0, math.sin(yaw_rad / 2))
            else:
                orient = Gf.Quatd(1, 0, 0, 0)

            xform = UsdGeom.Xform(cup_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)

            # Rigid body + collision (convexDecomposition preserves hollow interior)
            UsdPhysics.RigidBodyAPI.Apply(cup_prim)
            for child in Usd.PrimRange(cup_prim):
                if child.GetTypeName() == "Mesh":
                    UsdPhysics.CollisionAPI.Apply(child)
                    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(child)
                    mesh_col.CreateApproximationAttr().Set("convexDecomposition")
                    # Tighten contactOffset to match robot side. PhysX sums
                    # the two shapes' contactOffsets when computing contact-
                    # generation zone, so leaving the cup at the 20 mm
                    # default (unauthored) swamps our 0.1 mm on the robot —
                    # effective zone becomes 20.1 mm and cups still get
                    # soft-contact pushes. 0.1 mm on both sides → 0.2 mm
                    # total zone → only real geometric contact. See
                    # DEBUG-GUIDE § 4.3.
                    px = PhysxSchema.PhysxCollisionAPI.Apply(child)
                    px.CreateContactOffsetAttr().Set(0.0001)  # 0.1 mm
                    px.CreateRestOffsetAttr().Set(0.0)

            placed += 1
            print(f"  Added cup_{color_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Physics material for cups
        physics_mat_path = f"{container_path}/PhysicsMaterial"
        material = UsdShade.Material.Define(stage, physics_mat_path)
        physics_mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_mat_api.CreateDynamicFrictionAttr().Set(self._object_dynamic_friction)
        physics_mat_api.CreateRestitutionAttr().Set(self._object_restitution)
        physics_mat_api.CreateStaticFrictionAttr().Set(self._object_static_friction)
        physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_mat_api.CreateFrictionCombineModeAttr().Set(self._object_friction_combine_mode)
        physx_mat_api.CreateRestitutionCombineModeAttr().Set(self._object_restitution_combine_mode)

        # Bind physics material to all cup collision prims
        from omni.physx.scripts import physicsUtils
        physics_mat_sdf_path = Sdf.Path(physics_mat_path)
        containers_prim = stage.GetPrimAtPath(container_path)
        if containers_prim.IsValid():
            for child in containers_prim.GetChildren():
                if child.GetName() == "PhysicsMaterial":
                    continue
                for desc in Usd.PrimRange(child):
                    if desc.HasAPI(UsdPhysics.CollisionAPI):
                        physicsUtils.add_physics_material_to_prim(stage, desc, physics_mat_sdf_path)

        print(f"[add_cups] Placed {placed} cups ({mode} mode, radius={CUP_LAYOUT['radius']}m)")

        # Override cup material colors to match OBJECT_COLORS palette
        for color_name in colors:
            if color_name not in OBJECT_COLORS:
                continue
            cup_prim_path = f"{container_path}/cup_{color_name}"
            self._set_block_material_color(cup_prim_path, OBJECT_COLORS[color_name])

        # Add ArUco markers on cup sides
        try:
            self._add_aruco_to_cups(container_path)
        except Exception as e:
            print(f"[add_cups] ArUco marker error: {e}")
            traceback.print_exc()

        return True

    def _add_aruco_to_cups(self, container_path="/World/Containers"):
        """Add pre-baked ArUco marker USD meshes onto cup surfaces.

        Each marker is a Blender-generated curved mesh (shrinkwrapped onto the
        actual cup STL) stored as a USDA in assets/aruco_markers/.  At runtime
        we simply add a USD reference as a child of the cup prim — the mesh
        already has the correct shape, UVs, and material.

        If a marker USD is missing, falls back to generating it via Blender
        headless (requires blender on PATH).
        """
        import omni.kit.commands
        from pxr import Sdf, UsdShade, UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()
        containers = stage.GetPrimAtPath(container_path)
        if not containers or not containers.IsValid():
            return

        cfg = CUP_ARUCO_CONFIG
        ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        aruco_dir = os.path.join(ext_dir, "assets", "aruco_markers")

        looks_path = f"{container_path}/Looks"
        if not stage.GetPrimAtPath(looks_path):
            UsdGeom.Scope.Define(stage, looks_path)

        unbaked = []
        total = 0
        for child in containers.GetChildren():
            cup_name = child.GetName()
            color = cup_name.replace("cup_", "")
            aruco_id = cfg["ids"].get(color)
            if aruco_id is None:
                continue

            marker_prim_path = f"{child.GetPath()}/aruco_{aruco_id:03d}"
            if stage.GetPrimAtPath(marker_prim_path).IsValid():
                continue

            # Locate the pre-baked marker USD
            marker_usda = os.path.join(aruco_dir, "meshes", f"aruco_marker_{aruco_id:03d}.usda")
            if not os.path.isfile(marker_usda):
                # The baked USDs are committed (assets/aruco_markers/meshes/), so this bake is
                # only a fallback for a missing file; it needs Blender 4.x (SOARM_BLENDER).
                self._generate_marker_usd(aruco_id, aruco_dir, ext_dir, cfg)
                if not os.path.isfile(marker_usda):
                    unbaked.append(cup_name)
                    continue

            # Add as a referenced prim under the cup
            prim = stage.DefinePrim(marker_prim_path)
            marker_usd_uri = "file://" + os.path.abspath(marker_usda)
            prim.GetReferences().AddReference(
                marker_usd_uri,
                "/root/aruco_marker"
            )

            # The Blender-exported material uses UsdPreviewSurface which Isaac
            # Sim's RTX renderer ignores.  Create an OmniPBR material and bind
            # it to the marker mesh so the ArUco texture actually renders.
            mesh_path = f"{marker_prim_path}/aruco_marker_mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if mesh_prim and mesh_prim.IsValid():
                # Ensure marker PNG exists for the texture
                png_path = os.path.join(aruco_dir, "pngs", f"aruco_4x4_{aruco_id:03d}.png")
                if not os.path.isfile(png_path):
                    try:
                        from .generate_aruco_marker import generate_marker_png
                    except ImportError:
                        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                        from generate_aruco_marker import generate_marker_png
                    generate_marker_png(cfg["dictionary"], aruco_id, png_path, cfg["marker_png_pixels"])

                out = []
                omni.kit.commands.execute(
                    "CreateAndBindMdlMaterialFromLibrary",
                    mdl_name="OmniPBR.mdl",
                    mtl_name="OmniPBR",
                    mtl_created_list=out,
                    select_new_prim=False,
                )
                if out:
                    target_mat_path = f"{looks_path}/aruco_{color}_{aruco_id:03d}"
                    omni.kit.commands.execute(
                        "MovePrim", path_from=out[0], path_to=target_mat_path
                    )
                    shader = stage.GetPrimAtPath(target_mat_path + "/Shader")
                    if shader:
                        shader.CreateAttribute(
                            "inputs:diffuse_texture", Sdf.ValueTypeNames.Asset
                        ).Set(Sdf.AssetPath(f"file:{png_path}"))
                        shader.CreateAttribute(
                            "inputs:reflection_roughness_constant", Sdf.ValueTypeNames.Float
                        ).Set(0.9)
                    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
                        UsdShade.Material(stage.GetPrimAtPath(target_mat_path))
                    )

            total += 1
            print(f"  [aruco] Loaded marker ID {aruco_id} onto {cup_name}")

        if total:
            print(f"[aruco] Loaded {total} ArUco markers onto cups")
        if unbaked:
            carb.log_warn(
                f"[aruco] {', '.join(unbaked)} have NO ArUco marker: the baked marker USD is missing "
                f"and could not be generated. The training data has markers on the cups, so camera "
                f"frames will differ. Bake with Blender 4.x: SOARM_BLENDER=/path/to/blender-4.x "
                f"(then call add_cups again), or restore assets/aruco_markers/meshes/ from git.")

    def _find_blender4(self):
        """Blender binary for the marker bake: $SOARM_BLENDER, else `blender` on PATH, and only if
        it is 4.x or newer (bake_aruco_on_cup.py uses bpy.ops.wm.stl_import, added in 4.0).
        Probed once per session; None when unavailable (the caller warns)."""
        if getattr(self, "_blender4_probe", None) is not None:
            return self._blender4_probe or None
        import re
        import subprocess
        binary = os.environ.get("SOARM_BLENDER") or shutil.which("blender")
        found = ""
        if binary:
            try:
                out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                     timeout=30).stdout
                m = re.search(r"Blender (\d+)\.(\d+)", out)
                if m and int(m.group(1)) >= 4:
                    found = binary
                else:
                    carb.log_warn(f"[aruco] {binary} is not Blender 4.x "
                                  f"({m.group(0) if m else 'version unknown'}); set SOARM_BLENDER")
            except (OSError, subprocess.TimeoutExpired) as e:
                carb.log_warn(f"[aruco] cannot run Blender at {binary}: {e}")
        self._blender4_probe = found
        return found or None

    def _generate_marker_usd(self, aruco_id, aruco_dir, ext_dir, cfg):
        """Generate a marker USD via Blender headless (fallback if not pre-baked)."""
        import subprocess

        # Ensure marker PNG exists
        try:
            from .generate_aruco_marker import generate_marker_png
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from generate_aruco_marker import generate_marker_png

        png_path = os.path.join(aruco_dir, "pngs", f"aruco_4x4_{aruco_id:03d}.png")
        generate_marker_png(cfg["dictionary"], aruco_id, png_path, cfg["marker_png_pixels"])

        # Cup STL: the in-repo mesh. ext_dir is <repo>/isaac/exts/soarm101-dt.
        repo_root = os.path.abspath(os.path.join(ext_dir, "..", "..", ".."))
        cup_stl = os.path.join(repo_root, "vla_SO-ARM101", "src", "so_arm101_description",
                               "meshes", "cup", "cup.stl")
        if not os.path.isfile(cup_stl):
            carb.log_info(f"[aruco] cup STL not in repo ({cup_stl}); markers not generated")
            return
        blender = self._find_blender4()
        if blender is None:
            return

        script = os.path.join(ext_dir, "scripts", "bake_aruco_on_cup.py")
        output = os.path.join(aruco_dir, "meshes", f"aruco_marker_{aruco_id:03d}.usda")

        cmd = [
            blender, "--background", "--python", script, "--",
            "--cup-stl", cup_stl,
            "--marker-png", png_path,
            "--output", output,
            "--marker-size", str(cfg["marker_size_m"]),
            "--height-fraction", str(cfg["height_fraction"]),
            "--subdivisions", "16",
        ]
        print(f"[aruco] Generating marker {aruco_id} via Blender...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            # Blender exits 0 even when the bake script raises, so judge by the output file.
            if result.returncode == 0 and os.path.isfile(output):
                print(f"[aruco] Generated: {output}")
            elif result.returncode == 0:
                carb.log_warn(f"[aruco] {blender} produced no {os.path.basename(output)}: "
                              f"{result.stdout[-300:]}")
            else:
                print(f"[aruco] Blender failed: {result.stderr[-200:]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[aruco] Blender generation failed: {e}")

    def delete_cups(self, container_path="/World/Containers"):
        """Delete all cups, ArUco markers, drop pose graph, and wrapper prims."""
        stage = omni.usd.get_context().get_stage()

        # Remove drop poses action graph FIRST (references container prims)
        drop_graph = stage.GetPrimAtPath("/Graph/ActionGraph_drop_poses")
        if drop_graph and drop_graph.IsValid():
            stage.RemovePrim("/Graph/ActionGraph_drop_poses")
            print("[delete_cups] Removed /Graph/ActionGraph_drop_poses")

        prim = stage.GetPrimAtPath(container_path)
        if prim and prim.IsValid():
            stage.RemovePrim(container_path)
            print(f"[delete_cups] Deleted {container_path} (including ArUco markers and drop wrappers)")
        else:
            print(f"[delete_cups] {container_path} does not exist")

        # Also clean up any stale ArUco materials left in /World/Looks
        world_looks = stage.GetPrimAtPath("/World/Looks")
        if world_looks and world_looks.IsValid():
            for child in world_looks.GetChildren():
                if "aruco" in child.GetName():
                    stage.RemovePrim(child.GetPath())
                    print(f"[delete_cups] Cleaned up stale {child.GetPath()}")

    def _get_cup_positions(self, container_path="/World/Containers"):
        """Return dict of {color: {center_x, center_y, rim_z, opening_radius}} for each cup."""
        stage = omni.usd.get_context().get_stage()
        positions = {}
        containers = stage.GetPrimAtPath(container_path)
        if not containers or not containers.IsValid():
            return positions
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for child in containers.GetChildren():
            name = child.GetName()  # e.g. "cup_red"
            color = name.replace("cup_", "")
            if color in BLOCK_COLORS:
                bbox = bbox_cache.ComputeWorldBound(child)
                r = bbox.ComputeAlignedRange()
                mn, mx = r.GetMin(), r.GetMax()
                positions[color] = {
                    "center_x": (mn[0] + mx[0]) / 2.0,
                    "center_y": (mn[1] + mx[1]) / 2.0,
                    "rim_z": mx[2],
                    "opening_radius": min(mx[0] - mn[0], mx[1] - mn[1]) / 2.0,
                }
        return positions

    def sort_into_cups(self, color=None, folder_path="/World/Objects"):
        """Sort lego blocks by dropping each into its matching color cup.

        Blocks are placed above the cup rim so they fall in when
        physics is running. Small XY offsets keep them from stacking
        perfectly (more natural drop).

        Args:
            color: Optional color filter ('red', 'green', 'blue').
            folder_path: Parent prim path for objects.
        """
        cup_info = self._get_cup_positions()
        if not cup_info:
            print("[sort_into_cups] No cups found. Run add_cups first.")
            return 0

        stage = omni.usd.get_context().get_stage()
        colors_to_sort = [color] if color else list(BLOCK_COLORS.keys())
        sorted_count = 0

        for clr in colors_to_sort:
            if clr not in cup_info:
                print(f"[sort_into_cups] No cup for color '{clr}'")
                continue

            cup = cup_info[clr]
            cx, cy = cup["center_x"], cup["center_y"]
            rim_z = cup["rim_z"]
            opening_r = cup["opening_radius"]

            blocks = self._get_blocks_by_color(clr, folder_path)

            # Keep XY offsets within ~40% of opening radius so blocks land inside
            max_offset = opening_r * 0.3

            for i, block_name in enumerate(sorted(blocks)):
                prim_path = self._get_prim_path(block_name, folder_path)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue

                # Drop from above the rim, stagger height so they don't collide mid-air
                drop_z = rim_z + 0.02 + 0.025 * i
                offset_x = np.random.uniform(-max_offset, max_offset)
                offset_y = np.random.uniform(-max_offset, max_offset)
                pos = Gf.Vec3d(cx + offset_x, cy + offset_y, drop_z)
                self._set_obj_prim_pose(prim_path, pos)
                sorted_count += 1

        print(f"[sort_into_cups] Dropped {sorted_count} blocks above cups (color={color or 'all'})")
        return sorted_count

    def disassemble_objects(self, folder_path="/World/Objects"):
        """Lay out objects in a row along lateral axis at forward=BLOCK_SPAWN_FORWARD."""
        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath(folder_path)
        if not objects_root.IsValid():
            print(f"[disassemble] Error: {folder_path} does not exist")
            return 0

        # Collect valid block prims and their lateral extents
        block_info = []
        for child in objects_root.GetChildren():
            obj_name = child.GetName()
            if obj_name.endswith("Material"):
                continue
            body_path = self._get_prim_path(obj_name, folder_path)
            body_prim = stage.GetPrimAtPath(body_path)
            if not body_prim or not body_prim.IsValid():
                continue
            sx, sy, _ = self._get_prim_bbox_size(body_path)
            lat_ext = sy if ROBOT_FORWARD_AXIS == "X" else sx
            block_info.append((body_path, lat_ext))

        if not block_info:
            return 0

        # Compute total span and center the row at lateral=0
        total_span = sum(le for _, le in block_info) + self._object_gap * (len(block_info) - 1)
        lat_cursor = -total_span / 2.0

        for body_path, lat_ext in block_info:
            lat_position = lat_cursor + lat_ext / 2.0

            z = self._get_ground_z(body_path)
            wx, wy = _to_world(BLOCK_SPAWN_FORWARD, lat_position)
            self._set_obj_prim_pose(body_path, Gf.Vec3d(wx, wy, z))

            lat_cursor += lat_ext + self._object_gap

        print(f"[disassemble] Disassembled {len(block_info)} objects")
        return len(block_info)

    def randomize_object_poses(self, color=None, folder_path="/World/Objects",
                                required_visible_color=None,
                                required_visible_count=1,
                                camera_fov_margin_px=30):
        """
        Randomize block poses. Optionally filter by color and/or require that
        N legos of a chosen color land inside the workspace camera's view.

        Args:
            color: Optional color filter ('red', 'green', 'blue') for which
                blocks get NEW positions. If None, all are re-randomized.
            folder_path: Parent prim path for objects.
            required_visible_color: If set, the sampler will reject placements
                where fewer than `required_visible_count` legos of this color
                project into the workspace camera's image. Crucial for
                VLA-style recordings — without it, target legos can land
                behind the arm or off-camera and the recorded observation
                doesn't show the object the policy is meant to act on.
            required_visible_count: How many of `required_visible_color` must
                be in frame after randomization (default 1). Capped at the
                number of that-color legos actually present in the scene.
            camera_fov_margin_px: distance from any image edge that the
                target legos must clear (default 30 px on a 640x480 frame).
        Returns:
            (randomized_count, 0)
        """
        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath(folder_path)
        if not objects_root.IsValid():
            print(f"[randomize] Error: {folder_path} does not exist")
            return 0, 0

        # Collect blocks to randomize and fixed blocks (other colors)
        to_randomize = []
        fixed_positions = []

        for child in objects_root.GetChildren():
            obj = child.GetName()
            if obj == "PhysicsMaterial":
                continue

            prim_path = self._get_prim_path(obj, folder_path)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue

            xform = UsdGeom.Xformable(prim)
            world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_pos = world_transform.ExtractTranslation()

            block_color = self._get_block_color(obj)
            if color is None or block_color == color:
                to_randomize.append({
                    "name": obj,
                    "prim_path": prim_path,
                    "current_z": world_pos[2],
                })
            else:
                # This block stays fixed - record its position for collision avoidance
                fixed_positions.append(world_pos)

        if not to_randomize:
            print(f"[randomize] No blocks found for color={color or 'all'}")
            return 0, 0

        # Add cup bbox footprints as exclusion zones (if cups exist)
        cup_exclusions = []  # [(center_x, center_y, half_w, half_h)]
        cup_info = self._get_cup_positions()
        for clr, info in cup_info.items():
            cup_exclusions.append((
                info["center_x"], info["center_y"],
                info["opening_radius"], info["opening_radius"],
            ))

        z_values = [info["current_z"] for info in to_randomize]

        # Compute per-object half-diagonal for pair-wise separation
        radii = []
        for info in to_randomize:
            sx, sy, _ = self._get_prim_bbox_size(info["prim_path"])
            radii.append((sx**2 + sy**2) ** 0.5 / 2.0)

        # Build must_be_visible_indices: pick the first N to_randomize entries
        # whose color matches required_visible_color. The sampler's retry loop
        # explores different orderings, so any one of them eventually lands.
        must_be_visible_indices = None
        if required_visible_color:
            target_indices = [
                i for i, info in enumerate(to_randomize)
                if self._get_block_color(info["name"]) == required_visible_color
            ]
            if not target_indices:
                print(f"[randomize] required_visible_color={required_visible_color!r} "
                      f"requested but no matching blocks in to_randomize — "
                      f"FOV constraint dropped")
            else:
                n = max(1, min(int(required_visible_count), len(target_indices)))
                must_be_visible_indices = target_indices[:n]
                print(f"[randomize] requiring {n} {required_visible_color} lego(s) "
                      f"in workspace-camera FOV "
                      f"(indices={must_be_visible_indices}, "
                      f"margin={camera_fov_margin_px}px)")

        poses = self._sample_non_overlapping_objects(
            num_objects=len(to_randomize),
            z_values=z_values,
            fixed_positions=fixed_positions,
            radii=radii,
            exclusion_zones=cup_exclusions,
            must_be_visible_indices=must_be_visible_indices,
            camera_fov_margin_px=camera_fov_margin_px,
        )

        for obj_info, pose in zip(to_randomize, poses):
            prim_path = obj_info["prim_path"]
            pos = Gf.Vec3d(pose["position"][0], pose["position"][1], pose["position"][2])
            self._set_obj_prim_pose(prim_path, pos, (0.0, 0.0, pose["yaw_deg"]))

        print(f"[randomize] Randomized {len(to_randomize)} blocks (color={color or 'all'})")
        return len(to_randomize), 0

    def randomize_single_object(self, object_name, folder_path="/World/Objects",
                                x_range=None, y_range=None,
                                min_sep=None, yaw_range=(-180.0, 180.0), max_attempts=1000):
        """Randomize a single object's pose in a clear workspace area, keeping all others fixed.

        Args:
            object_name: Name of the object to randomize.
            folder_path: Parent prim path for objects.
            x_range: X range for randomization (clear workspace area).
            y_range: Y range for randomization (clear workspace area).
            min_sep: Minimum separation from all other objects.
            yaw_range: Yaw rotation range in degrees.
            max_attempts: Maximum placement attempts.
        """
        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath(folder_path)
        if not objects_root.IsValid():
            raise ValueError(f"{folder_path} does not exist")

        # Verify the target object exists
        target_parent_path = f"{folder_path}/{object_name}"
        target_child_path = self._get_prim_path(object_name, folder_path)
        target_parent_prim = stage.GetPrimAtPath(target_parent_path)
        target_child_prim = stage.GetPrimAtPath(target_child_path)
        if not target_parent_prim.IsValid() or not target_child_prim.IsValid():
            raise ValueError(f"Object '{object_name}' not found at {target_child_path}")

        # Compute target half-diagonal
        target_sx, target_sy, _ = self._get_prim_bbox_size(target_child_path)
        target_radius = (target_sx**2 + target_sy**2) ** 0.5 / 2.0

        # Collect world positions AND radii of all OTHER objects
        fixed_xy = []
        fixed_radii = []
        for child in objects_root.GetChildren():
            name = child.GetName()
            if name == object_name or name == "PhysicsMaterial":
                continue
            child_path = self._get_prim_path(name, folder_path)
            child_prim = stage.GetPrimAtPath(child_path)
            if child_prim.IsValid():
                xform = UsdGeom.Xformable(child_prim)
                world_pos = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                fixed_xy.append(np.array([world_pos[0], world_pos[1]]))
                fsx, fsy, _ = self._get_prim_bbox_size(child_path)
                fixed_radii.append((fsx**2 + fsy**2) ** 0.5 / 2.0)

        # Collect cup exclusion zones as AABB (if cups exist)
        cup_exclusions = []  # [(cx, cy, half_w, half_h)]
        cup_info = self._get_cup_positions()
        for clr, info in cup_info.items():
            cup_exclusions.append((
                info["center_x"], info["center_y"],
                info["opening_radius"], info["opening_radius"],
            ))

        # Get current Z of target object
        target_xform = UsdGeom.Xformable(target_child_prim)
        target_world_pos = target_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        target_z = target_world_pos[2]

        # Per-pair separation: target_radius + fixed_radius + gap
        if min_sep is not None:
            fixed_seps = [min_sep] * len(fixed_xy)
        else:
            fixed_seps = [target_radius + fr + self._object_gap for fr in fixed_radii]

        # Sample one non-overlapping position (same workspace as randomize_object_poses
        # when no range is given)
        x_range, y_range = self._default_block_xy_ranges(x_range, y_range)
        pose = None
        for _ in range(max_attempts):
            candidate = np.array([
                np.random.uniform(*x_range),
                np.random.uniform(*y_range)
            ])
            # Check circular separation against other blocks
            if not all(np.linalg.norm(candidate - fxy) >= fsep
                       for fxy, fsep in zip(fixed_xy, fixed_seps)):
                continue
            # Check AABB exclusion against cup footprints
            in_cup = False
            for cx, cy, hw, hh in cup_exclusions:
                margin = target_radius + self._object_gap
                if (abs(candidate[0] - cx) < hw + margin and
                    abs(candidate[1] - cy) < hh + margin):
                    in_cup = True
                    break
            if in_cup:
                continue
            yaw_deg = np.random.uniform(*yaw_range)
            pose = {"position": np.array([candidate[0], candidate[1], target_z]),
                    "yaw_deg": yaw_deg}
            break

        if pose is None:
            raise RuntimeError(
                f"Could not place '{object_name}' — workspace too crowded with "
                f"{len(fixed_xy)} other objects. Use randomize_object_poses to "
                f"re-randomize all objects together.")
        poses = [pose]

        # Apply the pose
        pose = poses[0]
        parent_xform = UsdGeom.Xformable(target_parent_prim)
        parent_world_transform = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        target_world = Gf.Vec3d(pose["position"][0], pose["position"][1], pose["position"][2])
        local_pos = parent_world_transform.GetInverse().Transform(target_world)

        self._set_obj_prim_pose(target_child_path, local_pos, (0.0, 0.0, pose["yaw_deg"]))

        print(f"[randomize_single] Placed '{object_name}' at world ({pose['position'][0]:.3f}, {pose['position'][1]:.3f}, {pose['position'][2]:.4f}), yaw={pose['yaw_deg']:.1f}°")

    def sync_real_poses(self):
        """Subscribe to /objects_poses_real and update sim object poses to match."""
        import rclpy
        from tf2_msgs.msg import TFMessage

        rclpy.init(args=None)
        node = rclpy.create_node("_sync_real_poses_tmp")
        try:
            msg_received = [None]

            def _cb(msg):
                msg_received[0] = msg

            sub = node.create_subscription(TFMessage, "/objects_poses_real", _cb, 1)
            # Spin until we get one message (timeout after 3 seconds)
            import time
            t0 = time.time()
            while msg_received[0] is None and (time.time() - t0) < 3.0:
                rclpy.spin_once(node, timeout_sec=0.1)
            node.destroy_subscription(sub)

            if msg_received[0] is None:
                print("[SyncRealPoses] No message received on /objects_poses_real within 3s")
                return

            # Build dict of real poses by name
            real_poses = {}
            for tf in msg_received[0].transforms:
                t = tf.transform.translation
                r = tf.transform.rotation
                real_poses[tf.child_frame_id] = {
                    'position': (t.x, t.y, t.z),
                    'quat_wxyz': (r.w, r.x, r.y, r.z),
                }

            stage = omni.usd.get_context().get_stage()
            updated = []
            for name, pose in real_poses.items():
                prim_path = self._get_prim_path(name)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    print(f"[SyncRealPoses] Prim not found: {prim_path}")
                    continue

                # Convert quaternion (w,x,y,z) to euler XYZ degrees
                qw, qx, qy, qz = pose['quat_wxyz']
                euler_deg = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)
                self._set_obj_prim_pose(prim_path, pose['position'], tuple(euler_deg))
                updated.append(name)

            print(f"[SyncRealPoses] Updated {len(updated)} objects: {updated}")
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def add_aruco_markers(self, objects_path="/World/Objects"):
        """Add ArUco marker meshes to all objects that have matching aruco JSON files."""
        import omni.kit.commands
        from pxr import Sdf, UsdShade, UsdGeom, Gf

        ARUCO_DIR = os.path.expanduser("~/Projects/aruco-grasp-annotator/data/aruco")
        ARUCO_PNG_DIR = os.path.join(ARUCO_DIR, "pngs")

        stage = omni.usd.get_context().get_stage()
        objects_prim = stage.GetPrimAtPath(objects_path)
        if not objects_prim or not objects_prim.IsValid():
            print(f"Error: {objects_path} not found")
            return

        total_added = 0
        for child in objects_prim.GetChildren():
            obj_name = child.GetName()
            aruco_json = os.path.join(ARUCO_DIR, f"{obj_name}_aruco.json")
            if not os.path.exists(aruco_json):
                continue

            with open(aruco_json, 'r') as f:
                aruco_data = json.load(f)

            # Find the mesh prim (e.g. /World/Objects/base1/base1/base1)
            base_prim_path = self._get_prim_path(obj_name, objects_path)
            base_prim = stage.GetPrimAtPath(base_prim_path)
            if not base_prim or not base_prim.IsValid():
                print(f"Warning: mesh prim not found at {base_prim_path}, skipping")
                continue

            aruco_dict = aruco_data.get('aruco_dictionary', 'DICT_4X4_50')
            dict_name = aruco_dict.replace('DICT_', '').split('_')[0].lower()
            marker_size = aruco_data.get('size', 0.021)

            for marker in aruco_data['markers']:
                aruco_id = marker['aruco_id']
                position = marker['T_object_to_marker']['position']
                rotation = marker['T_object_to_marker']['rotation']

                # Compensate positions for parent prim's scale
                parent_scale_attr = base_prim.GetAttribute("xformOp:scale")
                inv_scale = 1.0 / (parent_scale_attr.Get()[0] if parent_scale_attr and parent_scale_attr.Get() else 1.0)
                scaled_x = position['x'] * inv_scale
                scaled_y = position['y'] * inv_scale
                scaled_z = position['z'] * inv_scale

                cube_prim_path = f"{base_prim_path}/aruco_{aruco_id:03d}"

                # Remove existing marker and recreate
                if stage.GetPrimAtPath(cube_prim_path):
                    stage.RemovePrim(cube_prim_path)

                # Create cube mesh
                omni.kit.commands.execute("CreateMeshPrimCommand",
                                          prim_path=cube_prim_path,
                                          prim_type="Cube")

                cube_prim = stage.GetPrimAtPath(cube_prim_path)
                if not cube_prim:
                    print(f"Error: failed to create cube at {cube_prim_path}")
                    continue

                # Set transforms using existing attributes created by CreateMeshPrimCommand
                cube_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(scaled_x, scaled_y, scaled_z))

                rotation_matrix = Gf.Matrix3d(
                    Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation['roll'] * 180.0 / 3.14159) *
                    Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation['pitch'] * 180.0 / 3.14159) *
                    Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation['yaw'] * 180.0 / 3.14159))
                quat = rotation_matrix.ExtractRotation().GetQuat()
                cube_prim.GetAttribute("xformOp:orient").Set(quat)

                # Compensate for parent prim's scale (e.g. 0.01 from Fusion360 export)
                parent_scale_attr = base_prim.GetAttribute("xformOp:scale")
                parent_scale = parent_scale_attr.Get()[0] if parent_scale_attr and parent_scale_attr.Get() else 1.0
                compensated = marker_size / parent_scale
                cube_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(compensated, compensated, 0.0001 / parent_scale))

                # Create OmniPBR material, then move it into /World/Objects/Looks
                looks_path = f"{objects_path}/Looks"
                if not stage.GetPrimAtPath(looks_path):
                    UsdGeom.Scope.Define(stage, looks_path)
                out = []
                omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary",
                                          mdl_name="OmniPBR.mdl",
                                          mtl_name="OmniPBR",
                                          mtl_created_list=out,
                                          select_new_prim=False)
                if not out:
                    print(f"Error: failed to create material for marker {aruco_id}")
                    continue

                # Move material from /World/Looks/ to /World/Objects/Looks/
                created_path = out[0]
                target_mat_path = f"{looks_path}/aruco_{obj_name}_{aruco_id:03d}"
                omni.kit.commands.execute("MovePrim",
                                          path_from=created_path,
                                          path_to=target_mat_path)

                # Assign aruco texture
                aruco_png = os.path.join(ARUCO_PNG_DIR, f"aruco_marker_{dict_name}_{aruco_id:03d}.png")
                if os.path.exists(aruco_png):
                    shader_prim = stage.GetPrimAtPath(target_mat_path + "/Shader")
                    if shader_prim:
                        texture_attr = shader_prim.CreateAttribute('inputs:diffuse_texture', Sdf.ValueTypeNames.Asset)
                        texture_attr.Set(Sdf.AssetPath(f"file:{aruco_png}"))

                # Bind material to cube
                omni.kit.commands.execute("BindMaterial",
                                          prim_path=cube_prim_path,
                                          material_path=target_mat_path)

                total_added += 1

            print(f"Added {len(aruco_data['markers'])} ArUco markers to {obj_name}")

        print(f"=== ArUco markers complete: {total_added} markers added ===")

    # ── Object Visibility (hide = delete + store, unhide = restore) ────────

    def _get_scene_object_names(self, folder_path="/World/Objects"):
        """Return names of objects currently under /World/Objects (excludes PhysicsMaterial)."""
        stage = omni.usd.get_context().get_stage()
        objects_prim = stage.GetPrimAtPath(folder_path)
        if not objects_prim or not objects_prim.IsValid():
            return []
        return [
            child.GetName() for child in objects_prim.GetChildren()
            if child.IsA(UsdGeom.Xformable) and child.GetName() != "PhysicsMaterial"
        ]

    def _build_visibility_ui(self):
        """Build (or rebuild) the combo boxes for exclude/include.

        Always reads current scene objects and excluded list so the UI
        is up-to-date on first open and after every action.
        """
        scene_objects = self._get_scene_object_names()
        excluded_names = list(self._hidden_objects.keys())
        with ui.VStack(spacing=5, height=0):
            with ui.HStack(spacing=5):
                ui.Label("Object:", alignment=ui.Alignment.LEFT, width=80)
                if scene_objects:
                    self._vis_object_combo = ui.ComboBox(0, *scene_objects, width=150)
                else:
                    self._vis_object_combo = ui.ComboBox(0, "(none)", width=150)
                ui.Button("Exclude", width=80, height=30, clicked_fn=self._on_hide_clicked)
            with ui.HStack(spacing=5):
                ui.Label("Excluded:", alignment=ui.Alignment.LEFT, width=80)
                if excluded_names:
                    self._vis_hidden_combo = ui.ComboBox(0, *excluded_names, width=150)
                else:
                    self._vis_hidden_combo = ui.ComboBox(0, "(none)", width=150)
                ui.Button("Include", width=80, height=30, clicked_fn=self._on_unhide_clicked)

    def _on_hide_clicked(self):
        scene_objects = self._get_scene_object_names()
        if not scene_objects:
            print("No objects to exclude")
            return
        idx = self._vis_object_combo.model.get_item_value_model().as_int
        if idx < len(scene_objects):
            self.hide_object(scene_objects[idx])
        self._visibility_frame.rebuild()

    def _on_unhide_clicked(self):
        hidden_names = list(self._hidden_objects.keys())
        if not hidden_names:
            print("No excluded objects to include")
            return
        idx = self._vis_hidden_combo.model.get_item_value_model().as_int
        if idx < len(hidden_names):
            self.unhide_object(hidden_names[idx])
        self._visibility_frame.rebuild()


    def _save_all_object_poses(self, folder_path="/World/Objects", exclude=None):
        """Snapshot body poses of all objects under folder_path (excluding *exclude*)."""
        poses = {}
        for name in self._get_scene_object_names(folder_path):
            if name == exclude:
                continue
            body_path = self._get_prim_path(name, folder_path)
            pose = self._read_prim_pose(body_path)
            if pose:
                poses[name] = pose
        return poses

    def _restore_object_poses(self, poses, folder_path="/World/Objects"):
        """Write back a snapshot produced by _save_all_object_poses.

        Uses two-step ChangeProperty (orient first, then translate) consistent
        with assemble_objects.
        """
        for name, pose_data in poses.items():
            body_path = self._get_prim_path(name, folder_path)
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(body_path)
            if not prim or not prim.IsValid():
                continue
            quat = pose_data.get("quaternion", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
            pos = pose_data.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
            omni.kit.commands.execute("ChangeProperty",
                                     prop_path=f"{body_path}.xformOp:orient",
                                     value=Gf.Quatf(float(quat["w"]), float(quat["x"]),
                                                    float(quat["y"]), float(quat["z"])),
                                     prev=None)
            omni.kit.commands.execute("ChangeProperty",
                                     prop_path=f"{body_path}.xformOp:translate",
                                     value=Gf.Vec3d(pos["x"], pos["y"], pos["z"]),
                                     prev=None)

    def hide_object(self, object_name, folder_path="/World/Objects"):
        """Hide an object: save its reference path + pose, then delete the prim.

        Stops the simulation, saves poses of all remaining objects, deletes the
        target, restores the remaining poses, and resumes playback.
        """
        stage = omni.usd.get_context().get_stage()

        prim_path = f"{folder_path}/{object_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            print(f"Object {object_name} not found at {prim_path}")
            return

        # Read the USD reference file path from the root layer spec
        ref_path = None
        prim_spec = stage.GetRootLayer().GetPrimAtPath(prim_path)
        if prim_spec:
            refs = prim_spec.referenceList.prependedItems
            if refs:
                ref_path = refs[0].assetPath
        if not ref_path:
            print(f"Warning: Could not read reference path for {object_name}, unhide will not work")

        # Read the body pose of the object being hidden
        body_prim_path = self._get_prim_path(object_name, folder_path)
        body_pose = self._read_prim_pose(body_prim_path)

        # All objects are blocks
        category = "block"

        # Snapshot poses of all *other* objects before stopping
        other_poses = self._save_all_object_poses(folder_path, exclude=object_name)

        # Stop simulation, delete prim, restore others, resume
        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            timeline.stop()

        self._hidden_objects[object_name] = {
            "ref_path": ref_path,
            "body_pose": body_pose,
            "category": category,
        }

        stage.RemovePrim(prim_path)
        print(f"Hidden (deleted) object: {object_name}")

        # Restore other objects to their pre-stop poses and resume
        self._restore_object_poses(other_poses, folder_path)
        if was_playing:
            timeline.play()

    def unhide_object(self, object_name, folder_path="/World/Objects"):
        """Unhide an object: recreate it from saved reference + pose and reapply physics.

        Saves poses of all existing objects, stops simulation, recreates the
        target prim (with physics), restores *all* object poses (including the
        newly-restored one), and resumes playback.
        """
        if object_name not in self._hidden_objects:
            print(f"Object {object_name} is not in the hidden list")
            return

        data = self._hidden_objects[object_name]
        ref_path = data["ref_path"]
        body_pose = data["body_pose"]
        category = data["category"]

        if not ref_path:
            print(f"Error: No reference path saved for {object_name}, cannot restore")
            del self._hidden_objects[object_name]
            return

        stage = omni.usd.get_context().get_stage()

        prim_path = f"{folder_path}/{object_name}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            print(f"Object {object_name} already exists in scene, removing from hidden list")
            del self._hidden_objects[object_name]
            return

        # Snapshot poses of all existing objects before stopping
        other_poses = self._save_all_object_poses(folder_path)

        # Stop simulation
        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            timeline.stop()

        # Ensure parent folder exists
        if not stage.GetPrimAtPath(folder_path):
            UsdGeom.Xform.Define(stage, folder_path)

        # Recreate prim with USD reference
        prim = stage.DefinePrim(prim_path)
        prim.GetReferences().AddReference(ref_path)

        # Rename Body1 → object_name (mirrors add_objects logic)
        def _rename_body1(search_prim):
            for child in search_prim.GetAllChildren():
                if child.GetName() == "Body1":
                    new_path = child.GetPath().GetParentPath().AppendChild(object_name)
                    omni.kit.commands.execute("MovePrim", path_from=child.GetPath(), path_to=new_path)
                    return True
                if _rename_body1(child):
                    return True
            return False
        _rename_body1(stage.GetPrimAtPath(prim_path))

        # Rebind physics material if it exists in the scene
        physics_mat_path = f"{folder_path}/PhysicsMaterial"
        if stage.GetPrimAtPath(physics_mat_path).IsValid():
            from omni.physx.scripts import physicsUtils
            physics_mat_sdf_path = Sdf.Path(physics_mat_path)
            obj_prim = stage.GetPrimAtPath(prim_path)
            bound = 0
            for desc in Usd.PrimRange(obj_prim):
                if desc.HasAPI(UsdPhysics.CollisionAPI):
                    physicsUtils.add_physics_material_to_prim(stage, desc, physics_mat_sdf_path)
                    bound += 1
            if bound == 0:
                nested_path = f"{folder_path}/{object_name}/{object_name}"
                nested_prim = stage.GetPrimAtPath(nested_path)
                if nested_prim and nested_prim.IsValid():
                    physicsUtils.add_physics_material_to_prim(stage, nested_prim, physics_mat_sdf_path)

        # Reapply block collision / physics settings
        prim_params = self._get_block_params()
        mesh_path = self._get_prim_path(object_name, folder_path)
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if mesh_prim and mesh_prim.IsValid():
            approx = prim_params["collision_approximation"]
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            if approx == "sdf":
                mesh_collision_api.CreateApproximationAttr(PhysxSchema.Tokens.sdf)
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
                sdf_api.CreateSdfResolutionAttr(self._sdf_resolution)
            else:
                mesh_collision_api.CreateApproximationAttr(approx)
            rest_offset = prim_params["rest_offset"]
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            physx_collision_api.CreateContactOffsetAttr().Set(self._contact_offset)
            physx_collision_api.CreateRestOffsetAttr().Set(rest_offset)
            angular_damping = prim_params["angular_damping"]
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(mesh_prim)
            physx_rb_api.CreateAngularDampingAttr().Set(angular_damping)

        del self._hidden_objects[object_name]
        print(f"Unhidden (restored) object: {object_name}")

        # Build the full pose set (existing objects + the restored one)
        all_poses = other_poses
        all_poses[object_name] = body_pose

        # Play first, then restore poses while the sim is running — same
        # mechanism as Assemble / Restore State which handles overlapping
        # objects without collision artifacts.
        if was_playing:
            timeline.play()
            async def _restore_after_play():
                app = omni.kit.app.get_app()
                for _ in range(3):
                    await app.next_update_async()
                self._restore_object_poses(all_poses, folder_path)
            asyncio.ensure_future(_restore_after_play())
        else:
            self._restore_object_poses(all_poses, folder_path)

    def delete_objects(self, folder_path="/World/Objects"):
        """Delete the Objects folder from the scene. Stops simulation first to avoid tensor view crash."""
        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if was_playing:
            timeline.stop()
            print("Stopped simulation before deleting objects")

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(folder_path)
        if prim and prim.IsValid():
            stage.RemovePrim(folder_path)
            print(f"Deleted {folder_path}")
        else:
            print(f"Warning: {folder_path} does not exist")

        # Also delete the object poses action graph if present
        pose_graph_path = "/Graph/ActionGraph_objects_poses"
        pose_graph_prim = stage.GetPrimAtPath(pose_graph_path)
        if pose_graph_prim and pose_graph_prim.IsValid():
            stage.RemovePrim(pose_graph_path)
            print(f"Deleted {pose_graph_path}")

        # Delete the bbox publisher graph if present
        self._stop_bbox_publisher()
        bbox_graph_path = "/Graph/ActionGraph_objects_bbox"
        bbox_graph_prim = stage.GetPrimAtPath(bbox_graph_path)
        if bbox_graph_prim and bbox_graph_prim.IsValid():
            stage.RemovePrim(bbox_graph_path)
            print(f"Deleted {bbox_graph_path}")

    def _get_or_make_lego_instance_usd(self, src_usd_path, instance_name):
        """Generate (or return cached) a per-instance USD copy where the
        inner body prim is renamed to instance_name.

        Per-instance USDs let multiple legos reference the SAME source
        asset without sharing inner-prim names — sidesteps both the
        PhysX friction-patch hang from duplicate inner names AND the
        ROS2PublishTransformTree child_frame_id collision (which would
        publish 'red_2x3' twice for two instances of the same source).

        Returns the path to the instance USD. If the source's inner body
        is already named instance_name (single-instance / matching name
        case), returns src_usd_path unchanged.
        """
        from pxr import Sdf, Usd
        # Load source as fresh stage (NOT a reference, so the inner prim
        # is local to that stage and renamable).
        src_stage = Usd.Stage.Open(src_usd_path)
        if not src_stage:
            print(f"[lego_instance] Failed to open {src_usd_path}")
            return src_usd_path
        src_layer = src_stage.GetRootLayer()
        default_name = src_layer.defaultPrim
        if not default_name:
            # Find first rigid-body prim as fallback
            for p in src_stage.Traverse():
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    default_name = p.GetName()
                    break
        if not default_name:
            print(f"[lego_instance] No body prim found in {src_usd_path}")
            return src_usd_path
        if default_name == instance_name:
            return src_usd_path  # already correctly named, no copy needed

        # Cached instance dir. _get_lego_folder() returns a file:// URI
        # (used by AddReference); for filesystem ops we need the bare
        # path. Recompute the local path directly to avoid URI parsing.
        ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(ext_dir, "assets", "legos", "instances")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, f"lego_{instance_name}.usda")
        if os.path.isfile(out_path):
            return out_path

        # Copy source layer in memory and rename every prim spec whose
        # name matches default_name (lego USDs are structured as
        # Xform "red_2x3" / Mesh "red_2x3" — both need renaming so the
        # inner mesh resolves at the legacy {wrapper}/{wrapper} body
        # path AND ROS2PublishTransformTree gets a unique child_frame_id
        # per instance).
        new_layer = Sdf.Layer.CreateAnonymous(".usda")
        new_layer.TransferContent(src_layer)

        def _rename_recursive(prim_spec, old_name, new_name):
            for child in list(prim_spec.nameChildren):
                _rename_recursive(child, old_name, new_name)
                if child.name == old_name:
                    child.name = new_name

        _rename_recursive(new_layer.pseudoRoot, default_name, instance_name)
        new_layer.defaultPrim = instance_name

        # Sdf rename doesn't rewrite path references inside relationship
        # targets or attribute connections (e.g. material:binding =
        # </red_2x3/green> still points to the pre-rename path).
        # Patch via USDA text replacement post-export — USDA wraps every
        # path reference in angle brackets, so a targeted prefix replace
        # is unambiguous.
        import re
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix=".usda", delete=False) as tmp:
            tmp_path = tmp.name
        new_layer.Export(tmp_path)
        with open(tmp_path) as f:
            text = f.read()
        os.unlink(tmp_path)
        # Match </default_name> exactly and </default_name/...>
        # (followed by '/' or '>') — leaves substrings inside
        # comments / userProperties strings alone since those aren't
        # wrapped in <>.
        pattern = re.compile(
            r"<(/" + re.escape(default_name) + r")(?=[/>])")
        text = pattern.sub(f"</{instance_name}", text)
        with open(out_path, "w") as f:
            f.write(text)
        print(f"[lego_instance] Generated {out_path}")
        return out_path

    def add_objects(self):
        """Import lego blocks (N per color) using per-instance USD copies.

        For each (basename, src_usd, count) tuple in LEGO_USDS, generates
        `count` named instances ("{basename}_{i}") with per-instance USDs
        whose inner body is renamed to match the wrapper. This keeps
        wrapper name == inner body name (legacy {name}/{name} convention)
        AND gives every instance a unique inner prim — no PhysX issue,
        no TF collision in the /objects_poses_sim publisher.

        Skips entirely if all expected blocks already exist in the scene.
        """
        stage = omni.usd.get_context().get_stage()
        target_path = "/World/Objects"

        # Build expected block list by expanding each (basename, usd, count)
        # tuple into `count` named instances. Each instance gets a
        # generated block_name like "{basename}_{i}" — the index-suffix
        # naming required by Cloner's PhysX replicator (root_path + index).
        # Each "spawn group" pairs (basename, usd_path, count) with its
        # generated block_names so the cloning pass below can find them.
        blocks = []
        spawn_groups = []  # [(basename, usd_path, [block_name_0, ...])]
        for color_name in BLOCK_COLORS:
            entries = LEGO_USDS.get(color_name, [])
            if not entries:
                print(f"Warning: No USD files configured for color '{color_name}'")
                continue
            for basename, usd_file, count in entries:
                usd_path = _get_lego_folder() + usd_file
                names = [f"{basename}_{i}" for i in range(count)]
                spawn_groups.append((basename, usd_path, names))
                for n in names:
                    blocks.append((n, color_name, usd_path))

        # Check if all blocks already exist
        all_exist = stage.GetPrimAtPath(target_path) is not None and stage.GetPrimAtPath(target_path).IsValid()
        if all_exist:
            for block_name, _, _ in blocks:
                prim = stage.GetPrimAtPath(f"{target_path}/{block_name}")
                if not prim or not prim.IsValid():
                    all_exist = False
                    break

        if all_exist:
            print(f"[add_objects] All {len(blocks)} blocks already exist, skipping")
            return False

        # Create Objects folder if needed
        if not stage.GetPrimAtPath(target_path):
            UsdGeom.Xform.Define(stage, target_path)

        # Create common physics material
        physics_mat_path = f"{target_path}/PhysicsMaterial"
        material = UsdShade.Material.Define(stage, physics_mat_path)
        physics_mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_mat_api.CreateDynamicFrictionAttr().Set(self._object_dynamic_friction)
        physics_mat_api.CreateRestitutionAttr().Set(self._object_restitution)
        physics_mat_api.CreateStaticFrictionAttr().Set(self._object_static_friction)
        physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_mat_api.CreateFrictionCombineModeAttr().Set(self._object_friction_combine_mode)
        physx_mat_api.CreateRestitutionCombineModeAttr().Set(self._object_restitution_combine_mode)

        print(f"Adding {len(blocks)} lego blocks from {_get_lego_folder()}")

        # First pass: per spawn-group, generate a per-instance USD per
        # name and add a plain reference. Per-instance USDs guarantee
        # unique inner-body prim names — sidesteps both the PhysX
        # friction-patch hang AND the ROS2 frame_id collision that
        # multi-reference of one source USD would cause.
        for basename, usd_path, names in spawn_groups:
            for n in names:
                inst_prim_path = f"{target_path}/{n}"
                if stage.GetPrimAtPath(inst_prim_path).IsValid():
                    continue
                inst_usd = self._get_or_make_lego_instance_usd(usd_path, n)
                inst_prim = stage.DefinePrim(inst_prim_path)
                inst_prim.GetReferences().AddReference(inst_usd)

        # Second pass: ensure convexHull collision and resolve body paths
        body_paths = []
        for block_name, color_name, usd_path in blocks:
            body_path = self._get_prim_path(block_name, target_path)
            body_paths.append(body_path)
            body_prim = stage.GetPrimAtPath(body_path)
            if body_prim and body_prim.IsValid():
                if body_prim.HasAPI(UsdPhysics.CollisionAPI):
                    mc = UsdPhysics.MeshCollisionAPI.Apply(body_prim)
                    mc.CreateApproximationAttr().Set("convexHull")

        # Uniformly distribute blocks in a grid within the workspace, skipping cup zones
        fwd_min, fwd_max = BLOCK_RANDOM_FORWARD
        lat_min, lat_max = BLOCK_RANDOM_LATERAL

        # Collect cup exclusion zones
        cup_exclusions = []
        cup_info = self._get_cup_positions()
        for clr, info in cup_info.items():
            cup_exclusions.append((
                info["center_x"], info["center_y"],
                info["opening_radius"], info["opening_radius"],
            ))

        # Generate grid candidates, filter out those overlapping cups
        n = len(blocks)
        cols = math.ceil(math.sqrt(n * 2))  # oversample to account for excluded cells
        rows = math.ceil(n * 2 / cols)
        fwd_step = (fwd_max - fwd_min) / max(rows, 1)
        lat_step = (lat_max - lat_min) / max(cols, 1)

        valid_cells = []
        for row in range(rows):
            for col in range(cols):
                fwd = fwd_min + fwd_step * (row + 0.5)
                lat = lat_min + lat_step * (col + 0.5)
                wx, wy = _to_world(fwd, lat)
                # Check AABB exclusion against cups
                blocked = False
                for cx, cy, hw, hh in cup_exclusions:
                    margin = 0.025  # half block size + gap
                    if abs(wx - cx) < hw + margin and abs(wy - cy) < hh + margin:
                        blocked = True
                        break
                if not blocked:
                    valid_cells.append((wx, wy))
                if len(valid_cells) >= n:
                    break
            if len(valid_cells) >= n:
                break

        # Second pass: position each block on its body prim
        for i, (block_name, color_name, usd_path) in enumerate(blocks):
            body_path = body_paths[i]
            if i < len(valid_cells):
                wx, wy = valid_cells[i]
            else:
                # Fallback: place at workspace center if grid exhausted
                wx, wy = _to_world((fwd_min + fwd_max) / 2, (lat_min + lat_max) / 2)
            z = self._get_ground_z(body_path)

            self._set_obj_prim_pose(body_path, Gf.Vec3d(wx, wy, z))
            print(f"  Added {block_name} ({color_name}) at ({wx:.3f}, {wy:.3f})")

        # Bind physics material to collision prims
        from omni.physx.scripts import physicsUtils
        physics_mat_sdf_path = Sdf.Path(physics_mat_path)
        objects_prim = stage.GetPrimAtPath(target_path)
        if objects_prim.IsValid():
            for child in objects_prim.GetChildren():
                if child.GetName() == "PhysicsMaterial":
                    continue
                for desc in Usd.PrimRange(child):
                    if desc.HasAPI(UsdPhysics.CollisionAPI):
                        physicsUtils.add_physics_material_to_prim(stage, desc, physics_mat_sdf_path)

        # Override lego colors to match the shared OBJECT_COLORS palette
        for block_name, color_name, _ in blocks:
            if color_name in OBJECT_COLORS:
                prim_path = f"{target_path}/{block_name}"
                self._set_block_material_color(prim_path, OBJECT_COLORS[color_name])

        print(f"[add_objects] Added {len(blocks)} lego blocks")
        return True

    def create_pose_publisher(self):
        """Create action graph for publishing object poses to ROS2"""
        import omni.kit.commands
        from pxr import Sdf, Usd, UsdGeom
        import omni.usd
        import omni.graph.core as og

        def get_objects_in_folder(stage, folder_path="/World/Objects"):
            """Scan Objects folder and return body prim paths for pose publishing.

            Uses self._get_prim_path which has a structural fallback (find
            first child with UsdPhysics.RigidBodyAPI) — handles both the
            legacy single-instance case (wrapper and inner share a name)
            AND the Cloner multi-instance case (wrapper "red_2x3_0"
            wrapping a USD whose inner body is still "red_2x3").
            """
            body_paths = []
            objects_prim = stage.GetPrimAtPath(folder_path)
            if not objects_prim or not objects_prim.IsValid():
                print(f"Warning: {folder_path} does not exist")
                return body_paths

            for child in objects_prim.GetChildren():
                object_name = child.GetName()
                if object_name == "PhysicsMaterial":
                    continue
                body_path = self._get_prim_path(object_name, folder_path)
                body_prim = stage.GetPrimAtPath(body_path)
                if body_prim and body_prim.IsValid():
                    body_paths.append(body_path)
                    print(f"Found: {body_path}")
                else:
                    print(f"Warning: no body prim resolved for {object_name}")

            return body_paths

        def create_action_graph_with_transforms(target_prims, parent_prim="/World", topic_name="objects_poses_sim"):
            """
            Create an action graph with OnPlaybackTick and ROS2PublishTransformTree nodes
            
            Args:
                target_prims: List of prim paths to publish transforms for
                parent_prim: Parent prim for the transform tree
                topic_name: ROS2 topic name
            """
            
            graph_path = "/Graph/ActionGraph_objects_poses"

            # Check if graph already exists
            stage = omni.usd.get_context().get_stage()
            existing_graph = stage.GetPrimAtPath(graph_path)
            if existing_graph and existing_graph.IsValid():
                print(f"Action graph already exists at {graph_path}, skipping creation")
                return

            # Create the action graph using OmniGraph API
            keys = og.Controller.Keys
            
            (graph, nodes, _, _) = og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                        ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("isaac_read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("ros2_publish_transform_tree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ],
                    keys.CONNECT: [
                        ("on_playback_tick.outputs:tick", "ros2_publish_transform_tree.inputs:execIn"),
                        ("ros2_context.outputs:context", "ros2_publish_transform_tree.inputs:context"),
                        ("isaac_read_simulation_time.outputs:simulationTime", "ros2_publish_transform_tree.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        ("ros2_publish_transform_tree.inputs:topicName", topic_name),
                        ("isaac_read_simulation_time.inputs:resetOnStop", False),
                    ],
                },
            )
            
            # Get the stage to set relationships
            stage = omni.usd.get_context().get_stage()
            
            # Get the ROS2 node prim
            ros2_node_path = f"{graph_path}/ros2_publish_transform_tree"
            ros2_prim = stage.GetPrimAtPath(ros2_node_path)
            
            if ros2_prim.IsValid():
                # Set parent prim as a relationship
                parent_rel = ros2_prim.GetRelationship("inputs:parentPrim")
                if not parent_rel:
                    parent_rel = ros2_prim.CreateRelationship("inputs:parentPrim", custom=True)
                parent_rel.SetTargets([Sdf.Path(parent_prim)])
                
                # Set target prims as a relationship
                target_rel = ros2_prim.GetRelationship("inputs:targetPrims")
                if not target_rel:
                    target_rel = ros2_prim.CreateRelationship("inputs:targetPrims", custom=True)
                target_paths = [Sdf.Path(path) for path in target_prims]
                target_rel.SetTargets(target_paths)
                
                print(f"Action graph created at {graph_path}")
                print(f"Publishing {len(target_prims)} objects to topic: {topic_name}")
            else:
                print(f"Error: Could not find ROS2 node at {ros2_node_path}")

        # Get the current USD stage
        stage = omni.usd.get_context().get_stage()
        
        if not stage:
            print("Error: No stage found")
            return
        
        # Get all Body1 prims from the Objects folder
        object_paths = get_objects_in_folder(stage, "/World/Objects")
        
        if not object_paths:
            print("No objects found in /World/Objects")
            return
        
        print(f"\nFound {len(object_paths)} objects:")
        for path in object_paths:
            print(f"  - {path}")
        
        # Create the action graph with all found objects
        create_action_graph_with_transforms(
            target_prims=object_paths,
            parent_prim="/World",
            topic_name="objects_poses_sim"
        )

    # ==================== BBox Publisher ====================

    def setup_bbox_publisher(self):
        """Publish object bounding box dimensions to ROS2 as JSON on 'objects_bbox_sim'.

        Creates an OmniGraph with a ROS2Publisher (std_msgs/String) and a physics
        callback that periodically publishes a JSON dict mapping object names to
        their world-aligned bounding box sizes {name: {sx, sy, sz}}.
        Dimensions are in meters.
        """
        import omni.physx
        import omni.graph.core as og

        print("Setting up BBox Publisher...")

        stage = omni.usd.get_context().get_stage()
        objects_root = stage.GetPrimAtPath("/World/Objects")
        if not objects_root or not objects_root.IsValid():
            print("Error: /World/Objects does not exist. Add objects first.")
            return

        graph_path = "/Graph/ActionGraph_objects_bbox"
        existing = stage.GetPrimAtPath(graph_path)
        if existing and existing.IsValid():
            print(f"BBox publisher already exists at {graph_path}")
            return

        # Clean up previous
        self._stop_bbox_publisher()

        keys = og.Controller.Keys
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("tick", "omni.graph.action.OnPlaybackTick"),
                    ("context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
                ],
                keys.CONNECT: [
                    ("tick.outputs:tick", "publisher.inputs:execIn"),
                    ("context.outputs:context", "publisher.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("publisher.inputs:messageName", "String"),
                    ("publisher.inputs:messagePackage", "std_msgs"),
                    ("publisher.inputs:topicName", "objects_bbox_sim"),
                ],
            },
        )

        self._bbox_graph_path = graph_path
        self._bbox_pub_attr_path = f"{graph_path}/publisher.inputs:data"
        self._bbox_publish_counter = 0

        # Publish at physics rate but only update the JSON every ~1s
        self._bbox_physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step_bbox
        )

        print(f"BBox publisher created at {graph_path}, topic: objects_bbox_sim")

    def _on_physics_step_bbox(self, dt):
        """Physics callback — publish object bbox dimensions as JSON.

        Uses mesh extent × scale to get rotation-invariant physical dimensions
        (not the world-aligned AABB which inflates with yaw rotation).
        """
        import json
        import omni.graph.core as og

        # Only update every ~60 steps (~1s at 60Hz) — dimensions are static
        self._bbox_publish_counter += 1
        if self._bbox_publish_counter % 60 != 1:
            return

        try:
            stage = omni.usd.get_context().get_stage()
            objects_root = stage.GetPrimAtPath("/World/Objects")
            if not objects_root or not objects_root.IsValid():
                return

            bbox_dict = {}

            for child in objects_root.GetChildren():
                name = child.GetName()
                if name == "PhysicsMaterial":
                    continue
                prim_path = f"/World/Objects/{name}/{name}"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue

                # Read intrinsic mesh extent (pre-transform, in mesh units)
                extent_attr = prim.GetAttribute("extent")
                if not extent_attr or not extent_attr.HasValue():
                    continue
                extent = extent_attr.Get()
                ex = float(extent[1][0] - extent[0][0])
                ey = float(extent[1][1] - extent[0][1])
                ez = float(extent[1][2] - extent[0][2])

                # Apply prim scale (mesh mm -> meters)
                xf = UsdGeom.Xformable(prim)
                sx = sy = sz = 1.0
                for op in xf.GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:scale":
                        s = op.Get()
                        sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
                        break

                bbox_dict[name] = {
                    "sx": round(ex * sx, 5),
                    "sy": round(ey * sy, 5),
                    "sz": round(ez * sz, 5),
                }

            json_str = json.dumps(bbox_dict)
            og.Controller.set(
                og.Controller.attribute(self._bbox_pub_attr_path),
                json_str
            )
        except Exception as e:
            print(f"[BBox callback] {e}")

    def _stop_bbox_publisher(self):
        """Stop the bbox physics callback."""
        if self._bbox_physx_sub is not None:
            self._bbox_physx_sub = None
        self._bbox_graph_path = None
        self._bbox_pub_attr_path = None

    # ==================== Tool handlers (_cmd_*) ====================
    # Exposed as soarm101.<name> through omni-kit-mcp by isaac/soarm101_tools;
    # the name -> method table lives in tool_table.py.

    async def _wait_physx_warmup(self, max_frames=900):
        """Await frames until SimulationManager finishes physics warmup after play (~15 s cap).

        Ported from isaac-sim-mcp 5fc3464 / b6485d4 (ur5e-dt): mutating the stage while PhysX is
        still warming up after timeline.play() wedged the main thread. Awaits frames (never
        pumps) so it is safe inside a bridge handler."""
        app = omni.kit.app.get_app()
        await app.next_update_async()
        try:
            from isaacsim.core.simulation_manager import SimulationManager
        except ImportError:
            for _ in range(120):
                await app.next_update_async()
            return
        for _ in range(max_frames):
            if not SimulationManager._warmup_needed:
                return
            await app.next_update_async()
        carb.log_warn("[SO-ARM101-DT] PhysX warmup still pending after "
                      f"{max_frames} frames; continuing")

    async def _cmd_play_scene(self) -> Dict[str, Any]:
        """Start/resume the simulation timeline and wait out PhysX warmup."""
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            await self._wait_physx_warmup()
            return {
                "status": "success",
                "message": "Simulation started"
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to start simulation: {str(e)}"
            }

    def _cmd_stop_scene(self) -> Dict[str, Any]:
        """Stop the simulation timeline."""
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            return {
                "status": "success",
                "message": "Simulation stopped"
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to stop simulation: {str(e)}"
            }

    # ==================== Scene Management Handlers ====================

    def _get_scene_state_dir(self) -> str:
        """Get the directory for scene state JSON files.

        Uses the output dir pushed by MCP server, or falls back to default resources dir.
        """
        if self._output_dir:
            scene_dir = os.path.join(self._output_dir, "resources", "scene_states")
        else:
            scene_dir = os.path.join(RESOURCES_DIR, "scene_states")
        os.makedirs(scene_dir, exist_ok=True)
        return scene_dir

    def _get_latest_scene_state_path(self) -> str:
        """Find the most recent scene state file by timestamp in the filename."""
        scene_dir = self._get_scene_state_dir()
        import glob as glob_mod
        files = glob_mod.glob(os.path.join(scene_dir, "scene_state_*.json"))
        if not files:
            return None
        files.sort()
        return files[-1]

    def _get_prim_path(self, object_name: str, folder_path: str = "/World/Objects") -> str:
        """Get the full nested prim path for an object's body prim.

        Convention: f"{folder_path}/{object_name}/{object_name}" (legacy
        single-instance case where wrapper name matches inner body name).
        Falls back to the first child of the wrapper that has
        UsdPhysics.RigidBodyAPI applied — needed for the multi-instance
        case where Cloner produces wrappers like "red_2x3_a" wrapping a
        USD whose inner body is still "red_2x3".
        """
        legacy = f"{folder_path}/{object_name}/{object_name}"
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath(legacy).IsValid():
            return legacy
        wrapper = stage.GetPrimAtPath(f"{folder_path}/{object_name}") if stage else None
        if wrapper and wrapper.IsValid():
            for child in wrapper.GetChildren():
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(child.GetPath())
        return legacy

    def _resolve_output_dir(self, output_dir=None):
        """Update internal output_dir if provided by MCP server."""
        if output_dir:
            self._output_dir = os.path.abspath(output_dir)

    def _read_prim_pose(self, prim_path: str) -> Dict[str, Any]:
        """Helper method to read pose (position, quaternion, scale) from a prim."""
        try:
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)

            if not prim.IsValid():
                return None

            translate_attr = prim.GetAttribute("xformOp:translate")
            rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
            orient_attr = prim.GetAttribute("xformOp:orient")
            scale_attr = prim.GetAttribute("xformOp:scale")

            position = {"x": 0.0, "y": 0.0, "z": 0.0}
            rotation = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
            scale = {"x": 1.0, "y": 1.0, "z": 1.0}

            if translate_attr and translate_attr.IsValid():
                v = translate_attr.Get()
                if v:
                    position = {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

            if rotate_attr and rotate_attr.IsValid():
                v = rotate_attr.Get()
                if v:
                    rotation = {"rx": float(v[0]), "ry": float(v[1]), "rz": float(v[2])}
            elif orient_attr and orient_attr.IsValid():
                v = orient_attr.Get()
                if v:
                    qw, qx, qy, qz = float(v.GetReal()), *[float(c) for c in v.GetImaginary()]
                    euler = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)
                    rotation = {"rx": float(euler[0]), "ry": float(euler[1]), "rz": float(euler[2])}

            if scale_attr and scale_attr.IsValid():
                v = scale_attr.Get()
                if v:
                    scale = {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

            return {
                "position": position,
                "rotation": rotation,
                "scale": scale
            }
        except Exception as e:
            carb.log_error(f"Error reading pose for {prim_path}: {e}")
            return None

    def _write_prim_pose(self, prim_path: str, pose_data: Dict[str, Any]) -> bool:
        """Helper method to write pose (position, quaternion, scale) to a prim."""
        try:
            from pxr import Gf

            pos = pose_data.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
            rot = pose_data.get("rotation", {"rx": 0.0, "ry": 0.0, "rz": 0.0})
            scale = pose_data.get("scale", {"x": 1.0, "y": 1.0, "z": 1.0})

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                carb.log_error(f"Prim not found: {prim_path}")
                return False

            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr:
                translate_attr.Set(Gf.Vec3d(pos["x"], pos["y"], pos["z"]))

            rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
            if rotate_attr:
                rotate_attr.Set(Gf.Vec3f(rot["rx"], rot["ry"], rot["rz"]))

            scale_attr = prim.GetAttribute("xformOp:scale")
            if scale_attr:
                scale_attr.Set(Gf.Vec3d(scale["x"], scale["y"], scale["z"]))

            return True
        except Exception as e:
            carb.log_error(f"Error writing pose for {prim_path}: {e}")
            traceback.print_exc()
            return False

    def _cmd_sort_objects(self, color: str = None) -> Dict[str, Any]:
        """MCP handler for sorting blocks by color."""
        try:
            color = color or None  # normalize empty string from MCP default
            if color and color not in BLOCK_COLORS:
                return {"status": "error", "message": f"Unknown color '{color}'. Valid: {list(BLOCK_COLORS.keys())}"}
            sorted_count, _ = self.sort_objects(color=color)
            return {"status": "success", "message": f"Sorted {sorted_count} blocks (color={color or 'all'})"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": f"Failed to sort objects: {str(e)}"}

    def _cmd_randomize_object_poses(self, color: str = None,
                                     required_visible_color: str = None,
                                     required_visible_count: int = 1,
                                     camera_fov_margin_px: int = 30) -> Dict[str, Any]:
        """MCP handler for randomizing block poses, optionally filtered by color
        and/or constrained to keep N target-color legos in workspace-camera FOV."""
        try:
            color = color or None  # normalize empty string from MCP default
            required_visible_color = required_visible_color or None
            if color and color not in BLOCK_COLORS:
                return {"status": "error", "message": f"Unknown color '{color}'. Valid: {list(BLOCK_COLORS.keys())}"}
            if required_visible_color and required_visible_color not in BLOCK_COLORS:
                return {"status": "error",
                        "message": f"Unknown required_visible_color '{required_visible_color}'. "
                                   f"Valid: {list(BLOCK_COLORS.keys())}"}
            randomized, _ = self.randomize_object_poses(
                color=color,
                required_visible_color=required_visible_color,
                required_visible_count=int(required_visible_count or 1),
                camera_fov_margin_px=int(camera_fov_margin_px or 30),
            )
            visible_msg = (f", visible={required_visible_count}×{required_visible_color}"
                           if required_visible_color else "")
            return {
                "status": "success",
                "message": f"Randomized {randomized} blocks (color={color or 'all'}{visible_msg})"
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": f"Failed to randomize object poses: {str(e)}"}

    def _cmd_randomize_single_object(self, object_name: str = None) -> Dict[str, Any]:
        """MCP handler for randomizing a single object's pose."""
        if not object_name:
            return {"status": "error", "message": "object_name is required"}
        try:
            self.randomize_single_object(object_name)
            return {
                "status": "success",
                "message": f"Randomized '{object_name}' in clear workspace area"
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": f"Failed to randomize '{object_name}': {str(e)}"}

    def _cmd_save_scene_state(self, json_file_path: str = None, output_dir: str = None) -> Dict[str, Any]:
        """Save scene state (object poses) to a JSON file.

        If json_file_path is provided, saves to that filename inside scene_states dir.
        Otherwise creates a timestamped file.

        Args:
            json_file_path: Optional filename (e.g. 'assembled.json'). If omitted, timestamped.
            output_dir: Optional output directory pushed by MCP server.

        Returns:
            Dictionary with execution result.
        """
        try:
            from pxr import UsdGeom
            from datetime import datetime

            self._resolve_output_dir(output_dir)
            scene_dir = self._get_scene_state_dir()
            if json_file_path:
                if not json_file_path.endswith(".json"):
                    json_file_path += ".json"
                save_path = os.path.join(scene_dir, os.path.basename(json_file_path))
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(scene_dir, f"scene_state_{timestamp}.json")

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return {
                    "status": "error",
                    "message": "No stage is currently open"
                }

            objects_prim = stage.GetPrimAtPath("/World/Objects")
            if not objects_prim.IsValid():
                return {
                    "status": "error",
                    "message": "/World/Objects path does not exist"
                }

            children = objects_prim.GetChildren()
            object_names = [child.GetName() for child in children if child.IsA(UsdGeom.Xformable)]

            if not object_names:
                return {
                    "status": "error",
                    "message": "No objects found in /World/Objects"
                }

            poses = {}
            saved_count = 0
            failed_names = []

            for object_name in object_names:
                prim_path = self._get_prim_path(object_name)
                pose = self._read_prim_pose(prim_path)
                if pose:
                    poses[object_name] = pose
                    saved_count += 1
                else:
                    failed_names.append(object_name)

            with open(save_path, 'w') as f:
                json.dump(poses, f, indent=4)

            abs_path = os.path.abspath(save_path)
            msg = f"[save_state] Saved {saved_count} objects to {abs_path}"
            if failed_names:
                msg += f", {len(failed_names)} failed: {failed_names}"
            print(msg)

            return {
                "status": "success",
                "message": msg,
                "saved_count": saved_count,
                "failed_names": failed_names,
                "json_file_path": save_path
            }

        except Exception as e:
            print(f"[save_state] Error: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to save scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _cmd_restore_scene_state(self, json_file_path: str = None, output_dir: str = None) -> Dict[str, Any]:
        """Restore scene state (object poses) from a JSON file.

        If json_file_path is provided, loads that specific file from scene_states dir.
        Otherwise finds the most recent timestamped save.

        Args:
            json_file_path: Optional filename (e.g. 'assembled.json'). If omitted, uses latest.
            output_dir: Optional output directory pushed by MCP server.

        Returns:
            Dictionary with execution result.
        """
        try:
            self._resolve_output_dir(output_dir)

            if json_file_path:
                if not json_file_path.endswith(".json"):
                    json_file_path += ".json"
                restore_path = os.path.join(self._get_scene_state_dir(), os.path.basename(json_file_path))
                if not os.path.exists(restore_path):
                    return {
                        "status": "error",
                        "message": f"Scene state file not found: {restore_path}"
                    }
                latest_path = restore_path
            else:
                latest_path = self._get_latest_scene_state_path()

            if not latest_path:
                return {
                    "status": "error",
                    "message": f"No scene state files found in {self._get_scene_state_dir()}"
                }

            try:
                with open(latest_path, 'r') as f:
                    poses = json.load(f)
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Invalid JSON file: {str(e)}"
                }

            if not poses:
                return {
                    "status": "error",
                    "message": "No objects found in scene state file"
                }

            restored_count = 0
            failed_names = []

            for object_name, pose_data in poses.items():
                prim_path = self._get_prim_path(object_name)
                if self._write_prim_pose(prim_path, pose_data):
                    restored_count += 1
                else:
                    failed_names.append(object_name)

            msg = f"[restore_state] Restored {restored_count} objects from {latest_path}"
            if failed_names:
                msg += f", {len(failed_names)} failed: {failed_names}"
            print(msg)

            return {
                "status": "success",
                "message": msg,
                "restored_count": restored_count,
                "failed_names": failed_names,
                "json_file_path": latest_path
            }

        except Exception as e:
            print(f"[restore_state] Error: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to restore scene state: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _cmd_add_objects(self) -> Dict[str, Any]:
        """MCP handler for adding lego block objects to the scene."""
        try:
            added = self.add_objects()
            if added is False:
                return {
                    "status": "success",
                    "message": "All blocks already exist in scene, skipping",
                }
            return {
                "status": "success",
                "message": f"Lego blocks added from {_get_lego_folder()}",
                "folder_path": _get_lego_folder()
            }
        except Exception as e:
            carb.log_error(f"Error in add_objects: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to add objects: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _cmd_delete_objects(self) -> Dict[str, Any]:
        """MCP handler for deleting objects from the scene.

        Returns:
            Dictionary with execution result.
        """
        try:
            self.delete_objects()
            return {
                "status": "success",
                "message": "Objects deleted from /World/Objects"
            }
        except Exception as e:
            carb.log_error(f"Error in delete_objects: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to delete objects: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _cmd_setup_pose_publisher(self) -> Dict[str, Any]:
        """MCP handler for setting up pose publisher action graph.

        Returns:
            Dictionary with execution result.
        """
        try:
            self.create_pose_publisher()
            return {
                "status": "success",
                "message": "Pose publisher action graph created at /Graph/ActionGraph_objects_poses"
            }
        except Exception as e:
            carb.log_error(f"Error in setup_pose_publisher: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to setup pose publisher: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _cmd_sync_real_poses(self) -> Dict[str, Any]:
        """MCP handler for syncing real object poses to sim."""
        try:
            self.sync_real_poses()
            return {"status": "success", "message": "Synced real poses to sim"}
        except Exception as e:
            carb.log_error(f"Error in sync_real_poses: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    async def _cmd_load_scene(self) -> Dict[str, Any]:
        """MCP handler: awaits load_scene (same as UI button)."""
        try:
            await self.load_scene()
            return {"status": "success", "message": "Scene loaded (physics, ground plane, frame rate)"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def _cmd_load_robot(self) -> Dict[str, Any]:
        """MCP handler: awaits load_robot (same as UI button)."""
        try:
            await self.load_robot()
            return {
                "status": "success",
                "message": "SO-ARM101 loaded with 5 arm joints + integrated gripper at /World/SO_ARM101"
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def _cmd_setup_action_graph(self) -> Dict[str, Any]:
        """MCP handler: awaits setup_action_graph (same as UI button)."""
        try:
            await self.setup_action_graph()
            return {"status": "success", "message": "ROS2 action graph created for SO-ARM101 joint control"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_setup_force_publisher(self) -> Dict[str, Any]:
        """MCP handler for setting up the force publisher."""
        try:
            self.setup_force_publish_action_graph()
            return {"status": "success", "message": "Force publisher action graph created for SO-ARM101"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_setup_bbox_publisher(self) -> Dict[str, Any]:
        """MCP handler for setting up the bbox publisher."""
        try:
            self.setup_bbox_publisher()
            return {"status": "success", "message": "BBox publisher created, topic: objects_bbox_sim"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_setup_wrist_camera_action_graph(self) -> Dict[str, Any]:
        """MCP handler for setting up the wrist camera action graph."""
        try:
            self.setup_wrist_camera_action_graph()
            return {"status": "success", "message": "Wrist camera action graph created, topics: wrist_camera_rgb_sim, wrist_camera_info"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_add_cups(self) -> Dict[str, Any]:
        """MCP handler for adding sorting cups."""
        try:
            self.add_cups()
            cup_info = self._get_cup_positions()
            positions_summary = {
                k: [v["center_x"], v["center_y"], v["rim_z"]]
                for k, v in cup_info.items()
            }
            return {
                "status": "success",
                "message": f"Added {len(cup_info)} cups in arc",
                "cup_positions": positions_summary
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _get_cup_body_center_offset(self, cup_prim):
        """Half of the cup's z-extent in its own local frame.

        Used to lift each ``drop_{id}`` wrapper prim from the cup's
        base-center (cup prim's own origin) to the cup's body center.
        This matches the lego convention where ``/objects_poses_sim``
        publishes each block's CENTER pose (not its base) — so downstream
        consumers can treat lego and cup poses the same way.

        Returns the half-extent in meters. Falls back to 0.0965/2 (known
        cup STL height ≈ 97 mm) if the bbox query fails.
        """
        try:
            from pxr import UsdGeom, Usd
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                      [UsdGeom.Tokens.default_])
            bbox = cache.ComputeUntransformedBound(cup_prim)
            r = bbox.ComputeAlignedRange()
            z_extent = r.GetMax()[2] - r.GetMin()[2]
            if z_extent > 1e-6:
                return z_extent / 2.0
        except Exception:
            pass
        return 0.0965 / 2.0

    def _compute_drop_wrapper_xform(self, cup_prim, half_height=None):
        """Compose cup's L2W with a local (0,0,+half_height) translation.

        The result is a 4x4 transform whose translation is the cup body
        center in world, and whose rotation equals the cup's world
        orientation. This is what the ``drop_{id}`` wrapper prim should
        hold so /drop_poses publishes cup-body-center pose (matching the
        lego convention).
        """
        from pxr import UsdGeom, Gf
        cup_L2W = UsdGeom.Xformable(cup_prim).ComputeLocalToWorldTransform(0)
        if half_height is None:
            half_height = self._get_cup_body_center_offset(cup_prim)
        offset_local = Gf.Matrix4d().SetTranslate(
            Gf.Vec3d(0.0, 0.0, float(half_height)))
        # Row-vector convention: p_world = p_local * offset_local * cup_L2W
        return offset_local * cup_L2W

    def _cmd_publish_drop_poses(self) -> Dict[str, Any]:
        """MCP handler: publish each cup's BODY-CENTER pose to /drop_poses.

        Creates wrapper Xform prims (drop_0, drop_1, drop_2), each whose
        world transform = cup_prim.L2W composed with a local +Z offset of
        half the cup's height. Then builds action graph
        /Graph/ActionGraph_drop_poses with ROS2PublishTransformTree targeting
        the wrapper prims so child_frame_ids become drop_0/1/2.

        Semantic contract: /drop_poses transforms encode cup BODY CENTER
        in world (same convention as /objects_poses_sim for legos — the
        published pose is the object's center, not its base or any marker
        surface). Downstream consumers in the VLA package are responsible
        for applying any rim/hover offsets they need.

        Historical note: prior to 2026-04-21 this published the ArUco
        marker mesh's world transform. That equaled cup-base-center only
        because the Blender-baked marker USDs happened to have identity
        local xforms — a latent coupling that would silently break if the
        marker USDs were ever regenerated with a non-zero mesh origin.
        Reading the cup prim directly removes that coupling and aligns the
        sim-side contract with what aruco_camera_localizer publishes on
        the real-world side (also cup-body-center, after it applies its
        marker→cup transform).

        Requires cups to be present in the scene (add_cups must be called first).
        """
        try:
            import omni.usd
            import omni.graph.core as og
            from pxr import Sdf, UsdGeom, Gf

            graph_path = "/Graph/ActionGraph_drop_poses"
            topic_name = "drop_poses"
            # parent_prim's BASENAME becomes the published TF header.frame_id.
            # Use "/World/base" (identity-xform Xform at world origin) so frame_id="base"
            # — matches /ee_pose convention and the URDF base_link, since SO-ARM101's
            # base is co-located with World origin.
            parent_prim = "/World/base"

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return {"status": "error", "message": "No stage found"}

            # Ensure /World/base exists as an identity-xform Xform (idempotent).
            base_prim = stage.GetPrimAtPath(parent_prim)
            if not (base_prim and base_prim.IsValid()):
                UsdGeom.Xform.Define(stage, parent_prim)

            # Color → aruco_id mapping is kept only for frame-id naming
            # (drop_{aruco_id}) and for aruco_camera_localizer symmetry on
            # the real-world side — the sim reads cup prims directly now.
            color_to_id = CUP_ARUCO_CONFIG["ids"]  # {red:3, green:2, blue:1}

            def _write_wrapper(color, aruco_id, require_wrapper_exists):
                """Compute cup-body-center world xform and write it onto
                the drop_{aruco_id} wrapper. Returns wrapper_path or None."""
                cup_path = f"/World/Containers/cup_{color}"
                cup_prim = stage.GetPrimAtPath(cup_path)
                if not (cup_prim and cup_prim.IsValid()):
                    return None
                wrapper_path = f"/World/Containers/drop_{aruco_id}"
                wrapper_prim = stage.GetPrimAtPath(wrapper_path)
                if require_wrapper_exists and not (
                        wrapper_prim and wrapper_prim.IsValid()):
                    return None
                if not (wrapper_prim and wrapper_prim.IsValid()):
                    wrapper_prim = UsdGeom.Xform.Define(
                        stage, wrapper_path).GetPrim()
                half_h = self._get_cup_body_center_offset(cup_prim)
                # Cache for the live-sync callback — avoids recomputing the
                # bbox every frame.
                self._cup_half_heights.setdefault(color, float(half_h))
                world_xform = self._compute_drop_wrapper_xform(
                    cup_prim, half_height=half_h)
                wx = UsdGeom.Xformable(wrapper_prim)
                wx.ClearXformOpOrder()
                wx.AddTransformOp().Set(world_xform)
                return wrapper_path

            # Initialize / reset per-cup half-height cache (populated by
            # _write_wrapper below and consumed by the live-sync callback).
            self._cup_half_heights = {}

            # If graph already exists, just refresh wrapper transforms + ensure
            # the live-sync callback is installed. This lets callers re-trigger
            # the publisher to pick up moved cups without rebuilding the graph.
            existing = stage.GetPrimAtPath(graph_path)
            if existing and existing.IsValid():
                refreshed = [wp for wp in
                             (_write_wrapper(c, i, require_wrapper_exists=True)
                              for c, i in color_to_id.items())
                             if wp is not None]
                self._install_drop_pose_live_sync()
                return {"status": "success",
                        "message": (f"Graph exists — refreshed "
                                    f"{len(refreshed)} wrappers + live-sync "
                                    f"active (cup body-center semantics)")}

            # Build wrapper Xform prims at each cup's body-center world pose
            target_prims = []
            found_colors = []
            for color, aruco_id in color_to_id.items():
                wrapper_path = _write_wrapper(
                    color, aruco_id, require_wrapper_exists=False)
                if wrapper_path is None:
                    print(f"[drop_poses] Warning: cup_{color} not found — "
                          f"skipping {color}")
                    continue
                target_prims.append(wrapper_path)
                found_colors.append(color)
                print(f"[drop_poses] Created wrapper {wrapper_path} at "
                      f"cup_{color} body-center")

            if not target_prims:
                return {
                    "status": "error",
                    "message": "No ArUco marker meshes found. Add cups first (add_cups / Add Cups button)."
                }

            # Create the action graph — same pattern as ActionGraph_objects_poses
            keys = og.Controller.Keys
            (graph, nodes, _, _) = og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                        ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("isaac_read_simulation_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("ros2_publish_transform_tree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ],
                    keys.CONNECT: [
                        ("on_playback_tick.outputs:tick", "ros2_publish_transform_tree.inputs:execIn"),
                        ("ros2_context.outputs:context", "ros2_publish_transform_tree.inputs:context"),
                        ("isaac_read_simulation_time.outputs:simulationTime", "ros2_publish_transform_tree.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        ("ros2_publish_transform_tree.inputs:topicName", topic_name),
                        ("isaac_read_simulation_time.inputs:resetOnStop", False),
                    ],
                },
            )

            # Wire USD relationships (must be done after graph creation)
            ros2_node_path = f"{graph_path}/ros2_publish_transform_tree"
            ros2_prim = stage.GetPrimAtPath(ros2_node_path)

            if not ros2_prim.IsValid():
                return {"status": "error", "message": f"Could not find ROS2 node at {ros2_node_path}"}

            parent_rel = ros2_prim.GetRelationship("inputs:parentPrim")
            if not parent_rel:
                parent_rel = ros2_prim.CreateRelationship("inputs:parentPrim", custom=True)
            parent_rel.SetTargets([Sdf.Path(parent_prim)])

            target_rel = ros2_prim.GetRelationship("inputs:targetPrims")
            if not target_rel:
                target_rel = ros2_prim.CreateRelationship("inputs:targetPrims", custom=True)
            target_rel.SetTargets([Sdf.Path(p) for p in target_prims])

            print(f"[drop_poses] Action graph created at {graph_path}")
            print(f"[drop_poses] Publishing {len(target_prims)} drop poses to topic: /{topic_name}")
            print(f"[drop_poses] Cups found: {found_colors}")

            # Install a per-frame callback that copies each ArUco marker's
            # live world transform into its wrapper prim. Without this, cups
            # moved after setup would not be reflected in /drop_poses (the
            # wrappers above are snapshotted at setup time only).
            self._install_drop_pose_live_sync()

            return {
                "status": "success",
                "message": f"Publishing /drop_poses for cups: {found_colors} (live sync active)",
                "graph_path": graph_path,
                "target_prims": target_prims,
            }

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _install_drop_pose_live_sync(self):
        """Register a per-frame update callback that syncs each drop_{id}
        wrapper prim's transform from its source CUP prim.

        The ROS2PublishTransformTree node in /Graph/ActionGraph_drop_poses
        publishes the wrappers' world poses on every playback tick. Without
        this sync, those poses are frozen at publish_drop_poses setup time.
        With this sync, cup movement in Isaac is reflected in /drop_poses
        one frame later — matching the live behavior of /objects_poses_sim
        for legos.

        Per-cup half-heights are cached in self._cup_half_heights (populated
        at setup time in _cmd_publish_drop_poses) so the callback doesn't
        pay a BBoxCache query on every frame.

        Idempotent: second+ calls are no-ops (subscription already held).
        """
        if getattr(self, "_drop_pose_live_sync_sub", None) is not None:
            return
        try:
            import omni.kit.app
            import omni.usd

            def _on_update(e):
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                color_to_id = CUP_ARUCO_CONFIG["ids"]
                for color, aruco_id in color_to_id.items():
                    cup_path = f"/World/Containers/cup_{color}"
                    wrapper_path = f"/World/Containers/drop_{aruco_id}"
                    cup_prim = stage.GetPrimAtPath(cup_path)
                    wrap_prim = stage.GetPrimAtPath(wrapper_path)
                    if not (cup_prim and cup_prim.IsValid()
                            and wrap_prim and wrap_prim.IsValid()):
                        continue
                    try:
                        half_h = self._cup_half_heights.get(color)
                        if half_h is None:
                            half_h = self._get_cup_body_center_offset(cup_prim)
                            self._cup_half_heights[color] = float(half_h)
                        wx_src = self._compute_drop_wrapper_xform(
                            cup_prim, half_height=half_h)
                        from pxr import UsdGeom
                        wx_dst = UsdGeom.Xformable(wrap_prim)
                        wx_dst.ClearXformOpOrder()
                        wx_dst.AddTransformOp().Set(wx_src)
                    except Exception:
                        pass

            app = omni.kit.app.get_app()
            self._drop_pose_live_sync_sub = (
                app.get_update_event_stream().create_subscription_to_pop(
                    _on_update, name="soarm101_drop_pose_live_sync"))
            print("[drop_poses] Live-sync callback installed (per-frame, "
                  "cup body-center)")
        except Exception as e:
            print(f"[drop_poses] Failed to install live sync: {e}")

    def _cmd_delete_cups(self) -> Dict[str, Any]:
        """MCP handler for deleting sorting cups."""
        try:
            self.delete_cups()
            return {"status": "success", "message": "Cups deleted"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_sort_into_cups(self, color: str = None) -> Dict[str, Any]:
        """MCP handler for sorting blocks into their matching color cups."""
        try:
            color = color or None
            count = self.sort_into_cups(color=color)
            return {
                "status": "success",
                "message": f"Sorted {count} blocks into cups (color={color or 'all'})"
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_update_cups(self) -> Dict[str, Any]:
        """MCP handler: reposition cups to current CUP_LAYOUT settings.

        Delegates to _add_cups_from_ui, which now teleports existing cups
        via RigidPrim rather than delete/add (action graph preserved).
        Falls back to delete+add on first-time creation.
        """
        try:
            self._add_cups_from_ui()
            return {"status": "success",
                    "message": "Cups updated from CUP_LAYOUT"}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_randomize_cups(self,
                             radius_min: float = 0.22,
                             radius_max: float = 0.30,
                             angle_min_deg: float = -50.0,
                             angle_max_deg: float = 50.0,
                             gap_min: float = 0.005,
                             gap_max: float = 0.030,
                             modes: list = None,
                             randomize_order: bool = True,
                             yaw_jitter_deg: float = 15.0,
                             cup_lego_clearance: float = 0.04,
                             cup_robot_clearance: float = 0.015,
                             max_attempts: int = 200,
                             seed: int = None) -> Dict[str, Any]:
        """MCP handler: randomize CUP_LAYOUT params (mode/radius/angle/gap),
        retry until a placement clears legos and robot-link AABBs.

        Thin wrapper over self.randomize_cups — see that docstring for the
        full parameter and constraint description.
        """
        try:
            return self.randomize_cups(
                radius_min=radius_min, radius_max=radius_max,
                angle_min_deg=angle_min_deg,
                angle_max_deg=angle_max_deg,
                gap_min=gap_min, gap_max=gap_max,
                modes=tuple(modes) if modes else ("arc", "line"),
                randomize_order=randomize_order,
                yaw_jitter_deg=yaw_jitter_deg,
                cup_lego_clearance=cup_lego_clearance,
                cup_robot_clearance=cup_robot_clearance,
                max_attempts=max_attempts, seed=seed,
            )
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def _cmd_quick_start(self) -> Dict[str, Any]:
        """MCP handler: full scene setup."""
        try:
            await self.quick_start()
            return {"status": "success", "message": "Quick start complete: scene, robot, action graphs, cameras, objects, cups, publishers, simulation running."}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    # --- Phase 13: ports from ur5e-dt ---

    async def _cmd_new_stage(self) -> Dict[str, Any]:
        """Clear the USD stage and await a few app frames so Hydra/PhysX tear down cleanly.

        Awaits frames instead of calling app.update(): a synchronous pump from inside a
        handler running on Kit's loop re-enters that loop."""
        try:
            # Stop our viewport publishers before tearing down the stage they reference
            try:
                _viewport_pub.stop_all()
            except Exception:
                pass
            # Stop recording if active
            try:
                if _recording_mgr.is_recording():
                    self._cmd_stop_recording()
            except Exception:
                pass
            import omni.usd, omni.kit.app
            await omni.usd.get_context().new_stage_async()
            app = omni.kit.app.get_app()
            for _ in range(30):
                await app.next_update_async()
            # Clear World singleton so next quick_start re-initializes fresh
            try:
                World.clear_instance()
            except Exception:
                pass
            return {"status": "success", "message": "Stage cleared. Call quick_start to rebuild."}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_start_viewport_publisher(self, camera_prim_path: str = None, topic_name: str = None,
                                       frame_id: str = None) -> Dict[str, Any]:
        if not camera_prim_path or not topic_name:
            return {"status": "error", "message": "camera_prim_path and topic_name are required"}
        try:
            _viewport_pub.start(camera_prim_path, topic_name, frame_id=frame_id or None)
            return {"status": "success", "message": f"Publishing {camera_prim_path} → /{topic_name}",
                    "status_snapshot": _viewport_pub.get_status()}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_stop_viewport_publisher(self, topic_name: str = None) -> Dict[str, Any]:
        if not topic_name:
            return {"status": "error", "message": "topic_name is required"}
        try:
            _viewport_pub.stop(topic_name)
            return {"status": "success", "message": f"Stopped /{topic_name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _cmd_list_viewport_publishers(self) -> Dict[str, Any]:
        return {"status": "success", "publishers": _viewport_pub.get_status()}

    def _cmd_start_recording(self, file_name: str = None, camera: str = None,
                              fps: int = 30) -> Dict[str, Any]:
        """Start real-time viewport recording to an MP4 via ffmpeg."""
        try:
            if _recording_mgr.is_recording():
                return {"status": "error", "message": "Recording already active. Call stop_recording first."}
            if not file_name:
                file_name = "realtime_recording.mp4"
            if not file_name.endswith(".mp4"):
                file_name += ".mp4"
            import subprocess
            import omni.kit.viewport.utility as vp_utils

            os.makedirs(VIDEOS_DIR, exist_ok=True)
            output_path = os.path.join(VIDEOS_DIR, file_name)

            viewport = vp_utils.get_active_viewport()
            if camera:
                viewport.camera_path = camera
            w, h = viewport.get_texture_resolution()

            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "18",
                output_path,
            ]
            mgr = _recording_mgr
            mgr.ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            mgr.recording = True
            mgr.output_path = output_path
            mgr.frame_count = 0
            mgr.start_time = time.time()
            mgr.fps = fps
            mgr.last_print = 0

            def on_capture(buffer, buffer_size, width, height, fmt):
                import ctypes
                if not mgr.recording or mgr.ffmpeg_proc is None:
                    return
                try:
                    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                    ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                    raw = (ctypes.c_ubyte * buffer_size).from_address(ptr)
                    mgr.ffmpeg_proc.stdin.write(bytes(raw))
                    mgr.frame_count += 1
                except (BrokenPipeError, OSError):
                    self._cmd_stop_recording()

            def on_frame_change(event):
                if not mgr.recording or mgr.ffmpeg_proc is None:
                    return
                vp_utils.capture_viewport_to_buffer(viewport, on_capture)

            mgr.frame_sub = viewport.subscribe_to_frame_change(on_frame_change)
            mgr.save_state()
            return {"status": "success", "message": f"Recording → {output_path} ({w}x{h} @ {fps}fps)",
                    "output_path": output_path, "resolution": [w, h], "fps": fps}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_stop_recording(self) -> Dict[str, Any]:
        try:
            mgr = _recording_mgr
            if not mgr.is_recording():
                return {"status": "error", "message": "No recording is active"}
            if mgr.frame_sub is not None:
                mgr.frame_sub = None
            if mgr.ffmpeg_proc is not None:
                try:
                    mgr.ffmpeg_proc.stdin.close()
                    mgr.ffmpeg_proc.wait(timeout=10)
                except Exception as e:
                    print(f"[Recording] ffmpeg close warning: {e}")
                mgr.ffmpeg_proc = None
            mgr.recording = False
            mgr.clear_state()
            result = {"status": "success", "output_path": mgr.output_path,
                      "frame_count": mgr.frame_count, "duration_s": mgr.frame_count / max(mgr.fps, 1)}
            return result
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _cmd_get_recording_status(self) -> Dict[str, Any]:
        mgr = _recording_mgr
        if not mgr.is_recording():
            return {"status": "success", "recording": False}
        elapsed = time.time() - mgr.start_time if mgr.start_time else 0
        return {"status": "success", "recording": True, "frame_count": mgr.frame_count,
                "elapsed_s": elapsed, "output_path": mgr.output_path, "fps": mgr.fps}

    # ------------------------------------------------------------------
    # Contact sensors — used by scripts/motion_logger.py at 50 Hz to
    # record every physical contact between a cup and any robot link.
    # Gives deterministic ground truth vs. the swept-volume / convex-
    # decomp approximations our offline motion_sweep.py uses (which can
    # miss sub-mm overhang in the convex decomposition pieces).
    # API: isaacsim.sensors.physics.ContactSensor (introspected live —
    # docs point to experimental module that isn't installed in 5.0).
    # ------------------------------------------------------------------

    def _setup_contact_sensors(self):
        """Subscribe directly to PhysX contact-report events via
        `omni.physx.get_physx_simulation_interface().subscribe_contact_report_events`.

        Rationale: the isaacsim.sensors.physics.ContactSensor wrapper
        depends on the `omni.physx.contact` extension, which is not
        present under that name in Isaac Sim 5.0. Every sensor we created
        via that path came back is_valid=False. The underlying PhysX
        simulation interface exposes contact reporting directly; it
        fires for any contact pair where at least one actor has
        PhysxContactReportAPI applied. We just need the API applied on
        the prims we care about (cups + the gripper/jaw/wrist rigid
        bodies) and a subscription to drain events into a deque.

        Reference material: docs/references/contact-sensor/ contains
        the three NVIDIA files that pinned down the wrapper diagnosis.

        Event fields captured per contact (resolved on the physics
        thread — cheap int->SdfPath lookup):
          - type: CONTACT_FOUND / CONTACT_PERSIST / CONTACT_LOST
          - actor0/1 path, collider0/1 path
          - position (world coords), normal, impulse (Vec3)
          - time (sim time at event)

        Idempotent. Safe on hot-reload — drops any previous subscription
        before installing a new one.
        """
        global _CONTACT_SUB, _CONTACT_EVENTS, _CONTACT_WATCHED_PATHS
        try:
            import omni.physx
            from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
            import omni.timeline
        except ImportError as e:
            print(f"[contact_events] SKIP: import unavailable ({e})")
            return

        stage = omni.usd.get_context().get_stage()

        # (1) Apply PhysxContactReportAPI with threshold=0 on all prims we
        # want contact events for. Cups + gripper/jaw/wrist links. If
        # already applied, skip (idempotent). threshold=0 means "report
        # any contact regardless of impulse."
        targets = []
        for color in ("red", "green", "blue"):
            targets.append(f"/World/Containers/cup_{color}")
        for link in ("gripper_link", "moving_jaw_so101_v1_link", "wrist_link"):
            targets.append(f"/World/SO_ARM101/{link}")

        watched = set()
        for path in targets:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                print(f"[contact_events] skip missing prim: {path}")
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"[contact_events] skip {path}: no RigidBodyAPI")
                continue
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                cr.CreateThresholdAttr().Set(0.0)
            else:
                cr = PhysxSchema.PhysxContactReportAPI(prim)
                thr_attr = cr.GetThresholdAttr()
                if thr_attr and (thr_attr.Get() is None or thr_attr.Get() > 0.0):
                    cr.CreateThresholdAttr().Set(0.0)
            watched.add(path)

        _CONTACT_WATCHED_PATHS = watched
        # Back-compat dict — populate so motion_logger sees "configured".
        _CONTACT_SENSORS.clear()
        for path in watched:
            name = path.rsplit("/", 1)[-1]
            _CONTACT_SENSORS[name] = path

        # (2) Drop any existing subscription. The Subscription holder
        # unsubscribes on __del__ / when reassigned.
        if _CONTACT_SUB is not None:
            try:
                _CONTACT_SUB = None
            except Exception:
                pass
        _CONTACT_EVENTS.clear()

        sim_iface = omni.physx.get_physx_simulation_interface()
        tl = omni.timeline.get_timeline_interface()

        def _on_contact(headers, data):
            # Runs on the PhysX thread after each sim step. Keep O(fast).
            try:
                t_now = float(tl.get_current_time())
            except Exception:
                t_now = 0.0
            for h in headers:
                try:
                    a0_int = int(h.actor0); a1_int = int(h.actor1)
                    c0_int = int(h.collider0); c1_int = int(h.collider1)
                    try:
                        a0 = str(PhysicsSchemaTools.intToSdfPath(a0_int))
                        a1 = str(PhysicsSchemaTools.intToSdfPath(a1_int))
                    except Exception:
                        a0, a1 = f"int:{a0_int}", f"int:{a1_int}"
                    try:
                        c0 = str(PhysicsSchemaTools.intToSdfPath(c0_int))
                        c1 = str(PhysicsSchemaTools.intToSdfPath(c1_int))
                    except Exception:
                        c0, c1 = a0, a1

                    # Filter: only events involving at least one watched
                    # prim. Ground plane + own-body self-contacts get
                    # dropped (there will be lots of them).
                    def _is_watched(p):
                        return any(p == w or p.startswith(w + "/") for w in _CONTACT_WATCHED_PATHS)
                    if not (_is_watched(a0) or _is_watched(a1)):
                        continue

                    # Sum contact-data impulses for this header's slice.
                    ix = iy = iz = 0.0
                    px = py = pz = 0.0
                    nc = int(h.num_contact_data)
                    off = int(h.contact_data_offset)
                    if nc > 0 and off >= 0:
                        for k in range(nc):
                            try:
                                d = data[off + k]
                                imp = d.impulse
                                ix += float(imp[0]); iy += float(imp[1]); iz += float(imp[2])
                                pos = d.position
                                px += float(pos[0]); py += float(pos[1]); pos_z = float(pos[2]); pz += pos_z
                            except Exception:
                                continue
                        if nc:
                            px /= nc; py /= nc; pz /= nc

                    ev_type = str(h.type).rsplit(".", 1)[-1]  # "CONTACT_FOUND"
                    _CONTACT_EVENTS.append({
                        "t": t_now,
                        "type": ev_type,
                        "a0": a0, "a1": a1,
                        "c0": c0, "c1": c1,
                        "impulse": [ix, iy, iz],
                        "impulse_mag": (ix * ix + iy * iy + iz * iz) ** 0.5,
                        "position": [px, py, pz],
                        "n_contacts": nc,
                    })
                except Exception:
                    # Swallow per-event errors — don't crash the physics
                    # thread if a single header is malformed.
                    continue

        _CONTACT_SUB = sim_iface.subscribe_contact_report_events(_on_contact)
        print(f"[contact_events] subscribed — watching {len(watched)} prims: "
              f"{sorted(_CONTACT_WATCHED_PATHS)}")

    def _cmd_get_contact_events(self, drain: bool = True,
                                 max_events: int = 256) -> Dict[str, Any]:
        """Drain buffered contact events. Returns a list of dicts, each
        with {t, type, a0, a1, c0, c1, impulse, impulse_mag, position,
        n_contacts}. When drain=True (default) the buffer is emptied;
        when False, a snapshot is returned and events remain.

        Used by motion_logger at 50 Hz to attribute cup displacement to
        specific robot-link / cup contact pairs. See _setup_contact_sensors
        for the subscription mechanism.
        """
        global _CONTACT_EVENTS, _CONTACT_SUB
        events = []
        if drain:
            n = min(len(_CONTACT_EVENTS), max_events)
            for _ in range(n):
                events.append(_CONTACT_EVENTS.popleft())
        else:
            events = list(_CONTACT_EVENTS)[-max_events:]
        return {
            "status": "success",
            "subscribed": _CONTACT_SUB is not None,
            "watched": sorted(_CONTACT_WATCHED_PATHS),
            "events": events,
            "buffer_remaining": len(_CONTACT_EVENTS),
        }

    def on_shutdown(self):
        """Clean shutdown"""
        global _INSTANCE
        if _INSTANCE is self:
            _INSTANCE = None
        self._teardown_log_redirect()
        print("[SO-ARM101-DT] Digital Twin shutdown")

        # Stop physics-rate callbacks
        self._stop_force_publish()
        self._stop_bbox_publisher()

        # Stop viewport publishers (they hold references to viewport/camera prims)
        try:
            _viewport_pub.stop_all()
        except Exception:
            pass

        # Finalize any in-flight video recording
        try:
            if _recording_mgr.is_recording():
                self._cmd_stop_recording()
        except Exception:
            pass

        # Isaac Sim handles ROS2 shutdown automatically
        print("ROS2 bridge shutdown handled by Isaac Sim")

        # Clear UI widget references to break closure references
        self._workspace_checkbox = None
        self._custom_checkbox = None
        self._custom_camera_prim_field = None
        self._custom_camera_topic_field = None
        self._resolution_combo = None
        self._assembly_combo = None

        # Destroy window - clear frame first to break lambda closure references
        if self._window:
            self._window.frame.clear()
            self._window.destroy()
            self._window = None