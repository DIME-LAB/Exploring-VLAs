#!/usr/bin/env python3
# Reference: https://so101-ros2.readthedocs.io/latest/imitation_learning.html
# Reference: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html
# Reference: lerobot/src/lerobot/scripts/lerobot_eval.py + lerobot/src/lerobot/policies/factory.py
"""smolvla_inference.py — closed-loop SmolVLA inference into Isaac Sim.

Subscribes to our existing recording-side topics, runs a SmolVLA checkpoint
at low rate (chunk-and-replan), and dispatches FollowJointTrajectory action
goals to ros2_control at controller rate.

Architecture
------------
Two threads on one rclpy node:
  1. ROS executor thread (rclpy.spin) drains subscriptions into shared state.
  2. Publisher timer (controller rate, default 25 Hz) pops the next action
     from a deque, hard-clamps to URDF limits, dispatches FJT action goals
     to /arm_controller/follow_joint_trajectory + /gripper_controller/...
  3. Inference thread (default 2 Hz) snapshots the latest observation,
     runs policy.predict_action_chunk, swaps the action deque under a lock.

Same topic + unit contract as record_sim_isaac.sh — the model sees exactly
what it was trained on. /joint_states (rad) → state in degrees per
use_degrees=True; model emits action in degrees → degrees back to rad
before publishing.

Model agnosticism
-----------------
Loads any SmolVLA checkpoint from a local path or HF repo. Full FT vs LoRA
auto-detected via adapter_config.json presence — the LoRA path resolves the
base via PeftConfig.base_model_name_or_path and applies PeftModel on top.
Camera-key drift between checkpoints is handled by --rename-map (dataset
key → model feature key); empty placeholder cameras are auto-filled with
black frames of the model-declared shape.

CLI
---
  smolvla_inference.py \
      --model.path=anirudhrani/smolvla_blue_sort_ven_50k \
      --model.checkpoint=050000 \
      --task="Pick a blue lego and place it in blue cup" \
      [--rename-map='{"wrist":"wrist","top":"top"}'] \
      [--inference-rate=2.0] \
      [--publish-rate=25.0] \
      [--device=cuda] \
      [--dry-run]      # load model, run one inference, print, exit (no ROS publish)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

# Imports below the CLI parser are deferred until after we've parsed args
# so --help / --dry-run don't pay the torch+lerobot startup cost.

logger = logging.getLogger("smolvla_inference")


# ---------------------------------------------------------------------------
# Constants — must match training-time + control_gui.py contract
# ---------------------------------------------------------------------------

# Joint order in observation.state / action — matches the so101_ros2 plugin
# default (config_so101_ros2.py:_default_joint_names) and the recorded
# datasets' feature names.
JOINT_NAMES_CANONICAL = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# URDF link name → canonical name. /joint_states publishes "gripper_joint"
# (URDF), but our datasets recorded the canonical "gripper" key.
JOINT_NAME_MAP = {"gripper_joint": "gripper"}

# Hard joint limits in radians — pulled from control_gui.py JOINT_LIMITS.
# Applied after deg→rad conversion and before publishing. Source of truth
# is the URDF; if the URDF changes, sync these.
JOINT_LIMITS_RAD = {
    "shoulder_pan":  (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex":    (-1.69, 1.69),
    "wrist_flex":    (-1.65806, 1.65806),
    "wrist_roll":    (-2.74385, 2.84121),
    # gripper key on the wire — URDF joint is gripper_joint
    "gripper":       (-0.174533, 1.74533),
}

# Default Isaac-Sim-side topic contract (matches record_sim_isaac.sh).
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_CAMERA_TOPICS = {
    "wrist": "/wrist_camera_rgb_sim",
    "top":   "/workspace_camera_sim",
}

# Action client targets — direct to ros2_controllers, bypassing control_gui's
# /joint_commands path (avoids executor saturation; mirrors drive_pick_place.py).
ARM_FJT_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_FJT_ACTION = "/gripper_controller/follow_joint_trajectory"
ARM_JOINTS_URDF = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_URDF = "gripper_joint"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SmolVLA closed-loop inference into Isaac Sim via ros2_control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model.path", dest="model_path", required=True,
                   help="HF repo id (e.g. anirudhrani/smolvla_blue_sort_ven_50k) "
                        "or local path to checkpoint folder containing config.json")
    p.add_argument("--model.checkpoint", dest="model_checkpoint", default=None,
                   help="Subfolder (e.g. 050000) when path is a HF repo with "
                        "checkpoints/<step>/pretrained_model layout. Ignored if path "
                        "already points at a pretrained_model dir.")
    p.add_argument("--device", default="cuda",
                   help="torch device (cuda / cuda:0 / cpu)")
    # Task
    p.add_argument("--task", required=True,
                   help="Natural-language task instruction passed to the VLM.")
    # Topic + rate
    p.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    p.add_argument("--camera-topics", default=json.dumps(DEFAULT_CAMERA_TOPICS),
                   help='JSON dict {feature_key: ros_topic}. Defaults match '
                        'record_sim_isaac.sh: wrist→/wrist_camera_rgb_sim, '
                        'top→/workspace_camera_sim.')
    p.add_argument("--rename-map", default="{}",
                   help='JSON dict mapping dataset feature key → model feature '
                        'key (e.g. {"wrist":"camera1","top":"camera2"} for '
                        'checkpoints trained on renamed keys).')
    p.add_argument("--inference-rate", type=float, default=1.0,
                   help="Hz: how often the model is invoked (chunk replan). "
                        "Default 1.0 matches NVIDIA so101-ros2 reference. "
                        "With chunk_size=50 actions and publish-rate=30 Hz, "
                        "1.0 Hz consumes ~60%% of each chunk (including the "
                        "descend-and-close phase) before replanning.")
    p.add_argument("--publish-rate", type=float, default=30.0,
                   help="Hz: how often actions are dispatched to ros2_control. "
                        "Default 30.0 matches dataset.fps=30 from "
                        "record_sim_isaac.sh — model was trained on 30 Hz "
                        "trajectories so per-step joint deltas only execute "
                        "at the correct real-time speed when publish-rate=30.")
    # Safety + smoke
    p.add_argument("--clamp-joint-limits", action="store_true", default=True,
                   help="Hard-clamp every joint command to URDF JOINT_LIMITS_RAD.")
    p.add_argument("--no-clamp-joint-limits", dest="clamp_joint_limits",
                   action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="Load model, run one inference on dummy obs, print, exit. "
                        "No ROS subscriptions, no action goals.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ---------------------------------------------------------------------------
# Model load — handles full FT and LoRA, both via local path or HF repo
# ---------------------------------------------------------------------------

def _resolve_pretrained_path(model_path: str, checkpoint: Optional[str]) -> str:
    """Return a LOCAL filesystem path to a folder containing config.json.

    Three cases:
      1. Local dir already at the pretrained_model folder (has config.json) → use as-is.
      2. Local dir at the repo root with checkpoints/<step>/pretrained_model
         subfolder → descend.
      3. Otherwise treat as HF repo id and snapshot_download the subfolder
         locally. We can't pass `repo/sub/path` to PreTrainedConfig.from_pretrained
         because huggingface_hub validates repo_id strictly (must be one or
         two segments). Downloading first uniforms the downstream code path.

    Why snapshot_download instead of passing `subfolder=` through every
    from_pretrained call: PolicyProcessorPipeline + PeftConfig + the policy
    class all take `from_pretrained` separately, and the kwarg is named
    differently in some places. Pulling the files locally once and pointing
    every API at the local path is simpler and matches what HF caching
    already does under the hood.
    """
    # ----- Local cases -----
    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "config.json")):
            return os.path.abspath(model_path)
        if checkpoint:
            sub = os.path.join(model_path, "checkpoints", checkpoint,
                               "pretrained_model")
            if os.path.exists(os.path.join(sub, "config.json")):
                return os.path.abspath(sub)
        # Try to find any pretrained_model subfolder if checkpoint not given
        for root, dirs, files in os.walk(model_path):
            if "config.json" in files and root.endswith("pretrained_model"):
                return os.path.abspath(root)
        raise FileNotFoundError(
            f"--model.path={model_path} has no config.json and no "
            f"checkpoints/{checkpoint or '<step>'}/pretrained_model subfolder."
        )

    # ----- HF repo case: download the subfolder we need -----
    from huggingface_hub import snapshot_download
    subfolder = (
        f"checkpoints/{checkpoint}/pretrained_model" if checkpoint else None
    )
    allow_patterns = (
        [f"{subfolder}/*"] if subfolder else
        # Top-level pretrained_model in repo root (no checkpoints/ layout).
        ["config.json", "*.safetensors", "*.json"]
    )
    logger.info(
        "snapshot_download repo=%s subfolder=%s (this caches to "
        "~/.cache/huggingface/hub on first run)",
        model_path, subfolder or "<root>",
    )
    local_root = snapshot_download(
        repo_id=model_path,
        allow_patterns=allow_patterns,
    )
    if subfolder:
        local_path = os.path.join(local_root, subfolder)
    else:
        local_path = local_root
    if not os.path.exists(os.path.join(local_path, "config.json")):
        raise FileNotFoundError(
            f"After snapshot_download({model_path}, subfolder={subfolder!r}) "
            f"the resolved path {local_path} has no config.json. "
            f"Verify --model.checkpoint matches a subfolder in the repo."
        )
    return local_path


def _is_lora_checkpoint(pretrained_path: str) -> bool:
    """Detect LoRA by adapter_config.json existence next to config.json.

    Works for local paths only. For HF repos we let PeftConfig.from_pretrained
    raise and catch — the load wrapper handles that fallback.
    """
    if os.path.isdir(pretrained_path):
        return os.path.exists(os.path.join(pretrained_path, "adapter_config.json"))
    return False


def load_policy(pretrained_path: str, device: str):
    """Load any SmolVLA checkpoint (full FT or LoRA) and its processor pipeline.

    Mirrors lerobot.scripts.lerobot_eval flow but bypasses make_policy's
    ds_meta requirement — we read input/output features straight from the
    checkpoint's config.json (saved at training time).

    Returns: (policy, preprocessor, postprocessor, policy_cfg)
    """
    import torch  # noqa: F401  (silences linter; checked via device)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import (
        get_policy_class, make_pre_post_processors,
    )

    logger.info("Loading policy config from %s", pretrained_path)
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    cfg.device = device  # override training-time device with deploy device
    cfg.pretrained_path = pretrained_path

    # LoRA detection — try the local-path heuristic first, then fall back to
    # PeftConfig probe (works for HF repo IDs too).
    is_lora = _is_lora_checkpoint(pretrained_path)
    if not is_lora:
        try:
            from peft import PeftConfig
            PeftConfig.from_pretrained(pretrained_path)
            is_lora = True
        except Exception:
            is_lora = False

    policy_cls = get_policy_class(cfg.type)
    if is_lora:
        from peft import PeftConfig, PeftModel
        peft_cfg = PeftConfig.from_pretrained(pretrained_path)
        base = peft_cfg.base_model_name_or_path
        if not base:
            raise ValueError(
                f"LoRA adapter at {pretrained_path} has no base_model_name_or_path; "
                "cannot resolve underlying SmolVLA checkpoint."
            )
        logger.info("LoRA detected — base=%s, adapter=%s", base, pretrained_path)
        # Load base policy with our deploy config (preserves input/output features
        # from the checkpoint config.json, which match what training saw).
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=base, config=cfg,
        )
        policy = PeftModel.from_pretrained(policy, pretrained_path, config=peft_cfg)
    else:
        logger.info("Full FT detected — loading from %s", pretrained_path)
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=pretrained_path, config=cfg,
        )

    policy.to(device)
    policy.eval()

    # Reset internal action queue (used by select_action; we use predict_action_chunk
    # but reset is cheap and idempotent).
    if hasattr(policy, "reset"):
        policy.reset()

    # Pre/post-processors are always loaded from the same path as the model
    # — they carry the normalization stats that were fit during training.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    logger.info("Policy + processors loaded. Input features: %s",
                list(cfg.input_features.keys()))
    logger.info("Output features: %s", list(cfg.output_features.keys()))
    logger.info("chunk_size=%s, n_action_steps=%s",
                getattr(cfg, "chunk_size", "?"),
                getattr(cfg, "n_action_steps", "?"))
    return policy, preprocessor, postprocessor, cfg


# ---------------------------------------------------------------------------
# Observation builder — assembles the model batch from latest sub'd messages
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """Turns the latest /joint_states + camera frames into a model-ready batch.

    Threading: read snapshots are taken under a lock then released — the
    inference thread copies once, then the ROS spin thread is free to keep
    updating. Numpy arrays from cv2 conversions are not aliased back into
    the spin thread.
    """

    def __init__(self, *, input_features: dict, rename_map: dict[str, str],
                 task: str, device: str):
        self.input_features = input_features
        self.rename_map = rename_map
        self.task = task
        self.device = device

        # Inverse map: model-feature-key → dataset-feature-key, so we know
        # which subscription provides each model-required image. Defaults
        # to identity for any unmapped key.
        self._dataset_to_model = dict(rename_map) if rename_map else {}
        self._model_to_dataset = {
            v: k for k, v in self._dataset_to_model.items()
        }

        # Pre-compute the canonical state joint order. Always 6 joints,
        # always degrees (dataset use_degrees=True), so we feed degrees in.
        self.state_joints = JOINT_NAMES_CANONICAL

        # Discover which model-feature image keys are real vs placeholder.
        # Real ones map to a dataset key we have a subscription for.
        # Placeholders (e.g. "empty_camera_0", "camera3" with only 2 sub'd
        # cams) get filled with zeros of the model-declared shape.
        self.image_feature_keys = [
            k for k in input_features
            if k.startswith("observation.images.")
        ]
        logger.info("Model expects image features: %s", self.image_feature_keys)

    def feature_for_dataset_key(self, dataset_key: str) -> Optional[str]:
        """Return the model-side feature key for a given dataset image key,
        or None if the model doesn't consume that key."""
        # If rename map says wrist→camera1, then dataset 'wrist' goes to
        # model 'camera1'. If no map, identity.
        model_key = self._dataset_to_model.get(dataset_key, dataset_key)
        feat_key = f"observation.images.{model_key}"
        if feat_key in self.input_features:
            return feat_key
        # Identity check (no rename for this key)
        feat_key = f"observation.images.{dataset_key}"
        if feat_key in self.input_features:
            return feat_key
        return None


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

def _make_node(args, policy_cfg):
    """Construct the rclpy node + state. Imports rclpy here so --dry-run can
    skip ROS entirely."""
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Image
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from builtin_interfaces.msg import Duration

    rad_to_deg = 180.0 / np.pi
    deg_to_rad = np.pi / 180.0

    class SmolVLANode(Node):
        def __init__(self):
            super().__init__("smolvla_inference")
            self._cb_group = ReentrantCallbackGroup()
            self._lock = threading.Lock()

            # Latest observation snapshots
            self._latest_state_deg: Optional[dict[str, float]] = None
            self._latest_state_t: float = 0.0
            self._latest_imgs: dict[str, Any] = {}  # dataset_key → np.ndarray HxWx3 uint8
            self._latest_img_t: dict[str, float] = {}

            # Action chunk (deque of np.ndarray (6,) in DEGREES, model output)
            self._action_chunk: deque = deque()
            self._chunk_lock = threading.Lock()

            self._stop_evt = threading.Event()

            # ----- Subscriptions -----
            self.create_subscription(
                JointState, args.joint_states_topic,
                self._on_joint_state, qos_profile_sensor_data,
                callback_group=self._cb_group,
            )
            self._cam_topics = json.loads(args.camera_topics)
            for dataset_key, topic in self._cam_topics.items():
                self.create_subscription(
                    Image, topic,
                    lambda msg, k=dataset_key: self._on_image(msg, k),
                    qos_profile_sensor_data,
                    callback_group=self._cb_group,
                )
                self.get_logger().info(
                    f"subscribed: {topic} → observation.images.{dataset_key}")

            # ----- Action clients (FollowJointTrajectory) -----
            self._arm_ac = ActionClient(self, FollowJointTrajectory, ARM_FJT_ACTION,
                                        callback_group=self._cb_group)
            self._grip_ac = ActionClient(self, FollowJointTrajectory, GRIPPER_FJT_ACTION,
                                         callback_group=self._cb_group)

            # ----- Publish timer (controller rate) -----
            self._publish_period = 1.0 / float(args.publish_rate)
            self.create_timer(self._publish_period, self._on_publish_tick,
                              callback_group=self._cb_group)

            self.get_logger().info(
                f"smolvla_inference up — publish={args.publish_rate} Hz, "
                f"inference={args.inference_rate} Hz, task={args.task!r}")

        # ---- Callbacks ----

        def _on_joint_state(self, msg):
            now = time.monotonic()
            with self._lock:
                state: dict[str, float] = {}
                for name, pos in zip(msg.name, msg.position):
                    canon = JOINT_NAME_MAP.get(name, name)
                    state[canon] = float(pos) * rad_to_deg  # rad → deg
                self._latest_state_deg = state
                self._latest_state_t = now

        def _on_image(self, msg, dataset_key):
            # Strip alpha if RGBA, decode by encoding. Same approach as
            # lerobot.cameras.ros2.ROS2Camera (consumer-side strip per
            # record_sim_isaac.sh comments).
            try:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding in ("rgb8", "bgr8"):
                    img = arr.reshape((msg.height, msg.width, 3))
                    if msg.encoding == "bgr8":
                        img = img[..., ::-1].copy()
                elif msg.encoding in ("rgba8", "bgra8"):
                    rgba = arr.reshape((msg.height, msg.width, 4))
                    img = rgba[..., :3].copy()
                    if msg.encoding == "bgra8":
                        img = img[..., ::-1].copy()
                else:
                    self.get_logger().warning(
                        f"unsupported image encoding {msg.encoding!r} on "
                        f"{dataset_key}; dropping frame")
                    return
            except Exception as e:
                self.get_logger().warning(f"image decode failed ({dataset_key}): {e}")
                return
            with self._lock:
                self._latest_imgs[dataset_key] = img
                self._latest_img_t[dataset_key] = time.monotonic()

        # ---- Action publishing ----

        def _on_publish_tick(self):
            with self._chunk_lock:
                if not self._action_chunk:
                    return
                action_deg = self._action_chunk.popleft()

            # Convert deg → rad, build per-joint dict, hard-clamp.
            cmd_rad: dict[str, float] = {}
            for j, val_deg in zip(JOINT_NAMES_CANONICAL, action_deg):
                rad = float(val_deg) * deg_to_rad
                if args.clamp_joint_limits:
                    lo, hi = JOINT_LIMITS_RAD[j]
                    if rad < lo or rad > hi:
                        self.get_logger().warning(
                            f"clamping {j}: {rad:.3f} → "
                            f"[{lo:.3f},{hi:.3f}]")
                    rad = max(lo, min(hi, rad))
                cmd_rad[j] = rad

            # Dispatch to arm + gripper FJT clients. Tiny duration so the
            # controller interpolates over publish_period; next tick we
            # send the next setpoint.
            arm_pos = [cmd_rad[j] for j in ARM_JOINTS_URDF]
            grip_pos = cmd_rad["gripper"]  # canonical name in our state dict
            secs = int(self._publish_period)
            nsecs = int((self._publish_period - secs) * 1e9)
            dur = Duration(sec=secs, nanosec=nsecs)

            arm_traj = JointTrajectory()
            arm_traj.joint_names = ARM_JOINTS_URDF
            pt = JointTrajectoryPoint()
            pt.positions = arm_pos
            pt.time_from_start = dur
            arm_traj.points.append(pt)

            grip_traj = JointTrajectory()
            grip_traj.joint_names = [GRIPPER_JOINT_URDF]
            gpt = JointTrajectoryPoint()
            gpt.positions = [grip_pos]
            gpt.time_from_start = dur
            grip_traj.points.append(gpt)

            arm_goal = FollowJointTrajectory.Goal()
            arm_goal.trajectory = arm_traj
            grip_goal = FollowJointTrajectory.Goal()
            grip_goal.trajectory = grip_traj

            # Fire-and-forget: don't await result. send_goal_async returns
            # a future we ignore — the next tick supersedes whatever's
            # running. This matches drive_pick_place.py's pattern at
            # publish rates.
            if self._arm_ac.server_is_ready():
                self._arm_ac.send_goal_async(arm_goal)
            if self._grip_ac.server_is_ready():
                self._grip_ac.send_goal_async(grip_goal)

        # ---- Observation snapshot for inference thread ----

        def snapshot(self):
            """Return (state_deg dict, images dict, ts) under lock. Returns
            None if any required camera or state hasn't arrived yet."""
            with self._lock:
                if self._latest_state_deg is None:
                    return None
                # All required cameras must have produced a frame.
                for cam in self._cam_topics:
                    if cam not in self._latest_imgs:
                        return None
                return (
                    dict(self._latest_state_deg),
                    {k: v.copy() for k, v in self._latest_imgs.items()},
                    self._latest_state_t,
                )

        def push_chunk(self, chunk_deg_array):
            """Replace the action queue wholesale. chunk_deg_array is shape
            (T, 6). Holding _chunk_lock briefly is fine — publisher tick
            just popleft-s; if it has to wait one tick, that's invisible at
            25 Hz with controller interpolation."""
            with self._chunk_lock:
                self._action_chunk.clear()
                for row in chunk_deg_array:
                    self._action_chunk.append(row)

        def stop(self):
            self._stop_evt.set()

    return rclpy, SmolVLANode


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def _build_batch(snap, obs_builder, policy_cfg, np, torch):
    """Convert (state_deg, images, t) into a torch batch the model expects.

    Returns a dict with keys exactly matching policy_cfg.input_features,
    plus the "task" key required by SmolVLA's preprocessor.
    """
    state_deg, images, _ = snap
    batch: dict[str, Any] = {}

    # observation.state — float32 tensor (6,)
    state_vec = np.array(
        [state_deg[j] for j in JOINT_NAMES_CANONICAL],
        dtype=np.float32,
    )
    batch["observation.state"] = torch.from_numpy(state_vec).unsqueeze(0)  # (1,6)

    # Images: every model-required image feature gets filled either from a
    # subscription or with a zeros placeholder of the model-declared shape.
    for feat_key, feat in policy_cfg.input_features.items():
        if not feat_key.startswith("observation.images."):
            continue
        model_cam_key = feat_key[len("observation.images."):]
        # Find which dataset key feeds this model key.
        dataset_key = obs_builder._model_to_dataset.get(model_cam_key, model_cam_key)
        img = images.get(dataset_key)
        if img is None:
            # Placeholder: black image of the shape the model expects.
            shape = feat.shape  # (3, H, W)
            placeholder = np.zeros((shape[1], shape[2], 3), dtype=np.uint8)
            img = placeholder
        # Resize if the captured frame doesn't match the model-declared size.
        target_h, target_w = feat.shape[1], feat.shape[2]
        if img.shape[0] != target_h or img.shape[1] != target_w:
            try:
                import cv2
                img = cv2.resize(img, (target_w, target_h),
                                 interpolation=cv2.INTER_AREA)
            except Exception as e:
                logger.warning("cv2 resize failed for %s: %s", feat_key, e)
        # HWC uint8 → CHW float32 in [0,1]. The preprocessor will further
        # normalize per training stats.
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[feat_key] = img_t.unsqueeze(0)  # (1, 3, H, W)

    # Task (required by SmolVLA's preprocessor — tokenized into language tokens
    # by the smolvla preprocessor pipeline).
    batch["task"] = [obs_builder.task]
    return batch


def inference_loop(node, policy, preprocessor, postprocessor, policy_cfg,
                   obs_builder, args):
    """Run the chunk-and-replan inference pattern at args.inference_rate Hz.

    Canonical wrappers (matching lerobot.scripts.lerobot_eval.eval_one):
      * `policy.reset()` clears the policy's internal action queue before
        the first forward pass — required when using `select_action`, and
        cheap insurance when using `predict_action_chunk`.
      * `torch.inference_mode()` is the eval-canonical context (slightly
        cheaper than `torch.no_grad()` since it also disables view tracking).
      * `torch.autocast(device_type=...)` enables mixed-precision when AMP
        is requested (eval calls this gated by `cfg.policy.use_amp`); we
        default to True for CUDA, False for CPU.
      * `torch.cuda.empty_cache()` after each forward pass releases the
        intermediate activations PyTorch keeps around for the autograd
        graph construction. Without this, repeated chunks let memory
        creep and inference latency drifts upward — observed 435 ms →
        2700 ms before adding this. Matches NVIDIA's so101-ros2 reference.

    Two-thread architecture (vs eval's single-threaded gym loop) is
    intentional: a separate publish timer at 25 Hz dispatches actions
    from the chunk while this thread runs forwards at 2 Hz; mirrors the
    NVIDIA so101-ros2 policy_node pattern (inference_rate decoupled from
    publish_rate).
    """
    import numpy as np
    import torch

    use_amp = (args.device.startswith("cuda"))
    period = 1.0 / float(args.inference_rate)
    n_steps = getattr(policy_cfg, "n_action_steps", 50)
    logger.info(
        "inference loop: %s Hz, chunk size = %s steps "
        "(consumes ~%.2fs of trajectory before next replan), "
        "amp=%s, autocast=%s",
        args.inference_rate, n_steps, n_steps / args.publish_rate,
        use_amp, args.device,
    )

    # Canonical: reset the policy before the first inference (clears the
    # internal action queue). lerobot_eval calls this before each rollout.
    policy.reset()

    while not node._stop_evt.is_set():
        t0 = time.monotonic()
        snap = node.snapshot()
        if snap is None:
            logger.debug("waiting for first /joint_states + cameras…")
            time.sleep(0.05)
            continue

        try:
            batch = _build_batch(snap, obs_builder, policy_cfg, np, torch)
            # Preprocessor runs the saved pipeline (normalize, tokenize, etc.)
            batch = preprocessor(batch)
            # Canonical: torch.inference_mode + autocast for mixed-precision
            # forward (matches lerobot_eval's `with torch.no_grad(),
            # torch.autocast(device_type=device.type) if use_amp else
            # nullcontext()`). torch.inference_mode is the newer,
            # slightly-faster equivalent of torch.no_grad.
            from contextlib import nullcontext
            autocast_ctx = (
                torch.autocast(device_type="cuda")
                if use_amp else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                chunk = policy.predict_action_chunk(batch)
            # chunk shape: (1, n_action_steps, action_dim=6); postprocessor
            # unnormalizes to dataset units (degrees, since use_degrees=True).
            chunk_post = postprocessor(chunk)
            # squeeze batch dim → (T, 6)
            chunk_arr = chunk_post.squeeze(0).detach().cpu().numpy()
            # Release intermediate CUDA tensors. Critical for sustained
            # inference rate — without this, latency drifts upward.
            if use_amp:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.exception("inference step failed: %s", e)
            time.sleep(period)
            continue

        node.push_chunk(chunk_arr)
        dt = time.monotonic() - t0

        # Per-chunk instrumentation: log the model's full plan envelope so we
        # can distinguish "model plans descent we cut off" vs "model never plans
        # to descend". Joint order matches JOINT_NAMES_CANONICAL.
        # Format: each joint shows [min..max] across all 50 chunk steps,
        #         plus action[49] (chunk tail = model's planned end-state).
        # state[3]=wrist_flex (descend signal), state[5]=gripper (grasp signal).
        joint_short = ["pan", "lift", "elb", "wf", "wr", "grip"]
        envelope = " ".join(
            f"{joint_short[i]}=[{chunk_arr[:, i].min():+6.1f},{chunk_arr[:, i].max():+6.1f}]"
            for i in range(6)
        )
        last_action = chunk_arr[-1]
        last_str = " ".join(
            f"{joint_short[i]}={last_action[i]:+6.1f}" for i in range(6)
        )
        first_action = chunk_arr[0]
        first_str = " ".join(
            f"{joint_short[i]}={first_action[i]:+6.1f}" for i in range(6)
        )
        logger.info(
            "inference ok: %s actions, took %.0f ms (replan every %.0f ms)",
            chunk_arr.shape[0], dt * 1000.0, period * 1000.0,
        )
        logger.info("  chunk envelope: %s", envelope)
        logger.info("  chunk[ 0]     : %s", first_str)
        logger.info("  chunk[49]     : %s", last_str)
        # Sleep the remainder of the period.
        sleep_s = max(0.0, period - dt)
        if sleep_s:
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Resolve the model path (supports both HF repo + checkpoint subfolder
    # and local pretrained_model dirs).
    pretrained = _resolve_pretrained_path(args.model_path, args.model_checkpoint)
    rename_map = json.loads(args.rename_map)
    logger.info("model = %s", pretrained)
    logger.info("rename_map = %s", rename_map)
    logger.info("task = %s", args.task)

    # Heavy imports here so --help and arg validation are fast.
    import numpy as np
    import torch

    policy, preprocessor, postprocessor, policy_cfg = load_policy(
        pretrained, args.device,
    )
    obs_builder = ObservationBuilder(
        input_features=policy_cfg.input_features,
        rename_map=rename_map,
        task=args.task,
        device=args.device,
    )

    if args.dry_run:
        # Synthesize an obs of the right shape and run one forward pass.
        # Useful to verify model load + processor pipeline without ROS.
        logger.info("DRY RUN — synthesizing dummy obs and running one inference")
        dummy_state = {j: 0.0 for j in JOINT_NAMES_CANONICAL}
        dummy_imgs = {
            k: np.zeros((480, 640, 3), dtype=np.uint8)
            for k in json.loads(args.camera_topics)
        }
        snap = (dummy_state, dummy_imgs, 0.0)
        batch = _build_batch(snap, obs_builder, policy_cfg, np, torch)
        batch = preprocessor(batch)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        chunk_post = postprocessor(chunk)
        arr = chunk_post.squeeze(0).detach().cpu().numpy()
        print(f"[dry-run] chunk shape = {arr.shape}")
        print(f"[dry-run] first 3 actions:\n{arr[:3]}")
        return

    # --- ROS ---
    rclpy_mod, NodeCls = _make_node(args, policy_cfg)
    rclpy_mod.init()
    node = NodeCls()

    # Run inference in a daemon thread; ROS spin owns the main thread so
    # subscriptions + timer are scheduled by the executor.
    inf_thread = threading.Thread(
        target=inference_loop,
        args=(node, policy, preprocessor, postprocessor, policy_cfg,
              obs_builder, args),
        daemon=True,
    )

    def _shutdown(*_a):
        logger.info("shutdown requested")
        node.stop()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy_mod.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Use a multi-threaded executor so subscriptions + timer + action-client
    # callbacks don't serialize through one callback at a time.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    inf_thread.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy_mod.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
