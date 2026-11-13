# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
import math
from dataclasses import dataclass, field
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot_ros2_context.qos import QoSConfig


@TeleoperatorConfig.register_subclass("ros2_joint")
@dataclass
class ROS2JointTeleoperatorConfig(TeleoperatorConfig):
    action_topic: str = "/joint_commands_lerobot"
    joint_names: list[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
    joint_name_map: dict[str, str] = field(default_factory=lambda: {"gripper_joint": "gripper"})
    use_degrees: bool = True
    action_warmup_s: float = 2.0
    action_stale_ms: float = 500.0
    command_mode: str = "on_change"
    qos: QoSConfig = field(default_factory=QoSConfig)

    def __post_init__(self):
        if not self.action_topic or not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("An action topic and distinct joint names are required")
        if self.command_mode not in ("on_change", "continuous"):
            raise ValueError("command_mode must be on_change or continuous")
        if not math.isfinite(self.action_warmup_s) or self.action_warmup_s < 0:
            raise ValueError("action_warmup_s must be finite and nonnegative")
        if not math.isfinite(self.action_stale_ms) or self.action_stale_ms <= 0:
            raise ValueError("action_stale_ms must be finite and positive")
