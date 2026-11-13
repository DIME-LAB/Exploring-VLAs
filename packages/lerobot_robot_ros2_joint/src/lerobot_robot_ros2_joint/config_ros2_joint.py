# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass, field
import math
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot_ros2_context.qos import QoSConfig

SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121), "gripper": (-0.174533, 1.74533),
}


@dataclass
class ControllerGroup:
    action: str
    joints: list[str]
    command_topic: str = ""

    def __post_init__(self):
        if not self.action or not self.joints or len(set(self.joints)) != len(self.joints):
            raise ValueError("Controller groups require an action and distinct joints")
        if not self.command_topic:
            if not self.action.endswith("/follow_joint_trajectory"):
                raise ValueError("Provide command_topic for a nonstandard action name")
            self.command_topic = self.action.removesuffix("/follow_joint_trajectory") + "/joint_trajectory"


@RobotConfig.register_subclass("ros2_joint")
@dataclass
class ROS2JointRobotConfig(RobotConfig):
    joint_names: list[str] = field(default_factory=list)
    controllers: list[ControllerGroup] = field(default_factory=list)
    joint_name_map: dict[str, str] = field(default_factory=dict)  # ROS name -> feature name
    joint_states_topic: str = "/joint_states"
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)  # radians
    urdf_path: str | None = None
    use_degrees: bool = True
    actuate: bool = False
    clamp_joint_limits: bool = True
    max_relative_target: float | dict[str, float] | None = None  # feature units
    state_max_age_ms: float = 500.0
    state_timeout_s: float = 5.0
    image_timeout_ms: float = 500.0
    action_duration_s: float = 0.1
    robot_type: str | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    qos: QoSConfig = field(default_factory=lambda: QoSConfig(depth=5))

    def __post_init__(self):
        super().__post_init__()
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must contain distinct names")
        if not self.joint_states_topic:
            raise ValueError("joint_states_topic is required")
        for value in (self.state_max_age_ms, self.state_timeout_s, self.image_timeout_ms, self.action_duration_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("State, image and action time limits must be finite and positive")
        grouped = [name for group in self.controllers for name in group.joints]
        if len(grouped) != len(set(grouped)) or set(grouped) - set(self.joint_names):
            raise ValueError("Controller joints must be configured and belong to one group")
        if self.actuate and set(grouped) != set(self.joint_names):
            raise ValueError("Actuation requires a controller for every joint")
        mapped = [self.joint_name_map.get(name, name) for name in self.joint_name_map]
        if len(mapped) != len(set(mapped)):
            raise ValueError("joint_name_map must have distinct targets")
        for name, bounds in self.joint_limits.items():
            if name not in self.joint_names or len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError("Joint limits require configured names and finite ordered bounds")
        if self.max_relative_target is not None:
            values = self.max_relative_target
            if isinstance(values, dict):
                if set(values) != set(self.joint_names):
                    raise ValueError("max_relative_target must cover every joint")
                values = values.values()
            else:
                values = [values]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("max_relative_target must be finite and nonnegative")


@RobotConfig.register_subclass("so101_ros2_joint")
@dataclass
class SO101ROS2JointRobotConfig(ROS2JointRobotConfig):
    joint_names: list[str] = field(default_factory=lambda: list(SO101_JOINTS))
    joint_name_map: dict[str, str] = field(default_factory=lambda: {"gripper_joint": "gripper"})
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(SO101_LIMITS))
    controllers: list[ControllerGroup] = field(default_factory=lambda: [
        ControllerGroup("/arm_controller/follow_joint_trajectory", SO101_JOINTS[:5]),
        ControllerGroup("/gripper_controller/follow_joint_trajectory", ["gripper"])])


def so101_config(**kwargs):
    """SO-101 preset matching Exploring-VLAs' six position features."""
    return SO101ROS2JointRobotConfig(**kwargs)
