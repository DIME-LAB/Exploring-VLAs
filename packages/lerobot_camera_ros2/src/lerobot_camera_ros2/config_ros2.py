# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot_ros2_context.qos import QoSConfig


@CameraConfig.register_subclass("ros2_image")
@dataclass
class ROS2CameraConfig(CameraConfig):
    topic: str = ""
    encoding: str = "passthrough"
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: float = 5.0
    compressed: bool = False
    use_cv_bridge: bool = False
    qos: QoSConfig = field(default_factory=QoSConfig)

    def __post_init__(self):
        if not self.topic or self.warmup_s <= 0:
            raise ValueError("topic and positive warmup_s are required")
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
