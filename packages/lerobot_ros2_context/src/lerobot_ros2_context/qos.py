# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
from dataclasses import dataclass


@dataclass
class QoSConfig:
    depth: int = 10
    reliability: str = "best_effort"
    durability: str = "volatile"

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("QoS depth must be positive")
        if self.reliability not in ("best_effort", "reliable"):
            raise ValueError("QoS reliability must be best_effort or reliable")
        if self.durability not in ("volatile", "transient_local"):
            raise ValueError("QoS durability must be volatile or transient_local")

    def profile(self):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        return QoSProfile(depth=self.depth, reliability=getattr(ReliabilityPolicy, self.reliability.upper()),
                          durability=getattr(DurabilityPolicy, self.durability.upper()),
                          history=HistoryPolicy.KEEP_LAST)
