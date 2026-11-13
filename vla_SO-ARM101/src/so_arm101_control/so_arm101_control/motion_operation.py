"""Immutable motion inputs shared by the GUI and headless execution adapters."""
from dataclasses import dataclass, replace
from typing import Optional, Tuple
import math


@dataclass(frozen=True)
class MotionOperation:
    epoch: Optional[int]
    target: Tuple[Tuple[str, float], ...]
    event: object
    duration_s: float
    speed_scale: float
    tag: str
    velocity_scale: float = .5
    planning_attempts: int = 50

    def __post_init__(self):
        if self.event.motion_epoch != self.epoch:
            raise ValueError('Motion target and completion event have different origins')
        if self.duration_s <= 0 or self.speed_scale <= 0 or not all(
                math.isfinite(v) for v in (self.duration_s, self.speed_scale)):
            raise ValueError('Invalid operation timing')
        if not all(math.isfinite(value) for _, value in self.target):
            raise ValueError('Non-finite operation target')

    @classmethod
    def create(cls, epoch, target, event, duration_s, speed_scale=1., tag='motion', velocity_scale=.5, planning_attempts=50):
        return cls(epoch, tuple(sorted((str(k), float(v)) for k,v in target.items())),
                   event, float(duration_s), float(speed_scale), str(tag), float(velocity_scale), int(planning_attempts))

    def joints(self):
        return dict(self.target)

    def retarget(self, target, event=None):
        return replace(self, target=tuple(sorted((str(k),float(v)) for k,v in target.items())),
                       event=self.event if event is None else event)

    def with_planning(self, **settings):
        return replace(self, **settings)
