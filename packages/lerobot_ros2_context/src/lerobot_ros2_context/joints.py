# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Merged joint cache with per-joint arrival ages."""
import math
import logging
import threading
import time


class JointCache:
    def __init__(self, names, name_map, use_degrees):
        self.names = list(names)
        self.name_map = name_map
        self.scale = 180.0 / math.pi if use_degrees else 1.0
        self.condition = threading.Condition()
        self.values = {}
        self.arrivals = {}
        self._last_error_log = -math.inf

    def update(self, msg):
        try:
            if not msg.position:
                return  # JointState permits effort/velocity-only messages.
            if len(msg.name) != len(msg.position):
                raise ValueError("JointState names and positions must have equal lengths")
            incoming = {}
            for name, value in zip(msg.name, msg.position, strict=True):
                name = self.name_map.get(name, name)
                if name not in self.names:
                    continue
                value = float(value) * self.scale
                if not math.isfinite(value):
                    raise ValueError("Joint positions must be finite")
                if name in incoming:
                    raise ValueError("JointState names must be unique after mapping")
                incoming[name] = value
            with self.condition:
                now = time.monotonic()
                self.values.update(incoming)
                self.arrivals.update(dict.fromkeys(incoming, now))
                self.condition.notify_all()
        except (ValueError, TypeError) as exc:
            now = time.monotonic()
            if now - self._last_error_log >= 5:
                logging.getLogger(__name__).warning("Joint message ignored: %s", exc)
                self._last_error_log = now

    def wait(self, timeout):
        with self.condition:
            ready = self.condition.wait_for(lambda: all(name in self.values for name in self.names), timeout)
            if not ready:
                raise TimeoutError("Waiting for all configured joints timed out")

    def snapshot(self, max_age_ms=None):
        with self.condition:
            missing = set(self.names) - self.values.keys()
            if missing:
                raise KeyError(f"Missing joints: {sorted(missing)}")
            if max_age_ms is not None:
                now = time.monotonic()
                old = [name for name in self.names if (now - self.arrivals[name]) * 1000 > max_age_ms]
                if old:
                    raise TimeoutError(f"Joint values too old: {old}")
            return {f"{name}.pos": self.values[name] for name in self.names}
