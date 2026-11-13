# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class TaskRequest:
    instruction: str
    max_duration: float
    model: str = ""

    def __post_init__(self):
        if not self.instruction.strip() or not math.isfinite(self.max_duration) or self.max_duration <= 0:
            raise ValueError("A nonempty instruction and finite positive max_duration are required")


@dataclass(frozen=True)
class TaskResult:
    outcome: str
    message: str = ""

    @property
    def success(self):
        return self.outcome == "success"


class Episode:
    """One goal owns home, policy activity, policy stop and return home."""
    def __init__(self, robot, policy, home, *, fps=17.4, home_duration_s=3, home_timeout_s=15):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.robot, self.policy, self.home = robot, policy, home
        self.period = 1 / fps
        self.home_duration_s, self.home_timeout_s = home_duration_s, home_timeout_s
        self.cancelled = threading.Event()
        self.succeeded = threading.Event()
        self._lock = threading.Lock()
        self._instruction = ""
        self._running = False
        self.missed_slots = 0

    def set_instruction(self, instruction):
        if not instruction.strip():
            raise ValueError("Instruction must be nonempty")
        with self._lock:
            if not self._running:
                raise RuntimeError("No policy is running")
            self.policy.set_instruction(instruction)
            self._instruction = instruction

    def _home(self, cancel_event=None):
        move = self.robot.move_to(self.home, duration_s=self.home_duration_s)
        move.wait(self.home_timeout_s, cancel_event=cancel_event)

    def run(self, request, feedback=lambda *_: None):
        result = TaskResult("error", "Episode did not start")
        began = time.monotonic()
        self.succeeded.clear()
        self.missed_slots = 0
        self._instruction = request.instruction
        try:
            feedback("home", self._instruction, request.model, 0.0)
            self._home(self.cancelled)
            if self.cancelled.is_set():
                raise InterruptedError("Goal cancelled during home")
            self.policy.start(request.instruction, request.model)
            with self._lock:
                self._running = True
            began = next_slot = time.monotonic()
            deadline = began + request.max_duration
            while True:
                now = time.monotonic()
                if self.cancelled.is_set():
                    result = TaskResult("cancel")
                    break
                if now >= deadline:
                    result = TaskResult("timeout")
                    break
                if self.succeeded.is_set():
                    result = TaskResult("success")
                    break
                with self._lock:
                    instruction = self._instruction
                feedback("run", instruction, request.model, now - began)
                if self.cancelled.is_set() or self.succeeded.is_set():
                    continue
                action = self.policy.next_action()
                if action is not None:
                    self.robot.send_action(action)
                next_slot += self.period
                now = time.monotonic()
                if next_slot < now:
                    skipped = int((now - next_slot) / self.period) + 1
                    self.missed_slots += skipped
                    self.policy.skip_steps(skipped)
                    next_slot += skipped * self.period
                self.cancelled.wait(max(0, min(next_slot, deadline) - now))
        except InterruptedError as exc:
            result = TaskResult("cancel", str(exc))
        except Exception as exc:
            result = TaskResult("error", f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._running = False
            try:
                self.policy.stop()
            except Exception as exc:
                result = TaskResult("error", f"Policy stop failed: {exc}")
            try:
                feedback("return_home", self._instruction, request.model, time.monotonic() - began)
                self._home()  # Return-home is allowed to finish after cancel.
            except Exception as exc:
                result = TaskResult("error", f"Return home failed: {exc}; prior outcome={result.outcome}")
        return result
