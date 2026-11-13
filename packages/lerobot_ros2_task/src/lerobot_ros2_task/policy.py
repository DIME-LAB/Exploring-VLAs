# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Observation/model work lives in a spawned process, away from dispatch."""
from collections import deque
from copy import deepcopy
import multiprocessing as mp
import queue
import time
import traceback
import threading


def _replace(queue_, value):
    try:
        queue_.put_nowait(value)
    except queue.Full:
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass
        try:
            queue_.put_nowait(value)
        except queue.Full:
            pass


class ActionBuffer:
    def __init__(self):
        self.step = 0
        self.generation = 0
        self.values = deque()

    def replace(self, generation, start, actions):
        if generation != self.generation:
            return
        end = start + len(actions)
        if end <= self.step:
            return
        if start > self.step:
            return  # No invented action in a gap.
        self.values = deque(actions[self.step - start:])

    def take(self):
        value = self.values.popleft() if self.values else None
        self.step += 1
        return value

    def skip(self, count):
        for _ in range(min(count, len(self.values))):
            self.values.popleft()
        self.step += count

    def change_instruction(self):
        self.generation += 1
        self.values.clear()


def _model_worker(config, model, device, command_text, command_generation, results, next_step, stop):
    robot = None
    try:
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot_robot_ros2_joint import ROS2JointRobot

        policy_config = PreTrainedConfig.from_pretrained(model)
        policy_config.device = device
        policy = get_policy_class(policy_config.type).from_pretrained(model, config=policy_config)
        policy.to(device).eval()
        policy.reset()
        pre, post = make_pre_post_processors(
            policy_config, pretrained_path=model,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        config = deepcopy(config)
        config.actuate = False
        robot = ROS2JointRobot(config)
        robot.connect()
        generation = -1
        while not stop.is_set():
            with command_text.get_lock():
                instruction = command_text.value.decode("utf-8")
                latest_generation = command_generation.value
            if generation != latest_generation:
                generation = latest_generation
                policy.reset()
                pre.reset()
                post.reset()
            start = next_step.value
            obs = robot.get_observation()
            prepared = {"observation.state": np.array([obs[key] for key in robot.action_features], dtype=np.float32)}
            prepared.update({f"observation.images.{key}": obs[key] for key in config.cameras})
            with torch.inference_mode():
                batch = prepare_observation_for_inference(prepared, torch.device(device), instruction, robot.robot_type)
                chunk = policy.predict_action_chunk(pre(batch))
                # Per-step postprocessing uses LeRobot's normal action pipeline.
                actions = [post(chunk[:, i, :]).squeeze(0).detach().cpu().tolist()
                           for i in range(chunk.shape[1])]
            formatted = [dict(zip(robot.action_features, row, strict=True)) for row in actions]
            _replace(results, ("chunk", generation, start, formatted))
    except BaseException:
        _replace(results, ("error", traceback.format_exc()))
    finally:
        if robot is not None:
            robot.disconnect()


class PolicyProcess:
    def __init__(self, robot_config, *, device="cuda", response_timeout_s=120):
        self.config = robot_config
        self.device = device
        self.response_timeout_s = response_timeout_s
        self.process = None
        self.buffer = ActionBuffer()
        self._lock = threading.RLock()

    def start(self, instruction, model):
        if self.process is not None:
            raise RuntimeError("Policy process is already running")
        context = mp.get_context("spawn")
        self.results = context.Queue(maxsize=2)
        self.command_text = context.Array("c", 65536)
        self.command_generation = context.Value("q", 0)
        self.next_step, self.stop_event = context.Value("q", 0), context.Event()
        self.buffer = ActionBuffer()
        self._write_instruction(instruction)
        self.process = context.Process(target=_model_worker, args=(
            self.config, model, self.device, self.command_text, self.command_generation,
            self.results, self.next_step, self.stop_event),
            name="lerobot_task_observation", daemon=True)
        self.process.start()
        self.last_response = time.monotonic()

    def set_instruction(self, instruction):
        encoded = instruction.encode("utf-8")
        if len(encoded) >= 65536 or b"\0" in encoded:
            raise ValueError("Instruction must fit in 65535 UTF-8 bytes and contain no null character")
        with self._lock:
            if self.process is None:
                raise RuntimeError("Policy process is not running")
            self.buffer.change_instruction()
            self._write_instruction(instruction)

    def _write_instruction(self, instruction):
        encoded = instruction.encode("utf-8")
        if len(encoded) >= 65536 or b"\0" in encoded:
            raise ValueError("Instruction must fit in 65535 UTF-8 bytes and contain no null character")
        with self.command_text.get_lock():
            self.command_text.value = encoded
            self.command_generation.value = self.buffer.generation

    def next_action(self):
        with self._lock:
            return self._next_action()

    def _next_action(self):
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            if result[0] == "error":
                raise RuntimeError(result[1])
            _, generation, start, actions = result
            self.buffer.replace(generation, start, actions)
            self.last_response = time.monotonic()
        if not self.process.is_alive():
            raise RuntimeError(f"Policy process exited with code {self.process.exitcode}")
        if time.monotonic() - self.last_response > self.response_timeout_s:
            raise TimeoutError("Policy process stopped providing action chunks")
        value = self.buffer.take()
        self.next_step.value = self.buffer.step
        return value

    def skip_steps(self, count):
        with self._lock:
            self.buffer.skip(count)
            self.next_step.value = self.buffer.step

    def stop(self):
        if self.process is None:
            return
        self.stop_event.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            raise RuntimeError("Policy process did not stop")
        self.process.close()
        self.process = None
        self.results.cancel_join_thread()
        self.results.close()
