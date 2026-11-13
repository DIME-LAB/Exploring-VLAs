import multiprocessing as mp
import queue
import sys
import threading
import types
import numpy as np
from fake_ros import Message
from lerobot_ros2_task.policy import _model_worker, PolicyProcess
from lerobot_robot_ros2_joint import so101_config


def test_worker_prepares_live_instruction_and_denormalized_actions(monkeypatch):
    import torch
    config = so101_config()
    config.cameras = {"wrist": Message()}
    keys = [f"{name}.pos" for name in config.joint_names]
    seen = {}
    class Robot:
        robot_type = "so_follower"
        action_features = dict.fromkeys(keys)
        def __init__(self, cfg):
            assert not cfg.actuate
        def connect(self):
            seen["connected"] = True
        def get_observation(self):
            return {**dict.fromkeys(keys, 2.0), "wrist": np.full((2, 3, 3), 51, dtype=np.uint8)}
        def disconnect(self):
            seen["disconnected"] = True
    class Pipeline:
        def __init__(self, post=False):
            self.post = post
        def reset(self):
            pass
        def __call__(self, value):
            return value + 1 if self.post else value
    class Model:
        def to(self, _):
            return self
        def eval(self):
            return self
        def reset(self):
            pass
        def predict_action_chunk(self, batch):
            seen["batch"] = batch
            return torch.zeros((1, 3, 6))
    def prepare(obs, device, instruction, robot_type):
        assert instruction == "place blue" and robot_type == "so_follower"
        assert obs["observation.state"].dtype == np.float32
        np.testing.assert_array_equal(obs["observation.state"], np.full(6, 2))
        assert obs["observation.images.wrist"].shape == (2, 3, 3)
        return obs
    def module(name, **kwargs):
        obj = types.ModuleType(name)
        obj.__dict__.update(kwargs)
        monkeypatch.setitem(sys.modules, name, obj)
    module("lerobot.configs.policies", PreTrainedConfig=Message(from_pretrained=lambda _: Message(type="fake")))
    module("lerobot.policies.factory", get_policy_class=lambda _: Message(from_pretrained=lambda *a, **kw: Model()),
           make_pre_post_processors=lambda *a, **kw: (Pipeline(), Pipeline(post=True)))
    module("lerobot.policies.utils", prepare_observation_for_inference=prepare)
    import lerobot_robot_ros2_joint
    monkeypatch.setattr(lerobot_robot_ros2_joint, "ROS2JointRobot", Robot)
    stop = threading.Event()
    class Results(queue.Queue):
        def put_nowait(self, value):
            super().put_nowait(value)
            stop.set()
    ctx = mp.get_context("spawn")
    text = ctx.Array("c", 65536)
    text.value = b"place blue"
    generation, next_step = ctx.Value("q", 4), ctx.Value("q", 17)
    results = Results()
    _model_worker(config, "model", "cpu", text, generation, results, next_step, stop)
    value = results.get_nowait()
    assert value[:3] == ("chunk", 4, 17), value
    assert value[3] == [dict.fromkeys(keys, 1.0)] * 3
    assert seen["connected"] and seen["disconnected"]


def test_prompt_shared_memory_keeps_latest_generation():
    ctx = mp.get_context("spawn")
    policy = PolicyProcess(so101_config())
    policy.command_text = ctx.Array("c", 65536)
    policy.command_generation = ctx.Value("q", 0)
    policy.process = Message()
    for prompt in ["pick", "place red", "place blue"]:
        policy.set_instruction(prompt)
    assert policy.command_text.value.decode() == "place blue"
    assert policy.command_generation.value == 3
    assert policy.buffer.generation == 3
    import pytest
    with pytest.raises(ValueError):
        policy.set_instruction("x" * 65536)
    assert policy.command_generation.value == 3
