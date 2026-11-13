import os
import time
import threading
import pytest
from lerobot_ros2_task.policy import PolicyProcess
from lerobot_robot_ros2_joint import so101_config
from spawn_policy import chunks, blocked


def action(policy):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = policy.next_action()
        if value is not None:
            return value
        threading.Event().wait(.005)
    raise TimeoutError("Spawned policy did not provide an action")


def test_real_spawn_alignment_prompt_generation_and_clean_stop():
    policy = PolicyProcess(so101_config(), worker_target=chunks, device="cpu", stop_timeout_s=.5)
    policy.start("pick", "stub")
    child = policy.process
    try:
        value = action(policy)
        assert value["axis.pos"] == policy.buffer.step - 1
        policy.skip_steps(4)
        value = action(policy)
        assert value["axis.pos"] == policy.buffer.step - 1
        policy.set_instruction("place blue")
        value = action(policy)
        assert value["generation"] == 1 and value["prompt"] == "place blue"
        assert value["axis.pos"] == policy.buffer.step - 1
    finally:
        policy.stop()
    assert policy.process is None and policy.results._closed
    with pytest.raises(ValueError, match="closed"):
        child.is_alive()


def test_stop_ends_a_blocked_spawned_worker():
    policy = PolicyProcess(so101_config(), worker_target=blocked, device="cpu", stop_timeout_s=.05)
    policy.start("pick", "stub")
    policy.stop()
    assert policy.process is None and policy.results._closed


@pytest.mark.skipif(__import__('sys').platform != 'linux', reason="process state check uses Linux /proc")
def test_worker_exits_when_parent_is_killed(tmp_path):
    import subprocess
    import sys
    pid_file = tmp_path / "worker.pid"
    code = '''
import sys,threading
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from lerobot_ros2_task.policy import PolicyProcess
from lerobot_robot_ros2_joint import so101_config
from spawn_policy import blocked
policy=PolicyProcess(so101_config(),worker_target=blocked)
policy.start("pick","stub")
Path(sys.argv[2]).write_text(str(policy.process.pid))
threading.Event().wait(60)
'''
    tests_dir = str(__import__('pathlib').Path(__file__).parent)
    parent = subprocess.Popen([sys.executable, "-c", code, tests_dir, str(pid_file)])
    def active(pid):
        try:
            return __import__('pathlib').Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            threading.Event().wait(.01)
        assert pid_file.exists()
        worker_pid = int(pid_file.read_text())
        assert active(worker_pid)
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 8
        while active(worker_pid) and time.monotonic() < deadline:
            threading.Event().wait(.02)
        assert not active(worker_pid), "worker survived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
