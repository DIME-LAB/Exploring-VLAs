import threading
from types import SimpleNamespace as NS
import pytest
from lerobot_ros2_task import Episode, TaskRequest
from lerobot_ros2_task.policy import ActionBuffer


class Move:
    def wait(self, timeout, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("cancel during home")


class Robot:
    def __init__(self):
        self.homes = []
        self.actions = []

    def move_to(self, action, **kwargs):
        self.homes.append(dict(action))
        return Move()

    def send_action(self, action):
        self.actions.append(dict(action))


class Policy:
    def __init__(self):
        self.starts, self.updates = [], []
        self.stops = self.skipped = 0

    def start(self, instruction, model):
        self.starts.append((instruction, model))

    def set_instruction(self, instruction):
        self.updates.append(instruction)

    def next_action(self):
        return {"axis.pos": 1.0}

    def skip_steps(self, count):
        self.skipped += count

    def stop(self):
        self.stops += 1


@pytest.mark.parametrize("outcome", ["success", "cancel", "timeout", "error"])
def test_home_run_outcome_stop_home(outcome):
    robot, policy = Robot(), Policy()
    episode = Episode(robot, policy, {"axis.pos": 0}, fps=1000)
    phases = []
    def feedback(phase, *_):
        phases.append(phase)
        if phase == "run":
            if outcome == "success":
                episode.succeeded.set()
            elif outcome == "cancel":
                episode.cancelled.set()
            elif outcome == "error":
                raise RuntimeError("policy failed")
    result = episode.run(TaskRequest("pick", 0.003, "model"), feedback)
    assert result.outcome == outcome
    assert len(robot.homes) == 2 and robot.homes[0] == robot.homes[1]
    assert policy.stops == 1
    assert phases[0] == "home" and phases[-1] == "return_home"


def test_cancel_before_execute_is_preserved_and_home_failure_reported():
    robot, policy = Robot(), Policy()
    episode = Episode(robot, policy, {"axis.pos": 0})
    episode.cancelled.set()
    result = episode.run(TaskRequest("pick", 1))
    assert result.outcome == "cancel" and not policy.starts
    assert len(robot.homes) == 2
    class FailedRobot(Robot):
        def move_to(self, *_args, **_kwargs):
            raise RuntimeError("controller absent")
    episode = Episode(FailedRobot(), Policy(), {"axis.pos": 0})
    result = episode.run(TaskRequest("pick", 1))
    assert result.outcome == "error" and "Return home failed" in result.message


def test_live_instruction_and_slot_skipping(monkeypatch):
    import lerobot_ros2_task.episode as module
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    robot, policy = Robot(), Policy()
    episode = Episode(robot, policy, {"axis.pos": 0}, fps=10)
    class Cancel:
        def is_set(self):
            return now[0] > 0.5
        def wait(self, duration):
            now[0] += duration
    episode.cancelled = Cancel()
    def slow_action():
        episode.set_instruction("place red")
        now[0] += 0.25
        return {"axis.pos": 1}
    policy.next_action = slow_action
    result = episode.run(TaskRequest("pick", 1))
    assert result.outcome == "cancel"
    assert policy.updates == ["place red", "place red"]
    assert policy.skipped == episode.missed_slots == 4
    with pytest.raises(RuntimeError, match="No policy"):
        episode.set_instruction("later")


def test_action_buffer_ignores_old_prompt_and_old_time_steps():
    buffer = ActionBuffer()
    assert buffer.take() is None
    buffer.replace(0, 0, ["old", "one", "two", "three"])
    assert buffer.take() == "one"
    buffer.skip(1)
    assert buffer.take() == "three"
    buffer.change_instruction()
    buffer.replace(0, 4, ["old instruction"])
    assert buffer.take() is None
    buffer.replace(1, 4, ["late", "fresh"])
    assert buffer.take() == "fresh"
    buffer.replace(1, 50, ["gap"])
    assert buffer.take() is None


@pytest.mark.parametrize("instruction,duration", [("", 1), (" ", 1), ("x", 0), ("x", float("nan"))])
def test_invalid_goals(instruction, duration):
    with pytest.raises(ValueError):
        TaskRequest(instruction, duration)
