import sys
import types
from fake_ros import Message
from test_task import Robot, Policy


class RunTask:
    class Goal(Message):
        def __init__(self, **kwargs):
            super().__init__(instruction="pick", max_duration=0.002, model="", update_instruction=False)
            self.__dict__.update(kwargs)
    Result = Message
    Feedback = Message


class Goal:
    def __init__(self, request, callback=lambda _: None):
        self.request, self.callback = request, callback
        self.is_cancel_requested = False
        self.status = "active"
        self.feedback = []
    def publish_feedback(self, value):
        self.feedback.append(value)
        self.callback(value)
    def succeed(self):
        self.status = "success"
    def canceled(self):
        self.status = "cancel"
    def abort(self):
        self.status = "abort"


def node(ros, monkeypatch):
    package = types.ModuleType("lerobot_ros2_interfaces")
    action = types.ModuleType("lerobot_ros2_interfaces.action")
    action.RunTask = RunTask
    monkeypatch.setitem(sys.modules, "lerobot_ros2_interfaces", package)
    monkeypatch.setitem(sys.modules, "lerobot_ros2_interfaces.action", action)
    from lerobot_ros2_task.node import create_node
    robot, policy = Robot(), Policy()
    result = create_node(robot=robot, policy_factory=lambda: policy)
    result.params.update(model="checkpoint", fps=1000.0)
    return result, robot, policy


def test_real_node_goal_accept_timeout_and_busy(ros, monkeypatch):
    server, robot, policy = node(ros, monkeypatch)
    request = RunTask.Goal()
    assert server._accept(request) == 1
    assert server._accept(request) == 0
    goal = Goal(request)
    result = server._execute(goal)
    assert result.outcome == "timeout" and goal.status == "abort"
    assert len(robot.homes) == 2 and policy.stops == 1
    assert not server._reserved and server._episode is None
    assert server._accept(RunTask.Goal(instruction="")) == 0
    server.destroy_node()


def test_real_node_prompt_goal_topic_success_and_model_selection(ros, monkeypatch):
    server, robot, policy = node(ros, monkeypatch)
    server.params["models"] = '{"base-ft":"/models/base-ft"}'
    request = RunTask.Goal(model="base-ft")
    assert server._accept(request) == 1
    def feedback(value):
        if value.phase == "run":
            update = RunTask.Goal(instruction="place red", update_instruction=True)
            assert server._accept(update) == 1
            update_goal = Goal(update)
            result = server._execute(update_goal)
            assert result.success and result.outcome == "instruction_updated"
            assert server._reserved  # Prompt update does not release the active episode.
            server._instruction(Message(data="place blue"))
            sent = server._update_client.goals[-1]
            assert sent.update_instruction and sent.instruction == "place blue"
            server._success(Message(data=True))
    goal = Goal(request, feedback)
    result = server._execute(goal)
    assert result.success and goal.status == "success"
    assert policy.starts == [("pick", "/models/base-ft")]
    assert policy.updates == ["place red"]
    assert len(robot.homes) == 2
    server.destroy_node()


def test_real_node_cancel_is_seen_during_home(ros, monkeypatch):
    server, robot, policy = node(ros, monkeypatch)
    request = RunTask.Goal()
    assert server._accept(request) == 1
    goal = Goal(request)
    goal.is_cancel_requested = True
    result = server._execute(goal)
    assert result.outcome == "cancel" and goal.status == "cancel"
    assert len(robot.homes) == 2 and not policy.starts
    server.destroy_node()


def test_finish_supplies_result_for_transition_time_response_api():
    from lerobot_ros2_task.node import finish_goal
    class ImmediateGoal:
        def __init__(self):
            self.published = None
        def abort(self, response=None):
            self.published = response or RunTask.Result()
    goal = ImmediateGoal()
    result = RunTask.Result(outcome="timeout", success=False, message="")
    finish_goal(goal, "abort", result)
    assert goal.published is result
    stock = Goal(RunTask.Goal())
    finish_goal(stock, "abort", result)
    assert stock.status == "abort"
