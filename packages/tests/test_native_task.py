"""Real ROS action transport with fake robot/policy owners, on an isolated domain."""
import os
import threading
import time
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NATIVE_ROS") != "1", reason="set RUN_NATIVE_ROS=1 with built action overlay")


def wait(future, timeout=5):
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    assert event.wait(timeout), "ROS future did not complete"
    return future.result()


def test_real_task_action_timeout_cancel_prompt_and_success():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import String, Bool
    from lerobot_ros2_interfaces.action import RunTask
    from lerobot_ros2_task.node import create_node
    from test_task import Robot, Policy
    rclpy.init()
    robot, policies = Robot(), []
    def policy():
        value = Policy()
        policies.append(value)
        return value
    server = create_node(robot=robot, policy_factory=policy)
    server.set_parameters([Parameter("model", value="fake-model"), Parameter("fps", value=100.0)])
    client_node = rclpy.create_node("phase2_task_transport_client")
    client = ActionClient(client_node, RunTask, "run_task")
    prompt_pub = client_node.create_publisher(String, "run_task/instruction", 10)
    success_pub = client_node.create_publisher(Bool, "run_task/success", 10)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(server)
    executor.add_node(client_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=5)
        goal = RunTask.Goal(instruction="pick", max_duration=0.08, model="")
        handle = wait(client.send_goal_async(goal))
        assert handle.accepted
        result = wait(handle.get_result_async())
        assert result.result.outcome == "timeout"
        assert len(robot.homes) == 2
        running = threading.Event()
        def feedback(message):
            if message.feedback.phase == "run":
                running.set()
        goal.max_duration = 2.0
        handle = wait(client.send_goal_async(goal, feedback_callback=feedback))
        assert handle.accepted and running.wait(5)
        update = RunTask.Goal(instruction="place red", max_duration=0.0, model="", update_instruction=True)
        update_handle = wait(client.send_goal_async(update))
        assert update_handle.accepted
        assert wait(update_handle.get_result_async()).result.outcome == "instruction_updated"
        assert policies[-1].updates == ["place red"]
        # Topic updates are routed back through this same action interface.
        changed = threading.Event()
        original = policies[-1].set_instruction
        def on_update(instruction):
            original(instruction)
            if instruction == "place green":
                changed.set()
        policies[-1].set_instruction = on_update
        prompt_pub.publish(String(data="place green"))
        assert changed.wait(5)
        assert "place green" in policies[-1].updates
        wait(handle.cancel_goal_async())
        assert wait(handle.get_result_async()).result.outcome == "cancel"
        assert len(robot.homes) == 4
        running.clear()
        handle = wait(client.send_goal_async(goal, feedback_callback=feedback))
        assert handle.accepted and running.wait(5)
        success_pub.publish(Bool(data=True))
        result = wait(handle.get_result_async())
        assert result.result.success and result.result.outcome == "success"
        assert len(robot.homes) == 6 and all(value.stops == 1 for value in policies)
    finally:
        if server._episode is not None:
            server._episode.cancelled.set()
        executor.shutdown(timeout_sec=20)
        thread.join(timeout=5)
        client.destroy()
        server.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


def test_real_node_reads_robot_and_model_json_parameter_overrides():
    import json
    from pathlib import Path
    import rclpy
    from rclpy.parameter import Parameter
    from lerobot_ros2_task.node import create_node
    from test_task import Robot, Policy
    example = Path(__file__).resolve().parents[1] / "lerobot_ros2_task/examples/so101.json"
    rclpy.init()
    node = create_node(robot=Robot(), policy_factory=Policy, parameter_overrides=[
        Parameter("robot_config", value=example.read_text()),
        Parameter("models", value=json.dumps({"base-ft": "/models/base-ft"})),
        Parameter("model", value="base-ft")])
    try:
        assert node._model("") == "/models/base-ft"
        assert list(node._policy_config.cameras) == ["top", "wrist"]
        assert node._policy_config.robot_type == "so_follower"
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("state", ["abort", "succeed", "canceled"])
def test_early_result_request_survives_work_after_terminal_transition(state):
    import inspect
    import rclpy
    from rclpy.action import ActionServer, ActionClient, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from lerobot_ros2_interfaces.action import RunTask
    from lerobot_ros2_task.node import finish_goal
    rclpy.init()
    node = rclpy.create_node("phase2_transition_result")
    proceed, transitioned, may_return = threading.Event(), threading.Event(), threading.Event()
    def execute(goal):
        assert proceed.wait(5)
        result = RunTask.Result(success=state == "succeed", outcome=state, message="completed result")
        finish_goal(goal, state, result)
        transitioned.set()
        assert may_return.wait(5)
        return result
    server = ActionServer(node, RunTask, "transition_result", execute_callback=execute,
                          cancel_callback=lambda _: CancelResponse.ACCEPT,
                          callback_group=ReentrantCallbackGroup())
    client = ActionClient(node, RunTask, "transition_result")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=5)
        handle = wait(client.send_goal_async(RunTask.Goal(instruction="pick", max_duration=1.)))
        result_future = handle.get_result_async()
        # Wait for the server's GetResult callback to be registered before allowing
        # the transition. This deliberately covers the installed rclpy race.
        server_future = server._result_futures[bytes(handle.goal_id.uuid)]
        deadline = time.monotonic() + 5
        while not server_future._callbacks and time.monotonic() < deadline:
            threading.Event().wait(.001)
        assert server_future._callbacks
        if state == "canceled":
            assert wait(handle.cancel_goal_async()).goals_canceling
        proceed.set()
        assert transitioned.wait(5)
        from rclpy.action.server import ServerGoalHandle
        immediate_api = "response" in inspect.signature(getattr(ServerGoalHandle, state)).parameters
        if not immediate_api:
            may_return.set()
        response = wait(result_future)
        assert response.result.outcome == state
        assert response.result.message == "completed result"
        may_return.set()
    finally:
        proceed.set()
        may_return.set()
        executor.shutdown(timeout_sec=10)
        thread.join(2)
        client.destroy()
        server.destroy()
        node.destroy_node()
        rclpy.shutdown()


def test_next_goal_is_accepted_when_previous_result_is_published(monkeypatch):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from lerobot_ros2_interfaces.action import RunTask
    import lerobot_ros2_task.node as module
    from test_task import Robot, Policy
    rclpy.init()
    published, return_allowed = threading.Event(), threading.Event()
    original = module.finish_goal
    first = [True]
    def finish(goal, state, result):
        original(goal, state, result)
        if first[0]:
            first[0] = False
            published.set()
            assert return_allowed.wait(5)
    monkeypatch.setattr(module, "finish_goal", finish)
    node = module.create_node(robot=Robot(), policy_factory=Policy)
    node.set_parameters([Parameter("model", value="stub"), Parameter("fps", value=1000.)])
    client = ActionClient(node, RunTask, "run_task")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_server(timeout_sec=5)
        goal = RunTask.Goal(instruction="pick", max_duration=.01)
        first_handle = wait(client.send_goal_async(goal))
        first_result = first_handle.get_result_async()
        assert published.wait(5)
        second_handle = wait(client.send_goal_async(goal))
        assert second_handle.accepted
        return_allowed.set()
        assert wait(first_result).result.outcome == "timeout"
        assert wait(second_handle.get_result_async()).result.outcome == "timeout"
    finally:
        return_allowed.set()
        executor.shutdown(timeout_sec=10)
        thread.join(2)
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()
