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
