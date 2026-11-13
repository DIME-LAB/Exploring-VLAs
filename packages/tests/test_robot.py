from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
from fake_ros import Message
from lerobot_robot_ros2_joint import ROS2JointRobot, ROS2JointRobotConfig, ControllerGroup, so101_config
from lerobot_robot_ros2_joint.config_ros2_joint import SO101_JOINTS


def state(names=None, values=None):
    return Message(name=names or SO101_JOINTS[:5] + ["gripper_joint"],
                   position=values if values is not None else [0.1] * 6)


@pytest.fixture
def make_robot(ros, monkeypatch):
    rclpy, _ = ros
    create = rclpy.create_node
    def create_node(*args, **kwargs):
        node = create(*args, **kwargs)
        subscribe = node.create_subscription
        def subscription(kind, topic, callback, qos):
            result = subscribe(kind, topic, callback, qos)
            if "joint_states" in topic:
                callback(state())
            return result
        node.create_subscription = subscription
        return node
    monkeypatch.setattr(rclpy, "create_node", create_node)
    made = []
    def make(**kwargs):
        robot = ROS2JointRobot(so101_config(**kwargs))
        made.append(robot)
        robot.connect()
        return robot
    yield make
    for robot in made:
        robot.disconnect()


def test_stream_and_observation_equivalence_to_fork(make_robot):
    fork_module = pytest.importorskip("lerobot.robots.so101_ros2")
    fork = fork_module.SO101ROS2Robot(fork_module.SO101ROS2RobotConfig(actuate=True, action_duration_s=1/17.4))
    fork._is_connected = True
    fork._arm_command_pub = NS(messages=[])
    fork._arm_command_pub.publish = fork._arm_command_pub.messages.append
    fork._gripper_command_pub = NS(messages=[])
    fork._gripper_command_pub.publish = fork._gripper_command_pub.messages.append
    robot = make_robot(actuate=True, action_duration_s=1/17.4)
    fork._on_joint_state(state())
    assert robot.get_observation() == fork.get_observation()
    assert robot.observation_features == fork.observation_features
    rng = np.random.default_rng(21)
    for _ in range(40):
        action = dict(zip(robot.action_features, rng.uniform(-220, 220, 6), strict=True))
        expected, actual = fork.send_action(action), robot.send_action(action)
        assert actual == expected
        for pub, fork_pub in zip(robot._publishers, [fork._arm_command_pub, fork._gripper_command_pub], strict=True):
            result, comparison = pub.messages[-1], fork_pub.messages[-1]
            assert result.joint_names == comparison.joint_names
            assert result.points[0].positions == comparison.points[0].positions
            assert result.points[0].time_from_start.__dict__ == comparison.points[0].time_from_start.__dict__
            assert result.header.stamp.sec == 0
    fork._is_connected = False


def test_measured_state_merged_cache_and_per_joint_age(make_robot):
    robot = make_robot(joint_states_topic="/joint_states_measured")
    assert robot._sub.topic == "/joint_states_measured"
    robot._on_joint_state(state(["shoulder_pan"], [0.25]))
    obs = robot.get_observation()
    assert obs["shoulder_pan.pos"] == 0.25 * 180 / math.pi
    assert len(obs) == 6
    robot._cache.arrivals["gripper"] -= 1
    robot._on_joint_state(state(["shoulder_pan"], [0.5]))
    with pytest.raises(TimeoutError, match="gripper"):
        robot.get_observation()


def test_no_actuation_and_relative_targets(make_robot):
    robot = make_robot()
    action = {"gripper.pos": 70}
    assert robot.send_action(action) is action and not robot._publishers
    robot.disconnect()
    robot = make_robot(actuate=True, max_relative_target=2.0)
    action = {key: 50.0 for key in robot.action_features}
    result = robot.send_action(action)
    assert all(abs(value - (0.1 * 180/math.pi + 2)) < 1e-12 for value in result.values())
    with pytest.raises(ValueError, match="finite"):
        robot.send_action({"gripper.pos": float("nan")})


def test_urdf_limits_and_custom_controller(ros, tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot><joint name="raw" type="revolute"><limit lower="-0.2" upper="0.3"/></joint></robot>')
    config = ROS2JointRobotConfig(joint_names=["axis"], joint_name_map={"raw": "axis"},
                                 urdf_path=str(urdf), use_degrees=False, actuate=True,
                                 controllers=[ControllerGroup("/custom/follow_joint_trajectory", ["axis"])])
    robot = ROS2JointRobot(config)
    assert robot._targets({"axis.pos": 5}) == {"axis": 0.3}
    trajectory = robot._trajectory(config.controllers[0], {"axis": 0.3}, 0.125)
    assert trajectory.joint_names == ["raw"]
    assert trajectory.points[0].time_from_start.nanosec == 125000000


def test_tracked_move_success_timeout_and_late_cancel(make_robot):
    from concurrent.futures import Future
    from fake_ros import GoalHandle
    from lerobot_robot_ros2_joint.moves import Move
    robot = make_robot(actuate=True)
    action = dict.fromkeys(robot.action_features, 0)
    move = robot.move_to(action)
    for client in robot._clients:
        client.handles[-1].result.set_result(Message(status=4, result=Message(error_code=0)))
    assert len(move.wait(1)) == 2
    move = robot.move_to(action)
    with pytest.raises(TimeoutError):
        move.wait(0)
    assert all(client.handles[-1].cancels == 1 for client in robot._clients)
    pending = Future()
    move = Move([pending])
    move.cancel()
    handle = GoalHandle()
    pending.set_result(handle)
    assert handle.cancels == 1


def test_declined_and_failed_moves_cancel_other_groups(make_robot):
    robot = make_robot(actuate=True)
    for status, code in [(5, 0), (4, -1)]:
        move = robot.move_to(dict.fromkeys(robot.action_features, 0))
        for client in robot._clients:
            client.handles[-1].result.set_result(Message(status=status, result=Message(error_code=code)))
        with pytest.raises(RuntimeError, match="failed"):
            move.wait(1)
    move = robot.move_to(dict.fromkeys(robot.action_features, 0))
    robot._clients[0].handles[-1].accepted = False
    with pytest.raises(RuntimeError, match="declined"):
        move.wait(1)
    assert robot._clients[1].handles[-1].cancels == 1


def test_missing_and_invalid_joint_data(make_robot):
    robot = make_robot()
    robot._on_joint_state(state(["shoulder_pan"], []))
    with pytest.raises(ValueError, match="Invalid"):
        robot.get_observation()
    robot._on_joint_state(state())
    assert len(robot.get_observation()) == 6


def test_failed_connect_releases_resources(ros):
    robot = ROS2JointRobot(so101_config(actuate=True, state_timeout_s=0.001))
    with pytest.raises(TimeoutError):
        robot.connect()
    assert not robot.is_connected and robot._lease is None
    assert not ros[1].ok()
