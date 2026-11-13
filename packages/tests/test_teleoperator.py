import pytest
from fake_ros import Message
from lerobot_teleoperator_ros2_joint import ROS2JointTeleoperator, ROS2JointTeleoperatorConfig


def message():
    return Message(name=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper_joint"],
                   position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_fork_command_equivalence(ros):
    module = pytest.importorskip("lerobot.teleoperators.so101_ros2")
    fork = module.SO101ROS2Teleoperator(module.SO101ROS2TeleoperatorConfig())
    fork._is_connected = True
    plugin = ROS2JointTeleoperator(ROS2JointTeleoperatorConfig(action_warmup_s=0))
    plugin.connect()
    assert plugin._sub.qos.depth == 5
    fork._on_action(message())
    plugin._on_action(message())
    assert plugin.get_action() == fork.get_action()
    assert plugin.action_features == fork.action_features
    fork._is_connected = False
    plugin.disconnect()


def test_on_change_hold_and_continuous_age(ros):
    plugin = ROS2JointTeleoperator(ROS2JointTeleoperatorConfig(action_warmup_s=0, command_mode="on_change"))
    plugin.connect()
    with pytest.raises(KeyError, match="Missing"):
        plugin.get_action()
    plugin._on_action(message())
    expected = plugin.get_action()
    for key in plugin._cache.arrivals:
        plugin._cache.arrivals[key] -= 30
    assert plugin.get_action() == expected
    plugin.config.command_mode = "continuous"
    with pytest.raises(TimeoutError):
        plugin.get_action()
    plugin.disconnect()
    plugin.connect()
    with pytest.raises(KeyError):
        plugin.get_action()  # reconnect never reuses the preceding command
    plugin.disconnect()


def test_robot_and_teleoperator_share_context(ros):
    from lerobot_ros2_context import acquire
    robot_lease = acquire()
    node = robot_lease.node
    plugin = ROS2JointTeleoperator(ROS2JointTeleoperatorConfig(action_warmup_s=0))
    plugin.connect()
    assert plugin._lease.node is node
    robot_lease.close()
    assert not node.destroyed and ros[1].ok()
    plugin.disconnect()
    assert node.destroyed and not ros[1].ok()
