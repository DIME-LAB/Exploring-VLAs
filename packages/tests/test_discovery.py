import importlib.metadata
import draccus
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins, make_device_from_device_class


def test_installed_distribution_discovery_and_real_factory(ros):
    register_third_party_plugins()
    expected = ["lerobot_camera_ros2", "lerobot_robot_ros2_joint", "lerobot_teleoperator_ros2_joint"]
    assert all(importlib.metadata.metadata(name)["Name"] == name for name in expected)
    camera_cfg = draccus.decode(CameraConfig, {"type": "ros2_image", "topic": "/cam"})
    robot_cfg = draccus.decode(RobotConfig, {"type": "ros2_joint", "joint_names": ["axis"]})
    teleop_cfg = draccus.decode(TeleoperatorConfig, {"type": "ros2_joint", "joint_names": ["axis"]})
    for config in [camera_cfg, robot_cfg, teleop_cfg]:
        device = make_device_from_device_class(config)
        assert device.config_class is type(config)
        assert not device.is_connected
