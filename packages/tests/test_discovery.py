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
    preset_cfg = draccus.decode(RobotConfig, {"type": "so101_ros2_joint", "actuate": True})
    assert len(preset_cfg.joint_names) == 6 and len(preset_cfg.controllers) == 2
    for config in [camera_cfg, robot_cfg, teleop_cfg, preset_cfg]:
        device = make_device_from_device_class(config)
        assert device.config_class is type(config)
        assert not device.is_connected


def test_discovery_in_fresh_interpreter():
    import subprocess
    import sys
    code = '''
from lerobot.utils.import_utils import register_third_party_plugins, make_device_from_device_class
register_third_party_plugins()
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
import draccus
for cls, value in [
    (CameraConfig, {"type":"ros2_image", "topic":"/test"}),
    (RobotConfig, {"type":"so101_ros2_joint"}),
    (TeleoperatorConfig, {"type":"ros2_joint"})]:
    device = make_device_from_device_class(draccus.decode(cls, value))
    assert not device.is_connected
print("fresh-interpreter discovery and factories passed")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "factories passed" in result.stdout


def test_documented_task_robot_config_uses_registered_presets(ros):
    import json
    from pathlib import Path
    register_third_party_plugins()
    config_path = Path(__file__).resolve().parents[1] / "lerobot_ros2_task/examples/so101.json"
    config = draccus.decode(RobotConfig, json.loads(config_path.read_text()))
    assert config.type == "so101_ros2_joint"
    assert config.robot_type == "so_follower"
    assert list(config.cameras) == ["top", "wrist"]
    assert all(camera.type == "ros2_image" and camera.encoding == "rgb8" for camera in config.cameras.values())
