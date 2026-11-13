"""ROS-only lifecycle checks; no publishers, controllers or simulator required."""
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NATIVE_ROS") != "1", reason="set RUN_NATIVE_ROS=1 in a ROS env")


def test_real_context_reconnect_and_stopped_context_cleanup():
    import rclpy
    from lerobot_ros2_context import acquire
    from lerobot_camera_ros2 import ROS2Camera, ROS2CameraConfig
    for _ in range(2):
        camera = ROS2Camera(ROS2CameraConfig(topic="/phase2_test/image"))
        camera.connect(warmup=False)
        node = camera._lease.node
        camera.disconnect()
        assert not rclpy.ok()
    camera = ROS2Camera(ROS2CameraConfig(topic="/phase2_test/image"))
    camera.connect(warmup=False)
    rclpy.shutdown()
    camera.disconnect()
    assert camera._lease is None
    with acquire() as lease:
        assert lease.node.context.ok()


def test_real_robot_failed_connect_cleanup_then_reconnect():
    from lerobot_robot_ros2_joint import ROS2JointRobot, so101_config
    robot = ROS2JointRobot(so101_config(actuate=True, state_timeout_s=0.01,
                                       joint_states_topic="/phase2_test/absent"))
    for _ in range(2):
        with pytest.raises(TimeoutError):
            robot.connect()
        assert robot._lease is None and not robot._publishers and not robot.is_connected


def test_real_external_executor_is_left_running():
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from lerobot_ros2_context import acquire
    context = Context()
    rclpy.init(context=context)
    executor = SingleThreadedExecutor(context=context)
    try:
        with acquire(context=context, executor=executor) as lease:
            assert lease.node in executor.get_nodes()
        assert context.ok() and not executor.get_nodes()
    finally:
        executor.shutdown()
        context.shutdown()
