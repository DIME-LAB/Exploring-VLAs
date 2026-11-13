# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_ros2_context import acquire
from lerobot_ros2_context.joints import JointCache
from .config_ros2_joint import ROS2JointTeleoperatorConfig


class ROS2JointTeleoperator(Teleoperator):
    config_class = ROS2JointTeleoperatorConfig
    name = "ros2_joint"

    def __init__(self, config, *, context=None, executor=None):
        super().__init__(config)
        self.config = config
        self._context, self._executor = context, executor
        self._lease = self._sub = None
        self._cache = JointCache(config.joint_names, config.joint_name_map, config.use_degrees)

    @property
    def action_features(self):
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return self._lease is not None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Teleoperator is already connected")
        from sensor_msgs.msg import JointState
        self._cache = JointCache(self.config.joint_names, self.config.joint_name_map, self.config.use_degrees)
        self._lease = acquire(context=self._context, executor=self._executor)
        try:
            self._sub = self._lease.node.create_subscription(
                JointState, self.config.action_topic, self._on_action, self.config.qos.profile())
            try:
                self._cache.wait(self.config.action_warmup_s)
            except TimeoutError:
                pass  # A producer may start later; get_action requires a complete first command.
        except BaseException:
            self.disconnect()
            raise

    def _on_action(self, msg):
        self._cache.update(msg)

    def get_action(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Teleoperator is disconnected")
        self._lease.check()
        age = self.config.action_stale_ms if self.config.command_mode == "continuous" else None
        return self._cache.snapshot(age)

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        lease, self._lease = self._lease, None
        if lease is not None:
            try:
                if self._sub is not None:
                    lease.node.destroy_subscription(self._sub)
            finally:
                self._sub = None
                lease.close()
