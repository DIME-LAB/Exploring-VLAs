# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import math
import logging
import xml.etree.ElementTree as ET
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_ros2_context import acquire
from lerobot_ros2_context.joints import JointCache
from .config_ros2_joint import ROS2JointRobotConfig, SO101ROS2JointRobotConfig
from .moves import Move


class ROS2JointRobot(Robot):
    config_class = ROS2JointRobotConfig
    name = "ros2_joint"

    def __init__(self, config, *, context=None, executor=None):
        super().__init__(config)
        self.config = config
        if config.robot_type is not None:
            self.name = self.robot_type = config.robot_type
        self._context, self._executor = context, executor
        self._lease = self._sub = None
        self._connected = False
        self._publishers, self._clients = [], []
        self._move = None
        self._cache = JointCache(config.joint_names, config.joint_name_map, config.use_degrees)
        self.cameras = make_cameras_from_configs(config.cameras)
        if context is not None or executor is not None:
            for camera in self.cameras.values():
                if hasattr(camera, "attach_context"):
                    camera.attach_context(context=context, executor=executor)
        self._wire_names = {name: raw for raw, name in config.joint_name_map.items()}
        self._limits = self._load_limits()

    def _load_limits(self):
        limits = {}
        if self.config.urdf_path:
            root = ET.parse(self.config.urdf_path).getroot()
            for joint in root.findall("joint"):
                name = self.config.joint_name_map.get(joint.attrib["name"], joint.attrib["name"])
                bound = joint.find("limit")
                if name in self.config.joint_names and joint.get("type") != "continuous" and bound is not None:
                    lo, hi = float(bound.attrib["lower"]), float(bound.attrib["upper"])
                    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                        raise ValueError("URDF joint limits must be finite and ordered")
                    limits[name] = (lo, hi)
        limits.update(self.config.joint_limits)
        return limits

    @property
    def observation_features(self):
        return {**self.action_features, **{key: (cfg.height, cfg.width, 3) for key, cfg in self.config.cameras.items()}}

    @property
    def action_features(self):
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self._lease is not None:
            raise DeviceAlreadyConnectedError("Robot is already connected")
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        self._cache = JointCache(self.config.joint_names, self.config.joint_name_map, self.config.use_degrees)
        self._lease = acquire(context=self._context, executor=self._executor)
        try:
            self._sub = self._lease.node.create_subscription(JointState, self.config.joint_states_topic,
                                                            self._on_joint_state, self.config.qos.profile())
            if self.config.actuate:
                for group in self.config.controllers:
                    self._publishers.append(self._lease.node.create_publisher(JointTrajectory, group.command_topic, 1))
            for camera in self.cameras.values():
                camera.connect()
            self._cache.wait(self.config.state_timeout_s)
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def _on_joint_state(self, msg):
        self._cache.update(msg)

    def _check(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is disconnected")
        self._lease.check()

    def get_observation(self):
        self._check()
        obs = self._cache.snapshot(self.config.state_max_age_ms)
        for key, camera in self.cameras.items():
            obs[key] = camera.async_read(timeout_ms=self.config.image_timeout_ms)
        return obs

    def _targets(self, action, *, relative=True):
        goal = {}
        for key, value in action.items():
            name = key.removesuffix(".pos")
            if name in self.config.joint_names:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Action positions must be finite")
                goal[name] = value
        if relative and self.config.max_relative_target is not None:
            present = self._cache.snapshot(self.config.state_max_age_ms)
            for name, value in goal.items():
                cap = self.config.max_relative_target
                cap = cap[name] if isinstance(cap, dict) else cap
                current = present[f"{name}.pos"]
                goal[name] = max(current - cap, min(current + cap, value))
        factor = math.pi / 180.0 if self.config.use_degrees else 1.0
        goal = {name: value * factor for name, value in goal.items()}
        if self.config.clamp_joint_limits:
            for name, value in goal.items():
                lo, hi = self._limits.get(name, (-math.inf, math.inf))
                goal[name] = max(lo, min(hi, value))
        return goal

    def _trajectory(self, group, goal, duration):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        trajectory = JointTrajectory()
        trajectory.joint_names = [self._wire_names.get(name, name) for name in group.joints]
        point = JointTrajectoryPoint()
        point.positions = [goal[name] for name in group.joints]
        seconds = int(duration)
        point.time_from_start = Duration(sec=seconds, nanosec=int((duration - seconds) * 1e9))
        trajectory.points.append(point)
        return trajectory

    def send_action(self, action):
        self._check()
        if not self.config.actuate:
            return action
        goal = self._targets(action)
        for group, publisher in zip(self.config.controllers, self._publishers, strict=True):
            if all(name in goal for name in group.joints):
                publisher.publish(self._trajectory(group, goal, self.config.action_duration_s))
        factor = 180.0 / math.pi if self.config.use_degrees else 1.0
        return {key: goal[key.removesuffix(".pos")] * factor if key.removesuffix(".pos") in goal else value
                for key, value in action.items()}

    def move_to(self, action, *, duration_s=3.0, server_timeout_s=5.0):
        """Send a discrete, tracked FJT move; caller must wait or cancel it."""
        self._check()
        if not self.config.actuate:
            raise RuntimeError("move_to requires actuate=True")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Move duration must be finite and positive")
        goal = self._targets(action, relative=False)
        if set(goal) != set(self.config.joint_names):
            raise ValueError("A discrete move requires every joint")
        if self._move is not None:
            self._move.cancel()
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        if not self._clients:
            try:
                for group in self.config.controllers:
                    self._clients.append(ActionClient(self._lease.node, FollowJointTrajectory, group.action))
            except BaseException:
                for client in self._clients:
                    client.destroy()
                self._clients = []
                raise
        for client in self._clients:
            if not client.wait_for_server(timeout_sec=server_timeout_s):
                raise TimeoutError(f"Controller action unavailable: {client}")
        futures = []
        try:
            for group, client in zip(self.config.controllers, self._clients, strict=True):
                message = FollowJointTrajectory.Goal()
                message.trajectory = self._trajectory(group, goal, duration_s)
                futures.append(client.send_goal_async(message))
        except BaseException:
            Move(futures).cancel()
            raise
        self._move = Move(futures)
        self._move.targets = dict(goal)
        return self._move

    def disconnect(self):
        self._connected = False
        lease, self._lease = self._lease, None
        sub, self._sub = self._sub, None
        clients, self._clients = self._clients, []
        publishers, self._publishers = self._publishers, []
        errors = []
        def clean(callback):
            try:
                callback()
            except Exception as exc:
                errors.append(exc)
        if self._move is not None:
            clean(self._move.cancel)
            self._move = None
        for client in clients:
            clean(client.destroy)
        for camera in self.cameras.values():
            clean(camera.disconnect)
        if lease is not None:
            try:
                if sub is not None:
                    clean(lambda: lease.node.destroy_subscription(sub))
                for publisher in publishers:
                    clean(lambda: lease.node.destroy_publisher(publisher))
            finally:
                clean(lease.close)
        if errors:
            raise RuntimeError("Robot cleanup encountered an error after releasing all handles") from errors[0]


class SO101ROS2JointRobot(ROS2JointRobot):
    config_class = SO101ROS2JointRobotConfig
    name = "so101_ros2_joint"
