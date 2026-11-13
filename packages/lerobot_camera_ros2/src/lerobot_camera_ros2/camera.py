# Copyright 2026 The HuggingFace Inc. team and Exploring-VLAs contributors.
# Licensed under the Apache License, Version 2.0.
# Original ROS camera: Yadunund, huggingface/lerobot PR #866.
# CvBridge fallback adapted from fiveages-sim/lerobot_ros2 (Apache-2.0).
from __future__ import annotations

import logging
import os
import threading
import time

import cv2
import numpy as np
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_ros2_context import acquire
from .config_ros2 import ROS2CameraConfig

_LOG = logging.getLogger(__name__)
_ENCODINGS = {
    "rgb8": ("u1", 3), "bgr8": ("u1", 3), "rgba8": ("u1", 4), "bgra8": ("u1", 4),
    "rgb16": ("u2", 3), "bgr16": ("u2", 3), "rgba16": ("u2", 4), "bgra16": ("u2", 4),
    "mono8": ("u1", 1), "mono16": ("u2", 1), "16uc1": ("u2", 1), "32fc1": ("f4", 1),
}


def _try_create_cv_bridge():
    if os.getenv("LEROBOT_ROS2_DISABLE_CV_BRIDGE", "0") == "1":
        return None
    try:
        from cv_bridge import CvBridge
        return CvBridge()
    except Exception as exc:
        _LOG.warning("cv_bridge unavailable (%s); using numpy conversion", exc)
        return None


def decode_image(msg):
    """Decode declared source layout, including row padding and byte order."""
    source = msg.encoding.lower()
    if source not in _ENCODINGS:
        raise ValueError(f"Unsupported image encoding: {source}")
    kind, channels = _ENCODINGS[source]
    dtype = np.dtype(kind).newbyteorder(">" if msg.is_bigendian else "<")
    h, w = int(msg.height), int(msg.width)
    row = w * channels * dtype.itemsize
    step = int(msg.step)
    if h <= 0 or w <= 0 or step < row or len(msg.data) != h * step:
        raise ValueError("Image dimensions, step and payload length do not agree")
    shape = (h, w, channels)
    img = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                     strides=(step, channels * dtype.itemsize, dtype.itemsize))
    return img.astype(dtype.newbyteorder("="), copy=True)


class ROS2Camera(Camera):
    config_class = ROS2CameraConfig

    def __init__(self, config, *, context=None, executor=None):
        super().__init__(config)
        self.config = config
        self._context, self._executor = context, executor
        self._lease = self._sub = None
        self._condition = threading.Condition()
        self._msg = None
        self._sequence = self._consumed = 0
        self._received = 0.0
        self._stamp_ns = None
        self._bridge = None

    @property
    def is_connected(self):
        return self._lease is not None

    def attach_context(self, *, context=None, executor=None):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Disconnect before attaching another context")
        self._context, self._executor = context, executor

    @property
    def last_header_stamp_ns(self):
        """Stamp paired with the last returned frame; simulation stamps stay in simulation time."""
        return self._stamp_ns

    @staticmethod
    def find_cameras():
        with acquire() as lease:
            return [{"topic": topic, "type": kind} for topic, kinds in lease.node.get_topic_names_and_types()
                    for kind in kinds if kind in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")]

    def connect(self, warmup=True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Camera is already connected")
        from sensor_msgs.msg import Image, CompressedImage
        with self._condition:
            self._msg = None
            self._sequence = self._consumed = 0
            self._stamp_ns = None
            self._lease = acquire(context=self._context, executor=self._executor)
        try:
            self._bridge = _try_create_cv_bridge() if self.config.use_cv_bridge else None
            self._sub = self._lease.node.create_subscription(
                CompressedImage if self.config.compressed else Image, self.config.topic,
                self._on_message, self.config.qos.profile())
            if warmup:
                with self._condition:
                    if not self._condition.wait_for(lambda: self._msg is not None or not self.is_connected,
                                                    self.config.warmup_s):
                        raise TimeoutError(f"No image on {self.config.topic} during warmup")
                frame = self.read_latest(max_age_ms=self.config.warmup_s * 1000)
                self.height, self.width = frame.shape[:2]
        except BaseException:
            self.disconnect()
            raise

    def _on_message(self, msg):
        with self._condition:
            self._msg = msg
            self._received = time.monotonic()
            self._sequence += 1
            self._condition.notify_all()

    def _convert(self, msg):
        if self.config.compressed:
            # OpenCV decodes compressed color into BGR, including PNG alpha.
            img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Compressed image cannot be decoded")
            source = "bgr8"
            if img.ndim == 2:
                img = img[:, :, None]
                source = "mono8"
        else:
            source = msg.encoding.lower()
            img = None
            if self._bridge is not None:
                try:
                    img = self._bridge.imgmsg_to_cv2(msg, "passthrough")
                    if img.ndim == 2:
                        img = img[:, :, None]
                except Exception as exc:
                    _LOG.warning("cv_bridge conversion failed (%s); using numpy conversion", exc)
                    self._bridge = None
            if img is None:
                img = decode_image(msg)
        # encoding is a conversion request, never a reinterpretation of source bytes.
        requested = self.config.encoding.lower()
        if requested != "passthrough" and requested not in _ENCODINGS:
            raise ValueError(f"Unsupported requested encoding: {requested}")
        if requested != "passthrough" and _ENCODINGS[requested][0] != img.dtype.str[-2:]:
            raise ValueError("Requested encoding must preserve source element type")
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        if source.startswith("bgr") and self.config.color_mode == ColorMode.RGB:
            img = img[:, :, ::-1]
        elif source.startswith("rgb") and self.config.color_mode == ColorMode.BGR:
            img = img[:, :, ::-1]
        rotations = {Cv2Rotation.NO_ROTATION: 0, Cv2Rotation.ROTATE_90: -1,
                     Cv2Rotation.ROTATE_180: 2, Cv2Rotation.ROTATE_270: 1}
        return np.ascontiguousarray(np.rot90(img, rotations[self.config.rotation]))

    def _get(self, *, fresh, timeout_ms=0, max_age_ms=None, consume=True):
        if timeout_ms < 0 or (max_age_ms is not None and max_age_ms < 0):
            raise ValueError("Frame time limits must be nonnegative")
        with self._condition:
            if not self.is_connected:
                raise DeviceNotConnectedError("Camera is disconnected")
            self._lease.check()
            if fresh and not self._condition.wait_for(
                lambda: not self.is_connected or self._sequence > self._consumed, timeout_ms / 1000
            ):
                self._lease.check()
                raise TimeoutError(f"No fresh frame on {self.config.topic}")
            if not self.is_connected:
                raise DeviceNotConnectedError("Camera disconnected while waiting")
            if self._msg is None:
                raise RuntimeError("Camera has no frame")
            if max_age_ms is not None and (time.monotonic() - self._received) * 1000 > max_age_ms:
                raise TimeoutError(f"Latest frame on {self.config.topic} is too old")
            msg = self._msg
            sequence = self._sequence
            frame = self._convert(msg)
            if consume:
                self._consumed = sequence
            self._stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            return frame

    def read(self):
        return self._get(fresh=False)

    def async_read(self, timeout_ms=200):
        return self._get(fresh=True, timeout_ms=timeout_ms)

    def read_latest(self, max_age_ms=500):
        return self._get(fresh=False, max_age_ms=max_age_ms, consume=False)

    def disconnect(self):
        with self._condition:
            lease, self._lease = self._lease, None
            self._condition.notify_all()
        if lease is not None:
            try:
                if self._sub is not None:
                    lease.node.destroy_subscription(self._sub)
            finally:
                self._sub = None
                lease.close()
