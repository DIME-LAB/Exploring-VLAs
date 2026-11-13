import threading
from types import SimpleNamespace as NS
import cv2
import numpy as np
import pytest
from lerobot_camera_ros2 import ROS2Camera, ROS2CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def image(encoding="rgb8", *, padding=0, big=False):
    channels = 4 if "a8" in encoding else (1 if encoding.startswith("mono") else 3)
    dtype = np.dtype(">u2" if big else "u1")
    values = np.arange(2 * 3 * channels, dtype=dtype).reshape(2, 3, channels)
    row = 3 * channels * dtype.itemsize
    payload = b"".join(values[i].tobytes() + b"x" * padding for i in range(2))
    return NS(encoding=encoding, height=2, width=3, step=row + padding,
              is_bigendian=big, data=payload, header=NS(stamp=NS(sec=7, nanosec=123)))


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8"])
@pytest.mark.parametrize("mode", [ColorMode.RGB, ColorMode.BGR])
@pytest.mark.parametrize("rotation", list(Cv2Rotation))
def test_fork_pixel_equivalence(ros, encoding, mode, rotation):
    ForkCamera = pytest.importorskip("lerobot.cameras.ros2.camera_ros2").ROS2Camera
    config = ROS2CameraConfig(topic="/test", color_mode=mode, rotation=rotation)
    camera = ROS2Camera(config)
    fork = ForkCamera.__new__(ForkCamera)
    fork.config, fork.color_mode, fork.rotation = config, mode, rotation
    msg = image(encoding)
    actual, expected = camera._convert(msg), fork._msg_to_ndarray(msg)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype == np.uint8
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8"])
@pytest.mark.parametrize("mode", [ColorMode.RGB, ColorMode.BGR])
@pytest.mark.parametrize("rotation", list(Cv2Rotation))
def test_raw_color_and_rotation(ros, encoding, mode, rotation):
    camera = ROS2Camera(ROS2CameraConfig(topic="/test", color_mode=mode, rotation=rotation))
    msg = image(encoding, padding=7)
    channels = 4 if "a8" in encoding else 3
    expected = np.arange(2 * 3 * channels, dtype=np.uint8).reshape(2, 3, channels)[:, :, :3]
    if (encoding.startswith("bgr") and mode == ColorMode.RGB) or (encoding.startswith("rgb") and mode == ColorMode.BGR):
        expected = expected[:, :, ::-1]
    turns = {0: 0, 90: -1, 180: 2, -90: 1}
    expected = np.rot90(expected, turns[rotation.value])
    np.testing.assert_array_equal(camera._convert(msg), expected)


def test_padding_and_big_endian_mono(ros):
    camera = ROS2Camera(ROS2CameraConfig(topic="/mono"))
    msg = image("mono16", padding=4, big=True)
    actual = camera._convert(msg)
    assert actual.shape == (2, 3, 3) and actual.dtype == np.uint16
    np.testing.assert_array_equal(actual[:, :, 0], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(actual[:, :, 0], actual[:, :, 2])


def test_independent_subscriptions_and_peek_does_not_consume(ros):
    one, two = [ROS2Camera(ROS2CameraConfig(topic="/same")) for _ in range(2)]
    one.connect(warmup=False)
    two.connect(warmup=False)
    assert len(one._lease.node.subs) == 2
    one._on_message(image())
    np.testing.assert_array_equal(one.read_latest(), one.async_read(timeout_ms=0))
    assert one.last_header_stamp_ns == 7000000123
    with pytest.raises(TimeoutError):
        one.async_read(timeout_ms=1)
    with pytest.raises(RuntimeError, match="no frame"):
        two.read_latest()
    one._received -= 1
    with pytest.raises(TimeoutError, match="too old"):
        one.read_latest(max_age_ms=10)
    one.disconnect()
    assert two.is_connected and len(two._lease.node.subs) == 1
    two.disconnect()


def test_condition_wakes_for_new_frame_and_disconnect(ros):
    camera = ROS2Camera(ROS2CameraConfig(topic="/test"))
    camera.connect(warmup=False)
    result = []
    started = threading.Event()
    def read():
        started.set()
        try:
            result.append(camera.async_read(timeout_ms=500))
        except Exception as exc:
            result.append(exc)
    worker = threading.Thread(target=read)
    worker.start()
    assert started.wait(1)
    camera._on_message(image())
    worker.join(1)
    assert isinstance(result.pop(), np.ndarray)
    worker = threading.Thread(target=read)
    worker.start()
    camera.disconnect()
    worker.join(1)
    from lerobot.utils.errors import DeviceNotConnectedError
    assert isinstance(result.pop(), DeviceNotConnectedError)


def test_compressed_png_and_cv_bridge_failure(ros):
    camera = ROS2Camera(ROS2CameraConfig(topic="/compressed", compressed=True))
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    np.testing.assert_array_equal(camera._convert(NS(data=encoded.tobytes())), bgr[:, :, ::-1])
    raw = ROS2Camera(ROS2CameraConfig(topic="/raw"))
    class Bridge:
        def imgmsg_to_cv2(self, *_):
            raise RuntimeError("conversion unavailable")
    raw._bridge = Bridge()
    np.testing.assert_array_equal(raw._convert(image()), np.arange(18).reshape(2, 3, 3))
    assert raw._bridge is None


def test_bad_layout_and_warmup_cleanup(ros):
    camera = ROS2Camera(ROS2CameraConfig(topic="/test", warmup_s=0.001))
    with pytest.raises(TimeoutError):
        camera.connect()
    assert not camera.is_connected
    msg = image()
    msg.step = 1
    with pytest.raises(ValueError, match="payload"):
        camera._convert(msg)
