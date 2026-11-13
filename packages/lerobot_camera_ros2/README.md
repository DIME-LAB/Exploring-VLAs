# ROS 2 camera (2.2)

Install the shared context package then `pip install ./packages/lerobot_camera_ros2`
in a ROS environment. LeRobot discovers distribution `lerobot_camera_ros2` and
registers camera type **ros2_image**. That distinct type permits comparison with
the fork's `ros2` camera in one process.

```python
from lerobot_camera_ros2 import ROS2Camera, ROS2CameraConfig
camera = ROS2Camera(ROS2CameraConfig(topic="/wrist_camera_rgb_sim", width=640, height=480, fps=30))
camera.connect()
image = camera.async_read(timeout_ms=500)
stamp_ns = camera.last_header_stamp_ns
camera.disconnect()
```

`async_read` consumes the latest unconsumed frame, waiting on a Condition when
needed. `read` consumes the cached frame. `read_latest(max_age_ms=500)` peeks
without consuming and checks monotonic arrival age; header stamps remain in the
publisher's clock domain. Every instance has its own subscription/cache, including
instances on the same topic. Optional constructor `context`/`executor` attaches
to externally managed ROS resources.

Output is contiguous H×W×3 in `color_mode` (RGB default). Alpha is discarded;
mono is repeated into three channels. Raw row padding and byte order are honored;
8-bit, 16-bit and float element types are retained. `compressed=True` accepts
CompressedImage JPEG/PNG via OpenCV. Invalid buffers raise a clear error.
`encoding` accepts a compatible element type as a conversion hint; channel order
of output is always `color_mode`. QoSConfig controls depth/reliability/durability.
NumPy decoding is the default. `use_cv_bridge=True` optionally tries CvBridge,
disables it on conversion failure and uses the NumPy decoder instead.

Apache-2.0; see LICENSE and NOTICE. The original ROS 2 camera was contributed by
Yadunund in [huggingface/lerobot PR #866](https://github.com/huggingface/lerobot/pull/866).
CvBridge fallback pattern is adapted from
[fiveages-sim/lerobot_ros2](https://github.com/fiveages-sim/lerobot_ros2), revision
f750af764a8539d321d54955fe58441d94f93e73 (Apache-2.0).
PyPI's normalized name `lerobot-camera-ros2` returned 404 on 2026-11-13; availability
is an observation, not a name reservation. No package was uploaded.

For CycloneDDS, set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and an explicit
`CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml`
(loopback transport). The context module refuses to initialize CycloneDDS with
that setting absent. The bare installed Jazzy/Cyclone node teardown can hang
without this explicit transport config. Other projects may supply their own XML.
