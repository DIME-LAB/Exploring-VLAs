# LeRobot ROS 2 packages

These independent distributions are opt-in. Install them in a separate ROS-aware
Python 3.12 environment; the deployment scripts and fork submodule retain their
current behavior until ROADMAP 2.7.

| Item | Distribution / ROS package | LeRobot type |
|---|---|---|
| 2.1 | lerobot_ros2_context | shared context, executor, QoS and joint cache |
| 2.2 | lerobot_camera_ros2 | camera `ros2_image` |
| 2.3 | lerobot_robot_ros2_joint | robot `ros2_joint`, SO-101 `so101_ros2_joint` |
| 2.4 | lerobot_teleoperator_ros2_joint | teleoperator `ros2_joint` |
| 2.5, 2.5b | lerobot_ros2_task | action server, policy worker, client, optional Isaac panel |
| 2.5 | lerobot_ros2_interfaces | generated ROS RunTask action |
| 2.6 | tests, ../.github/workflows/ros2-plugins.yml | fake ROS and native ROS checks |

```bash
python -m pip install ./packages/lerobot_ros2_context ./packages/lerobot_camera_ros2 \
  ./packages/lerobot_robot_ros2_joint ./packages/lerobot_teleoperator_ros2_joint \
  ./packages/lerobot_ros2_task
python -m pytest packages/tests -q
```

ROS (`rclpy`, sensor_msgs, trajectory_msgs, control_msgs, std_msgs) is supplied by
your ROS distribution/RoboStack. Normal wheel names normalize underscores into
hyphens, while metadata/import names keep the underscores required by the
installed LeRobot scanner. Each plugin registers its config on root import,
exports Config and device classes, and supports LeRobot's unmodified factory.
See [LeRobot hardware integration](https://huggingface.co/docs/lerobot/integrate_hardware),
[third-party cameras](https://huggingface.co/docs/lerobot/third_party_sensors), and
[third-party robots](https://huggingface.co/docs/lerobot/third_party_robots).
The mechanism was checked against installed fork e1e20109 and PyPI 0.7.0.

Tests replace ROS messages/executors only, while using real LeRobot base classes,
registries, parsing and factories. Fork-only comparison cases skip when the fork
is absent; all input/output contract tests still run on PyPI. Fresh-interpreter
discovery catches import-order issues. Native tests require built RunTask interfaces
and an explicit opt-in; see the task and context READMEs. CI builds wheels and runs
both fake and native checks with PyPI and the pinned fork. Test artifacts retain
JUnit results and built wheels. Native transport tests use fake robot/model owners,
so they require no simulator, controller or GPU and do not prove task success.

For CycloneDDS set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and
`CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml`.
The shared context refuses to start CycloneDDS without an explicit URI. Run isolated
native checks in a domain assigned by the lead (90–99 here), never the deployment
or collector domain. Use `pixi run --manifest-path linux-env/pixi.toml <test-python>`
on dual-a4500 so native shared libraries use the ROS environment.

All packages are Apache-2.0. Each distribution carries LICENSE and, where code
was derived from a published source, NOTICE. Camera credits include Yadunund's
huggingface/lerobot PR #866 and the fiveages-sim CvBridge fallback pattern.
