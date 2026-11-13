# ROS 2 joint robot (2.3)

Install `lerobot_ros2_context`, then `pip install ./packages/lerobot_robot_ros2_joint`
inside a ROS environment with sensor_msgs, trajectory_msgs, control_msgs and
builtin_interfaces. Discovery registers generic robot type **ros2_joint** and SO-101 preset
type **so101_ros2_joint** (`--robot.type=so101_ros2_joint`).

```python
from lerobot_robot_ros2_joint import ROS2JointRobot, so101_config
robot = ROS2JointRobot(so101_config(
    actuate=True, joint_states_topic="/joint_states_measured", action_duration_s=1/30))
robot.connect()
obs = robot.get_observation()
robot.send_action({key: 0.0 for key in robot.action_features})
move = robot.move_to({key: 0.0 for key in robot.action_features}, duration_s=3)
move.wait(timeout_s=10)
robot.disconnect()
```

Use `/joint_states` for models trained on the old reference-state data. Select
`/joint_states_measured` for models trained on physical feedback. State is merged
across partial JointState messages; every joint must be present and younger than
`state_max_age_ms` (default 500). Units are degrees by default, radians with
`use_degrees=False`. Observation keys are `<joint>.pos` plus camera keys;
cameras use fresh `async_read`. Width/height/fps must be provided for robot cameras.
`robot_type` overrides both dataset name fields when needed.

Generic configuration uses `ROS2JointRobotConfig(joint_names=[...],
controllers=[ControllerGroup(action="/.../follow_joint_trajectory", joints=[...])])`.
Each group owns its action name, ordered canonical joints and optional
`command_topic`. `joint_name_map` maps ROS names into feature names and is inverted
for commands. The SO-101 preset provides six joint names, gripper remapping,
limits, and the arm/gripper controller groups. Explicit joint limits are radians;
`urdf_path` loads URDF limits, with explicit config taking precedence.
`max_relative_target` bounds streaming changes in feature units against fresh measured
state. `actuate=False` returns actions unchanged and creates no command publishers.

Streaming uses depth-one positions-only JointTrajectory publishers, zero header
stamp, one point with `action_duration_s`, and no action futures. It performs no
camera reads or waits. The caller owns its absolute dispatch clock and observation
process; this package does not replace the Phase 1 dispatcher. Incomplete groups
are omitted, as in the fork. `move_to` requires every joint and uses tracked FJT
goals for discrete moves (home). Discrete moves bypass the per-tick relative cap; `Move.targets` exposes sent
radian targets. `Move.wait` checks acceptance, action status and
controller result; a failure/timeout cancels every pending group, including late
acceptance. Constructor context/executor arguments attach ROS resources.

Apache-2.0; see LICENSE/NOTICE. Derived from the Hugging Face LeRobot interfaces
and Exploring-VLAs' LeRobot fork e1e20109 (`so101_ros2`), Copyright 2026 The
HuggingFace Inc. team. SO-101 limit values match the deployment fork.

For CycloneDDS, set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and an explicit
`CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml`
(loopback transport). The context module refuses to initialize CycloneDDS with
that setting absent. The bare installed Jazzy/Cyclone node teardown can hang
without this explicit transport config. Other projects may supply their own XML.
