# ROS 2 commanded-joint teleoperator (2.4)

Install `lerobot_ros2_context`, then
`pip install ./packages/lerobot_teleoperator_ros2_joint` in your ROS environment.
LeRobot discovers teleoperator type **ros2_joint**. It subscribes to JointState
commands on `/joint_commands_lerobot` by default and returns six `<joint>.pos`
features in degrees, matching the SO-101 robot preset. Change action_topic,
joint_names, joint_name_map and use_degrees for another robot.

```python
from lerobot_teleoperator_ros2_joint import ROS2JointTeleoperator, ROS2JointTeleoperatorConfig
teleop = ROS2JointTeleoperator(ROS2JointTeleoperatorConfig(command_mode="continuous"))
teleop.connect()
action = teleop.get_action()
teleop.disconnect()
```

`command_mode="on_change"` (default) holds the last complete command until
another arrives. This matches control_gui publishers that send only changed
commands: an unmoving target remains a valid target. It cannot detect a stopped
publisher; use `continuous` with `action_stale_ms=500` for the recording mirror,
which publishes at a fixed rate. In that mode every joint must receive updates
within the age limit. Both modes merge partial messages and require all configured
joints before get_action succeeds. connect waits up to action_warmup_s but permits
a producer to arrive later; no zero-valued command is invented. QoS is configurable;
transient_local helps only when the producer also offers transient_local durability.
Default volatile QoS requires one new command after connecting. Context/executor
constructor arguments attach to external ROS resources. Feedback is empty.

Apache-2.0; see LICENSE/NOTICE. Derived from the Hugging Face LeRobot interfaces
and Exploring-VLAs' LeRobot fork e1e20109 (`so101_ros2`), Copyright 2026 The
HuggingFace Inc. team.
