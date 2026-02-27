# SO-ARM101 ROS2 MoveIt Control Stack

ROS2 Humble packages for the SO-ARM101 5-DOF robot arm with MoveIt2 motion planning, GUI control, and LeRobot/VLA integration readiness.

## Packages

| Package | Description |
|---|---|
| `so_arm101_description` | URDF, meshes, and display launch |
| `so_arm101_moveit_config` | MoveIt2 config (SRDF, kinematics, controllers, RViz) |
| `so_arm101_control` | GUI, servo driver, EE pose publisher, IK/planning tests |

## Robot

- **5-DOF arm**: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll (STS3215 servos)
- **1 gripper**: Jaw
- **IK solver**: [pick_ik](https://github.com/PickNikRobotics/pick_ik) (`rotation_scale: 0.5` for 5-DOF)
- **Planner**: OMPL (RRTConnect)

## Prerequisites

```bash
# ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop

# Required packages
sudo apt install \
  ros-humble-moveit \
  ros-humble-pick-ik \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers

# CycloneDDS (recommended for Docker, fixes DDS discovery issues)
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
source ~/.bashrc
```

## Build

```bash
# Create workspace and symlink/copy src packages
mkdir -p ~/ros2_ws/src
ln -s /path/to/vla_SO-ARM101/src/so_arm101_description ~/ros2_ws/src/
ln -s /path/to/vla_SO-ARM101/src/so_arm101_control ~/ros2_ws/src/
ln -s /path/to/vla_SO-ARM101/src/so_arm101_moveit_config ~/ros2_ws/src/

# Build
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Launch

### MoveIt Demo (RViz + planning + simulated controllers)

```bash
ros2 launch so_arm101_moveit_config demo.launch.py
```

This launches move_group, RViz with MotionPlanning plugin, and ros2_control with fake hardware. Drag the interactive marker arrows in RViz to set goal positions, then click **Plan & Execute**.

### Full Stack (GUI + MoveIt + RViz + EE publisher)

```bash
ros2 launch so_arm101_control control.launch.py
```

This launches everything: MoveIt move_group, ros2_control, RViz, control GUI, and EE pose publisher in a single command.

The GUI has two modes:
- **Direct**: sliders move the robot immediately via joint trajectory controller
- **Planning**: sliders set a goal, then "Plan & Execute" uses MoveIt to plan and execute

### IK / Planning Tests

```bash
# IK benchmark (10 poses, position-only vs full)
ros2 run so_arm101_control test_ik_solvers

# IK + collision check + OMPL planning diagnostic
ros2 run so_arm101_control test_planning
```

## Key Configuration

### Kinematics (`so_arm101_moveit_config/config/kinematics.yaml`)

```yaml
arm:
  kinematics_solver: pick_ik/PickIkPlugin
  rotation_scale: 0.5           # Low orientation weight (5-DOF can't do full 6-DOF)
  orientation_threshold: 0.1
  minimal_displacement_weight: 0.001
```

- `rotation_scale: 0.5` keeps orientation influence low — the 5-DOF arm has 3 position DOFs + 2 orientation DOFs (pitch + tool roll), insufficient for full 6-DOF pose control

### RViz (`so_arm101_moveit_config/config/moveit.rviz`)

- `MoveIt_Allow_Approximate_IK: true` — allows approximate IK solutions
- `MoveIt_Use_Constraint_Aware_IK: true` — prevents self-collision IK solutions

### Hardware

For real hardware, edit `so_arm101_moveit_config/config/so_arm101.ros2_control.xacro` to set the serial port for the STS3215 servo bus. The `servo_driver` node in `so_arm101_control` handles communication.
