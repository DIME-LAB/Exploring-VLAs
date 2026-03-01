# SO-ARM101 ROS2 MoveIt Control Stack

ROS2 Humble packages for the SO-ARM101 5-DOF robot arm with MoveIt2 motion planning, GUI control, and LeRobot/VLA integration readiness.

## Packages

| Package | Description |
|---|---|
| `so_arm101_description` | URDF, meshes, and display launch |
| `so_arm101_moveit_config` | MoveIt2 config (SRDF, kinematics, controllers, RViz) |
| `so_arm101_control` | GUI, servo driver, geometric IK, grasp pipeline, EE/camera pose publishers |

## Robot

- **5-DOF arm**: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll (STS3215 servos)
- **1 gripper**: Jaw (single moving jaw)
- **Grasp IK**: Geometric IK solver — analytical 2-link law-of-cosines with gripper-down constraint (θ₂+θ₃+θ₄=90°), FK refinement step (<0.5mm error)
- **MoveIt IK**: [pick_ik](https://github.com/PickNikRobotics/pick_ik) (`rotation_scale: 0.5`) for non-grasp motion planning
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

### Control Stack (GUI + MoveIt + mock hardware)

```bash
# Default: simulation mode, no RViz
ros2 launch so_arm101_control control.launch.py

# With RViz
ros2 launch so_arm101_control control.launch.py rviz:=true

# With Isaac Sim topics (sets default object pose/bbox topics)
ros2 launch so_arm101_control control.launch.py use_sim:=true

# Real hardware
ros2 launch so_arm101_control control.launch.py real_hardware:=true serial_port:=/dev/ttyACM0
```

Launches MoveIt move_group, ros2_control (mock or real), control GUI, and EE/camera pose publishers. RViz is off by default — enable with `rviz:=true`.

### Gazebo Simulation (Ignition + full stack)

```bash
# Default: Gazebo + RViz
ros2 launch so_arm101_control gazebo.launch.py

# Without RViz
ros2 launch so_arm101_control gazebo.launch.py rviz:=false
```

Spawns the robot in Ignition Gazebo with physics, loads ros2_control controllers, and launches the full stack (MoveIt + GUI + RViz).

### MoveIt Demo (RViz + planning only)

```bash
ros2 launch so_arm101_moveit_config demo.launch.py
```

RViz with MotionPlanning plugin and ros2_control with fake hardware. Drag interactive markers to set goals, click **Plan & Execute**.

### URDF Viewer

```bash
ros2 launch so_arm101_description display.launch.py
```

Robot state publisher + joint slider GUI + RViz. No MoveIt, no control.

### Utilities

```bash
# Compute workspace bounds (general + top-down grasp)
ros2 run so_arm101_control compute_workspace

# Calibrate jaw gap model from STL mesh
ros2 run so_arm101_control calibrate_jaw

# IK benchmark
ros2 run so_arm101_control test_ik_solvers
```

## Key Configuration

### Geometric IK (`so_arm101_control/compute_workspace.py`)

The grasp pipeline uses an analytical geometric IK solver instead of MoveIt/pick_ik. Constants are derived from the URDF FK chain and verified by `calibrate_ik.py`:

- Link lengths: `L_UPPER=0.116m`, `L_LOWER=0.135m` (shoulder-to-elbow, elbow-to-wrist)
- Gripper-down constraint: `θ₂+θ₃+θ₄=90°` forces top-down grasp orientation
- Pan decoupling: `θ₁=atan2(-y, x-X_PAN)` reduces to 2D arm-plane IK
- Wrist roll: analytically computed from desired grasp yaw
- FK refinement: one Newton step corrects cross-plane coupling to <0.5mm

### MoveIt Kinematics (`so_arm101_moveit_config/config/kinematics.yaml`)

```yaml
arm:
  kinematics_solver: pick_ik/PickIkPlugin
  rotation_scale: 0.5           # Low orientation weight (5-DOF can't do full 6-DOF)
  orientation_threshold: 0.1
  minimal_displacement_weight: 0.001
```

- `rotation_scale: 0.5` keeps orientation influence low — the 5-DOF arm has 3 position DOFs + 2 orientation DOFs (pitch + tool roll), insufficient for full 6-DOF pose control
- Used for non-grasp motion planning (joint-space moves, Cartesian path following)

### RViz (`so_arm101_moveit_config/config/moveit.rviz`)

- `MoveIt_Allow_Approximate_IK: true` — allows approximate IK solutions
- `MoveIt_Use_Constraint_Aware_IK: true` — prevents self-collision IK solutions

### Hardware

For real hardware, edit `so_arm101_moveit_config/config/so_arm101.ros2_control.xacro` to set the serial port for the STS3215 servo bus. The `servo_driver` node in `so_arm101_control` handles communication.
