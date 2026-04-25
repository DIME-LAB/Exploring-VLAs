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
- **Motion planner**: tiered deterministic — linear joint-space → retract-pan-settle decomposition → OMPL fallback (opt-in per primitive). See "Motion Planning" below.

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

### Pick-and-place test harness (`scripts/`)

End-to-end automated cycle testing without GUI interaction. Three scripts, all relying on the auto-registered `_cmd_*` debug services:

```bash
# Single QS cycle for one named lego (PASS/FAIL + per-cycle planner counts)
scripts/test_qs_cycle.sh red_2x3                    # default log: /tmp/qs_cycle_test.log

# Reset Isaac Sim scene + control_gui state to a clean starting point
scripts/sim_reset.sh                                # qs_restart → detach → MCP update_cups
                                                    # → MCP randomize_object_poses
                                                    # → qs_refresh_all → grasp_home

# Sequentially pick-and-place every available lego, log per-lego verdicts
scripts/test_pick_all.sh /tmp/pick_all_$(date +%Y%m%dT%H%M%S)
                                                    # writes per-lego logs + summary.csv
                                                    # rows: lego, PASS/FAIL, elapsed_s,
                                                    #   tier1_n, tier2_n, ompl_n,
                                                    #   refused_n, halt_step, halt_reason
```

`test_qs_cycle.sh` polls `/so_arm101_control_gui/get_log` for terminal sentinels (`pick-and-drop cycle complete`, `Quickstart halted at`, `Quickstart aborted`) and trims the captured log to per-cycle scope before exiting, so downstream consumers (e.g. `test_pick_all.sh`) see only this-cycle counts. `sim_reset.sh` calls the Isaac Sim MCP socket on `localhost:8767` for `update_cups` and `randomize_object_poses`. All scripts work over ROS2 services on port 7400 — no GUI focus required.

## Key Configuration

### Motion Planning — tiered deterministic planner (`control_gui.py: _joint_space_collision_free_execute`)

Every arm motion (grasp_home, grasp_move, drop_point, drop_sweep, IK targets) routes through `_joint_space_collision_free_execute`, which tries planners in order and dispatches the first one that returns a clean trajectory:

1. **Tier 1 — direct linear joint-space interp.** `_plan_linear_joint_path` builds an N-waypoint (default 50) `JointTrajectory` interpolating from start to target, validates every waypoint with `_check_state_valid_with_contacts` (same `/check_state_validity` service the OMPL post-check uses — no coherence gap), and dispatches the *same* trajectory to `_execute_full_trajectory`. Earlier tier-1 attempts in the codebase validated N waypoints but executed only `(start, end, duration)`, letting the controller's spline interpolate paths that clipped collision objects between knots — fixed by sending the full N-waypoint trajectory.
2. **Tier 2 — retract-pan-settle decomposition.** `_plan_retract_pan_settle` splits into three linear sub-segments: (a) retract from current to `(current_pan, NEUTRAL_NON_PAN)`, (b) pan across to `(target_pan, NEUTRAL_NON_PAN)`, (c) settle from there to target. Each sub-segment passes through tier-1 individually; concatenated into one trajectory if all three clear. Handles "need to pan past a cup" and "post-drop grasp_home swings near another cup" cases geometrically without sampling.
3. **Tier 3 — OMPL fallback** (`_ompl_plan_validate_execute`). Opt-in per caller via `allow_ompl_fallback=True` (default). `_cmd_grasp_home` opts OUT — RRTConnect's RNG variance has produced cup-clipping plans from far-pan start poses, and tier-1/tier-2 must suffice for grasp_home. On both-tier failure with fallback disabled, returns a `tiered_planner_exhausted` verdict.

Each motion emits a `tracer.event('planner_used', which='linear|retract_pan_settle|ompl_fallback|none')` record so post-run analysis can see which planner actually ran.

### Drop motion — attach-offset-aware target + pan lock (`_cmd_drop_sweep`)

The held lego's pose in `tcp_link` frame is captured at `_attach_lego_to_gripper` time and stored on `self._attached_lego_tcp_offset`. `_cmd_drop_sweep` reads this and uses the measured `|ax|` as the gap→tcp shift magnitude, instead of the theoretical `half_gap` from `BASELINE_JAW_GAP + JAW_GAP_RATE * gripper_joint`. Off-center grasps (typical: −11 to −13 mm in TCP X) shift the block by 1-2 mm vs the theoretical jaw center, which previously surfaced as cup-wall penetration during the wrist sweep.

Drop sweep also passes `lock_pan=current_pan` to `_plan_collision_free_execute`, suppressing the ~1° base yaw that otherwise appears because IK on the shifted tcp_target gives a slightly different pan than `drop_point` set. Mirrors the "single yaw across stages" pattern in `find_reachable_grasp_yaw`.

### Cup collision padding (`_CUP_COLLISION_PADDING`, default 1.05 = 5%)

The cup collision objects in the MoveIt planning scene are scaled by `_CUP_COLLISION_PADDING` at load time. Default 5% — absorbs the controller's tracking-lag overshoot during fast pan motions (post-drop `grasp_home` rotates ~107° at peak velocity, where 1% wasn't enough). Padding only enlarges cups in the *planning* scene; Isaac Sim's physics simulator continues to use the real cup geometry, so contact telemetry remains honest. User-tunable via the "Collision padding %" spinbox in the RViz tab → Cups section.

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
