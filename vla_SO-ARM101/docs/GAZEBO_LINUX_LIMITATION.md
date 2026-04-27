# Gazebo Backend on Linux — Per-Backend Controller Config Split

> **Status (2026-04-27): RESOLVED.** Gazebo simulation runs end-to-end on
> Linux via a per-backend `ros2_controllers_*.yaml` split. `arm_controller`
> activates, `grasp_home` executes (verified: 51-waypoint tier-1 trajectory
> completed against `gz_ros2_control` → Gazebo physics). Verified pipeline:
> Gazebo cameras (`/wrist_camera` + `/top_camera`, ~7 Hz on Linux Fortress)
> → `ros_gz_bridge` → `aruco_camera_localizer` + YOLOE → `/aruco_poses_real`
> + `/objects_poses_real` + `/yoloe_annotated`. Same `gazebo.launch.py`
> entry point; the user just needs to set `GZ_SIM_SYSTEM_PLUGIN_PATH` to the
> locally-built `gz_ros2_control` lib (Mac via pixi handles this through
> `CONDA_PREFIX` automatically).

## TL;DR — what changed

Gazebo and Isaac Sim now use separate ros2_control YAMLs:

| YAML | Used by | `arm_controller.command_interfaces` |
|---|---|---|
| `config/ros2_controllers.yaml` | Isaac Sim path (`control.launch.py` + `ros2_control_node`) | `[position, velocity]` — Phase 9 PD feed-forward intact |
| `config/ros2_controllers_gazebo.yaml` | Gazebo path (`gazebo.launch.py` → URDF `<plugin>` `<parameters>` tag) | `[position]` — Linux gz_ros2_control compatible |

The Gazebo URDF (`so_arm101.gazebo.xacro`) explicitly references the
gazebo-specific YAML in its `<gazebo>` plugin block. Isaac Sim's path is
untouched.

## Why a split, not a single yaml

The Phase 9 commit `eb13c10` added `velocity` to `arm_controller`'s
command interfaces because Isaac Sim's action graph reads
`velocityCommand` off `/joint_states` and writes it to
`UsdPhysics.DriveAPI.targetVelocity`, giving the PhysX PD drive its
feed-forward term. That eliminates `B*v/K` steady-state ramp lag (~0.9°
wrist_flex lag measured pre-fix → ~2.5 mm TCP error during drop_sweep).

Gazebo's `gz_ros2_control/GazeboSimSystem` has **no analogous PD-feedforward
feedback loop**. The `velocity` command interface would be silently
ignored by the plugin even on a multi-mode-capable version, providing zero
benefit to Gazebo. So semantically, Gazebo *should* use position-only.

The local Linux build of `gz_ros2_control` (from `~/ros2_ws/install/`,
Humble + Ignition Fortress era) goes one step further: it doesn't tolerate
multi-command-interface joints at all. When the URDF declares more than
one `<command_interface>` per joint, the plugin silently fails URDF
parsing and exposes ZERO interfaces — breaking `joint_state_broadcaster`,
`arm_controller`, and `gripper_controller` simultaneously. So a single
shared YAML with `[position, velocity]` doesn't work on Linux.

The fix is therefore **both architecturally appropriate** (per-backend
config matches the underlying conceptual model — Gazebo and Isaac Sim
aren't the same hardware) **and toolchain-pragmatic** (sidesteps the
gz_ros2_control parse limitation without regressing Phase 9's Isaac Sim
optimization).

## Files involved

| File | Role |
|---|---|
| `src/so_arm101_moveit_config/config/ros2_controllers.yaml` | Default — Isaac Sim path, position+velocity |
| `src/so_arm101_moveit_config/config/ros2_controllers_gazebo.yaml` | NEW — Gazebo path, position-only |
| `src/so_arm101_description/urdf/so_arm101.gazebo.xacro` | UPDATED — `<gazebo><plugin><parameters>` now points at the gazebo-specific yaml |

CMakeLists already installs the entire `config/` directory, so both
YAMLs are picked up by `colcon build` without explicit per-file install
rules.

## Linux launch invocation

```bash
cd ~/Projects/Exploring-VLAs/vla_SO-ARM101
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash       # for the locally-built gz_ros2_control
source install/setup.bash                  # for so_arm101_*

# point Gazebo at the locally-built gz_ros2_control plugin lib
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ros2_ws/install/gz_ros2_control/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

ros2 launch so_arm101_control gazebo.launch.py headless:=true rviz:=false
```

(Mac via `mac-env/scripts/bootstrap.sh` + pixi handles the plugin path
automatically through `CONDA_PREFIX`. The xacro reads `CONDA_PREFIX`
when set and falls back to the existing `GZ_SIM_SYSTEM_PLUGIN_PATH`
otherwise — so this Linux invocation is the manual equivalent.)

## Operating notes (gotchas)

### 1. Orphan `robot_state_publisher` from a prior Isaac Sim launch

If you've been running the Isaac Sim control stack
(`ros2 launch so_arm101_control control.launch.py`) before switching to
Gazebo, the launch's `robot_state_publisher` child can survive the
parent dying and become reparented to PID 1. It keeps publishing the
**moveit URDF** (with `mock_components/GenericSystem` plugin) on
`/robot_description`. When `gz_ros2_control` connects to
"the" robot_state_publisher and asks for `robot_description`, it can
hit the orphan and try to load `mock_components/GenericSystem` as a
GazeboSimSystem — which fails with:

```
[gz_ros2_control]: The plugin failed to load for some reason.
  Error: According to the loaded plugin descriptions the class
  mock_components/GenericSystem with base class type
  gz_ros2_control::GazeboSimSystemInterface does not exist.
```

**Detection**: `ps -ef | grep robot_state_publisher` after stopping the
Isaac Sim launch — if there are entries with elapsed time > a few minutes
that have no parent (PPID = 1), kill them.

**Cleanup**: `kill -SIGTERM <pid>` (graceful), then SIGKILL only if it
survives. robot_state_publisher is headless, no X11 cleanup concerns.

### 2. Linux Fortress camera frame rate

The wrist + top cameras are declared at 30 Hz in the SDF but actually
publish at ~7 Hz on Linux/Fortress. This is the Ogre2 renderer's cap on
this hardware with two cameras sharing the render context. Mac via
Harmonic + Vulkan typically hits closer to 25-30 Hz. Detection latency
will feel different between the platforms — that's the renderer, not the
detection code.

### 3. SDF camera_info topic naming asymmetry

The wrist camera SDF declares
`<camera_info_topic>wrist_camera/camera_info</camera_info_topic>` but
older Fortress versions flatten relative paths in this declaration, so
the actual GZ topic ends up at `/camera_info` (root level). The
`ros_gz_bridge` config in `gazebo.launch.py` was written to match this
flattening (`/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo`).
If `aruco_camera_localizer` auto-discovers `<camera_topic>/camera_info`
it'll look for `/wrist_camera/camera_info` and miss; the localizer falls
back to its own `robot_config.yaml` intrinsics (which we already use, so
this is harmless in current config). If pose accuracy ever looks off,
the workaround is `--ros-args -r /wrist_camera/camera_info:=/camera_info`
on the localizer launch.

## When the split could be collapsed back to one yaml

The split exists primarily to sidestep the locally-built gz_ros2_control
limitation. If/when one of these happens, the gazebo-specific yaml can
be deleted and the gazebo xacro can point at the default:

1. **Linux upgrades to a Jazzy-era `gz_ros2_control`** (≥ ~1.2.x) that
   tolerates multi-command-interface joints — same single YAML works
   for both backends because newer gz_ros2_control just ignores the
   unused `velocity` declaration. Likely path: upgrade entire ROS2
   distro to Jazzy, or build Jazzy version from source on Humble (ABI
   risk).
2. **Phase 9's `eb13c10` velocity command interface stops being
   load-bearing on Isaac Sim** (e.g., a different tracking-lag fix
   replaces PD feed-forward) — at that point dropping `velocity` from
   the default YAML restores backend symmetry without regression.

Until then, the split stays.

## Verification status (2026-04-27)

- **Isaac Sim path**: `scripts/test_qs_cycle.sh red_2x3` — full
  pick-and-drop cycle PASSED on the merged code (tier-1 4×, tier-2 2×,
  pan-lock, `_attached_lego_tcp_offset` capture).
- **Gazebo Linux path**: `gazebo.launch.py headless:=true` →
  `ros2 service call /so_arm101_control_gui/grasp_home` →
  `success=True, message='_cmd_grasp_home: trajectory complete (51 wps)'`.
  Camera bridge alive (`/wrist_camera`, `/top_camera` both publishing
  rgb8 640×480 at ~7 Hz). Detection pipeline verified end-to-end:
  YOLOE → `/objects_poses_real` (1.4 Hz),
  ArUco → `/aruco_poses_real` (~93 Hz with Kalman extrapolation),
  drop poses → `/drop_poses_real`.
- **Gazebo Mac path**: still untested. Mac uses RoboStack jazzy's
  `gz_ros2_control` 1.2.17 which likely tolerates multi-command-mode,
  but the per-backend yaml split works on it too — the position-only
  yaml is correct for Gazebo on either platform regardless of plugin
  version.
- **Real hardware**: untested (separate scope).

## Pre-merge rollback anchors (in case of need)

- Pre-merge git tag in `Exploring-VLAs/`: `pre-merge-snapshot` (points
  to `f131e48`) — the state before the upstream merge but also before
  this Gazebo fix.
- Quarantined merge originals (base / local / upstream / merged-with-markers
  for every conflicted file plus the auto-merged `control.launch.py`):
  `~/Documents/merge-quarantine/vla-SO-ARM101-2026-04-27/`.
- Decision log for each merge resolution:
  `~/Documents/merge-quarantine/vla-SO-ARM101-2026-04-27/DECISIONS.md`.
