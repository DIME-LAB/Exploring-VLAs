# vla_SO-ARM101 — agent context

This file gives agents the per-package mental model for the SO-ARM101 ROS2 stack.

> **Inherits**: see `../CLAUDE.md` for repo topology, build paths, macOS gotchas.

> **README.md is stale.** It still references ROS2 Humble + Ubuntu 22.04 and only 3 packages — that predates Phase 6. Don't trust it for current build instructions; use `docs/ROS2_MAC_SETUP.md` and `docs/LEROBOT_ROS2_MAC_SETUP.md` instead. Cleaning up README.md is a separate ticket.

## Six packages, one role each

| Package | Role | Built into `/tmp/soarm-ws/install/`? |
| --- | --- | --- |
| `so_arm101_description` | URDF + SDF (Gazebo `lego_world.sdf`), meshes, robot_state_publisher launch | Yes |
| `so_arm101_moveit_config` | MoveIt config (SRDF, kinematics, OMPL/Pilz), `move_group.launch.py`, `demo.launch.py` | Yes |
| `so_arm101_control` | `control_gui` (tk-based), `gazebo.launch.py` (the master sim launcher), ros2_control config, ee_pose_publisher, camera_pose_publisher | Yes |
| `jointstatereader` | Real-hardware Feetech serial bridge — reads leader/follower servos, publishes `/joint_states` + `/joint_commands`, optional mirror or topic-driven follower write | Yes |
| `so_arm101_bringup` | **Real-camera bringup** — `camera_publisher` (cv2.VideoCapture → ROS2 Image), `real_cameras.launch.py` (wrist + optional top). Self-contained — no aruco_camera_localizer dependency | Yes (Phase 6-02) |
| `sim_ground_truth` | **Sim ground-truth** — gz.transport13 → `/objects_poses_sim`, `/objects_bbox_sim`, `/drop_poses` mirroring aruco real-side contract. Catalog at `config/lego_world_objects.yaml`. | Yes (Phase 6-04) |

## Sim-world layout (`so_arm101_description/worlds/lego_world.sdf`)

The world contains:

- **`ground_plane`** + lights + GUI camera
- **3 lego models** (`red_lego_2x4`, `green_lego_2x3`, `blue_lego_2x2`) — graspables
- **3 cup models** (`cup_red`, `cup_green`, `cup_blue`) — drop containers, 50 mm dia × 50 mm tall, static cylinders
- **3 sibling drop frames** (`drop_red`, `drop_green`, `drop_blue`) — top-level frame-only models at the ArUco-marker world positions, rpy `(0, π/2, 0)` so local +Z = world +X (marker normal radially outward)

> **Why drop_<color> are sibling top-level models, not child links of the cups:** Gazebo's `Pose_V` publishes child-link poses **relative to the parent model**. Top-level models get **absolute world poses**. Isaac Sim does the equivalent with wrapper Xform prims at the marker's world transform — same pattern, different substrate. Documented in the SDF comments + `sim_ground_truth/sim_ground_truth/ground_truth_publisher.py`.

The world name in the SDF is **`so_arm101_lego_world`** (the SDF *filename* is `lego_world.sdf`, but `<world name="...">` differs — affects `gz topic` namespacing). When configuring `sim_ground_truth`, use `world_name:=so_arm101_lego_world`.

## isaac-sim-mcp parity

This stack mirrors the **`inbarajaldrin/isaac-sim-mcp@so-arm101` phase 01** topic contract so a single `control_gui` consumes either source:

| Topic | Producer (sim, ours) | Producer (sim, isaac) | Producer (real) |
| --- | --- | --- | --- |
| `/objects_poses_sim` (TFMessage) | `sim_ground_truth` | `isaac-sim-mcp` action graph | `aruco_camera_localizer` (with `objects_poses_topic:=/objects_poses_real`) |
| `/objects_bbox_sim` (String JSON) | `sim_ground_truth` | `isaac-sim-mcp` | `aruco_camera_localizer` (`objects_bbox_topic:=/objects_bbox_real`) |
| `/drop_poses` (TFMessage) | `sim_ground_truth` | `isaac-sim-mcp` action graph | `aruco_camera_localizer --drop` |

`child_frame_id` convention is uniform: object name (`red_lego_2x4`, etc.) for `/objects_poses_*`, `drop_red`/`drop_green`/`drop_blue` for `/drop_poses`. Drop-height offset math (above rim, inward to cup center) lives in the **control package**, not the publisher — same code works on sim or real.

## Conventions

- **Joint names (canonical, used across so101_ros2 plugin + jointstatereader):** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_joint`. URDF/Gazebo also publishes `gripper_joint`; the `lerobot/so101_ros2` plugin's `joint_name_map` remaps to upstream's `gripper` for HF-dataset parity.
- **Use `pixi run --manifest-path /tmp/mac-env/pixi.toml ...`** to invoke `ros2`, `colcon`, `python` — never the system binaries. `mac-env/scripts/*.sh` already wrap this.
- **Edit packages here, build at `/tmp/soarm-ws/`** — bootstrap symlinks source. After editing C++ or `setup.py`, `colcon build --packages-select <pkg>` from `/tmp/soarm-ws`. After editing pure-Python in an installed package, the install is *not* a symlink-install (setuptools 80+ removed `develop --editable`), so a rebuild is needed.

## Related docs

- `docs/ROS2_MAC_SETUP.md` — bootstrap (project-agnostic)
- `docs/LEROBOT_ROS2_MAC_SETUP.md` — record runbook (sim + real)
- `docs/grasp_pipeline.md` — control_gui → MoveIt grasp flow
- `docs/AGENT_DEBUG_GUIDE.md` — debugging the sim stack
- `../lerobot/ROS2_PLUGINS.md` — porting the `so101_ros2` plugin to a new arm
