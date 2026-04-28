# vla_SO-ARM101 — SO-ARM101 ROS2 Control Stack

ROS2 Humble packages for the SO-ARM101 5-DOF arm: MoveIt2 motion
planning, tkinter control GUI, geometric IK, drop motion pipeline.

> **Inherits**: see `../CLAUDE.md` for repo topology, build paths, macOS gotchas.

> **README.md is stale.** It still references ROS2 Humble + Ubuntu 22.04 and only 3 packages — that predates Phase 6. Don't trust it for current build instructions; use `docs/ROS2_MAC_SETUP.md` and `docs/LEROBOT_ROS2_MAC_SETUP.md` instead. Cleaning up README.md is a separate ticket.

## Cross-repo

Isaac Sim digital twin lives at `~/Documents/isaac-sim-mcp/` (branch
`so-arm101`). MCP socket on `localhost:8767`. The pick-and-drop
pipeline spans both repos — this one plans + executes motion;
isaac-sim-mcp provides physics, the cup/lego scene, and the
`/drop_poses` + `/objects_poses_sim` topics. Bring-up sequence and
MCP tool inventory live in `~/Documents/isaac-sim-mcp/CLAUDE.md`.

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

## Build & launch

The canonical launchers live in `mac-env/scripts/` and wrap pixi + ROS2 +
the right env. **Use them instead of calling `ros2 launch` directly** —
they handle CycloneDDS config, `KMP_DUPLICATE_LIB_OK`, the workspace
sourcing, and Terminal.app handoff for cv2 viewers.

```bash
# === Sim stack (Gazebo / Isaac / mock) — single mode-aware launcher ===
# stack.launch.py: core (RSP + move_group + control_gui + RViz) + ONE backend.
bash mac-env/scripts/stack_start.sh --mode gazebo --gazebo-gui --rviz   # full sim
bash mac-env/scripts/stack_start.sh --mode gazebo                       # headless server only
bash mac-env/scripts/stack_start.sh --mode isaac --rviz                 # core+RViz, Isaac runs externally
bash mac-env/scripts/stack_start.sh --mode real --rviz \                # real arm + cameras + aruco/yoloe
    follower_port:=/dev/tty.usbmodem<NNN>
bash mac-env/scripts/stack_stop.sh                                      # SIGINT + per-binary cleanup

# Legacy positional args still work (back-compat with scripts/test_*.sh,
# isaac-sim-mcp's restart-control-stack.sh, etc.):
bash mac-env/scripts/stack_start.sh headless    # = --mode gazebo
bash mac-env/scripts/stack_start.sh all         # = --mode gazebo --gazebo-gui --rviz

# Attach RViz to a running headless stack (uses moveit_rviz.launch.py
# so SRDF actually loads — direct rviz2 invocation skips the SRDF param)
bash mac-env/scripts/rviz_attach.sh
bash mac-env/scripts/rviz_stop.sh

# === Real-side perception (camera_publisher + ArUco + YOLOE) ===
# Auto-detects wrist USB cam by signature in robot_config.yaml.
bash mac-env/scripts/camera_pipeline_start.sh                       # auto-detect
bash mac-env/scripts/camera_pipeline_start.sh --camera-id 0         # explicit
bash mac-env/scripts/camera_pipeline_stop.sh
bash mac-env/scripts/fix_camera_permissions.sh                      # if Terminal can't open the cam

# === Builds ===
# Single package, fast iteration:
cd /tmp/soarm-ws && pixi run --manifest-path /tmp/mac-env/pixi.toml \
    colcon build --packages-select so_arm101_control
# (so_arm101_control is python — needs full rebuild after source edits because
# colcon's --symlink-install no longer works with setuptools 80+, see Conventions)

# === Trigger commands once a stack is up ===
ros2 service call /so_arm101_control_gui/grasp_home std_srvs/srv/Trigger
# Other ~/_cmd_* services auto-register from control_gui — see AGENT_DEBUG_GUIDE.md

# === Cross-repo restart (legacy, pre-dates stack_start.sh) ===
~/Documents/isaac-sim-mcp/scripts/restart-control-stack.sh
```

Launch graph (after the 2026-04-27 refactor):

```
mac-env/scripts/stack_start.sh
  └── ros2 launch so_arm101_control stack.launch.py mode:=…
        ├── core.launch.py            ← always: RSP + move_group + control_gui + ee/camera_pose_publisher + RViz
        └── ONE of:
            ├── backend_gazebo.launch.py  ← gz sim + parameter_bridge + controllers + sim_ground_truth
            ├── backend_isaac.launch.py   ← empty placeholder (Isaac publishes externally)
            └── backend_real.launch.py    ← jointstatereader + real_cameras + localize_aruco + localize_yoloe
                                            (motion-execution shim NOT YET IMPLEMENTED — planning works,
                                             execute path needs the trajectory_shim node)
                                            (pass `read_only:=true` for first-contact bringup —
                                             jointstatereader observation-only, no follower writes)
```

The old monolithic `gazebo.launch.py` is preserved unchanged for
back-compat — anything still calling it directly works. New code
should use `stack.launch.py` via `stack_start.sh`.

## Conventions

- **Joint names (canonical, used across so101_ros2 plugin + jointstatereader):** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_joint`. URDF/Gazebo also publishes `gripper_joint`; the `lerobot/so101_ros2` plugin's `joint_name_map` remaps to upstream's `gripper` for HF-dataset parity.
- **Use `pixi run --manifest-path /tmp/mac-env/pixi.toml ...`** to invoke `ros2`, `colcon`, `python` — never the system binaries. `mac-env/scripts/*.sh` already wrap this. (Mac/pixi path; Linux uses sourced ROS2 humble normally.)
- **Edit packages here, build at `/tmp/soarm-ws/`** — bootstrap symlinks source. After editing C++ or `setup.py`, `colcon build --packages-select <pkg>` from `/tmp/soarm-ws`. After editing pure-Python in an installed package, the install is *not* a symlink-install (setuptools 80+ removed `develop --editable`), so a rebuild is needed.

## Tiered deterministic motion planner

Every arm motion routes through `_joint_space_collision_free_execute`
(`src/so_arm101_control/so_arm101_control/control_gui.py:5817`):

1. **Tier 1** — `_plan_linear_joint_path` (line 5923) builds an
   N-waypoint linear interp, validates each via
   `_check_state_valid_with_contacts`, and dispatches the SAME N
   waypoints to `_execute_full_trajectory`. The validated path IS
   the executed path — no spline divergence between knots.
2. **Tier 2** — `_plan_retract_pan_settle` (line 5960) decomposes
   into 3 linear sub-segments: retract to `NEUTRAL_NON_PAN`, pan
   across, settle to target. Each sub-segment passes through
   tier-1; concatenated into one trajectory if all three clear.
3. **OMPL fallback** — `_ompl_plan_validate_execute` (line 6133).
   Opt-in per caller via `allow_ompl_fallback=True` (default).
   `_cmd_grasp_home` (line 5755) passes `False` — must use
   deterministic tiers, surface failure loudly instead of rolling
   RNG dice.

Per-motion `tracer.event('planner_used', which='linear|retract_pan_settle|ompl_fallback|none')`
records which planner ran. Read in /tmp/arm_traj/*.json or via
`/so_arm101_control_gui/get_log` service.

## Drop motion specifics

- **Attach offset captured at grasp time**: `_attach_lego_to_gripper`
  (line 5168) computes block-in-tcp_link pose and stores it on
  `self._attached_lego_tcp_offset`. `_cmd_drop_sweep` (line 6793)
  uses MEASURED `|ax|` for the gap→tcp shift, NOT theoretical
  `half_gap`. Off-center grasps shift the block 1-2 mm vs theoretical
  jaw center; planner compensates.
- **Pan lock on drop_sweep**: `_plan_collision_free_execute`
  (line 2176) accepts `lock_pan=<rad>` kwarg. drop_sweep passes
  current shoulder_pan so geometric IK on the shifted tcp_target
  doesn't introduce ~1° base yaw vs drop_point's pan. Mirrors
  `find_reachable_grasp_yaw`'s "single yaw across stages" pattern.
- **Hover above cup rim default 50 mm**. Block clears cup wall
  during the wrist_flex 90° → 55° sweep arc. User-tunable via
  Grasp tab → Drop section spinbox.
- **Cup padding default 5%** (`_CUP_COLLISION_PADDING = 1.05` at
  line 172). MoveIt planning scene only — Isaac Sim physics sees
  the real cup geometry. Absorbs controller tracking-lag overshoot
  during fast pan motions (post-drop grasp_home is the worst case
  at ~36°/s peak pan rate).

## Test harness (`scripts/`)

```bash
# Single QS cycle for one named lego
scripts/test_qs_cycle.sh red_2x3 [/tmp/log_path]

# Reset Isaac Sim scene + control_gui state to a clean baseline
scripts/sim_reset.sh
# Internally: qs_restart → detach_lego → MCP update_cups
#           → MCP randomize_object_poses → qs_refresh_all → grasp_home

# Sequentially pick-and-place every available lego
scripts/test_pick_all.sh /tmp/run_$(date +%Y%m%dT%H%M%S)
# Writes per-lego logs + summary.csv with PASS/FAIL + planner counts
```

`test_qs_cycle.sh` polls `/so_arm101_control_gui/get_log` for terminal
sentinels (`pick-and-drop cycle complete | Quickstart halted at |
Quickstart aborted`) and trims the captured log to per-cycle scope at
exit, so downstream consumers see only this-cycle counts.

## Key files

| Method / constant | File:Line | Role |
|---|---|---|
| `_joint_space_collision_free_execute` | `control_gui.py:5817` | Tiered planner entry point |
| `_plan_linear_joint_path` | `control_gui.py:5923` | Tier-1 linear interp + validate |
| `_plan_retract_pan_settle` | `control_gui.py:5960` | Tier-2 decomposition |
| `_ompl_plan_validate_execute` | `control_gui.py:6133` | Tier-3 OMPL with Mode A/B classification |
| `_plan_collision_free_execute` | `control_gui.py:2176` | IK + collision check + dispatch (drop_sweep, IK targets) |
| `_cmd_grasp_home` | `control_gui.py:5755` | Pure deterministic; `allow_ompl_fallback=False` |
| `_cmd_drop_sweep` | `control_gui.py:6793` | IK-planned with measured offset + pan lock |
| `_attach_lego_to_gripper` | `control_gui.py:5168` | Captures `_attached_lego_tcp_offset` |
| `_QS_SEQUENCE` | `control_gui.py:3424` | The 9-step pick-and-drop player flow |
| `_CUP_COLLISION_PADDING` | `control_gui.py:172` | Module-level cup pad multiplier |

## Gotchas

- **Feetech `Homing_Offset` (reg 31) is 12-bit sign-magnitude in EEPROM,
  NOT int16.** Bit 11 = sign, bits 0-10 = magnitude. lerobot's
  `motors_bus.py` decodes this internally before storing in JSON.
  Raw byte readers must apply the same decode or you'll see false
  "EEPROM ≠ JSON" mismatches on every joint with negative homing
  (most of them after a real calibration). Reference implementation:
  `mac-env/scripts/feetech_calib_audit.py:read_signed_word`.
- **Calibration MUST run with the stack stopped.** Even though
  `calibrate_arm.sh` `pkill`'s jsr first, control_gui's auto-Connect
  respawns its own jsr ~3s after `stack_start`. If lerobot-calibrate
  is still running then, jsr's 50 Hz reads of register 56 collide
  with lerobot's writes to register 31 → corrupted EEPROM. Workflow:
  `stack_stop → calibrate_arm.sh → feetech_calib_audit.py → stack_start`.
  Never run calibration with the stack up.
- **Calibration step 1's pose IS the new /joint_states zero**, regardless
  of the wizard's "middle of its range of motion" wording. The wizard
  records WHATEVER pose you press ENTER at as the homing reference.
  Press ENTER at URDF zero (straight horizontal) for /joint_states to
  match RViz at URDF zero. wrist_roll has no end-stop so it's the
  joint most likely to be off-zero by 90°+ if you don't visually align
  the gripper parallel to the arm before pressing ENTER.
- **`recenter_one` requires prior freedrive (torque off).** Direct call
  with torque ON makes the servo PD drive to a wrong position because
  Goal_Position is interpreted in post-Homing coords; changing Homing
  while torque is on means the same Goal value points to a new physical
  position. **For bulk EEPROM drift, use `lerobot-calibrate` (via
  `calibrate_arm.sh`)**, not recenter. An in-place "recenter to a
  non-zero target" operation without hand-positioning is architecturally
  impossible on Feetech servos for cable-loaded joints — Feetech doesn't
  expose a "rebase Goal with Homing" command, and torque-off cycles
  trade PD-driven motion for cable/gravity sag.
- **Tier-1 spline-divergence trap**: an earlier tier-1 implementation
  validated N waypoints but dispatched only `(start, end, duration)`
  — controller's spline took different paths. Current tier-1
  dispatches the FULL N-waypoint trajectory. Don't "optimize" by
  collapsing back to (start, end), and don't reuse `_execute_trajectory`
  (line 7167) for tier-1 — it's the (start, end, duration) path.
  Use `_execute_full_trajectory` (line 6249).
- **Padding asymmetry is intentional**: the MoveIt planning scene
  has padded cups, Isaac Sim physics doesn't. Don't compare
  `/check_state_validity` results to omni.physx contact telemetry
  as if they should agree — they intentionally don't.
- **`home_velocity_scale` is OMPL-only**: the param overrides
  `velocity_scale_var`, but `velocity_scale_var` is consumed only
  by `_ompl_plan_sync` (line 6228). Tier-1 / tier-2 use `duration_s`
  directly. So `home_velocity_scale` is dead code for the
  deterministic planner path. To slow grasp_home, edit `duration_s`
  on the `_joint_space_collision_free_execute` call inside
  `_cmd_grasp_home`.
- **MCP cup tools live across the repo boundary**: `update_cups`,
  `randomize_object_poses`, `sort_into_cups` are MCP tools in the
  isaac-sim-mcp repo, called via socket on localhost:8767. See
  `scripts/sim_reset.sh` for the call pattern.
- **Convention-based debug services**: every `_cmd_*` method on the
  control_gui node auto-registers as a `/so_arm101_control_gui/<name>`
  Trigger service (the leading `_cmd_` is stripped). GUI button
  click and service call hit the SAME code via `_make_trigger_callback`
  (line 864) which schedules the call on the Tk thread. Both paths
  clear `_motion_event` / `_last_motion_status` / `_cmd_error`
  before dispatch. So the script harness and GUI buttons produce
  identical behavior.

## TODO: cv2 toolbar crash on macOS (2026-04-27)

`localize_aruco`'s "Merged Detection" cv2 window keeps the default OpenCV
toolbar (zoom / save / copy / properties buttons), but **clicking any of
those buttons crashes the process** on macOS due to a long-standing
OpenCV-Qt threading bug — `NSUndoManager._endTopLevelGroupings` is called
from the toolbar handler but is only legal on the AppKit main thread, and
rclpy's daemon spin thread violates that assumption.

References (all open / wontfix): opencv-python#134, opencv-python#376,
opencv#18087, opencv#12903, opencv#22561.

Mitigations in place today:
- Press `s` in the window to save a timestamped PNG to `/tmp/aruco_captures/`
  (custom keypress handler in `merged_localization_aruco.py`)
- macOS `Cmd+Shift+4` for arbitrary screenshots

Proper fix later: either (a) split the viewer into a separate
single-threaded subprocess that doesn't share rclpy's runloop, or
(b) replace cv2.imshow with a tk/Qt viewer we control. Until one of
those lands, just don't click the toolbar buttons.

## Camera resolution: 640×480 throughout (2026-04-27)

Project-wide standard for all wrist + top cameras (sim and real) is
**640×480** — set in `aruco_camera_localizer/config/robot_config.yaml`
under `so_arm101.camera.width/height`. The decision rationale: YOLOE
inference saturates 4 macOS cores at 720p but sits at ~100 % of a single
core at 480p, and 480p is still plenty for ArUco marker detection +
lego/cup color blobs at typical wrist-cam working distances.

The earlier 1280×720 setting (committed before 2026-04-27) tried to
match RealSense's native res for sim/real parity. We're explicitly
choosing 480p over that parity for performance reasons; downstream
consumers (lerobot dataset features, MoveIt planning scene, control_gui
overlay scaling) all read intrinsics from this same yaml, so the matrix
stays self-consistent.

**TODO**: make sure every place in the pipeline that pins a
resolution agrees with 640×480 — as of 2026-04-27 the live perception
path does (single source of truth via this yaml), but verify on next
pass:

- `vla_SO-ARM101/src/so_arm101_description/worlds/lego_world.sdf` —
  sim `wrist_camera` and `top_camera` sensor `<width>` / `<height>`
- `vla_SO-ARM101/src/so_arm101_bringup/launch/real_cameras.launch.py` —
  real cam publisher launch-arg defaults (`width:=640 height:=480`)
- any lerobot `dataset_features` that pin width/height for recording

## Tracking-lag context

`docs/notes/2026-04-24-forum-findings-tracking-lag.md` (in
isaac-sim-mcp repo) documents the SO-ARM101 controller tracking-lag
investigation that motivated the deterministic planner + 5% padding
combination. wrist_flex tracks 10-13° behind commanded position at
high velocities; shoulder_pan accumulates 22° at peak ref_vel. The
planner produces conservative paths, padding absorbs residual lag,
and tier-2 decomposition handles the "post-drop grasp_home swings
near another cup" scenario.

## Related docs

- `docs/ROS2_MAC_SETUP.md` — bootstrap (project-agnostic)
- `docs/LEROBOT_ROS2_MAC_SETUP.md` — record runbook (sim + real)
- `docs/grasp_pipeline.md` — control_gui → MoveIt grasp flow
- `docs/AGENT_DEBUG_GUIDE.md` — debugging the sim stack
- `docs/GAZEBO_LINUX_LIMITATION.md` — Gazebo on Linux is broken by Phase 9's PD feed-forward yaml; revisit when needed
- `../lerobot/ROS2_PLUGINS.md` — porting the `so101_ros2` plugin to a new arm
