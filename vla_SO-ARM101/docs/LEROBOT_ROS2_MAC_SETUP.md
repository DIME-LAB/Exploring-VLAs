# LeRobot + ROS2 Integration on macOS (Apple Silicon)

Living guide for the lerobot + ROS2 camera/dataset work inside `Exploring-VLAs`.
The target is a single unified data path: sim frames (Gazebo) and real frames
(USB) both enter lerobot recording via ROS2 topics, producing datasets ready
for SmolVLA / Pi0.5 co-training.

## Who this doc serves (pick your path)

| If you want to... | Read sections |
|---|---|
| Bring up the **SO-ARM101 Gazebo sim stack** with nothing else | `Setup from scratch on Mac` (steps 1–4). Then run `stack_start.sh gz`. Everything after that is optional. |
| **Record a lerobot v3 dataset** end-to-end (sim or real) | `Setup from scratch on Mac` → `Record a dataset from scratch (end-to-end runbook)` |
| Debug a **stuck controller / missing topic** | `Known gotchas` + `SO-ARM101 topic reference` |
| Set up the **real SO-ARM101** on ROS2 topics (V2-REAL-UNIFIED) | Not yet documented — Linux follow-up. `jointstatereader` package already bridges real hardware; see scope plan in REQUIREMENTS.md |

## Target

One recording pipeline that works for **sim and real** via ROS2 topics:

```
SO-ARM101 (sim or real)   →  ROS2 topics  →  lerobot-record  →  HF dataset (v3)
  • /wrist_camera         →  lerobot.cameras.ros2.ROS2Camera
  • /joint_states         →  (upcoming) lerobot_robot_so101_ros2
  • /joint_commands       →  (upcoming) lerobot_teleoperator_so101_ros2
```

## Progress

| Step | State | Notes |
|---|---|---|
| Fork of huggingface/lerobot | ✅ | `github.com/inbarajaldrin/lerobot` — branches: `ros2_camera` (PR #866 archive), `ros2-camera-on-main` (current work) |
| Submodule in `Exploring-VLAs/lerobot` | ✅ | remotes: `origin` = user fork, `yadunund`, `upstream` = huggingface/lerobot |
| Rebase onto upstream main (Phase 1 — FOUND-01) | ✅ | `ros2-camera-on-main` branched off `upstream/main`, planning cherry-picked |
| ROS2 camera ported to new layout (FOUND-02) | ✅ | `src/lerobot/cameras/ros2/` — matches zmq subpackage pattern |
| Pixi env inside this repo (FOUND-03) | ✅ | `Exploring-VLAs/mac-env/` (Python 3.12 + ROS2 Jazzy + Gazebo + MoveIt + CycloneDDS + lerobot editable). Env **installs at `/tmp/mac-env`** because RoboStack conda can't live at a spaced path — see bootstrap section below. |
| L1 smoke on rebased tree | ✅ | 10 unique `(480, 640, 3) uint8` frames, connect <0.1s, `async_read(timeout_ms)` verified |
| L2 smoke (v2.1 dataset) | ✅ | historical — was run on PR #866 branch; will re-run as v3 in Phase 4 |
| Phase 2: top camera SDF + bridge (FOUND-04, OBS-03) | ✅ | `top_camera` sensor added to `so_arm101.gazebo.xacro`, bridge added to `gazebo.launch.py`. FOUND-04 was a no-op — URDF was already using HF-canonical joint names. |
| Phase 2: runtime verification on Mac | ✅ | Live Gazebo published `/wrist_camera`, `/top_camera`, `/joint_states`, `/clock`, `/tf`, etc. RViz confirmed both camera feeds. User verified visually. |
| Stack start/stop/restart/status scripts | ✅ | `mac-env/scripts/stack_*.sh` with 4 modes: `headless` / `gz` / `rviz` / `all` |
| ROS2 Robot + Teleop BYOH plugins (Phase 3) | ✅ | `src/lerobot/robots/so101_ros2/` + `src/lerobot/teleoperators/so101_ros2/`. Dual-camera verified live: `/wrist_camera` + `/top_camera` both 640×480 rgb8 through the plugin, `/joint_states` @ 20 Hz with `gripper_joint → gripper` remap, arm tracks goals on `/arm_controller/joint_trajectory`. `mac-env/scripts/lerobot-record-mode.sh --mode sim\|real` shim in place. |
| `lerobot-record --mode sim` end-to-end (Phase 4) | ✅ | Live record produces v3 datasets; `inbarajaldrin/so_arm101_sim_smoke_v0` pushed to Hub; 9-feature schema equality vs real target PASS; rerun integration live. |
| Pick-and-place + schema parity (Phase 5) | ✅ | `verify_parity.py` in `.planning/checks/` (exits 0 on match, 1 on drift + readable diff). `inbarajaldrin/so_arm101_sim_base_yaw_v0` (3 episodes, 527 frames, action ±90° yaw sweep) verified against the real target — PASS. Full pick-and-place trajectory deferred to the Linux+IsaacSim side (Mac Gazebo lacks the necessary contact physics tuning for reliable grasp). |
| Linux setup notes | ☐ | to be written when we move there |

---

## Repo layout (what lives where)

| Path | Purpose |
|---|---|
| `Exploring-VLAs/lerobot/` | Fork of huggingface/lerobot (submodule). Active branch: `ros2-camera-on-main` |
| `Exploring-VLAs/lerobot/.planning/` | GSD project planning (PROJECT / REQUIREMENTS / ROADMAP / STATE / phases) |
| `Exploring-VLAs/lerobot/src/lerobot/cameras/ros2/` | ROS2 camera subpackage |
| `Exploring-VLAs/mac-env/` | Pixi env + smoke-test scripts (tracked — `.pixi/` is gitignored) |
| `Exploring-VLAs/vla_SO-ARM101/` | SO-ARM101 ROS2 stack (URDF, MoveIt, control GUI, Gazebo) |
| `Exploring-VLAs/vla_SO-ARM101/docs/` | SO-ARM101 docs — **this file lives here** alongside `AGENT_DEBUG_GUIDE.md` and `grasp_pipeline.md` |
| `Exploring-VLAs/vla_SO-ARM101/docs/LEROBOT_GUIDE.md` | Reference guide from the Windows / RTX4090 flow (untracked, reference only — not Mac-specific) |

The pixi env (`mac-env/`) lives **inside Exploring-VLAs** so anyone cloning the
repo can reproduce it. The `.pixi/` build directory is gitignored (multi-GB,
regenerated by `pixi install`). On Linux / real hardware the env is built
differently (conda, per the `LEROBOT_GUIDE.md` flow) — see the Linux section
below when written.

---

## Setup from scratch on Mac (Apple Silicon)

### 1. Prerequisites

- pixi installed (`curl -fsSL https://pixi.sh/install.sh | bash`)
- gh auth'd (`gh auth status`)
- Repo cloned with submodule populated:
  ```bash
  git clone git@github.com:DIME-LAB/Exploring-VLAs.git
  cd Exploring-VLAs
  git submodule update --init --recursive
  ```
- Switch the lerobot submodule to the active branch:
  ```bash
  cd lerobot
  git fetch origin
  git checkout ros2-camera-on-main   # active working branch
  cd ..
  ```

### 2. Bootstrap — one command does the whole env

RoboStack conda packages (shebangs, colcon's `local_setup.sh`) break when the
env lives at a path with spaces — and our repo sits under `…/untitled folder/…`.
Workaround: the pixi env is materialized at **`/tmp/mac-env`** (spaceless copy
of the committed `mac-env/pixi.toml`), and the colcon workspace at
**`/tmp/soarm-ws`** (symlinks to `vla_SO-ARM101/src/`). One script handles it:

```bash
bash mac-env/scripts/bootstrap.sh
```

What `bootstrap.sh` does (first-time ~10–15 min; idempotent on re-run):
1. Copy `pixi.toml` + `pixi.lock` + `cyclonedds.xml` to `/tmp/mac-env/`.
2. `pixi install` → `/tmp/mac-env/.pixi/envs/default` (Python 3.12, ROS2 Jazzy desktop, Gazebo Harmonic, MoveIt, CycloneDDS, full toolchain).
3. Create `/tmp/soarm-ws/src/` with symlinks to the four SO-ARM101 packages and `colcon build`.
4. `pip install -e ../lerobot` into the pixi env (for Phase 4 recording).

### 3. Key conda deps (reference — `mac-env/pixi.toml` is authoritative)

- `python=3.12`, `opencv`, `ffmpeg`, `numpy` (let lerobot pin ≥2.0), `pillow`
- `ros-jazzy-desktop`, `ros-jazzy-ros-gz`, `ros-jazzy-gz-ros2-control`
- `ros-jazzy-ros2-control`, `ros-jazzy-ros2-controllers`, `ros-jazzy-joint-trajectory-controller`, `ros-jazzy-joint-state-broadcaster`, `ros-jazzy-gripper-controllers`
- `ros-jazzy-moveit` + `ros-jazzy-pilz-industrial-motion-planner`
- `ros-jazzy-xacro`, `ros-jazzy-robot-state-publisher`, `ros-jazzy-tf2-*`
- `ros-jazzy-rmw-cyclonedds-cpp` (FastDDS fails discovery on macOS)
- `colcon-common-extensions`

**Not on RoboStack osx-arm64 (known):** `ros-jazzy-pick-ik`. SO-ARM101's MoveIt
config references it; we skip pick-ik and let MoveIt fall back — topic-level
verification works without it. Source-build is a future task.

**`cv_bridge` at runtime:** present via conda but not imported by our code.
Our `ROS2Camera` decodes `sensor_msgs/Image` with pure numpy (works under
numpy ≥ 2.0 where cv_bridge's conda binaries break).

### 4. Running the stack

All four scripts live in `mac-env/scripts/` and handle env vars, `/tmp` paths,
and single-instance enforcement internally. Invoke from anywhere:

| Script | What it does |
|---|---|
| `stack_start.sh [MODE]` | Launch ONE stack; refuses if anything's running. Mode: `headless` (default) · `gz` (Gazebo GUI) · `rviz` (RViz only) · `all` (both). Aliases: `record`→headless, `sim`→gz, `full`→all. |
| `stack_stop.sh` | SIGINT the launch then SIGKILL stragglers by exact name. Never broad wildcards. |
| `stack_restart.sh [MODE]` | `stack_stop.sh` then `stack_start.sh`, forwarding the mode. |
| `stack_status.sh` | List running SO-ARM / Gazebo / smoke processes; exit code = process count. |
| `verify_sim_topics.sh` | Launch headless, wait 45 s, assert `/wrist_camera` + `/top_camera` + `/joint_states` + `/clock` publish, then shut down. |

Log file: `/tmp/soarm_stack.log`. Pidfile: `/tmp/soarm_stack.pid`.

Runtime env vars (set by the scripts; listed here for reference):
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///tmp/mac-env/cyclonedds.xml"
export KMP_DUPLICATE_LIB_OK=TRUE                                # torch 2.10 + conda libomp co-exist
export AMENT_PYTHON_EXECUTABLE=/tmp/mac-env/.pixi/envs/default/bin/python
```

The `cyclonedds.xml` ships with the repo (`mac-env/cyclonedds.xml`, copied to
`/tmp/mac-env/` by bootstrap) and restricts DDS discovery to `lo0`. Don't use
FastDDS on macOS — discovery silently fails between separate Python processes.

---

## Smoke tests (what we ran, how to re-run)

All scripts live in `mac-env/scripts/`.

### L1 — `ROS2Camera` reads a live image topic  (✅ passing on rebased tree)

```bash
cd Exploring-VLAs/mac-env

# Terminal 1 — synthetic publisher
pixi run bash -c "\
  export KMP_DUPLICATE_LIB_OK=TRUE; \
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  export CYCLONEDDS_URI='file://$(pwd)/cyclonedds.xml'; \
  python -u scripts/smoke_publisher.py --topic /smoke/image --fps 30"

# Terminal 2 — ROS2Camera subscriber
pixi run bash -c "\
  export KMP_DUPLICATE_LIB_OK=TRUE; \
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  export CYCLONEDDS_URI='file://$(pwd)/cyclonedds.xml'; \
  python -u scripts/smoke_l1_camera.py --topic /smoke/image --frames 10"
```

Expected output: `[L1 PASS] 10 frames, shape=(480, 640, 3), 10/10 unique`.

Pass criteria: 10 frames of shape `(480, 640, 3)` uint8, all unique pixel sums.

### L2 — full LeRobotDataset written from ROS2 topics, then reloaded

Same publisher in T1; T2 runs:

```bash
pixi run bash -c "\
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  export CYCLONEDDS_URI='file://$(pwd)/cyclonedds.xml'; \
  python -u scripts/smoke_l2_dataset.py --frames 30 --fps 15"
```

What it exercises:
- `/smoke/image` (sensor_msgs/Image) → `ROS2Camera` → `observation.images.top`
- `/smoke/joint_states` (sensor_msgs/JointState, 6-DOF) — synthesized inside the script → `observation.state`
- `action` = next-step state + Gaussian noise
- `LeRobotDataset.create() → add_frame() × N → save_episode()`
- `LeRobotDataset(repo_id, root=…)` reload + `dataset[0]` verification

On-disk structure produced (written to `/tmp/lerobot_smoke_dataset/`):

| Path | Verified |
|---|---|
| `meta/info.json` | `codebase_version: v2.1`, `fps: 15`, `total_frames: 30` |
| `meta/episodes.jsonl` | 1 episode, length 30 |
| `meta/tasks.jsonl` | 1 task string registered |
| `meta/episodes_stats.jsonl` | stats computed by save_episode |
| `data/chunk-000/episode_000000.parquet` | 30 rows; `observation.state`/`action` as `fixed_size_list<float>[6]` |
| `videos/chunk-000/observation.images.top/episode_000000.mp4` | AV1 / yuv420p / 15 fps |

Reload via `LeRobotDataset(...)` returns `sample[0]["observation.images.top"]` as `(3, 480, 640) float32` (channel-first, normalized), matching real lerobot datasets.

> **Status after rebase to upstream main**: `smoke_l2_dataset.py` still targets
> the pre-v3 `LeRobotDataset` API; it has not been re-run against the rebased
> tree. L2 in `v3` format (file-based, multi-episode parquet/mp4) will be
> produced in Phase 4 using the `lerobot-record` CLI rather than this smoke
> script. Keep this script as a reference for the feature-key contract, not as
> the recording path.

---

## Record a dataset from scratch (end-to-end runbook)

This is the canonical path a fresh user follows to go from a just-cloned repo to a recorded `LeRobotDataset v3` pushed to Hugging Face. Every command is copy-pasteable. If a step fails, the **Known gotchas** section below has the fix.

### 0. Prerequisites (one time)

```bash
# Pixi on PATH
export PATH="$HOME/.pixi/bin:$PATH"

# Clone with submodules
git clone <your-fork> Exploring-VLAs && cd Exploring-VLAs
git submodule update --init --recursive

# Bootstrap: materialize pixi env at /tmp/mac-env + build colcon ws at /tmp/soarm-ws
# (one-time; ~10-15 min first run)
bash mac-env/scripts/bootstrap.sh

# CRITICAL: pin numpy<2 so controller spawners don't die on macOS Accelerate ILP64
pixi run --manifest-path /tmp/mac-env/pixi.toml \
  pip install --force-reinstall --no-deps 'numpy<2'

# Install rerun (for --display_data=true live view + lerobot-dataset-viz replay)
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install rerun-sdk
# If rerun install bumps numpy again, re-pin:
pixi run --manifest-path /tmp/mac-env/pixi.toml \
  pip install --force-reinstall --no-deps 'numpy<2'

# HF write-scoped token (token is persisted to ~/.cache/huggingface/token, outside the repo)
pixi run --manifest-path /tmp/mac-env/pixi.toml hf auth login
```

### 1. Start the sim stack

```bash
# Options: headless (no GUI), gz (Gazebo GUI), rviz, all
bash mac-env/scripts/stack_start.sh headless

# Verify all processes up — expect ~7 including gz sim, bridge, move_group,
# control_gui, robot_state_publisher
bash mac-env/scripts/stack_status.sh
```

**What to look for:** `control_gui` python process alive (it spawns after controllers — if it's missing, the controller chain is broken; check `/tmp/soarm_stack.log` for numpy import errors from the spawners).

### 2. Start a motion driver

Two options. Pick one.

**Option A — interactive (human in the loop):** open the Gazebo GUI mode (`stack_start.sh gz`), click through the control_gui tkinter window, drive the arm by hand.

**Option B — automated (reproducible, no human):** run the base-yaw sweep driver in a second terminal. It publishes to both `/arm_controller/joint_trajectory` (moves the arm) and `/joint_commands` (captured as `action` column).

```bash
export CYCLONEDDS_URI=file:///tmp/mac-env/cyclonedds.xml
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
pixi run --manifest-path /tmp/mac-env/pixi.toml \
  python mac-env/scripts/drive_base_yaw_sweep.py
# Leave running — Ctrl-C after recording.
```

### 3. Record the dataset

In a third terminal:

```bash
bash mac-env/scripts/record_sim.sh \
  --dataset.repo_id=<your-hf-user>/<dataset-name> \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=15 \
  --dataset.single_task="Your task string here" \
  --dataset.push_to_hub=true
```

`record_sim.sh` sets the CycloneDDS/RMW/KMP env, wires up both cameras (`/wrist_camera` + `/top_camera` @ 640×480), passes `--robot.robot_type=so_follower` (parity override — matches the real target dataset's `meta/info.json`), enables `use_degrees=true` (parity — real dataset stores joint positions in degrees), and enables `--display_data=true` for a live rerun viewer.

Any flag you append overrides a default (e.g. `--display_data=false` to disable rerun).

### 4. Stop the driver + stack

```bash
# Ctrl-C the driver (or kill by PID)
bash mac-env/scripts/stack_stop.sh
```

### 5. Verify schema parity against the real target dataset

```bash
pixi run --manifest-path /tmp/mac-env/pixi.toml python \
  lerobot/.planning/checks/verify_parity.py \
  --ours   <your-hf-user>/<dataset-name> \
  --target arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep
# Exit 0 = schema matches. Exit 1 = drift; readable diff printed.
```

Every feature key, dtype, shape, joint name, `codebase_version`, `fps`, and `robot_type` must match. The script does not compare task string, episode count, or pixel content — those legitimately differ.

### 6. Replay / visualize the recorded dataset

```bash
pixi run --manifest-path /tmp/mac-env/pixi.toml \
  lerobot-dataset-viz --repo-id=<your-hf-user>/<dataset-name> --episode-index=0
```

Rerun spawns with video panels for both cameras. **To see action/state plots:** in the rerun left panel, expand the `action` and `state` entries, right-click → "Add to new space view" → Time Series.

### 7. Troubleshooting quick table

| Symptom | Fix |
|---|---|
| `controller_manager: No clock received` spams the log; `/joint_states` at 0 Hz | Python spawners crashed on `import numpy` — pin `numpy<2` per step 0 and `stack_stop.sh && stack_start.sh` |
| `--robot.type: invalid choice: 'so101_ros2'` | Wrong lerobot install path; make sure `lerobot/src/lerobot/scripts/lerobot_record.py` has `so101_ros2` in its import block |
| `/wrist_camera` at 0 Hz (topic exists but no frames) | `so_arm101.gazebo.xacro` should attach the sensor to `usb_camera` (not `camera_link` — that's a geometry-less URDF convention frame); rebuild `so_arm101_description` |
| `push_to_hub` 403 | Token lacks Write scope — regenerate at https://huggingface.co/settings/tokens, rerun `hf auth login` |
| `verify_parity.py` FAIL on `action`/`observation.state` names | `use_degrees` or `joint_name_map` in plugin config got out of sync with the real dataset's naming; check `.planning/real_dataset_probe/PARITY_CONTRACT.md` |

### Reference datasets produced by this runbook

- `inbarajaldrin/so_arm101_sim_smoke_v0` — Phase 4 smoke (2 episodes, synthetic `/joint_commands`)
- `inbarajaldrin/so_arm101_sim_base_yaw_v0` — Phase 5 motion test (3 episodes, automated base-yaw + gripper sweep)

Both pass `verify_parity.py` against `arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep`.

---

## SO-ARM101 topic reference (for the real connection)

Grep'd out of `vla_SO-ARM101/src`:

| Topic | Type | Source | Notes |
|---|---|---|---|
| `/wrist_camera` | `sensor_msgs/msg/Image` | `so_arm101_description/urdf/so_arm101.gazebo.xacro` (camera SDF) → bridged in `so_arm101_control/launch/gazebo.launch.py` via `ros_gz_bridge parameter_bridge` | 1280×720 @ 30 fps R8G8B8 |
| `/camera_info` | `sensor_msgs/msg/CameraInfo` | same bridge | |
| `/joint_states` | `sensor_msgs/msg/JointState` | `joint_state_broadcaster` (sim) or `servo_driver.py` / `jointstatereader.py` (real) | **Canonical SO-101 names** (`jointstatereader.py`): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_joint`. URDF currently publishes `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw` — scheduled to rename in Phase 2. |

Reference: `vla_SO-ARM101/README.md` and `vla_SO-ARM101/docs/AGENT_DEBUG_GUIDE.md`.

---

## Known gotchas (learned the hard way)

### Still relevant after rebase
- **FastDDS on macOS:** silent discovery failure between separate Python processes. Use CycloneDDS + `lo0` xml.
- **OpenMP duplicate runtime:** torch + conda's libomp collide on macOS → `OMP: Error #15`. Set `KMP_DUPLICATE_LIB_OK=TRUE`.
- **ROS2Camera owns `rclpy.init()`:** `ROS2Camera.__init__` initializes rclpy unconditionally (guarded by a class-level flag). Don't pre-init rclpy in caller code; let the camera bootstrap it.
- **`rclpy.spin` across threads races** on the shared default executor (`ValueError: generator already executing`). Use one `SingleThreadedExecutor` for all your non-camera nodes.
- **Pixi editable with space in path:** `[pypi-dependencies] { path = "…" }` breaks because our repo lives under `untitled folder/`. Install lerobot via `pixi run pip install -e ../lerobot` instead.
- **NumPy 2 uint8 overflow:** `uint8_array + python_int` now raises on numeric promotion. Our `smoke_publisher.py` uses int32 intermediates.

### Obsolete after rebase to upstream main
- ~~`numpy<2` pin~~ — upstream requires numpy ≥ 2.0 for opencv-python-headless.
- ~~`datasets<4` pin~~ — upstream moved to `LeRobotDataset v3`, the pre-v3 reload bug is gone.
- ~~cv_bridge ABI under numpy 2~~ — our `ROS2Camera` decodes `sensor_msgs/Image` with a pure-numpy reshape; cv_bridge no longer required at runtime.

---

## TODO — things to fill in next sessions

1. **URDF joint rename (Phase 2 — FOUND-04):** rename so_arm101_description joints to canonical SO-101 names; update MoveIt SRDF, controller YAML, launch files.
2. **Top camera (Phase 2 — OBS-03):** add `top_camera` SDF sensor + bridge entry; pick matching resolution/pose to the real dataset's top view.
3. **ROS2 Robot + Teleop BYOH plugins (Phase 3):** author `lerobot_robot_so101_ros2` (subscribes `/wrist_camera`, `/top_camera`, `/joint_states`) + `lerobot_teleoperator_so101_ros2` (subscribes action topic). Hook up `--mode sim|real` CLI alias.
4. **`lerobot-record` end-to-end (Phase 4):** unmodified CLI drives both modes; dataset writes as v3 via `dataset.finalize()` + `push_to_hub()`.
5. **Pick-and-place + schema parity (Phase 5):** capture 1 real sim episode; assert schema equality against colleague's real SO-101 HF dataset.
6. **Linux setup:** sibling `LEROBOT_ROS2_LINUX_SETUP.md` under `vla_SO-ARM101/docs/`. Expected differences: apt ROS2, no numpy/OpenMP drama, FastDDS usually fine.

---

## Changelog

- **2026-04-23 (latest)** `/wrist_camera` fixed. Root cause: `<gazebo reference="camera_link">` attached the sensor to a geometry-less URDF convention frame, so Ogre2 never created a render context. Re-parented to `usb_camera` (mesh geometry present) with the URDF transform folded into the SDF `<pose>`; `ignition_frame_id` kept as `camera_link` so downstream TF/frame_ids unchanged. Resolution dropped from 1280×720 → 640×480 to match the real RealSense wrist view and keep Ogre2 render rate ≥10 Hz. Dual-camera checkpoint passes through the Phase 3 plugin (both 480×640×3 uint8). Edit in `so_arm101_description/urdf/so_arm101.gazebo.xacro`; rebuild that package.
- **2026-04-23 (later)** Phase 3 done. ROS2 BYOH plugins shipped inside the fork: `src/lerobot/robots/so101_ros2/` (reads `/joint_states` + camera topics, `joint_name_map={gripper_joint: gripper}` default, `send_action` is a no-op) and `src/lerobot/teleoperators/so101_ros2/` (reads `/joint_commands`, same remap, fail-loud on stale actions). Both piggyback on `ROS2Camera`'s rclpy singleton — one node, one spin thread process-wide. Registered as `--robot.type=so101_ros2` / `--teleop.type=so101_ros2`. `mac-env/scripts/lerobot-record-mode.sh --mode sim|real` dispatches to the right type pair. Live-Gazebo checkpoint: plugin reads 20 observations cleanly, arm moves +0.355 rad under a trajectory goal, `control_gui` alive. **Gotcha found + fixed:** `pip install -e lerobot` bumps numpy to 2.2.6 (PyPI wheel, Accelerate `NEWLAPACK$ILP64`) which kills the Python controller spawners on this Mac; downgrade with `pip install --force-reinstall --no-deps 'numpy<2'`. Follow-up: pin `numpy<2` in `mac-env/pixi.toml` so fresh bootstrap doesn't regress. **Known sim-stack issue, unrelated to Phase 3:** `/wrist_camera` publishes 0 Hz on the ROS2 side despite gz-side publisher being wired; `/top_camera` works. Plugin handles it correctly (descriptive `TimeoutError`).
- **2026-04-23 (late)** Phase 2 done. Top camera SDF sensor added to `so_arm101.gazebo.xacro`; `/top_camera` + `/top_camera/camera_info` bridges in `gazebo.launch.py`. FOUND-04 (URDF joint rename) was already canonical — no-op. Runtime verification on Mac: live Gazebo published all expected topics; cameras visible in RViz. Pixi env expanded to full sim stack (ros-jazzy-desktop + ros-gz + ros2-control + moveit). Env materialized at `/tmp/mac-env` to work around RoboStack's spaced-path issues. Stack `start/stop/restart/status/verify` scripts shipped in `mac-env/scripts/`. User's March 2026-era macOS compat fixes (`gz sim -s`/`-g` split, `GZ_SIM_SYSTEM_PLUGIN_PATH`, `controller_manager/spawner` instead of `ros2 control load_controller`, tkinter on main thread, joint acceleration limits) committed together as one commit.
- **2026-04-23** Phase 1 done. Rebased fork onto upstream main (`ros2-camera-on-main` branch). ROS2 camera ported to `src/lerobot/cameras/ros2/` (new layout). Dropped `cv_bridge` runtime dependency in favor of pure-numpy `Image` decoding (numpy 2 compat). Pixi env moved inside repo at `mac-env/`. L1 smoke green on rebased tree.
- **2026-04-22** Initial smoke-tests passing (L1, L2). Submodule, pixi env, CycloneDDS config, two smoke scripts committed. Fork = `inbarajaldrin/lerobot @ ros2_camera` (PR #866 snapshot).
