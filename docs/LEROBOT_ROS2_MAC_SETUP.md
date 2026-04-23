# LeRobot + ROS2 Integration on macOS (Apple Silicon)

Living guide for the lerobot + ROS2 camera/dataset work inside `Exploring-VLAs`.
We'll keep appending sections as each piece is wired up. The target is a
single unified data path: sim frames (Gazebo) and real frames (USB) both enter
lerobot recording via ROS2 topics, producing datasets ready for SmolVLA / Pi0.5
co-training.

## Target

One recording pipeline that works for **sim and real** via ROS2 topics:

```
SO-ARM101 (sim or real)   →  ROS2 topics  →  lerobot-record  →  HF dataset (v2.1)
  • /wrist_camera         →  ROS2Camera   (PR #866)
  • /joint_states         →  (future) ROS2-backed Robot
  • action controllers    →  (future) ROS2-backed Robot
```

## Progress

| Step | State | Notes |
|---|---|---|
| Fork PR #866 `ros2_camera` branch | ✅ | `github.com/inbarajaldrin/lerobot` @ `b1095d2` |
| Submodule in `Exploring-VLAs/lerobot` | ✅ | remotes: `origin` = fork, `yadunund`, `upstream` = huggingface/lerobot |
| Pixi env on Mac (outside this repo) | ✅ | `agents/ros2/lerobot/` — ROS2 Jazzy Python + CycloneDDS + lerobot editable |
| L1 smoke: `ROS2Camera.read()` live topic | ✅ | 10/10 unique 640×480×3 uint8 frames, connect <1.1 s |
| L2 smoke: full LeRobotDataset round-trip | ✅ | v2.1 layout, AV1 MP4, parquet — reload matches |
| Real SO-ARM101 Gazebo topics end-to-end | ☐ | next |
| ROS2-backed Robot (state + actions via topics) | ☐ | PR #866 only handles camera side |
| `lerobot-record` CLI working unchanged | ☐ | needs the Robot above |
| Real USB path unchanged (OpenCV/RealSense) | ☐ | verify PR #866 doesn't regress |
| Training-ready dataset (≥20 episodes, stats) | ☐ | |
| Linux setup notes | ☐ | to be written when we move there |

---

## Repo layout (what lives where)

| Path | Purpose |
|---|---|
| `Exploring-VLAs/lerobot/` | Fork of huggingface/lerobot with PR #866 applied (submodule) |
| `Exploring-VLAs/vla_SO-ARM101/` | SO-ARM101 ROS2 stack (URDF, MoveIt, control GUI, Gazebo) |
| `Exploring-VLAs/docs/` | Cross-package integration docs — this file lives here |
| `Exploring-VLAs/LEROBOT_GUIDE.md` | Reference guide from the Windows / RTX4090 flow (untracked) |
| `…/ros2/lerobot/` (outside this repo) | Smoke-test pixi env + scripts on Mac |

The pixi env for lerobot is **intentionally outside `Exploring-VLAs`** — it's
only for Mac sim-side testing. On Linux / real hardware the env is built
differently (conda, per the `LEROBOT_GUIDE.md` flow).

---

## Setup from scratch on Mac (Apple Silicon)

### 1. Prerequisites

- pixi installed (`curl -fsSL https://pixi.sh/install.sh | bash`)
- gh auth'd (`gh auth status`)
- Repo already cloned with `--recurse-submodules`, or after clone:
  ```bash
  cd Exploring-VLAs
  git submodule update --init --recursive
  ```

### 2. Pixi env (outside Exploring-VLAs)

The smoke-test env lives next to Exploring-VLAs so that the pixi lock and
.pixi cache are not committed into the VLA repo:

```bash
cd /path/to/agents/ros2/lerobot       # directory with pixi.toml
pixi install                          # creates .pixi/envs/default
```

Key conda deps (see `pixi.toml`):
- `ros-jazzy-rclpy`, `ros-jazzy-cv-bridge`, `ros-jazzy-sensor-msgs-py`
- `ros-jazzy-ros2cli`, `ros-jazzy-ros2topic`
- `ros-jazzy-rmw-cyclonedds-cpp` — FastDDS fails discovery on macOS
- `python=3.11`, `opencv`, `ffmpeg`, `numpy<2`, `pillow`

### 3. Install lerobot + pinned pip deps

pixi + editable path fails when the repo path contains a space
("untitled folder"), so lerobot is installed via pip inside the pixi env:

```bash
pixi run pip install -e /path/to/Exploring-VLAs/lerobot \
                     "numpy<2" \
                     "datasets<4"
```

- `numpy<2` — `cv_bridge` binaries were built against numpy 1.x ABI.
- `datasets<4` — `LeRobotDataset.__init__` uses `torch.stack(hf_dataset["timestamp"])` which errors on the `Column` type introduced in datasets 4.x.

### 4. DDS config (CycloneDDS, localhost only)

Put `cyclonedds.xml` alongside `pixi.toml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config" …>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="lo0"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
      <EnableMulticastLoopback>true</EnableMulticastLoopback>
      <DontRoute>true</DontRoute>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>200</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Export before every ROS2-related command:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$(pwd)/cyclonedds.xml"
```

---

## Smoke tests (what we ran, how to re-run)

Both scripts live in `agents/ros2/lerobot/scripts/`.

### L1 — `ROS2Camera` reads a live image topic

```bash
# Terminal 1
cd /path/to/agents/ros2/lerobot
pixi run bash -c "\
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  export CYCLONEDDS_URI='file://$(pwd)/cyclonedds.xml'; \
  python -u scripts/smoke_publisher.py --topic /smoke/image --fps 30"

# Terminal 2
pixi run bash -c "\
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; \
  export CYCLONEDDS_URI='file://$(pwd)/cyclonedds.xml'; \
  python -u scripts/smoke_l1_camera.py --topic /smoke/image --frames 10"
```

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

---

## SO-ARM101 topic reference (for the real connection)

Grep'd out of `vla_SO-ARM101/src`:

| Topic | Type | Source | Notes |
|---|---|---|---|
| `/wrist_camera` | `sensor_msgs/msg/Image` | `so_arm101_description/urdf/so_arm101.gazebo.xacro` (camera SDF) → bridged in `so_arm101_control/launch/gazebo.launch.py` via `ros_gz_bridge parameter_bridge` | 1280×720 @ 30 fps R8G8B8 |
| `/camera_info` | `sensor_msgs/msg/CameraInfo` | same bridge | |
| `/joint_states` | `sensor_msgs/msg/JointState` | `joint_state_broadcaster` (sim) or `servo_driver.py` (real) | Joint names: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw` |

Next pointer: when we flip `smoke_l2_dataset.py` to the real topics, use
`--image-topic /wrist_camera --js-topic /joint_states` and remove the in-script
`JointStatePublisher` block.

Two deltas to handle before that works:
- **Resolution:** `/wrist_camera` is 1280×720, not 640×480. Match the config shape.
- **Action source:** `JointStatePublisher` was a stand-in for actions. We need either (a) a command-topic to subscribe to, or (b) a teleop-driven Robot that writes the action column at record time. TBD once the ROS2 Robot backend is decided.

Reference: `vla_SO-ARM101/README.md` and `vla_SO-ARM101/docs/AGENT_DEBUG_GUIDE.md`.

---

## Known gotchas (learned the hard way)

- **NumPy 2 ABI:** cv_bridge segfaults; pin `numpy<2` after every full env rebuild.
- **datasets 4.x Column API:** breaks `LeRobotDataset.__init__` reload; pin `datasets<4`.
- **FastDDS on macOS:** silent discovery failure between separate Python processes. Use CycloneDDS + `lo0` xml.
- **PR #866 owns `rclpy.init()`:** `ROS2Camera.__init__` calls it unconditionally; don't pre-init from the caller.
- **`rclpy.spin` across threads races** on the shared default executor (`ValueError: generator already executing`). Use one `SingleThreadedExecutor` for all your non-camera nodes.
- **Pixi editable with space in path:** `pypi-dependencies { path = "…" }` breaks because our repo lives under `untitled folder/`. Install lerobot via `pixi run pip install -e` instead.
- **`pixi install` wipes pip packages:** any time we change `pixi.toml` we have to re-run the pip install line in step 3.

---

## TODO — things to fill in next sessions

1. **Real sim connection.** Launch `so_arm101_control/launch/gazebo.launch.py`, point `smoke_l2_dataset.py` at `/wrist_camera` and `/joint_states`, drop the synthetic publisher. Capture 1 real episode. Verify MP4 + parquet.
2. **ROS2 Robot backend.** Evaluate: (a) adopt `sacovo/lerobot_ros` wholesale, (b) write a minimal `SO101ROS2Follower` that implements lerobot's `Robot` interface over topics. Decide before touching `lerobot-record`.
3. **`lerobot-record` end-to-end.** Once the Robot backend is in, record ≥3 episodes via the CLI using `--robot.cameras="{ wrist: {type: ros2, topic: /wrist_camera, ...} }"`.
4. **Real-arm parity.** Verify the existing OpenCV/RealSense paths in PR #866 still work on the Windows/4090 side — i.e. the lerobot fork is usable in both worlds.
5. **Dataset quality.** Run `compute_stats` explicitly, load with a real SmolVLA config and dry-run one batch through the dataloader.
6. **Linux setup.** Separate section here (or a sibling file under `docs/`) once we move to Linux for training. The key differences we expect: no RoboStack (use apt ROS2), no numpy/cv_bridge ABI drama, FastDDS often fine.

---

## Changelog

- **2026-04-22** Initial smoke-tests passing (L1, L2). Submodule, pixi env, CycloneDDS config, two smoke scripts committed. Fork = `inbarajaldrin/lerobot @ ros2_camera`.
