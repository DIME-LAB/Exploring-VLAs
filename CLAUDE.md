# Exploring-VLAs — agent context

This file provides guidance to AI agents (Claude Code, Cursor, etc.) working in this repo.

## What this project is

End-to-end Vision-Language-Action stack for the **SO-ARM101** 5-DOF arm on **macOS Apple Silicon**:

- **Sim** — Gazebo Harmonic + MoveIt 2 + control_gui driving a URDF/SDF model of the arm.
- **Real-hardware bringup** — `jointstatereader` over Feetech serial + USB camera publisher.
- **Data collection** — fork of HuggingFace `lerobot` with new ROS2-backed Robot/Teleop/Camera plugins, drives the same `lerobot-record` CLI in sim and real.
- **Real-side perception** — `aruco_camera_localizer` publishes object poses + drop poses from ArUco markers; sim-side `sim_ground_truth` mirrors the same topic contract from Gazebo world poses, so `control_gui` consumes identical data sim and real.

Native ROS2 Jazzy via pixi + RoboStack. **No Docker.** Everything is ARM64 binaries.

## Repo topology (four moving parts)

| Where | Role | How to get it |
| --- | --- | --- |
| `vla_SO-ARM101/` | Robot description, Gazebo world, MoveIt config, control_gui, jointstatereader, so_arm101_bringup, sim_ground_truth | In-tree directory (part of this clone) |
| `lerobot/` | HF `lerobot` fork with `so101_ros2` Robot/Teleop + `ros2` Camera plugins | Git submodule (`git submodule update --init --recursive`) |
| `aruco_camera_localizer/` | Real-side ArUco / YOLO object localization → `/objects_poses_real`, `/objects_bbox_real`, `/drop_poses` | Auto-cloned by `bootstrap.sh` to `/tmp/aruco_camera_localizer` (not a submodule — git's submodule-add hits a pack-index bug on macOS spaced paths; documented in `vla_SO-ARM101/docs/ROS2_MAC_SETUP.md`) |
| `vla_cartpole/` | Tangential — separate experiment, not wired into the SO-ARM101 stack | In-tree |

## Build layout

Source-of-truth lives in this repo. Build artefacts live at **spaceless `/tmp/` paths** because RoboStack conda packages choke on `-isystem /Path With Spaces/`:

- `/tmp/mac-env/` — pixi env (`pixi.toml` materialized from `mac-env/pixi.toml`)
- `/tmp/soarm-ws/` — colcon workspace; `src/` is symlinks back to this repo
- `/tmp/aruco_camera_localizer/` — fresh clone, symlinked into the workspace

Recreate after a reboot or to start clean: `bash mac-env/scripts/bootstrap.sh`.

## Doc map

Read in this order on first contact:

1. `vla_SO-ARM101/docs/ROS2_MAC_SETUP.md` — project-agnostic Mac+ROS2 bootstrap (pixi, RoboStack, CycloneDDS)
2. `vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md` — end-to-end recording runbook (sim **and** real)
3. `lerobot/ROS2_PLUGINS.md` — fork plugin architecture + how to author a new `<arm>_ros2` plugin
4. `lerobot/.planning/STATE.md`, `ROADMAP.md`, `REQUIREMENTS.md` — what's built, what's next, why
5. `lerobot/.planning/phases/*/SUMMARY.md` — per-phase ship notes (`gsd`-style planning)
6. `vla_SO-ARM101/docs/AGENT_DEBUG_GUIDE.md` — debugging the sim stack

Per-area conventions also live in nested CLAUDE.md files:

- `vla_SO-ARM101/CLAUDE.md` — package inventory + sim-world conventions
- `mac-env/CLAUDE.md` — Mac script lifecycle (stack_start, record, drive)
- `linux-env/CLAUDE.md` — **Linux** record stack (mirror, lerobot-record wrapper, Record Sim tab integration). Recording on Linux pairs with Isaac Sim digital twin at `~/Documents/isaac-sim-mcp/` instead of Gazebo.
- `lerobot/CLAUDE.md` (= `AGENTS.md` symlinked) — fork plugin rules, rclpy singleton, numpy pin

## macOS gotchas (top 5)

| Gotcha | Workaround |
| --- | --- |
| `ros2 daemon` hangs | Always pass `--no-daemon` |
| `pip install -e lerobot` bumps `numpy>=2` → controller spawners die on Accelerate ILP64 | After any pip-touch-numpy: `pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall --no-deps 'numpy<2'` |
| FastDDS silently fails between Python processes | CycloneDDS — exports `CYCLONEDDS_URI=file:///tmp/mac-env/cyclonedds.xml`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| OpenMP duplicate runtime (`OMP: Error #15`) | `KMP_DUPLICATE_LIB_OK=TRUE` |
| Spaces in repo path break colcon C++ builds | Build at `/tmp/soarm-ws/` (the bootstrap symlinks source from here) |

## Bringing it up

```bash
# Once after cloning
git submodule update --init --recursive
export PATH="$HOME/.pixi/bin:$PATH"
bash mac-env/scripts/bootstrap.sh

# Each session
bash mac-env/scripts/stack_start.sh headless     # sim — Gazebo + MoveIt + control_gui
bash mac-env/scripts/stack_status.sh             # confirm processes
bash mac-env/scripts/record.sh \                  # record a sim episode
  --dataset.repo_id=<you>/<dataset> \
  --dataset.num_episodes=3 --dataset.episode_time_s=5 \
  --dataset.single_task="..."
bash mac-env/scripts/stack_stop.sh               # tear down (SIGINT propagates)
```

For real-hardware data collection see the runbook (#2 above).

### Bringing it up (Linux + Isaac Sim — sim recording)

```bash
# 1. Isaac Sim + control stack: see ~/Documents/isaac-sim-mcp/CLAUDE.md
#    (the canonical bring-up doc — covers preflight, MCP socket on 8767,
#    quick_start, control.launch.py)

# 2. Recording layer: see linux-env/CLAUDE.md
#    (mirror, lerobot-record wrapper, or the Record Sim tab in control_gui)
```

The Linux path uses Isaac Sim as the physics backend (Gazebo on Mac) and
the `linux-env/` pixi-Jazzy env for the lerobot consumer side.

## Conventions for agents working here

- **Don't add docs that already live elsewhere.** Cross-link instead. Each doc has one home.
- **Do prefer editing existing files.** New files are friction unless the scope is genuinely new.
- **Don't create `.md` files unless the user asks** (or the work landing genuinely needs one — e.g., a new architectural doc).
- **Trust the planning dir.** `.planning/STATE.md` reflects current truth; if something contradicts it, suspect drift and ask.
- **Match commit style** — the existing `feat(scope):`, `fix(scope):`, `docs(scope):`, `chore:` convention with multi-paragraph bodies covering motivation + implementation + verification.
