# Exploring-VLAs — agent context

This file provides guidance to AI agents (Claude Code, Cursor, etc.) working in this repo.

## What this project is

End-to-end Vision-Language-Action stack for the **SO-ARM101** 5-DOF arm. Two
supported platforms with **different physics backends but identical topic
contracts**:

| Platform | Physics | Producer-side ROS2 | Consumer-side ROS2 |
|---|---|---|---|
| **Linux** (Ubuntu 22.04 + NVIDIA) | Isaac Sim 5.x | system Humble | pixi-Jazzy (lerobot) |
| **macOS** (Apple Silicon) | Gazebo Harmonic | pixi-Jazzy via RoboStack | same pixi-Jazzy |

The control stack (MoveIt 2 + `control_gui` tkinter + ros2_control) is the
same on both. Only the physics backend differs.

- **Sim** — Isaac Sim (Linux) or Gazebo Harmonic (Mac), driving a URDF/SDF model of the arm.
- **Real-hardware bringup** — `jointstatereader` over Feetech serial + USB camera publisher.
- **Data collection** — fork of HuggingFace `lerobot` with new ROS2-backed Robot/Teleop/Camera plugins, drives the same `lerobot-record` CLI in sim and real.
- **Real-side perception** — `aruco_camera_localizer` publishes object poses + drop poses from ArUco markers; sim-side `sim_ground_truth` (Mac/Gazebo) and `soarm101-dt` Isaac extension (Linux) mirror the same topic contract, so `control_gui` consumes identical data on either backend.

**No Docker** on either platform.

## Repo topology

This CLAUDE.md is the **source of truth** for layout. Everything else cross-links from here.

| Where | Role | How to get it |
| --- | --- | --- |
| `vla_SO-ARM101/` | Robot description, Gazebo world, MoveIt config, control_gui, jointstatereader, so_arm101_bringup, sim_ground_truth. Colcon workspace builds in-tree on Linux, at `/tmp/soarm-ws/` on Mac (spaceless-path constraint). | In-tree directory (part of this clone) |
| `lerobot/` | HF `lerobot` fork with `so101_ros2` Robot/Teleop + `ros2` Camera plugins | Git submodule (`git submodule update --init --recursive`) |
| `isaac-sim-mcp/` | Isaac Sim digital twin extension (`soarm101-dt`) — physics backend on Linux. MCP socket on `localhost:8767`. Provides `/joint_states`, `/clock`, `/objects_poses_sim`, `/drop_poses`, `/wrist_camera_rgb_sim`, `/workspace_camera_sim`. | Git submodule |
| `linux-env/` | Linux-side: pixi-Jazzy env for lerobot, bootstrap + stack_*.sh scripts, vendored `isaacsim_launch.sh`, mirror node, `cyclonedds.xml` for cross-Python DDS | In-tree directory |
| `mac-env/` | Mac-side: pixi env (RoboStack) for the WHOLE stack, bootstrap + stack_*.sh scripts | In-tree directory |
| `aruco_camera_localizer/` | Real-side ArUco / YOLO object localization → `/objects_poses_real`, `/objects_bbox_real`, `/drop_poses`. **Not a submodule** (git submodule-add hits a pack-index bug on macOS spaced paths). On Mac auto-cloned to `/tmp/aruco_camera_localizer` by bootstrap; on Linux currently expected at `~/Desktop/ros2_ws/src/aruco_camera_localizer` (location being normalized — see cleanup tally). | Manual clone (Linux) / auto-cloned (Mac) |
| `vla_cartpole/` | Tangential — separate experiment, not wired into the SO-ARM101 stack | In-tree |

## Build layout

| Platform | Pixi env lives at | Colcon workspace | Why |
|---|---|---|---|
| **Linux** | `linux-env/.pixi/` (in-tree) | `vla_SO-ARM101/install/` (in-tree) | Linux paths are spaceless by convention; in-tree keeps build artefacts tied to source for symlink-install |
| **macOS** | `/tmp/mac-env/` | `/tmp/soarm-ws/` (src/ symlinks back) | RoboStack conda packages choke on `-isystem /Path With Spaces/` — repo lives under spaced path; `/tmp/` is the workaround |

Recreate after a reboot or to start clean: re-run the bootstrap for your platform.

## Doc map

This file (root `CLAUDE.md`) is the entry point. Per-area docs cross-link from here:

**First-time setup:**
- **Linux**: [`vla_SO-ARM101/docs/ROS2_LINUX_SETUP.md`](vla_SO-ARM101/docs/ROS2_LINUX_SETUP.md) — full from-zero (Humble + pixi + Isaac Sim + colcon)
- **Mac**: [`vla_SO-ARM101/docs/ROS2_MAC_SETUP.md`](vla_SO-ARM101/docs/ROS2_MAC_SETUP.md) — full from-zero (pixi, RoboStack, CycloneDDS)
- **Mac recording**: [`vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md`](vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md) — sim + real recording runbook

**Architecture / planner internals:**
- [`vla_SO-ARM101/CLAUDE.md`](vla_SO-ARM101/CLAUDE.md) — package inventory + tiered planner + sim-world conventions
- [`vla_SO-ARM101/docs/grasp_pipeline.md`](vla_SO-ARM101/docs/grasp_pipeline.md) — control_gui → MoveIt grasp flow
- [`vla_SO-ARM101/docs/AGENT_DEBUG_GUIDE.md`](vla_SO-ARM101/docs/AGENT_DEBUG_GUIDE.md) — debugging the sim stack
- [`isaac-sim-mcp/CLAUDE.md`](isaac-sim-mcp/CLAUDE.md) — Isaac Sim extension internals + MCP tool inventory (Linux backend)
- [`isaac-sim-mcp/docs/DEBUG-GUIDE.md`](isaac-sim-mcp/docs/DEBUG-GUIDE.md) — physics + planner integration debugging

**Per-area conventions:**
- [`linux-env/CLAUDE.md`](linux-env/CLAUDE.md) — Linux record stack (mirror, lerobot-record wrapper, Record Sim tab)
- [`mac-env/CLAUDE.md`](mac-env/CLAUDE.md) — Mac script lifecycle (stack_start, record, drive)
- [`lerobot/CLAUDE.md`](lerobot/CLAUDE.md) (symlinked from `AGENTS.md`) — fork plugin rules, rclpy singleton, numpy pin
- [`lerobot/ROS2_PLUGINS.md`](lerobot/ROS2_PLUGINS.md) — how to author a new `<arm>_ros2` plugin

**Project state (lerobot fork):**
- `lerobot/.planning/STATE.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `phases/*/SUMMARY.md` — `gsd`-style planning artefacts.

## Bringing it up — Linux

```bash
# Once after cloning
cd /your/path/to/Exploring-VLAs
git submodule update --init --recursive
export PATH="$HOME/.pixi/bin:$PATH"
bash linux-env/scripts/bootstrap.sh        # preflight + submodules + pixi env + colcon build

# Each session
bash linux-env/scripts/stack_start.sh      # Isaac Sim + control stack + RViz + tkinter GUI
bash linux-env/scripts/stack_status.sh     # health check across all layers
# … record episodes via control_gui's Record Sim tab (see linux-env/CLAUDE.md) …
bash linux-env/scripts/stack_stop.sh       # graceful tear-down (SIGINT-first)
```

**Hard requirements** (the bootstrap preflight checks these): Ubuntu 22.04, NVIDIA driver compatible with Isaac Sim 5.x, ROS2 Humble at `/opt/ros/humble`, pixi installed, ~50 GB free, X11 session. Full walk-through: [`ROS2_LINUX_SETUP.md`](vla_SO-ARM101/docs/ROS2_LINUX_SETUP.md).

## Bringing it up — macOS

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

For real-hardware data collection see [`LEROBOT_ROS2_MAC_SETUP.md`](vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md).

## Linux gotchas (top 5)

| Gotcha | Workaround |
|---|---|
| `ros2 param set` returns "empty node name returned by RMW layer" intermittently | Retry, or use `linux-env/scripts/_param_set_string.sh` (rclpy direct, forces STRING typing for numeric values) |
| Isaac Sim launch fails with `Failed to create any GPU devices` (CUDA 999) | `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` (UVM module reload — safe; doesn't touch display) |
| Cross-Python DDS discovery silently fails between Humble producers and Jazzy consumers | `linux-env/cyclonedds.xml` enables multicast on `lo`. Sourced automatically by `record_sim_isaac.sh` and the stack_start path |
| `kill -9` on Isaac Sim / RViz / tkinter cascades KWin BadWindow errors | Always SIGTERM/SIGINT. `stack_stop.sh` handles this; never `pkill -9` X-window-owning procs |
| Recording loop runs at 13–22 Hz vs 30 Hz target | Known perf issue (Isaac Sim camera bridge + RGBA→RGB cost). Doesn't block recording correctness |

## macOS gotchas (top 5)

| Gotcha | Workaround |
|---|---|
| `ros2 daemon` hangs | Always pass `--no-daemon` |
| `pip install -e lerobot` bumps `numpy>=2` → controller spawners die on Accelerate ILP64 | After any pip-touch-numpy: `pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall --no-deps 'numpy<2'` |
| FastDDS silently fails between Python processes | CycloneDDS — exports `CYCLONEDDS_URI=file:///tmp/mac-env/cyclonedds.xml`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| OpenMP duplicate runtime (`OMP: Error #15`) | `KMP_DUPLICATE_LIB_OK=TRUE` |
| Spaces in repo path break colcon C++ builds | Build at `/tmp/soarm-ws/` (the bootstrap symlinks source from here) |

## Conventions for agents working here

- **Don't add docs that already live elsewhere.** Cross-link instead. Each doc has one home.
- **Do prefer editing existing files.** New files are friction unless the scope is genuinely new.
- **Don't create `.md` files unless the user asks** (or the work landing genuinely needs one — e.g., a new architectural doc).
- **Trust the planning dir.** `.planning/STATE.md` reflects current truth; if something contradicts it, suspect drift and ask.
- **Match commit style** — the existing `feat(scope):`, `fix(scope):`, `docs(scope):`, `chore:` convention with multi-paragraph bodies covering motivation + implementation + verification.
