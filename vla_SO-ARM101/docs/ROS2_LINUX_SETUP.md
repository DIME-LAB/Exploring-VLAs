# ROS2 on Linux + Isaac Sim — full setup

Living runbook for bringing up the SO-ARM101 stack on Ubuntu Linux. The Linux
path uses **Isaac Sim** as the physics backend (Mac uses Gazebo) and pairs
**system Humble** for the producer side with a **pixi-Jazzy** env for the
lerobot consumer side.

> Mac counterpart: [`ROS2_MAC_SETUP.md`](./ROS2_MAC_SETUP.md). Recording
> runbook: see [`linux-env/CLAUDE.md`](../../linux-env/CLAUDE.md) and
> the parent [`CLAUDE.md`](../../CLAUDE.md).

## Hard requirements

- **Ubuntu 22.04** (the target tested distro). Other distros may work but
  Isaac Sim 5.x officially supports Ubuntu 22.04 / 24.04.
- **NVIDIA GPU** with a driver compatible with Isaac Sim 5.x (RTX 30-series
  or newer; 8 GB+ VRAM; driver ≥ 535).
- **ROS2 Humble** at `/opt/ros/humble` (system install, debs).
- **~50 GB free disk** (Isaac Sim cache + USDs + colcon build + pixi env).
- **X11 session** (Wayland is untested). The stack opens RViz + tkinter +
  Isaac Sim windows; killing any of them with `kill -9` cascades KWin
  BadWindow errors — always SIGTERM/SIGINT.

## The core tools

| Tool | Role | Install |
|---|---|---|
| ROS2 Humble (system) | Producer side: Isaac Sim, control_gui, MoveIt, mirror | [Ubuntu Install Debs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) |
| [`pixi`](https://pixi.sh) | Conda env manager for the Jazzy/lerobot consumer side | `curl -fsSL https://pixi.sh/install.sh \| bash` then `export PATH="$HOME/.pixi/bin:$PATH"` |
| RoboStack-Jazzy (via pixi) | Provides Python 3.12 + rclpy + ROS2 message types for lerobot | `pixi install --manifest-path linux-env/pixi.toml` (handled by `bootstrap.sh`) |
| [Isaac Sim 5.x](https://developer.nvidia.com/isaac-sim) | Physics + USD scene + ROS2 publishers | NVIDIA Omniverse Launcher or `pip install isaacsim` (whichever your environment uses). Default expected at `$HOME/env_isaaclab/bin/isaacsim`; override via `ISAACSIM_BIN` env var. |
| `colcon` | ROS2 workspace builder | `sudo apt install python3-colcon-common-extensions` (already pulled in by ros-humble-desktop) |

## Why two Pythons (don't skip this)

The producer side (Isaac Sim, control_gui, MoveIt) runs in **system Humble
Python 3.10**. The lerobot consumer side requires **Python 3.12+** per
lerobot's `pyproject.toml`. RoboStack-Jazzy via pixi gives us a Python 3.12
+ rclpy that's binary-compatible with the message types Humble producers
emit. Cross-Python DDS discovery requires multicast on the loopback
interface — `linux-env/cyclonedds.xml` enables it (system default disables
multicast, which silently breaks discovery between the two participant
graphs).

## Bootstrap (one command)

```bash
cd /your/path/to/Exploring-VLAs           # wherever you cloned
git submodule update --init --recursive
bash linux-env/scripts/bootstrap.sh
```

`bootstrap.sh` runs:

1. Preflight: confirms `/opt/ros/humble` is installed, `pixi` is on `$PATH`,
   Isaac Sim is findable (warns but doesn't fail if not), `colcon` is available.
2. `git submodule update --init --recursive` (idempotent; safe to re-run).
3. `pixi install --manifest-path linux-env/pixi.toml` — materializes the
   Jazzy + lerobot env at `linux-env/.pixi/`.
4. `colcon build --symlink-install` from `vla_SO-ARM101/`.
5. Prints a "next steps" summary.

Expect 10–20 minutes on first run (mostly the pixi download + colcon C++
compiles).

### Isaac Sim install notes

The launcher script (`linux-env/scripts/isaac/isaacsim_launch.sh`) defaults
to `$HOME/env_isaaclab/bin/isaacsim`. If yours lives elsewhere, set the
override before running anything:

```bash
export ISAACSIM_BIN=/path/to/your/isaacsim
```

The extension folder defaults to the in-repo `isaac-sim-mcp/exts/`; that's
where the `soarm101-dt` extension lives (the submodule). If you keep
isaac-sim-mcp checked out elsewhere, set `EXT_FOLDER`.

## Each session

```bash
bash linux-env/scripts/stack_start.sh        # Isaac Sim + RViz + tkinter GUI (default)
bash linux-env/scripts/stack_start.sh no-rviz   # without RViz (lighter)

bash linux-env/scripts/stack_status.sh       # health check across all layers

# … drive the robot via the tkinter GUI's Quickstart tab, or record episodes
#   via the Record Sim tab (CLI-driveable — see linux-env/CLAUDE.md) …

bash linux-env/scripts/stack_stop.sh         # graceful tear-down
```

`stack_start.sh` is idempotent — if Isaac Sim or the control stack are
already running, it skips that step and reuses them.

## Recording (lerobot dataset capture)

Once the stack is up, the Record Sim tab in `control_gui` orchestrates the
whole recording session. Driveable from CLI:

```bash
ros2 service call /so_arm101_control_gui/rec_start std_srvs/srv/Trigger
# … wait for episodes to complete …
ros2 service call /so_arm101_control_gui/rec_stop std_srvs/srv/Trigger
```

To configure (Episodes count, Block color, etc.), use the widget service.
The ROS2 CLI int-coerces numeric strings, so use the helper:

```bash
bash linux-env/scripts/_param_set_string.sh widget_id Episodes
bash linux-env/scripts/_param_set_string.sh widget_value 4
ros2 service call /so_arm101_control_gui/set_widget_value std_srvs/srv/Trigger
```

Datasets land at `~/.cache/huggingface/lerobot/local/<repo_id>/`. Full
recording reference: [`linux-env/CLAUDE.md`](../../linux-env/CLAUDE.md).

## Linux gotchas (top 5)

| Gotcha | Workaround |
|---|---|
| `ros2 param set` returns "empty node name returned by RMW layer" intermittently | Retry, or use `linux-env/scripts/_param_set_string.sh` (rclpy direct) |
| Isaac Sim launch fails with `Failed to create any GPU devices` (CUDA 999) | `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` (UVM module reload — safe; doesn't touch display) |
| Cross-Python DDS discovery silently fails between Humble producers and Jazzy consumers | `linux-env/cyclonedds.xml` enables multicast on `lo` — sourced by `record_sim_isaac.sh`. System default disables multicast. |
| `kill -9` on Isaac Sim / RViz / tkinter cascades KWin BadWindow errors | Always SIGTERM/SIGINT. `stack_stop.sh` handles this; see global `~/.claude/CLAUDE.md` X11 safety rules |
| Recording loop runs at 13–22 Hz vs 30 Hz target | Known perf issue (Isaac Sim camera bridge + RGBA→RGB conversion). Tracked as cleanup item; doesn't block recording correctness |

## Real-hardware mode (optional)

Real-mode adds ArUco / YOLO perception via `aruco_camera_localizer`. Clone
to wherever you want and ensure colcon can find it:

```bash
# (no canonical location yet — historically lived at ~/Desktop/ros2_ws/src/
# but this is being normalized; see top-level CLAUDE.md cleanup tally)
git clone https://github.com/inbarajaldrin/aruco_camera_localizer.git ~/Desktop/ros2_ws/src/aruco_camera_localizer
cd ~/Desktop/ros2_ws && colcon build --symlink-install --packages-select aruco_camera_localizer
```

Then in another terminal: `bash scripts/restart_aruco_localizer.sh` (or
`restart_yoloe.sh` for YOLO mode). See `isaac-sim-mcp/CLAUDE.md` for the
real-mode topic contract.

## Related docs

- [`../../CLAUDE.md`](../../CLAUDE.md) — repo topology, doc map, source-of-truth bring-up
- [`../../linux-env/CLAUDE.md`](../../linux-env/CLAUDE.md) — pixi-Jazzy env + recording layer
- [`../CLAUDE.md`](../CLAUDE.md) — package inventory + planner architecture
- [`../../isaac-sim-mcp/CLAUDE.md`](../../isaac-sim-mcp/CLAUDE.md) — Isaac Sim extension internals + MCP tool inventory
- [`AGENT_DEBUG_GUIDE.md`](./AGENT_DEBUG_GUIDE.md) — debugging the sim stack
- [`GAZEBO_LINUX_LIMITATION.md`](./GAZEBO_LINUX_LIMITATION.md) — why Gazebo isn't the Linux backend
