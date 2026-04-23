# ROS2 on macOS (Apple Silicon) — general setup

Living runbook for bringing up a native ROS2 Jazzy workspace on an
Apple-Silicon Mac. This doc is project-agnostic: once it completes, you
have a working `ros2` CLI, a colcon-buildable workspace, and the four
repos this project uses (SO-ARM101, lerobot, aruco_camera_localizer,
RoboSort). Project-specific runbooks (recording a lerobot dataset,
building the grasp pipeline, etc.) link here for the bootstrap and then
branch into their own steps.

> For the lerobot-specific record/train workflow, see
> [`LEROBOT_ROS2_MAC_SETUP.md`](./LEROBOT_ROS2_MAC_SETUP.md).

## Why native (no Docker)

Docker on macOS adds a Linux VM, which defeats the point of an ARM-native
workflow and breaks camera / USB-serial access. Native RoboStack via pixi
gets you pre-built ARM64 binaries for every ROS2 package we use — Gazebo
Harmonic, MoveIt, ros2_control, etc. No emulation, no VM, no hardware
passthrough nonsense.

## The hard requirements

- **Apple Silicon Mac** (M1/M2/M3/M4). Intel Macs aren't tested.
- **macOS 14+** (Sonoma or newer).
- **No spaces in the path** where the workspace and pixi env live. RoboStack's
  conda shebangs don't survive `-isystem /Path With Spaces/...`; most
  build/run scripts in this project materialize the env and ws under
  `/tmp/` specifically to avoid this.
- **Xcode command-line tools** installed (`xcode-select --install`).
- **20 GB free on the boot disk** for the pixi env + colcon build artefacts.

## The core tools

| Tool | Role | Install |
| --- | --- | --- |
| [`pixi`](https://pixi.sh) | Conda environment manager. Reads `pixi.toml`, resolves ARM64 conda packages including RoboStack's ROS2. | `curl -fsSL https://pixi.sh/install.sh \| bash` → `export PATH="$HOME/.pixi/bin:$PATH"` |
| [RoboStack](https://robostack.github.io) | Conda channel that packages ROS2 for macOS Apple Silicon. Pulled in via `pixi.toml`, not directly. | No separate install — pixi handles it. |
| [`colcon`](https://colcon.readthedocs.io) | ROS2 workspace builder. Pulled in by the pixi env. | — |
| [CycloneDDS](https://cyclonedds.io) | ROS2 middleware that actually discovers topics on macOS (FastDDS silently fails between Python processes). | `pip install cyclonedds` is **not** what you want — RoboStack ships `ros-jazzy-rmw-cyclonedds-cpp`. Configured via `cyclonedds.xml` + env vars. |
| [`ros-jazzy-desktop`](https://index.ros.org/p/ros2/) | ROS2 base meta-package. Pulled in by `pixi.toml`. | — |

## Bootstrap (one command)

This project ships a `mac-env/scripts/bootstrap.sh` that does:

1. Validates pixi is on `$PATH`.
2. Copies `pixi.toml` + `cyclonedds.xml` to `/tmp/mac-env/`.
3. Runs `pixi install --manifest-path /tmp/mac-env/pixi.toml` to materialize the env (~1.5 GB download on first run).
4. Sets up `/tmp/soarm-ws/src/` with symlinks to every package in the four repos (SO-ARM101, lerobot, aruco_camera_localizer, RoboSort).
5. Runs `colcon build --symlink-install` against that workspace.

```bash
cd /path/to/Exploring-VLAs   # or wherever you cloned the project
export PATH="$HOME/.pixi/bin:$PATH"
git submodule update --init --recursive
bash mac-env/scripts/bootstrap.sh
```

Expect 10–15 minutes on first run, mostly on the pixi download + the
initial colcon build.

Crucial post-bootstrap pin (macOS Accelerate ILP64 / numpy 2 drama):

```bash
pixi run --manifest-path /tmp/mac-env/pixi.toml \
  pip install --force-reinstall --no-deps 'numpy<2'
```

Without this pin, Python-based `controller_manager` spawners crash silently
on import, leaving `/joint_states` at 0 Hz.

## Runtime environment

Every shell that talks to ROS2 needs:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export CYCLONEDDS_URI="file:///tmp/mac-env/cyclonedds.xml"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export KMP_DUPLICATE_LIB_OK=TRUE
```

Source the workspace on top:

```bash
source /tmp/soarm-ws/install/setup.bash
```

The scripts under `mac-env/scripts/` (stack_start, record, etc.) do all of
this for you inside a `pixi run --manifest-path /tmp/mac-env/pixi.toml bash -c ...`
block — use them whenever possible instead of re-doing the env dance by hand.

## macOS-specific gotchas

### `ros2` daemon hangs

The default `ros2 daemon` hangs on macOS with CycloneDDS. Pass
`--no-daemon` to every CLI invocation (`ros2 topic list --no-daemon`,
`ros2 topic echo --no-daemon /foo`, etc.).

### CycloneDDS affinity warnings (benign)

Every `ros2` invocation on macOS emits several lines like:

```
[SYSTEM Error] Problem to set affinity of thread ... 'Protocol family not supported'
```

These come from CycloneDDS trying to pin threads on macOS, which isn't
supported. They are harmless — ignore.

### OpenMP duplicate runtime

PyTorch + conda's libomp collide on macOS → `OMP: Error #15: Initializing
libomp.dylib, but found libiomp5.dylib already initialized`. Always export
`KMP_DUPLICATE_LIB_OK=TRUE`.

### Spaces in the repo path

If your clone is under `~/Documents/Projects/my cool project/...`, colcon
will build ambient packages fine but any package that uses `-isystem`
(everything C++ in RoboStack) will fail with quoting errors. The
bootstrap script works around this by hosting `/tmp/mac-env` and
`/tmp/soarm-ws` — never build directly in-tree.

### Camera / USB-serial permissions

On first access:

- **Camera:** macOS prompts via a banner; accept it in *System Settings → Privacy & Security → Camera*.
- **Serial (ACM):** nothing extra usually needed, but the device may enumerate as `/dev/tty.usbmodem*` instead of `/dev/ttyACM*` — check with `ls /dev/tty.usb*`.

## The four repos (in this project)

| Repo | Role |
| --- | --- |
| `vla_SO-ARM101/` | Robot description, Gazebo world, MoveIt config, `control_gui`, `jointstatereader`, `so_arm101_bringup`, `sim_ground_truth` |
| `lerobot/` | The Hugging Face `lerobot` fork (rebased to upstream main) with our `so101_ros2` Robot + Teleop plugins |
| `aruco_camera_localizer/` | Real-side object localization (ArUco/YOLO) — publishes `/objects_poses`, `/objects_bbox` consumed by `control_gui` |
| `RoboSort/` | Mobile-robot side (JETANK) — separate product line, bootstrapped together for infra reuse |

The bootstrap symlinks every package under `/tmp/soarm-ws/src/`, so
`colcon build` builds them all in one pass. Each repo keeps its own git
history.

## Quick sanity checks after bootstrap

```bash
# Pixi env alive
pixi run --manifest-path /tmp/mac-env/pixi.toml ros2 --version

# Workspace has our packages
source /tmp/soarm-ws/install/setup.bash
ros2 pkg list --no-daemon | grep -E "so_arm101|sim_ground_truth|jointstatereader|lerobot"

# ROS ↔ ROS can discover (run in two shells)
ros2 topic list --no-daemon                 # shell 1
ros2 topic pub --no-daemon /hello std_msgs/String "{data: hi}"  # shell 2
```

If `ros2 pkg list` is missing a package, the colcon build for that package
failed — `cat /tmp/soarm-ws/log/latest_build/<pkg>/stderr.log` has the
reason.

## Where to go next

- **Record a lerobot dataset (sim or real):** [`LEROBOT_ROS2_MAC_SETUP.md`](./LEROBOT_ROS2_MAC_SETUP.md).
- **Debug the sim stack:** [`AGENT_DEBUG_GUIDE.md`](./AGENT_DEBUG_GUIDE.md).
- **Grasp pipeline (control_gui → MoveIt):** [`grasp_pipeline.md`](./grasp_pipeline.md).
- **Big-picture diagram:** [`pipeline_diagram.html`](./pipeline_diagram.html).

---

## Changelog

- **2026-04-23** First version. Extracted the Mac-specific bootstrap
  material out of `LEROBOT_ROS2_MAC_SETUP.md` so non-lerobot flows
  (aruco, RoboSort, standalone sim) have a clean entry point.
