# linux-env — agent context

Linux-side counterpart to `mac-env/`. Provides the **pixi-Jazzy environment**
that runs `lerobot-record` against Isaac Sim's existing `_sim`-suffixed topics,
plus the Linux-side mirror + drive scripts and the bring-up automation
(`bootstrap.sh`, `deploy_sim.sh`, `stack_start.sh`, `stack_stop.sh`, `stack_status.sh`).
Human-facing quick start (incl. watching from a Mac): [`../README.md`](../README.md).

> **Inherits**: see [`../CLAUDE.md`](../CLAUDE.md) for repo topology, doc map, and
> Linux gotchas; [`../vla_SO-ARM101/CLAUDE.md`](../vla_SO-ARM101/CLAUDE.md) for the control stack
> the recorder rides on; [`../vla_SO-ARM101/docs/ROS2_LINUX_SETUP.md`](../vla_SO-ARM101/docs/ROS2_LINUX_SETUP.md)
> for first-time install (Humble + pixi + Isaac Sim + colcon).

## First-time setup

```bash
cd /your/path/to/Exploring-VLAs
git submodule update --init --recursive
bash linux-env/scripts/bootstrap.sh
```

`bootstrap.sh` preflights, inits submodules, installs the pixi env, and runs
colcon. Idempotent — safe to re-run.

## Each session

```bash
# Deploy: Isaac + scene + control stack + SmolVLA async policy (GPU 1)
bash linux-env/scripts/deploy_sim.sh up [--rate-hz HZ] [--stream] [--gui|--no-gui] [--rviz|--no-rviz] [--model <preset|hf_id>] [--no-policy]
bash linux-env/scripts/deploy_sim.sh status
bash linux-env/scripts/deploy_sim.sh down [--policy]

# Or the stack without the policy (recording sessions)
bash linux-env/scripts/stack_start.sh        # Isaac Sim + RViz + tkinter GUI
bash linux-env/scripts/stack_start.sh no-rviz   # without RViz (lighter)
bash linux-env/scripts/stack_status.sh       # all layers' health
# … record / drive …
bash linux-env/scripts/stack_stop.sh         # graceful tear-down
```

## Why linux-env exists

On Linux the producer side (Isaac Sim + control stack) runs in **system
Humble Python 3.10**; the consumer side (lerobot's `so101_ros2`
Robot/Teleop/Camera plugins) needs **Python 3.12+** per lerobot's pyproject.
RoboStack-Jazzy on Linux gives us that 3.12 + `rclpy` + the message types
without touching the system Humble install.

`record_sim_isaac.sh` runs lerobot inside this pixi env; everything else
(Isaac Sim, control_gui, mirror) stays on system Humble. Cross-boundary DDS
discovery between the two participant graphs needs `cyclonedds.xml`'s
multicast-on-`lo` config (system default disables multicast).

## Files

| File | Role | Runs in |
|---|---|---|
| `pixi.toml` / `pixi.lock` | ros-jazzy-ros-base + lerobot deps | (env definition) |
| `cyclonedds.xml` | Multicast-on-lo for cross-Python DDS discovery (Humble producer ↔ Jazzy consumer) | (DDS config) |
| `scripts/_lib.sh` | Shared paths + env (`REPO_ROOT`, `ROS2_SETUP`, `SOARM_WS`, `PIXI_MANIFEST`, `ISAAC_LAUNCHER`, `KIT_MCP_DIR`, `SOARM_KIT_MCP_HOST/PORT` → also exported as `OMNI_KIT_MCP_PORT` for the launcher) and the shared bring-up steps used by stack_start + deploy_sim: `stack_preflight`, `stack_source_ros`, `kit_mcp_call <tool> [json] [timeout]`, `kit_mcp_ready`, `scene_is_live`, `isaac_up`, `scene_up`, `stack_detect_display`, `stack_require_display` (xdpyinfo check, fails with the fix; no Xvfb fallback), `controllers_active`, `control_stack_up <rviz> <gui>` (ready = the three controllers active, + control_gui when asked), `stop_pid` | (lib) |
| `scripts/bootstrap.sh` | One-shot setup: preflight → `git submodule sync/update` → pixi install → colcon build. Does NOT run omni-kit-mcp's `install.py` (the Isaac 6.1 install is shared; the bridge is enabled per launch) | foreground |
| `scripts/deploy_sim.sh up\|down\|status` | One command for the twin + policy: Isaac (`--stream` = headless WebRTC) → `soarm101.quick_start` → control stack (`control.launch.py gui:=` — control_gui/RViz opt-in via `--gui`/`--rviz`, default on locally and off with `--stream`) → `inference_async_lerobot.sh` with `CUDA_VISIBLE_DEVICES`=GPU `SOARM_POLICY_GPU` (default 1, pinned by UUID). `--model` takes a preset (`sim`, `mix-75s25r`, `mix-25s75r`, `mix-50s50r`) or an HF id. State + pidfiles in `/tmp/soarm_deploy/`; `down` stops only what `up` started; `down --policy` stops just the policy | detached |
| `scripts/stack_start.sh [no-rviz] [--stream]` | Bring up Isaac Sim + quick_start scene + ROS2 control stack with control_gui (no policy; fails fast without a usable display). Idempotent; skips quick_start when the scene is already live | foreground |
| `scripts/stack_stop.sh` | Graceful SIGINT-first tear-down. Doesn't `kill -9` anything that owns an X11 window | foreground |
| `scripts/stack_status.sh` | Health across Isaac (launcher status: process, bridge, stream; scene live?), ROS nodes, sim topics, /clock | foreground |
| `scripts/isaac/isaacsim_launch.sh` | Isaac Sim 6.1 + `soarm101-dt` + omni-kit-mcp bridge, enabled for this launch only. Subcommands: `launch [ext] [--stream] / wait [timeout] / status / close / kill / restart`; readiness = `soarm101.*` answering on the bridge; `close` only stops the Isaac it started. Extension folder is fixed to `$REPO/isaac/exts` (no `EXT_FOLDER` override). Bridge port: `OMNI_KIT_MCP_PORT` > `SOARM_KIT_MCP_PORT` > 8767. Log `/tmp/isaacsim-soarm101.log`. GPU wedge (no bridge + no Kit log progress for `ISAACSIM_WEDGE_S`=120 s) → exit 3 with a message, no automatic relaunch, Isaac left running for inspection. `wait [idle]` is RTX-shader-compile aware (idle = no-progress limit, cap `ISAACSIM_WAIT_CAP`, default 900 s); the first-run compile usually lands during the first `soarm101.quick_start`, so `scene_up` gives it `SOARM_QUICK_START_TIMEOUT` (default 900 s) then falls back to `wait`. Extension outputs (scene_states, videos) go to `MCP_CLIENT_OUTPUT_DIR` (default `/tmp/soarm101-resources`); `SOARM_BLENDER` is only for rebaking ArUco markers. Other env overrides (`ISAACSIM_BIN`, `ISAACSIM_GPU`, `ISAACSIM_STREAM_IP`, `ISAACSIM_LOG`) in its header | foreground |
| `scripts/_param_set_string.sh <name> <value> [node]` | Set a string-typed ROS2 param without int-coercion. Workaround for `ros2 param set ... '4'` parsing as integer 4 (breaks STRING-typed params like `widget_value`) | foreground |
| `scripts/eval/run_episodes.sh` | Home → blue-visible randomize → persistent-client policy episode → cup/lift score → viewport MP4 + CSV/JSONL. Requires the running scene/controllers; never launches/stops Isaac. [Instructions](scripts/eval/README.md), [RTC and timing](scripts/eval/rtc-report.md). Default client/server fps 17.4, duration 1/fps; deployment and inference wrappers accept `--fps`, `--action-duration-s`, `--no-rtc` | Humble orchestrator + pixi client/server |
| `scripts/record_sim_isaac.sh` | lerobot-record wrapper. Subscribes to `/wrist_camera_rgb_sim` + `/workspace_camera_sim` @ 640×480, action topic `/joint_commands_lerobot` | pixi-Jazzy (via bash wrapper) |
| `scripts/joint_states_to_commands.py` | Publishes controller-reference (commanded) positions → `/joint_commands_lerobot` at 30 Hz so lerobot's `action` column genuinely leads `observation.state`. Filename is historical; reads `/{arm,gripper}_controller/controller_state.reference.positions`, not `/joint_states` | system Humble Python 3.10 |
| `scripts/drive_pick_place.py` | Direct-publish driver via FollowJointTrajectory action client (alternative to control_gui's `qs_*` services for smoke tests) | system Humble |
| `scripts/drive_pick_place_loop.sh` | Thin wrapper around drive_pick_place.py; sources ROS2 then execs | system Humble |
| `scripts/drive_base_yaw_sweep.py` | Symlink to `../mac-env/scripts/drive_base_yaw_sweep.py` (synthetic motion smoke) | system Humble |

## CRITICAL: don't publish to `/joint_commands` from the mirror

control_gui (`vla_SO-ARM101/.../control_gui.py:_ext_cmd_callback`)
**subscribes to `/joint_commands`** and dispatches a FollowJointTrajectory
action goal per message. Mirroring `/joint_states → /joint_commands` at
30 Hz creates a feedback loop: ~60 action goals/sec saturate control_gui's
4-thread MultiThreadedExecutor → all `qs_*` services time out → pick-place
halts under recording load.

**Fix landed Apr 27, 2026:** mirror publishes to `/joint_commands_lerobot`,
record_sim_isaac.sh's teleop reads from there. `/joint_commands` stays
reserved for control_gui's "external teleop" semantic. Don't revert this
without first wiring an `enable_external_teleop=False` gate on
`_ext_cmd_callback`.

## Standard recording session (manual)

```bash
# 1. Bring up Isaac Sim + control stack first: bash scripts/stack_start.sh
# 2. Start the mirror (system Humble Python 3.10)
source /opt/ros/humble/setup.bash
nohup python3 -u scripts/joint_states_to_commands.py > /tmp/jstojc.log 2>&1 &
# Verify: ros2 topic info /joint_commands_lerobot → Publisher count: 1

# 3. Start lerobot-record (pixi-Jazzy via bash wrapper)
nohup bash scripts/record_sim_isaac.sh \
  --dataset.repo_id=local/<name> \
  --dataset.num_episodes=4 \
  --dataset.episode_time_s=120 \
  --dataset.single_task="Pick a blue lego and place it in blue cup" \
  --dataset.push_to_hub=false \
  --display_data=false \
  > /tmp/lerobot.log 2>&1 &
# Wait for 'Recording episode 0' line in /tmp/lerobot.log

# 4. Drive pick-place (option A: control_gui's qs_* services)
ros2 param set /so_arm101_control_gui ik_target blue_2x3
ros2 service call /so_arm101_control_gui/qs_refresh_all std_srvs/srv/Trigger
sleep 2.5
ros2 service call /so_arm101_control_gui/qs_select std_srvs/srv/Trigger
ros2 service call /so_arm101_control_gui/qs_play std_srvs/srv/Trigger
# Poll get_log for "pick-and-drop cycle complete" sentinel.

# 5. On stop: pkill -SIGINT -f lerobot-record (graceful dataset finalize)
```

## Preferred path: Record Sim tab

Better than the manual flow above — the **Record Sim tab in control_gui**
(added Apr 27, 2026) orchestrates the entire session: spawns mirror +
lerobot-record, runs the QS pick-place loop, retries on halt, finalizes
dataset on stop.

```bash
# Drive entirely from CLI, no GUI focus needed:
# 1. set Episodes (defaults 16). NOTE: ros2 param CLI int-coerces numeric strings,
#    so we use the rclpy helper to force STRING typing on widget_value.
bash linux-env/scripts/_param_set_string.sh widget_id Episodes
bash linux-env/scripts/_param_set_string.sh widget_value 4
ros2 service call /so_arm101_control_gui/set_widget_value std_srvs/srv/Trigger
# 2. start (auto-named dataset rec_<HHMMSS>)
ros2 service call /so_arm101_control_gui/rec_start std_srvs/srv/Trigger
# 3. monitor — Record Sim tab status reads via get_widget_value, or watch:
tail -F $(ls -t /tmp/control_stack_*.log | head -1) | grep "Record:"
# 4. stop (clean SIGINT to subprocs, finalize dataset)
ros2 service call /so_arm101_control_gui/rec_stop std_srvs/srv/Trigger
```

The Record Sim tab uses `REC_LEGOS_BY_COLOR` (module-level constant in
`control_gui.py`) — currently 2 of each color × 2x3 only:
`red_2x3_0/1`, `green_2x3_0/1`, `blue_2x3_0/1`. To add 2x2s back, edit
the constant AND update the soarm101-dt scene contract (`isaac/exts/soarm101-dt`) together — they
have to stay in sync (the scene must spawn whatever the orchestrator
expects to grasp).

## Datasets

Saved to `~/.cache/huggingface/lerobot/local/<repo_id>/` — three subdirs
(`data/`, `videos/`, `meta/`). **Use `lerobot-dataset-viz` to view —
DON'T write a custom rerun loader** (see gotcha below).

```bash
# from anywhere — REPO_ROOT/linux-env/pixi.toml is the manifest
pixi run --manifest-path "$(git rev-parse --show-toplevel)/linux-env/pixi.toml" \
  lerobot-dataset-viz \
  --repo-id local/<name> --root <full_path> --episode-index 0 --mode local
```

Local mode pushes into the rerun viewer at `localhost:9876` (start one
first: `pixi run rerun --port=9876 --memory-limit=10% --expect-data-soon`).
Re-running with a different `--episode-index` while the viewer is alive
attaches the new recording to the dropdown — don't kill/restart between
episodes.

## Gotchas

- **`/joint_commands` vs `/joint_commands_lerobot`** — see CRITICAL section
  above. Mirror MUST publish to the `_lerobot` variant.
- **Multiple mirror processes** — Record Sim tab dedups by checking
  `/joint_commands_lerobot` publisher count before spawning. Manual
  invocations should `pkill -f joint_states_to_commands` first.
- **Cross-boundary DDS** — record_sim_isaac.sh forces
  `CYCLONEDDS_URI=file://<linux-env>/cyclonedds.xml` so multicast on `lo`
  is enabled (system default is multicast off, which prevents Jazzy
  consumers from discovering Humble producers across process trees).
- **SIGINT, not SIGTERM** — lerobot-record needs SIGINT to finalize the
  dataset chunk cleanly. Record Sim tab sends `os.killpg(getpgid, SIGINT)`
  via `start_new_session=True` Popen so the signal propagates to the
  pixi-spawned child.
- **lerobot teleop warm-up** — `action_warmup_s=2.0`, so mirror MUST be
  publishing before lerobot starts subscribing or recording dies with
  `DeviceNotConnectedError`.
- **Don't run dataset viz in parallel with the live ROS→rerun bridge** —
  both push into port 9876 and the streams interleave. Kill the bridge
  (`pkill -f ros_to_rerun`) before pushing dataset frames.
- **NEVER write a custom rerun viewer for lerobot datasets.** Use the
  documented `lerobot-dataset-viz` command above. Reasons:
  - The official tool decodes videos via lerobot's pyav/torchvision
    path inside the pixi-Jazzy env, which works regardless of system
    ffmpeg version.
  - rerun's own `rr.AssetVideo` route requires **system ffmpeg ≥ 5.1**;
    Ubuntu 22.04 ships ffmpeg 4.4.2, which yields:
    `Failed to decode video: FFmpeg version is 4.4.2-0ubuntu0.22.04.1.
    Only versions >= 5.1 are officially supported.`
    Going down the AssetVideo path led to a 100-line cv2-frame-by-frame
    fallback that re-implemented what `lerobot-dataset-viz` already
    does. Don't repeat that mistake.
  - The viewer accumulates recordings as you re-invoke with different
    `--episode-index`, so per-episode review is one CLI call per ep.
  - If the dataset is on a remote machine, use `--mode distant`
    + `scp` the `.rrd` and open with local `rerun foo.rrd`.
- **action mirror semantics** — `joint_states_to_commands.py` no longer
  mirrors `/joint_states` despite the filename. It reads
  `/{arm,gripper}_controller/controller_state.reference.positions`
  (the JTC's per-tick commanded setpoint) and republishes as JointState
  on `/joint_commands_lerobot` at 30 Hz. This makes action genuinely
  lead state in the saved parquet (real-world property; previous mirror
  produced action == state to 14 decimals). Note: in sim, the magnitude
  of the lead is sub-degree because Isaac Sim physics tracks the
  smoothed planner trajectory closely. Real-world teleop sees 2-4° lead
  due to motor lag + human-jitter on the leader arm. For BC training
  this is still informative; for tighter sim2real fidelity, add a
  `reference.positions + reference.velocities × lookahead_s` projection
  (~100 ms) or detune the joint drives.

Sim pacing: the launcher's `DEFAULT_SIM_RATE_HZ` is the single default cadence (17.4 Hz, from ROADMAP 1.1).
`deploy_sim.sh up --rate-hz HZ` applies Kit pacing after World initialization
and prints the readback; `0` is unthrottled. At 60 Hz physics, RTF = Hz/60 only
when the box has enough CPU headroom. Do not change physics dt to tune wall pacing.

Pose graphs feed `ROS2PublishTransformTree` its external arrays: ordinary USD
Xforms (`/World`, `/World/base`, cup-center wrappers) are not physics tensor
objects. Preserve `World` / `base` and the `drop_1/2/3` frame names. Strip scale
with `Gf.Transform(matrix).GetRotation()` when extracting mesh orientations.
The log UI uses a bounded 500-row model; do not rebuild 500 labels per line.
Physics callback diagnostics are limited to one per source every five seconds.

Kit 110.3 requires `/app/runLoops/main/syncToPresent=false` for the main-loop
rate limiter to take effect. The launcher sets it per launch and on `rate` calls.
Keep physics at 60 Hz. The default 17.4 Hz ceiling targets RTF 0.29; overrides
19.4 and 0 select the alternate timing sweep and unthrottled profiling.
