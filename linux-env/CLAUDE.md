# linux-env — agent context

Linux-side counterpart to `mac-env/`. Provides the **pixi-Jazzy environment**
that runs `lerobot-record` against Isaac Sim's existing `_sim`-suffixed topics,
plus the Linux-side mirror + drive scripts.

> **Inherits**: see `../CLAUDE.md` for repo topology and shared conventions,
> and `../vla_SO-ARM101/CLAUDE.md` for the control stack the recorder rides on.

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

| Script | Role | Runs in |
|---|---|---|
| `pixi.toml` / `pixi.lock` | ros-jazzy-ros-base + lerobot deps | (env definition) |
| `cyclonedds.xml` | Multicast-on-lo for cross-boundary discovery | (DDS config) |
| `scripts/record_sim_isaac.sh` | lerobot-record wrapper. Subscribes to `/wrist_camera_rgb_sim` + `/workspace_camera_sim` @ 640×480, action topic `/joint_commands_lerobot` | pixi-Jazzy (via bash wrapper) |
| `scripts/joint_states_to_commands.py` | Publishes controller-reference (commanded) positions → `/joint_commands_lerobot` at 30 Hz so lerobot's `action` column genuinely leads `observation.state` (not action == state). Filename is historical; *reads* `/{arm,gripper}_controller/controller_state.reference.positions`, not `/joint_states`. See gotcha below. | system Humble Python 3.10 |
| `scripts/drive_pick_place.py` | Mac-style direct-publish driver via FollowJointTrajectory action client (alternative to control_gui's `qs_*` services for smoke tests) | system Humble |
| `scripts/drive_pick_place_loop.sh` | Thin wrapper around drive_pick_place.py; sources ROS2 then exec's the python | system Humble |
| `scripts/_lib.sh` | Shared paths/env (currently mac-style stub; Linux scripts source ROS2 directly) | (lib) |
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
# 1. Bring up Isaac Sim + control stack first (see ../../isaac-sim-mcp/CLAUDE.md)
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
# 1. set Episodes (defaults 16)
ros2 param set /so_arm101_control_gui widget_id Episodes
ros2 param set /so_arm101_control_gui widget_value '4'
ros2 service call /so_arm101_control_gui/set_widget_value std_srvs/srv/Trigger
# 2. start (auto-named dataset rec_<HHMMSS>)
ros2 service call /so_arm101_control_gui/rec_start std_srvs/srv/Trigger
# 3. monitor — Record Sim tab status reads via get_widget_value, or watch:
tail -F $(ls -t /tmp/control_stack_*.log | head -1) | grep "Record:"
# 4. stop (clean SIGINT to subprocs, finalize dataset)
ros2 service call /so_arm101_control_gui/rec_stop std_srvs/srv/Trigger
```

The Record Sim tab uses `REC_LEGOS_BY_COLOR` (module-level constant in
`control_gui.py`) — currently `{red: [red_2x2, red_2x3], green: [...],
blue: [blue_2x2, blue_2x3]}` to give 2 episodes per scene. Update the
constant to change which legos cycle.

## Datasets

Saved to `~/.cache/huggingface/lerobot/local/<repo_id>/` — three subdirs
(`data/`, `videos/`, `meta/`). **Use `lerobot-dataset-viz` to view —
DON'T write a custom rerun loader** (see gotcha below).

```bash
cd ~/Projects/Exploring-VLAs/linux-env
pixi run --manifest-path ./pixi.toml lerobot-dataset-viz \
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
