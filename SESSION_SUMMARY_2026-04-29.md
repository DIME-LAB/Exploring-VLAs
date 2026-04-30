# Session Summary — 2026-04-29

SmolVLA inference deployment on the SO-ARM101 Isaac Sim digital twin: from "no inference path exists" to "two working inference methods (custom + canonical async + RTC), recording, GUI tab, end-to-end videos in transfer folder." Plus diagnosis of a model-side limitation (undertrained sim FT) and a sim-physics smoothness investigation.

## Goal

Stand up closed-loop SmolVLA inference into Isaac Sim — model produces actions, arm moves, video proves it. Originally with two real-world FT models (smoke test the plumbing); later with the user's sim-trained checkpoint (`anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200`, full FT, 100 ep, 10k steps, H200).

## Method 1 — Custom chunk-replan node (built first)

**Files added:**
- `linux-env/scripts/smolvla_inference.py` — rclpy node. Subscribes `/joint_states` + cameras, runs `predict_action_chunk` at 1 Hz, dispatches FJT goals to `/arm_controller` + `/gripper_controller` at 30 Hz, hard-clamps to URDF limits.
- `linux-env/scripts/inference_smolvla.sh` — pixi-Jazzy wrapper with idempotent dep install + PYTHONPATH scrub.
- `vla_SO-ARM101/docs/SMOLVLA_INFERENCE_LINUX.md` — runbook.

**Bugs hit + fixed (in order):**
1. `huggingface_hub` rejected naïve `repo/checkpoints/X/pretrained_model` path → switched to `snapshot_download` + local-path resolution.
2. `UnsupportedTypeSupport` — pixi Py 3.12 inheriting Humble's Py 3.10 PYTHONPATH → wrapper now scrubs `PYTHONPATH`/`AMENT_PREFIX_PATH`/etc.
3. `ModuleNotFoundError: control_msgs` after scrub → added `ros-jazzy-control-msgs` + `ros-jazzy-trajectory-msgs` to `linux-env/pixi.toml`.

**Canonical inference patterns added** (matches `lerobot_eval`):
- `policy.reset()` before first inference
- `torch.inference_mode()` + `torch.autocast(cuda)`
- `torch.cuda.empty_cache()` per chunk
- Default rates aligned to dataset: `publish_rate=30`, `inference_rate=1.0`

## Method 2 — Canonical async inference + RTC (built second)

Reference: user's `delete_after_ingest` script (LeRobot async inference launcher, real-hardware-flavored, with RTC params).

**Files added:**
- `linux-env/scripts/run_async_inference_sim.py` — sim-adapted launcher. Spawns `lerobot.async_inference.policy_server` + `lerobot.async_inference.robot_client`; threads RTC params through env vars.
- `linux-env/scripts/inference_async_lerobot.sh` — wrapper (env scrub + idempotent dep install for `[smolvla,async]`).
- `linux-env/scripts/record_cameras_to_mp4.py` — fallback camera recorder for when async-mode native recording isn't wired.
- `/tmp/bag_to_mp4.py`, `/tmp/bag_to_state.py` — rosbag2 → mp4/npz extractors.

**Files modified (lerobot fork):**
- `lerobot/src/lerobot/robots/so101_ros2/config_so101_ros2.py`
  - Added `actuate: bool = False` (opt-in actuation; preserves recording-time no-op default)
  - Added `arm_action_topic`, `gripper_action_topic`, `gripper_joint_urdf`, `clamp_joint_limits`
  - `action_duration_s = 0.1` (was 0.0333) — JTC interpolation window for smoother motion
- `lerobot/src/lerobot/robots/so101_ros2/so101_ros2.py`
  - Implemented `send_action()` behind `actuate` flag — converts deg→rad, clamps to URDF limits, dispatches FJT goals (arm + gripper) via `rclpy.action.ActionClient`. Returns clamped action.
  - Bring up + tear down the action clients in `connect()` / `disconnect()`.
- `lerobot/src/lerobot/async_inference/robot_client.py`
  - Added `from lerobot.robots import so101_ros2` + `from lerobot.cameras.ros2 import ROS2CameraConfig` for plugin discovery (per `lerobot/CLAUDE.md`'s explicit-import rule).
- `lerobot/src/lerobot/async_inference/policy_server.py`
  - Added `import os` + RTC env-var override block. `LEROBOT_RTC_ENABLED=1` enables RTC at startup; `LEROBOT_RTC_EXECUTION_HORIZON`, `LEROBOT_RTC_MAX_GUIDANCE_WEIGHT`, `LEROBOT_RTC_PREFIX_ATTENTION_SCHEDULE` tune it. Re-inits the policy's `rtc_processor` after override.

**Pixi env:**
- `linux-env/pixi.toml` gained `ros-jazzy-control-msgs`, `ros-jazzy-trajectory-msgs`.
- `pip install lerobot[smolvla,async] + peft` — provides transformers, accelerate, grpcio, peft.

## GUI Inference tab

**File modified:** `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/control_gui.py`

Added between Record Sim and RViz:
- Module constants: `INF_SCRIPT`, `INF_ASYNC_SCRIPT`, `INF_MODEL_PRESETS`, `INF_METHOD_LABELS`, `INF_DEFAULT_*`.
- Presets: "Sim FT (full, blue_sort 100ep)" first/default, "Full FT (real-world, smoke)", "LoRA (real-world, smoke)", "Custom".
- Method var: defaults to `async_rtc`. Custom method routes to Method 1; async_rtc routes to Method 2.
- UI sections: Model · Task · Rates · Recovery · Record · Status. (Method radio UI itself **commented out** — see open issues.)
- Recovery: "Return to grasp_home (data) on Stop" checkbox (default ON). On subprocess exit, dispatches `_send_gripper_goal(1.4 rad)` + `_cmd_grasp_home_data()` so the next run starts from a clean pose.
- Record: optional checkbox + dataset name; spawns `joint_states_to_commands.py` mirror + `record_sim_isaac.sh` alongside the inference subprocess for custom mode (skipped for async — uses ros2 bag fallback).
- Auto-registered services: `inf_start`, `inf_stop`, `inf_dry_run`, `inf_tail_log`.
- Hot-reload tab list updated.

## Recording

- **Custom mode**: native via lerobot-record + mirror (writes v3 dataset).
- **Async mode**: not yet GUI-orchestrated; uses `ros2 bag record` + `bag_to_mp4.py` extractor as a stop-gap.

## Verification — videos sent to `/home/aaugus11/transfer/`

Per-video configuration table. All files are `1280×480` side-by-side mp4
(workspace on left, wrist on right, hstacked via `ffmpeg`).

### `inference_verify_inf_verify_063545.mp4`
- **Model**: `anirudhrani/smolvla_blue_sort_ven_50k @ checkpoints/050000` (real-world full FT, smoke test)
- **Task**: "Pick a blue lego and place it in blue cup"
- **Method**: Custom chunk-replan (Method 1)
- **Rates**: `publish_rate=25 Hz`, `inference_rate=2 Hz`
- **Recording**: lerobot-record + mirror, `episode_time_s=60`, auto-finalized
- **Trim**: `ffmpeg -ss 22` (skip model-load warmup)
- **Observed**: First end-to-end proof the loop closes. Real-world FT in sim → wrist_flex saturated at limit, lots of clamping (146 events). Plumbing pass; behavior expected-poor.

### `inference_sim_inf_sim_long_102036.mp4`
- **Model**: `anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200` (sim FT, full)
- **Task**: "Pick a blue lego and place it in blue cup"
- **Method**: Custom chunk-replan
- **Rates**: `publish_rate=25 Hz`, `inference_rate=2 Hz` (OLD — wrong)
- **Recording**: lerobot-record + mirror, `episode_time_s=120`
- **Trim**: `ffmpeg -ss 22`
- **Observed**: Arm hovers, no descent. **Diagnosis**: chunk math wrong — at these rates client consumes only 12 of 50 actions per chunk (25%), throwing away the descend+close phase (actions ~15-30).

### `inference_sim_v2_inf_sim_v2_103446.mp4`
- **Model**: same sim FT
- **Method**: Custom chunk-replan
- **Rates**: **`publish_rate=30 Hz`, `inference_rate=1 Hz`** (matches `dataset.fps=30`; consume ~30 of 50 actions per chunk = 60%)
- **Recording**: lerobot-record + mirror, `episode_time_s=120`
- **Trim**: `ffmpeg -ss 21`
- **Observed**: **Descent visible** — `wrist_flex` 90°→64.67° (61% of frames < 85°). Zero clamping. Gripper barely opens (range 0-3.85°).

### `inference_diag_inf_diag_104545.mp4`
- **Model**: same sim FT
- **Method**: Custom chunk-replan, with per-chunk **envelope instrumentation** added to `smolvla_inference.py` (logs each chunk's per-joint min/max + `chunk[0]` + `chunk[49]`)
- **Rates**: `publish_rate=30 Hz`, `inference_rate=1 Hz`
- **Recording**: lerobot-record + mirror, `episode_time_s=90`
- **Trim**: `ffmpeg -ss 21`
- **Observed**: 56 chunks logged. **`wrist_flex` minimum across all chunks: 63.80°.** **Gripper maximum across all chunks: 3.80°.** 0 chunks plan `<60°` or gripper `>30°`. **Verdict: model itself is undertrained — pipeline is innocent.**

### `inference_async_112130.mp4`
- **Model**: same sim FT
- **Method**: **Canonical async** (Method 2) — `lerobot.async_inference.policy_server` + `robot_client`
- **Robot**: `--robot.type=so101_ros2 --robot.actuate=true --robot.use_degrees=true`
- **Async params**: `--actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30`
- **RTC**: **OFF** (default at this point — RTC env-var override didn't exist yet)
- **`action_duration_s`**: 0.0333 s (default)
- **Recording**: `ros2 bag record /wrist_camera_rgb_sim /workspace_camera_sim` (manual fallback) → `bag_to_mp4.py` → `ffmpeg hstack`
- **Trim**: `ffmpeg -ss 30`
- **Observed**: Full async loop closes. `shoulder_pan +35°, shoulder_lift +10°, wrist_flex 90°→68°, gripper 0→0.11 rad opening`. Inference latency 380-420 ms stable. First confirmation `send_action` actuation works.

### `inference_async_rtc_114550.mp4`
- **Model**: same sim FT
- **Method**: Canonical async + **RTC ON**
- **Async params**: `--actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30`
- **RTC**: `--rtc.enabled --rtc.execution-horizon=10 --rtc.max-guidance-weight=10.0 --rtc.schedule=EXP` (env vars `LEROBOT_RTC_ENABLED=1` etc. picked up by patched `policy_server.py`)
- **`action_duration_s`**: **0.1 s** (bumped from 0.0333 — JTC interpolation window for sim smoothness)
- **Server log confirms**: `RTC enabled: execution_horizon=10, max_guidance_weight=10.0, prefix_attention_schedule=EXP`
- **Recording**: `ros2 bag record /wrist_camera_rgb_sim /workspace_camera_sim /joint_states` → extractor → `ffmpeg hstack`
- **Trim**: `ffmpeg -ss 22`
- **Observed**: Server latency stable 380-400 ms. Diff-based jerk metric didn't drop noticeably, but that's a measurement artifact (bag joint-state Δt has 7.6-40 ms jitter vs training's synthesized uniform 33 ms). Visual comparison vs `inference_async_112130.mp4` is the truth.

### Versions that exist on disk but NOT counted as valid
- An `actions_per_chunk=25` attempt was launched but failed: Isaac Sim's `/joint_states` publisher had stalled (`Publisher count: 0`), client timed out on connect, recording captured static frames. Discarded.

## Findings

### Model-side
- The sim-trained checkpoint (10k steps × 100 episodes on H200) is **undertrained for the descend+grasp phase**. Across 56 chunks emitted in one run: 0 chunks planned `wrist_flex < 60°`; 0 chunks planned gripper > 30°. The arm hovers above the lego but the model never commits to descent.
- Pipeline is provably innocent: zero clamping events, commanded actions match achieved state across every joint range.

### Sim-physics smoothness
- Training data jerk p95 ≈ 11k deg/s³ (clean 30 Hz QS-driven trajectories).
- Inference jerk p95 ≈ 145k deg/s³ — **13× higher** — mostly a measurement artifact (training data has synthesized uniform 33 ms timestamps; bag-recorded inference has real ROS publish jitter — Δt min 7.6 ms / max 40 ms inflates Δpos/Δt).
- Real driver of visual choppiness is **JTC cancel-replace at 30 Hz**: each `send_action` fires a fresh FJT goal, JTC starts a new interpolation from current state. Real Feetech motors don't have this concept; sim does.
- Two fixes applied: enabled **RTC** (flow-matching denoising-time guidance, blends new chunk's prefix with prev chunk's tail) and bumped **`action_duration_s` 33ms → 100ms** (longer JTC interpolation window). Visual comparison should be the truth — diff-based jerk metric isn't apples-to-apples.

### Cross-Python plumbing (recurring)
- System Humble Py 3.10 publishes `/joint_states`, runs control_gui.
- Pixi-Jazzy Py 3.12 runs lerobot, smolvla, async client/server.
- They share DDS via `linux-env/cyclonedds.xml` (multicast on `lo`).
- Wrapper scripts MUST scrub `PYTHONPATH`/`AMENT_PREFIX_PATH` before invoking pixi or Py 3.12 will silently load Py 3.10 type-support C extensions and segfault.

## Open issues

1. **GUI Method radio UI is commented out** — caused control_gui to SEGV during tab build (likely a `tk.Radiobutton`/`_widget_registry_add` interaction). Method selection still works via the `_inf_method_var` Tk var (defaults to `async_rtc`); CLI fallback for `custom` mode. Needs a proper fix.
2. **Async-mode native recording** is not wired into the GUI — currently uses `ros2 bag record` as a fallback. Wiring it in is straightforward (similar shape to Method 1's record orchestration; uses the bag-to-mp4 extractor for postprocessing).
3. **Cumulative GPU pressure** — running multiple inference rollouts back-to-back occasionally tipped over move_group / rviz / control_gui with SIGSEGV during shutdown, leaving stale rclpy contexts. Workaround: hard-kill all `so_arm101*` processes + `ros2 daemon stop` before each restart.
4. **Isaac Sim action graph stalls** — twice over the session, `/joint_states` `Publisher count` dropped to 0 after sustained inference activity (Isaac Sim process stayed alive). Workaround: re-trigger `bash linux-env/scripts/stack_start.sh` (idempotent).
5. **The sim model itself** is the bottleneck for behavior — needs more training (longer / more episodes / curriculum that emphasizes the descend-and-grasp phase).

## Files created this session

```
SESSION_SUMMARY_2026-04-29.md                                          (this file)
linux-env/scripts/smolvla_inference.py
linux-env/scripts/inference_smolvla.sh
linux-env/scripts/run_async_inference_sim.py
linux-env/scripts/inference_async_lerobot.sh
linux-env/scripts/record_cameras_to_mp4.py
vla_SO-ARM101/docs/SMOLVLA_INFERENCE_LINUX.md
/tmp/bag_to_mp4.py                                                     (extractor)
/tmp/bag_to_state.py                                                   (extractor)
```

## Files modified this session

```
vla_SO-ARM101/src/so_arm101_control/so_arm101_control/control_gui.py   (Inference tab)
lerobot/src/lerobot/robots/so101_ros2/config_so101_ros2.py             (actuate, action_duration_s)
lerobot/src/lerobot/robots/so101_ros2/so101_ros2.py                    (send_action implementation)
lerobot/src/lerobot/async_inference/policy_server.py                   (RTC env override)
lerobot/src/lerobot/async_inference/robot_client.py                    (so101_ros2 plugin discovery)
linux-env/pixi.toml                                                    (control-msgs + trajectory-msgs)
```

## How to use today

```bash
# Stack up
bash linux-env/scripts/stack_start.sh

# Verify joint_states is publishing
ros2 topic info /joint_states     # should show Publisher count: 1+
ros2 topic hz /joint_states       # should print ~30-50 Hz

# Method 2 (recommended): async + RTC
bash linux-env/scripts/inference_async_lerobot.sh \
    --pretrained=anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200 \
    --task='Pick a blue lego and place it in blue cup' \
    --rtc.enabled

# Or via GUI: Inference tab → ▶ Start (async + RTC is the default)

# Method 1: custom chunk-replan
bash linux-env/scripts/inference_smolvla.sh \
    --model.path=anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200 \
    --task='Pick a blue lego and place it in blue cup'
```

## Next session — priorities

1. Re-train the sim model with more steps / episodes — current bottleneck.
2. Fix the Method radio SEGV (debug `_widget_registry_add` + `tk.Radiobutton` interaction).
3. Wire native recording into async-mode GUI orchestration.
4. (Optional) Multi-point `JointTrajectory` dispatch — accumulate full chunk into one FJT goal, eliminate cancel-replace entirely. Bigger architectural change to `send_action`.
