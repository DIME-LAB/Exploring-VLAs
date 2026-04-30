# SmolVLA Closed-Loop Inference (Linux / Isaac Sim)

> **Inherits**: see [`../../CLAUDE.md`](../../CLAUDE.md) for repo topology;
> [`../../linux-env/CLAUDE.md`](../../linux-env/CLAUDE.md) for the pixi-Jazzy env;
> [`./ROS2_LINUX_SETUP.md`](./ROS2_LINUX_SETUP.md) for first-time install.

Loads any SmolVLA checkpoint (full FT or LoRA) and closes the loop in Isaac
Sim — model sees `/joint_states` + `/wrist_camera_rgb_sim` +
`/workspace_camera_sim`, dispatches `FollowJointTrajectory` action goals to
`/arm_controller` + `/gripper_controller`. The two checkpoints listed below
are real-world FT (won't perform well in sim — use them for plumbing
smoke tests). When the sim-trained checkpoint lands, point `--model.path`
at it; everything else stays the same.

## Architecture

Two-process design forced by the Python version split:

```
control_gui (system Humble Py 3.10)        inference_smolvla.sh
  └── Inference tab                          └── pixi-Jazzy Py 3.12
        ├── ▶ Start  ──spawns──▶                    ├── load SmolVLA (full FT or LoRA)
        ├── ⏹ Stop   ─SIGINT──▶                     ├── subscribe /joint_states + cams
        └── 🔍 Dry-run                              ├── inference @ 2 Hz → action chunk
                                                     └── publish @ 25 Hz → FJT action goals
                                                                            (arm + gripper)
```

The subprocess is the source of truth. control_gui is **not** on the
critical path — actions go straight to ros2_controllers, sidestepping the
`/joint_commands` 60 Hz saturation trap noted in
[`../CLAUDE.md`](../CLAUDE.md).

## Prereqs

1. Bootstrap done (`bash linux-env/scripts/bootstrap.sh`)
2. Stack up (`bash linux-env/scripts/stack_start.sh`)
3. Verify the input topics are alive:
   ```bash
   ros2 topic hz /joint_states
   ros2 topic hz /wrist_camera_rgb_sim
   ros2 topic hz /workspace_camera_sim
   ```

First-run only: the wrapper installs `lerobot[smolvla]` + `peft` into the
pixi env (idempotent — touch-marker skips re-runs). Takes 1-2 minutes.

## Quickest path: control_gui's Inference tab

After the stack is up, switch to the **Inference** tab in control_gui:

1. **Preset** dropdown:
   - `Full FT (real-world, smoke)` — anirudhrani/smolvla_blue_sort_ven_50k
   - `LoRA (real-world, smoke)` — anirudhrani/smolvla_blue_sort_v2_lora
   - `Sim FT (custom)` — for when your sim-trained checkpoint arrives
2. **Task** — natural-language instruction. Default matches the dataset:
   `Pick a blue lego and place it in blue cup`
3. Click **🔍 Dry-run** first — loads the model + runs one inference on
   synthetic obs without touching ROS. Confirms model + processors load
   cleanly. Output goes to `/tmp/inf_dryrun_*.log`.
4. Click **▶ Start** — spawns the inference subprocess. Status updates to
   RUNNING; log path shows in the Status section.
5. Click **⏹ Stop** to SIGINT the subprocess (clean rclpy + torch teardown).

## CLI path

```bash
# Smoke test — no ROS, just verify load + one inference
bash linux-env/scripts/inference_smolvla.sh \
  --model.path=anirudhrani/smolvla_blue_sort_ven_50k \
  --model.checkpoint=050000 \
  --task="Pick a blue lego and place it in blue cup" \
  --dry-run

# Closed-loop run
bash linux-env/scripts/inference_smolvla.sh \
  --model.path=anirudhrani/smolvla_blue_sort_ven_50k \
  --model.checkpoint=050000 \
  --task="Pick a blue lego and place it in blue cup"

# LoRA — needs rename map because it was trained on camera1/2/3 keys
bash linux-env/scripts/inference_smolvla.sh \
  --model.path=anirudhrani/smolvla_blue_sort_v2_lora \
  --model.checkpoint=050000 \
  --task="Pick a blue lego and place it in blue cup" \
  --rename-map='{"wrist": "camera1", "top": "camera2"}'

# Custom local sim-trained checkpoint (when it arrives)
bash linux-env/scripts/inference_smolvla.sh \
  --model.path=/path/to/your/sim_ft/checkpoints/050000/pretrained_model \
  --task="Pick a blue lego and place it in blue cup"
```

Stop with Ctrl-C (SIGINT). The script's signal handler tears down rclpy
and the torch context cleanly.

## Topic + unit contract

| Direction | Topic | Encoding | Unit |
|---|---|---|---|
| in | `/joint_states` | `sensor_msgs/JointState` | rad → deg (use_degrees=True) |
| in | `/wrist_camera_rgb_sim` | `sensor_msgs/Image` rgba8 640×480 | RGBA → RGB |
| in | `/workspace_camera_sim` | `sensor_msgs/Image` rgba8 640×480 | RGBA → RGB |
| out | `/arm_controller/follow_joint_trajectory` | `FollowJointTrajectory` action | deg → rad |
| out | `/gripper_controller/follow_joint_trajectory` | `FollowJointTrajectory` action | deg → rad |

Joint name mapping: URDF `gripper_joint` ↔ canonical `gripper`. Hard-clamp
to URDF limits before publishing (warn-and-clamp; doesn't error).

## Reset / safety

- **Joint limit clamping is on by default** (`--clamp-joint-limits`). When
  the model proposes an out-of-bounds target the script clamps and warns
  via the log. Disable with `--no-clamp-joint-limits` only for deliberate
  diagnostics.
- **Episode reset** — manual: stop, reset the scene from control_gui or
  `bash vla_SO-ARM101/scripts/sim_reset.sh`, then restart the inference
  subprocess (it calls `policy.reset()` on each load). The action queue is
  cleared on shutdown.
- **No physical safety story yet** — when you move to real hardware,
  consider wrapping with a watchdog that stops the controller on
  high-jerk action sequences, slow rates, or torque spikes.

## Verifying the loop closes

A model trained on real-world data will **not** perform the task in sim
(domain gap is huge). For smoke testing, the success criterion is:

- ✅ Subprocess starts cleanly (no model-load errors in the log)
- ✅ `inference ok: 50 actions, took ~XXX ms` log lines appear at ~2 Hz
- ✅ `ros2 topic echo /arm_controller/follow_joint_trajectory` shows
  goals arriving at ~25 Hz
- ✅ Joints visibly move in Isaac Sim (likely flailing, but moving)

If any of these fail, the plumbing is broken — fix that before assuming
the sim-trained checkpoint will save you. If all pass and behavior is
just bad: that's the domain gap, exactly what your sim-trained model is
designed to close.

## Gotchas

- **Hot-reload ON by default** — Ctrl+Shift+R rebuilds tabs in-place. The
  Inference tab is in the rebuild list, so you don't need to restart
  control_gui after editing the tab builder.
- **Editing control_gui.py needs a colcon rebuild** to land in the install
  tree (per [`../CLAUDE.md`](../CLAUDE.md): symlink-install no longer
  propagates pure-Python changes since setuptools 80+):
  ```bash
  cd vla_SO-ARM101
  colcon build --packages-select so_arm101_control --symlink-install
  source install/setup.bash
  ```
  Editing `linux-env/scripts/smolvla_inference.py` requires no rebuild
  (the wrapper invokes it by absolute path).
- **HF repo download is on first run** — the model and base VLM
  (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` for full FT,
  `lerobot/smolvla_base` for LoRA) cache to `~/.cache/huggingface/hub/`.
  Expect a few GB on first launch.
- **GPU memory** — SmolVLA-500M loads in ~3 GB; comfortable on any
  modern NVIDIA card. If you hit OOM, drop `--device=cpu` (much slower
  inference, but loop still closes).
- **`/joint_commands` is OFF the critical path** — control_gui's
  `_ext_cmd_callback` is not involved. This is intentional. Don't "fix"
  the inference script to publish on `/joint_commands` instead — that
  re-introduces the executor-saturation trap.

## When the sim-trained model lands

```bash
# Drop the checkpoint somewhere accessible
ls /path/to/sim_ft/checkpoints/050000/pretrained_model/
# config.json  model.safetensors  policy_preprocessor.json  train_config.json  ...

# Use it via control_gui Inference tab → preset = "Sim FT (custom)" →
# fill in Model path = /path/to/sim_ft, Checkpoint = 050000

# Or via CLI:
bash linux-env/scripts/inference_smolvla.sh \
  --model.path=/path/to/sim_ft \
  --model.checkpoint=050000 \
  --task="Pick a blue lego and place it in blue cup"
```

If the sim model uses identical camera keys (`wrist`, `top`,
`empty_camera_0`) — which it should, since it trains on our recorded
dataset — no rename map needed. If keys differ, set rename map and rely
on the same fallback as the LoRA case.

## Real-hardware deployment (later)

The script is platform-agnostic at the topic level. To deploy on real
hardware:

1. Replace the camera topic args with the real cameras (e.g. via
   `--camera-topics='{"wrist": "/wrist_camera/image_raw", "top":
   "/top_camera/image_raw"}'` matching `aruco_camera_localizer` /
   `so_arm101_bringup`'s real-side outputs).
2. The `/arm_controller/follow_joint_trajectory` +
   `/gripper_controller/follow_joint_trajectory` action targets are the
   **same on real and sim** (both use ros2_control with identical
   controller names — that's the whole point of the parity contract).
3. Add a hardware-side watchdog before turning the loop on for real.

The inference script doesn't need any changes between sim and real — the
control stack handles the substrate difference.

## Files

- `linux-env/scripts/smolvla_inference.py` — the inference node
- `linux-env/scripts/inference_smolvla.sh` — pixi env wrapper + idempotent
  dep top-up
- `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/control_gui.py`
  — Inference tab (`_build_inference_tab`, `_cmd_inf_*`)
