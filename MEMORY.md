# Project Memory — Exploring-VLAs

## Dataset viewer setup (rerun + lerobot-dataset-viz, 2026-04-28)

The `lerobot-dataset-viz` toolchain is fragile on macOS pixi. After it breaks
(symptoms: `datasets`, `av`, `rerun-sdk`, or `pyarrow` ImportError; `_iconv`
symbol-not-found from pyarrow), restore with this exact sequence:

```bash
# 1. pyarrow MUST come from conda (pip wheel hits libiconv linkage bug);
#    `pixi add` is the canonical way, but if a pip transitive pulled in a
#    different version, force-reinstall a known-good one with --no-deps:
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall \
    --no-deps "pyarrow==23.0.0"

# 2. `datasets` MUST be >=4.0 (older versions lack ParquetConfig.filters
#    which lerobot's load_nested_dataset uses):
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall \
    --no-deps "datasets==4.0.*"

# 3. PyAV for video decoding:
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install av

# 4. rerun-sdk for the viewer (will pull pyarrow as a transitive — uninstall
#    that one immediately and restore the conda one):
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install rerun-sdk
pixi run --manifest-path /tmp/mac-env/pixi.toml pip uninstall -y pyarrow
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall \
    --no-deps "pyarrow==23.0.0"

# 5. Restore numpy<2 (any pip install that doesn't use --no-deps bumps it):
pixi run --manifest-path /tmp/mac-env/pixi.toml pip install --force-reinstall \
    --no-deps "numpy<2"

# 6. Verify all imports clean:
pixi run --manifest-path /tmp/mac-env/pixi.toml python3 -c \
    "import pyarrow, datasets, rerun, av, lerobot; \
     print('ok:', pyarrow.__version__, datasets.__version__, rerun.__version__)"

# 7. Launch viewer:
cd /tmp/soarm-ws && pixi run --manifest-path /tmp/mac-env/pixi.toml bash -c '
  source install/setup.bash
  export KMP_DUPLICATE_LIB_OK=TRUE
  lerobot-dataset-viz \
    --repo-id <repo_id> \
    --episode-index 0 \
    --mode local
'
```

**Reference dataset (colleague's real SO-101 pick-and-place):**
`arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep` — 440 episodes,
162221 frames, 30 fps, dataset format v3. Schema metadata snapshot lives at
`lerobot/.planning/real_dataset_probe/{info.json,features.json,samples.json,
PARITY_CONTRACT.md}`.

## Motion-quality stack (2026-04-28) — what's where & why

Empirical-debug session that landed `trajectory_shim` + `jointstatereader` in a stable motion-quality regime. The journey is documented here because every change is non-obvious and tied to a specific observation about the **STS3215 firmware behavior** that you will rediscover otherwise.

### Hardware truths to know first

- **Feetech STS3215 firmware deadband**: ~0.88° (10 encoder counts at 0.088°/LSB). Goal_Position writes within this band of the previously-honored Goal are silently ignored. Source: Robo9 measurement (https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/).
- **Mechanical backlash**: ~0.87° (vendor spec said 0.5°; actual is 75% worse). Adds to the deadband to give a ~1° unrecoverable noise floor on static positioning.
- **Encoder LSB**: 0.088°. Visible "ticks" the user sees are servo motion at this resolution.
- **No upstream gravity-comp** in HuggingFace LeRobot for SO-ARM. Our trajectory_shim's stack is more sophisticated than anything documented upstream — we are in original-research territory.

### What's shipped in trajectory_shim.py

1. **Cubic Hermite interp + zero-velocity endpoints** (line ~88): smooths motion within a trajectory's waypoints; pinning first/last tangents to 0 eliminates the velocity-step jerk at trajectory boundaries.
2. **Soft-start prefix** (line ~497): if `/joint_states − traj.points[0]` exceeds threshold (default 0.5°), prepend a synthetic waypoint at current state and shift original waypoints by `soft_start_s` (0.2s default). Hermite then eases physical-actual → planner-seed-pose. **DEAD CODE** when MoveIt seeds from current state (gap=0), which is the normal case. Kept as defensive scaffolding.
3. **Continuous publishing during hold** (line ~402, `_idle_tick`): the shim publishes the last setpoint at 50 Hz forever between goals. Reason: when the bus goes silent, the STS3215 firmware enters its quiescent static-hold profile; the next goal accept causes a hold→motion firmware transition that briefly drops effective support torque → arm "releases" with gravity for a fraction of a second before catching. Continuous publish keeps the firmware out of that quiescent state.
4. **Wire freeze on `_play()` exit** (line ~752): after success or tolerance-violated abort, `_wire_frozen=True` so `_idle_tick` calls `_publish_frozen()` (republishes literal saved cmd values, bypassing FF/i_term re-evaluation). Reason: without this, the integrator continues drifting cmd post-trajectory at ~0.5°/sec; cumulative shift crosses the firmware deadband ~1 sec later, and the servo lurches in 1° LSB-snap "ticks" the user can see. Freeze stops the drift before it accumulates past the deadband.
5. **50 Hz integrator with proportionally-scaled alpha** (line ~241): integrator runs at the publish rate so the bus sees small per-tick increments rather than 2 Hz step changes. `alpha=0.008` and `motion_leak=0.0114` preserve the same cumulative learning rate as the original 2 Hz × 0.2 / 25% setup.
6. **Rate-limited i_term contribution** (line ~688): per-tick wire-cmd shift due to i_term capped at `max_i_change_per_tick_deg=0.04°` (½ LSB) so individual integrator updates can't cross encoder resolution and stair-step the servo.
7. **Predictive r_hat(q) model** (line ~218 onward): per-joint `K_res` learned online from converged hold integrator values, used as `cmd = q − ff + r_hat(q)` for trajectory-aware gravity correction. **DISABLED BY DEFAULT** (`r_hat_enabled=False`) after a GPT review identified critical signal-path bugs (i_term decoupled from plant → invalid label, learner runs twice per goal-accept, missing NLMS regularization, time-misaligned samples). Code retained as scaffolding for the offline-calibration follow-up. **Don't enable without redesigning the signal path.**

### What's shipped in jointstatereader.py

- **SyncWritePosEx (instruction 0x83)** (line ~263, `_sync_write_pos_ex`): writes Goal_Acc + Goal_Pos + Goal_Time + Goal_Speed for all 6 servos in ONE half-duplex bus packet. Replaces per-joint sequential WRITE_DATA writes (which had ~12 ms cumulative skew across 6 joints — exactly the window where gravity coupling shows up at trajectory start). Speed/Accel set to firmware max (3400/254) so each tick is "go now" rather than firmware-profiled.
- **Param defaults** (`use_sync_write=True`, `sync_write_acc=254`, `sync_write_speed=3400`).

### What was tried and didn't work

- **Freeze i_term during motion** (instead of motion-leak toward 0): made the perceived release WORSE because preserving the prior-pose integrator value into the new pose caused larger wire shifts at goal-accept.
- **Tighten tolerance to 1°**: many trajectories aborted with `tolerance_violated` because the integrator can't drive sub-deadband residuals (1° tolerance < 1° hardware floor). Combined with continuing post-abort drift, made the "ticks" worse and added GUI errors.
- **Online r_hat learning** (per-pose K_res LMS update from converged i_term): K_res oscillated wildly between samples because i_term was no longer in the wire (decoupled from plant), so the learner's "label" was an open-loop accumulator drift rather than a plant-coupled converged value. Trajectory itself became shaky as noisy K_res got injected per tick.

### Stable steady state (2026-04-28 evening)

| Setting | Value | Why |
|---|---|---|
| `tolerance_rad` | 3° | Below this, aborts proliferate (hardware can't deliver) |
| `completion_timeout_s` | 2 s | Original; longer doesn't help |
| `freeze_wire_at_t_end` | true | Stops post-trajectory drift from crossing deadband |
| `r_hat_enabled` | **false** | Online learner has critical signal-path bugs |
| `continuous_hold` | true | Eliminates trajectory-start "release" jerk |
| `use_sync_write` (jsr) | true | Eliminates per-joint bus skew |
| `_i_deadband` | 1° | Match firmware deadband floor |
| `max_i_change_per_tick_deg` | 0.04° | Sub-LSB per-tick wire shifts |

**Static accuracy:** ~3° at hold (per `tolerance_rad`). Acceptable for current experiments but limiting for grasping. Future work: offline calibration (move arm to N diverse poses, hold each ~10 s, measure settled residual command, fit `K_res` per joint via ridge regression on `τ_g` features, ship as static `K_res`). Then re-enable `r_hat_enabled=true` with the static `K_res`.

### Diagnostic tools that exist for next time

- `/tmp/start_jerk_recorder_v2.py` — stream-to-disk recorder for `/joint_commands` + `/joint_states`. SIGINT-safe. Use this for any future motion-quality investigation.
- `/tmp/tick_inspector.py` / `/tmp/tick_classifier.py` — find LSB-events in state stream and classify as during-motion vs idle-drift.
- `/tmp/tick_plots_v2.py` / `/tmp/tick_plots_v3.py` — generate annotated PNGs (events overlaid, shim trajectory boundaries marked).
- The empirical loop: record → inspect → propose → measure-diff. **Don't theory-fix without baseline measurement** — half this session's iterations chased the wrong root cause because we trusted intuition over data.

## Workflow preferences

- **Always restart the stack when a procedure requires it.** Don't ask "want me to restart?" — just run `bash mac-env/scripts/stack_stop.sh && bash mac-env/scripts/stack_start.sh --mode <mode> --rviz follower_port:=...` and wait for `/tmp/soarm_stack.log` to show ready signals. Reason: user runs `--dangerously-skip-permissions` and wants next-step actions taken without confirmation.
- **Real-mode follower port** is currently `/dev/tty.usbmodem5A680088871`. Pass it explicitly to `stack_start.sh`.
- **K_calib** lives at `~/.cache/so_arm101/gravity_comp/k_calib.json` and was refit 2026-04-27.

## Calibration / EEPROM workflow (SO-ARM101 follower)

These notes capture lessons from 2026-04-27 sessions where two agents successively chased phantom calibration bugs. Read these BEFORE touching `recenter_one`, `recenter_one_to_commanded`, or `lerobot-calibrate`.

### When `/joint_states` looks wrong

In priority order:

1. **Check the audit script's decoded values, not its raw EEPROM values.** `mac-env/scripts/feetech_calib_audit.py` was buggy until 2026-04-27: it read EEPROM Homing_Offset as `int16` but Feetech STS3215 firmware stores it as **12-bit sign-magnitude** (bit 11 = sign, bits 0-10 = magnitude). The fixed decoder now matches what lerobot's `motors_bus.py` does internally → `EEPROM == JSON` is the actual signal, not "raw int16 == JSON".
2. **Verify by physically positioning the arm to the same pose where you pressed ENTER at calibration step 1.** That pose is the calibration's "zero reference"; `/joint_states` should read ~0° on every joint there. The wizard's "middle of its range of motion" wording is ambiguous — it just records WHATEVER pose you press ENTER at, regardless of whether that matches URDF zero.
3. **Bus contention during calibration corrupts EEPROM writes.** Even though `calibrate_arm.sh` `pkill`'s jsr first, control_gui's auto-Connect respawns its own jsr ~3s after launch. If lerobot-calibrate is still running then, jsr's 50 Hz reads of register 56 collide with lerobot's writes to register 31. **Workflow:** `stack_stop` → `calibrate_arm.sh` → audit → `stack_start`. Never run calibration with the stack up.

### Recovering from accumulated EEPROM drift (the canonical sequence)

```bash
# 1. Stop everything; bus must be exclusive to lerobot-calibrate
bash mac-env/scripts/stack_stop.sh
pkill -f "joint_state_reader|trajectory_shim|control_gui" 2>/dev/null
sleep 4

# 2. Run lerobot-calibrate via the wrapper (kills jsr again as belt-and-suspenders)
bash mac-env/scripts/calibrate_arm.sh
# - Step 1: physically position arm at desired zero pose (URDF zero = straight horizontal),
#           press ENTER. THIS pose becomes /joint_states zero reference.
# - Step 2: sweep every joint through full range INCLUDING gripper. Don't skip the
#           gripper or it'll refuse to open until re-calibration.

# 3. Verify EEPROM == JSON for all 6 joints
pixi run --manifest-path /tmp/mac-env/pixi.toml python3 mac-env/scripts/feetech_calib_audit.py
# Expect all rows: Match? ✓

# 4. Restart stack
bash mac-env/scripts/stack_start.sh --mode real --rviz follower_port:=/dev/tty.usbmodem5A680088871

# 5. Move arm to URDF zero, verify /joint_states ≈ 0° for every joint in RViz.
```

### When to use which recenter tool

- **`recenter_one`** (per-joint): user freedrives ONE joint by hand (torque off), positions it to where they want zero, then calls. EEPROM gets a new offset such that current physical = 0 in /joint_states. Safe because freedrive turned torque off first, so changing Homing doesn't trigger PD-driven motion or sag. Use for: drift correction on a single joint where you can hand-position to the desired zero.
- **`arm_trim.py`**: integer ±delta tick adjustments per joint. Recommended range ±50 ticks (~±4.4°). Larger means re-run lerobot-calibrate.
- **`lerobot-calibrate`** (via `calibrate_arm.sh`): the safe path for bulk drift / EEPROM saturation cleanup. Resets all 6 joints from a single user-positioned reference pose.

NB: an in-place "recenter to a non-zero target" operation (rewrite EEPROM Homing while torque on, no hand-positioning) is **architecturally impossible** on Feetech servos for cable-loaded joints. `Goal_Position` is interpreted in post-Homing coords, so any Homing change implies a hold-target shift the firmware doesn't rebase. A torque-off cycle to avoid PD-driven motion just trades it for cable/gravity sag. Don't try to build this.

### Feetech STS3215 register encoding gotchas

- `Homing_Offset` (reg 31): **12-bit sign-magnitude** in EEPROM, NOT int16. `bit 11 = sign`, `bits 0-10 = magnitude`. Lerobot's stack decodes correctly; raw byte readers (e.g. older `feetech_calib_audit.py`) need the explicit decode.
- `Present_Position` (reg 56): unsigned 12-bit `0..4095`, with Homing_Offset already applied by firmware. jsr converts via `(ticks - 2048) / 2048 * π`.
- `Goal_Position` (reg 42): same coordinate system as Present_Position (post-Homing). Changing Homing while torque is ON makes the same Goal value point to a new physical position → PD drives the joint there → unwanted motion. This is why all recenter implementations need a torque-off window OR pre-existing freedrive.
- Standard `WRITE DATA (0x03)` takes effect immediately; no `ACTION` command needed for register 31. So no power-cycle / "save to EEPROM" step required after lerobot-calibrate.

### Lerobot 0.5.2 calibration-path quirks

- JSON saves to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` (NOT `so101_follower/`) regardless of `--robot.type=so101_follower`. The CLI flag configures the runtime CLASS; the JSON dir uses the BASE class name. (HANDOFF A.10 documented + reverted a path-drift assumption that broke this.)
- URDF/jsr/control_gui call the gripper joint `gripper_joint`; lerobot calibration JSON calls it `gripper`. Translation happens at the wrapper boundary.

## Gripper frame rename (2026-04-28)

The end-effector now has three distinct TF frames. Knowing which to use prevents calibration off-by-half-gap bugs:

- `gripper_tip` — static URDF link (was `tcp_link` pre-rename). **The IK chain tip** (SRDF `<chain tip_link>` points here). Use whenever you need the fixed-jaw tip pose; never refer to it as "TCP" in messages to agents.
- `jaw_tip` — static URDF link, child of `jaw`. Origin `(-0.010058, -0.081201, 0.018900)` derived from `moving_jaw_so101_v1.stl` via `calibrate_jaw.py`.
- `tcp_link` — **dynamic, broadcast at runtime by `grasp_center_publisher`** (NOT in URDF). Midpoint of `gripper_tip` + `jaw_tip` at the live `gripper_joint` angle. This is the "true TCP" / grasp center. `/ee_pose` reports this frame; vision `calibration_offset` should target this.

For "what-if" FK queries (e.g. collision-checking a planned grasp), request both `gripper_tip` and `jaw_tip` from MoveIt FK and average — same answer the live broadcaster would give. See `_compute_fk_to_spinboxes` in `control_gui.py`.

### TODO: Isaac Sim USD parity

Isaac Sim digital twin (`~/Documents/isaac-sim-mcp/`, branch `so-arm101`) still references `tcp_link` in the so-arm101-dt extension's USD. When we revisit Isaac work, port the same three-frame layout: rename existing `tcp_link` Xform → `gripper_tip`, add `jaw_tip` Xform under the moving-jaw prim, broadcast a dynamic `tcp_link` Xform via the action graph or a Python prim. Until then, real-world is correct, sim is on the old single-frame model — DO NOT trust Isaac-side `tcp_link` poses for grasp-center math.
