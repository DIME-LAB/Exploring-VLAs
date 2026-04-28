# Project Memory — Exploring-VLAs

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
