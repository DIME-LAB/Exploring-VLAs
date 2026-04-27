#!/usr/bin/env bash
# calibrate_arm.sh — wrapper around lerobot-calibrate with clear English prompts.
#
# lerobot's wizard says "move to the middle of its range of motion" which
# is genuinely confusing. This wrapper explains exactly what to do at each
# step BEFORE invoking the wizard, so you don't end up at the prompt
# wondering what "middle" means.
#
# What the wizard does (and what we explain better here):
#   Step 1: Position the arm at URDF-zero (arm fully extended horizontally)
#   Step 2: Move each joint through its full range of motion
#   Step 3: Wizard writes Homing_Offset to EEPROM + saves JSON
#
# Usage:
#   bash /tmp/calibrate_arm.sh
#
# Requires:
#   - jsr_write must NOT be running (calibrate needs the bus)
#   - control_gui can stay running, but Disconnect first

set -e

PORT="${PORT:-/dev/tty.usbmodem5A680088871}"
ARM_ID="${ARM_ID:-my_awesome_follower_arm}"

cat <<'BANNER'
═══════════════════════════════════════════════════════════════════════════
                  SO-ARM101 FOLLOWER ARM CALIBRATION
═══════════════════════════════════════════════════════════════════════════

This will calibrate the joint zero-references AND movement ranges.

What gets written to each servo's EEPROM:
  reg 31  Homing_Offset       — joint-zero reference
  reg  9  Min_Angle_Limit     — lowest allowed Goal_Position (in ticks)
  reg 11  Max_Angle_Limit     — highest allowed Goal_Position (in ticks)

⚠️  CRITICAL: the servo's firmware enforces Min_Angle_Limit / Max_Angle_Limit
itself. If a joint isn't swept here, its Min == Max == current position,
and the servo will SILENTLY REJECT every Goal_Position outside that range
(no bus error, no log line — just no motion). The gripper is the joint
people most often skip. Sweep it, or the Open Gripper button will refuse
to move.

Result: EEPROM updated + JSON saved to
~/.cache/huggingface/lerobot/calibration/robots/so_follower/

You will need TWO physical interactions with the arm.

───────────────────────────────────────────────────────────────────────────
STEP 1: PLACE ARM AT URDF-ZERO POSE  (lerobot calls this "the middle")
───────────────────────────────────────────────────────────────────────────

The arm should be FULLY EXTENDED HORIZONTALLY — like a straight stick
pointing forward. Specifically:

  shoulder_pan    →  Arm pointing STRAIGHT FORWARD from the base
                     (not rotated left or right)
  shoulder_lift   →  Upper arm HORIZONTAL, parallel to the ground
                     (NOT tilted up, NOT drooping)
  elbow_flex      →  Forearm COLLINEAR with upper arm (no bend at elbow)
  wrist_flex      →  Wrist NOT bent — gripper continues horizontally
  wrist_roll      →  Gripper jaws HORIZONTAL (parallel to ground)
  gripper         →  Closed (or your preferred zero opening)

Visual check: sight along the arm from the side — it should look like
a single straight horizontal line from base to gripper tip.

The arm is back-drivable (P=16) — push each joint into position by hand.

───────────────────────────────────────────────────────────────────────────
STEP 2: SWEEP EACH JOINT THROUGH ITS FULL RANGE
───────────────────────────────────────────────────────────────────────────

After you press ENTER in step 1, the wizard says
"Move all joints except 'wrist_roll' sequentially through their entire
ranges of motion. Recording positions. Press ENTER to stop..."

You MUST physically move every joint to its extremes. For each joint:

  shoulder_pan    →  Swing the whole arm LEFT then RIGHT to limits
  shoulder_lift   →  Lift upper arm UP, then DOWN to limits
  elbow_flex      →  Bend elbow fully UP, then fully DOWN
  wrist_flex      →  Flex wrist FULLY one way, then the other
  *** gripper     →  Open the jaws all the way, then close fully —
                     pump open/closed twice so both extremes register.
                     SKIP THIS AND THE GRIPPER WILL REFUSE TO OPEN UNTIL
                     YOU RE-RUN THIS SCRIPT. (The servo's firmware
                     enforces the Min/Max range you sweep here.)
  (wrist_roll is skipped — it's a continuous-rotation axis)

Take your time. Move EACH joint visibly to its end-stops.
Press ENTER only after every joint has been swept through its full range.

Common failure modes:
  - Not moving any joints → wizard crashes:
    "Some motors have the same min and max values."
  - Moving arm joints but skipping the gripper → wizard succeeds, but
    gripper open commands silently fail later (servo firmware-rejected).
    No error in jsr / control_gui logs — just a stuck gripper.

───────────────────────────────────────────────────────────────────────────

Press ENTER when you're ready to start.
BANNER

read -r _

# Pre-flight: kill jsr if running so calibrate has the bus
echo "[wrapper] killing jsr_write to free the bus..."
pkill -9 -f "run_jsr_write.sh" 2>/dev/null || true
pkill -9 -f "joint_state_reader" 2>/dev/null || true
sleep 1.5

echo "[wrapper] launching lerobot-calibrate..."
echo "          Port:    $PORT"
echo "          Arm ID:  $ARM_ID"
echo
echo "  Wizard prompt 'middle of its range of motion' = URDF-zero pose"
echo "  (the horizontal-extended pose described above)"
echo
echo "─────────────────────────────────────────────────────────────────────"

# macOS env-var gotchas (per project CLAUDE.md):
#   KMP_DUPLICATE_LIB_OK=TRUE  → torch+numpy both link OpenMP; without
#                                this, lerobot-calibrate abort()s on import
#   CYCLONEDDS_URI / RMW       → ROS2 DDS config
export KMP_DUPLICATE_LIB_OK=TRUE
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///tmp/mac-env/cyclonedds.xml"

# --robot.type=so101_follower (NOT so_follower).
# The local fork has a so_follower/ directory but it's not registered as
# a CLI-discoverable robot type in lerobot 0.5.2 — only so100_follower
# and so101_follower are exposed via plugin entry points. The SO-ARM101
# IS the SO-101 follower (same hardware), so this is the correct type.
# Calibration JSON will save to:
#   ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/$ARM_ID.json
cd /tmp/mac-env
pixi run --manifest-path /tmp/mac-env/pixi.toml bash -c "
  export KMP_DUPLICATE_LIB_OK=TRUE
  lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port='$PORT' \
    --robot.id='$ARM_ID'
"

CALIBRATE_RC=$?

echo "─────────────────────────────────────────────────────────────────────"
if [ "$CALIBRATE_RC" -eq 0 ]; then
  echo "[wrapper] calibration complete!"
  echo
  echo "Next steps:"
  echo "  1. Restart the stack:"
  echo "       bash mac-env/scripts/stack_stop.sh && bash mac-env/scripts/stack_start.sh --mode real --rviz"
  echo "  2. Re-fit the gravity-comp K_calib (uses the new joint zeros):"
  echo "       python3 /tmp/gravity_comp.py --urdf <urdf_path> --calibrate"
  echo "  3. Verify visually:"
  echo "       python3 /tmp/hold_zero.py"
else
  echo "[wrapper] calibration FAILED with exit code $CALIBRATE_RC"
  echo
  echo "If it crashed with 'Some motors have the same min and max values',"
  echo "you skipped step 2 (the range-of-motion sweep). Re-run this script"
  echo "and physically move each joint through its full range when prompted."
fi

exit $CALIBRATE_RC
