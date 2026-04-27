#!/usr/bin/env bash
# arm_finetune.sh — one-button finetuning workflow for the SO-ARM101 follower.
#
# Replaces the per-step sequence (lerobot-calibrate → write PID defaults →
# refit K_calib → verify) with a single guided run. Same outcome each time
# regardless of who's at the keyboard.
#
# Stages:
#   1. lerobot-calibrate     — joint zero references (Homing_Offset to EEPROM)
#   2. PID defaults to servos — P=16 D=32 I=0 (lerobot-recommended)
#   3. K_calib refit         — gravity-comp coefficients (Pinocchio + LS)
#   4. Visual verify         — drive to all-zero, hold 30s for inspection
#
# Skip stages with: --skip-calibrate, --skip-pid, --skip-kcalib, --skip-verify
# Hardware required (steps 1, 3, 4): SO-ARM101 follower at /dev/tty.usbmodem*
#
# Usage:
#   bash mac-env/scripts/arm_finetune.sh                     # full workflow
#   bash mac-env/scripts/arm_finetune.sh --skip-calibrate    # already calibrated lerobot
#   bash mac-env/scripts/arm_finetune.sh --skip-verify       # batch mode
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

PORT="${PORT:-/dev/tty.usbmodem5A680088871}"
ARM_ID="${ARM_ID:-my_awesome_follower_arm}"
SKIP_CALIBRATE=0
SKIP_PID=0
SKIP_KCALIB=0
SKIP_VERIFY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-calibrate) SKIP_CALIBRATE=1; shift ;;
    --skip-pid)       SKIP_PID=1; shift ;;
    --skip-kcalib)    SKIP_KCALIB=1; shift ;;
    --skip-verify)    SKIP_VERIFY=1; shift ;;
    --port)           PORT="$2"; shift 2 ;;
    --arm-id)         ARM_ID="$2"; shift 2 ;;
    --help|-h)
      sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "ERROR: unknown arg '$1'. Use --help."; exit 1 ;;
  esac
done

URDF_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)/vla_SO-ARM101/src/so_arm101_description/urdf/so_arm101.urdf"
KCALIB_FILE="$HOME/.cache/so_arm101/gravity_comp/k_calib.json"
mkdir -p "$(dirname "$KCALIB_FILE")"

echo "═══════════════════════════════════════════════════════════════════"
echo "                  SO-ARM101 ARM FINETUNE WORKFLOW"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Port:        $PORT"
echo "  Arm ID:      $ARM_ID"
echo "  URDF:        $URDF_PATH"
echo "  K_calib out: $KCALIB_FILE"
echo "═══════════════════════════════════════════════════════════════════"

# ===== Stage 1: lerobot-calibrate =====
if [ "$SKIP_CALIBRATE" -eq 0 ]; then
  echo ""
  echo "── STAGE 1/4: lerobot-calibrate (joint zero references) ──"
  echo "Will prompt you to: (a) place arm at URDF-zero pose (horizontal stick"
  echo "pointing forward), (b) sweep each joint through its full range."
  echo ""
  bash "$SCRIPT_DIR/calibrate_arm.sh"
else
  echo "── STAGE 1/4: skipped ──"
fi

# ===== Stage 2: PID defaults =====
if [ "$SKIP_PID" -eq 0 ]; then
  echo ""
  echo "── STAGE 2/4: write lerobot PID defaults (P=16 D=32 I=0) ──"
  pkill -9 -f run_jsr_write.sh 2>/dev/null || true
  pkill -9 -f joint_state_reader 2>/dev/null || true
  sleep 1.5
  pixi run --manifest-path "$MAC_ENV_MANIFEST" python3 \
    "$SCRIPT_DIR/arm_reset.py" --set-lerobot-defaults --port "$PORT"
else
  echo "── STAGE 2/4: skipped ──"
fi

# ===== Stage 3: K_calib refit =====
if [ "$SKIP_KCALIB" -eq 0 ]; then
  echo ""
  echo "── STAGE 3/4: refit gravity-comp K_calib ──"
  echo "Drives arm to URDF-zero + grasp_home, measures sag, fits K_calib."
  echo "Output: $KCALIB_FILE"
  pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
    source '$SOARM_WS_SETUP'
    export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
    export CYCLONEDDS_URI='$CYCLONEDDS_URI'
    export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
    ros2 run so_arm101_control gravity_comp_node \
      -- --calibrate \
      --urdf '$URDF_PATH' \
      --k-calib-file '$KCALIB_FILE'
  "
else
  echo "── STAGE 3/4: skipped ──"
fi

# ===== Stage 4: Visual verification =====
if [ "$SKIP_VERIFY" -eq 0 ]; then
  echo ""
  echo "── STAGE 4/4: visual verification ──"
  echo "Starting gravity_comp daemon + driving arm to all-zero pose."
  echo "Inspect physically: arm should be horizontal stick pointing forward."
  echo ""
  pkill -f gravity_comp 2>/dev/null || true
  sleep 1
  nohup pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
    source '$SOARM_WS_SETUP'
    export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
    export CYCLONEDDS_URI='$CYCLONEDDS_URI'
    export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
    ros2 run so_arm101_control gravity_comp_node \
      -- --urdf '$URDF_PATH' --k-calib-file '$KCALIB_FILE'
  " > /tmp/gravity_comp.log 2>&1 &
  disown
  sleep 4
  pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
    source '$SOARM_WS_SETUP'
    export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
    export CYCLONEDDS_URI='$CYCLONEDDS_URI'
    export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
    python3 '$SCRIPT_DIR/hold_zero.py'
  "
else
  echo "── STAGE 4/4: skipped ──"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                    FINETUNE WORKFLOW COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Saved to: $KCALIB_FILE"
echo ""
echo "Next steps:"
echo "  - bash mac-env/scripts/stack_start.sh --mode real --rviz"
echo "    (gravity_comp_node auto-spawns, loads K_calib from above path)"
echo "  - All motion via /joint_targets gets gravity-comp + integrator applied"
echo "    automatically before reaching the bus."
echo "═══════════════════════════════════════════════════════════════════"
