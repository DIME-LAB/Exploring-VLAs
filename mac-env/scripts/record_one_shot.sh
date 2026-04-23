#!/usr/bin/env bash
# record_one_shot.sh — set up, record, and tear down everything in a single call.
#
# Target user: anyone who just wants to record ONE dataset without remembering
# which producers have to be up. `record.sh` (and the original `record_sim.sh`)
# only own the subscriber side; they leave the sim stack, camera publisher,
# and drive script running after exit. This wrapper owns the whole session.
#
# Lifecycle:
#   1. If the sim stack isn't running, start it headless (skipped on --real).
#   2. Spawn camera_publisher and drive_joint_commands as tracked children.
#      (On --real, camera_publisher only — no drive script, the leader arm
#       publishes /joint_commands via jointstatereader instead.)
#   3. Run record.sh with the rest of the CLI passed through.
#   4. On ANY exit path (success, SIGINT, error), SIGINT every tracked child.
#      Leave the sim stack up if WE started it (user may want to record again);
#      use `stack_stop.sh` to take it all down.
#
# Usage:
#   bash record_one_shot.sh [--real] [--no-drive] -- <record.sh flags>
#
# Examples:
#   bash record_one_shot.sh -- \
#     --dataset.repo_id=me/smoke_v0 --dataset.num_episodes=3 \
#     --dataset.episode_time_s=5 --dataset.single_task="smoke"
#
#   bash record_one_shot.sh --real -- \
#     --mode=real --dataset.repo_id=me/real_smoke_v0 \
#     --dataset.num_episodes=3 --dataset.single_task="real smoke"
#
# Opinionated: this script does NOT try to do anything clever about which
# cameras/servos to use. Override camera indices or serial ports via the
# usual env vars (CAMERA_ID, LEADER_PORT, FOLLOWER_PORT) documented below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

MODE=sim
DRIVE=1
CAMERA_ID="${CAMERA_ID:-0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"

# Parse our own flags; everything after `--` goes to record.sh.
PASSTHROUGH=()
while (( $# )); do
  case "$1" in
    --real)      MODE=real; shift ;;
    --no-drive)  DRIVE=0; shift ;;
    --)          shift; PASSTHROUGH=("$@"); break ;;
    -h|--help)
      sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)           PASSTHROUGH+=("$1"); shift ;;
  esac
done

# Track PIDs so the trap can SIGINT them even if one fails to start.
CHILDREN=()
OWN_STACK=0

cleanup() {
  local rc=$?
  echo ""
  echo "== record_one_shot: cleanup (exit=$rc) =="
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -SIGINT "$pid" 2>/dev/null || true
    fi
  done
  # Give children a chance to shut down cleanly, then force any stragglers.
  sleep 1
  for pid in "${CHILDREN[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  # Stack stays up on purpose — `bash stack_stop.sh` to take it down.
  if (( OWN_STACK )); then
    echo "   note: sim stack was started by us and left running — "
    echo "         bash $SCRIPT_DIR/stack_stop.sh to stop it."
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

export PATH="$HOME/.pixi/bin:${PATH:-}"

# --- 1. Stack (sim mode only) ---
if [ "$MODE" = "sim" ]; then
  if ! pgrep -f "ros2 launch.*gazebo.launch.py" >/dev/null 2>&1; then
    echo "== record_one_shot: starting sim stack headless =="
    bash "$SCRIPT_DIR/stack_start.sh" headless
    OWN_STACK=1
    # Wait for /top_camera to appear (proxy for "stack ready").
    for _ in $(seq 1 60); do
      if pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
          source '$SOARM_WS_SETUP' 2>/dev/null
          ros2 topic list --no-daemon 2>/dev/null" | grep -q "/top_camera"; then
        break
      fi
      sleep 3
    done
  else
    echo "== record_one_shot: sim stack already running =="
  fi
fi

# --- 2. Camera publisher ---
echo "== record_one_shot: starting camera_publisher (cam_id=$CAMERA_ID) =="
pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  source '$SOARM_WS_SETUP' 2>/dev/null
  ros2 run so_arm101_bringup camera_publisher \
    --camera-id $CAMERA_ID --publish-topic /wrist_camera" \
  > /tmp/rec_cam.log 2>&1 &
CHILDREN+=("$!")
sleep 2

# --- 3. Drive script (sim + when requested) ---
if [ "$MODE" = "sim" ] && (( DRIVE )); then
  echo "== record_one_shot: starting drive_joint_commands =="
  pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
    source '$SOARM_WS_SETUP' 2>/dev/null
    python '$SCRIPT_DIR/drive_joint_commands.py'" \
    > /tmp/rec_drive.log 2>&1 &
  CHILDREN+=("$!")
  sleep 1
fi

# --- 4. Jointstatereader (real mode) ---
if [ "$MODE" = "real" ]; then
  echo "== record_one_shot: starting jointstatereader (leader=$LEADER_PORT follower=$FOLLOWER_PORT) =="
  pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
    source '$SOARM_WS_SETUP' 2>/dev/null
    ros2 run jointstatereader joint_state_reader --ros-args \
      -p leader_port:=$LEADER_PORT -p follower_port:=$FOLLOWER_PORT \
      -p publish_source:=follower -p publish_commands:=true \
      -p mirror_to_follower:=true" \
    > /tmp/rec_jsr.log 2>&1 &
  CHILDREN+=("$!")
  sleep 2
fi

# --- 5. Record ---
echo "== record_one_shot: running record.sh =="
bash "$SCRIPT_DIR/record.sh" "${PASSTHROUGH[@]}"

# (cleanup trap handles teardown)
