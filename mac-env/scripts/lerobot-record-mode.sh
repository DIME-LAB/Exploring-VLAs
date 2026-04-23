#!/usr/bin/env bash
# lerobot-record-mode.sh — thin --mode sim|real alias for `lerobot-record`.
#
# Usage:
#   lerobot-record-mode.sh --mode sim  [... any lerobot-record flags ...]
#   lerobot-record-mode.sh --mode real [... any lerobot-record flags ...]
#
# --mode sim   -> prepends --robot.type=so101_ros2    --teleop.type=so101_ros2
# --mode real  -> prepends --robot.type=so101_follower --teleop.type=so101_leader
#
# All other flags pass through unchanged. This wrapper exists so the record
# CLI stays platform-agnostic: one invocation style works across sim and real.
#
# Rationale for a shell script (not a lerobot-native alias): draccus config
# type dispatch is positional-by-flag; there's no first-class notion of
# "named mode presets" in lerobot yet. 20 lines of bash is clearer than
# adding an opinionated CLI layer upstream. Revisit if this shim accumulates
# more than mode mapping.
set -euo pipefail

MODE=""
PASSTHROUGH=()

while (( $# )); do
  case "$1" in
    --mode=*)    MODE="${1#--mode=}"; shift ;;
    --mode)      MODE="${2:-}"; shift 2 ;;
    *)           PASSTHROUGH+=("$1"); shift ;;
  esac
done

case "$MODE" in
  sim)
    TYPES=(--robot.type=so101_ros2 --teleop.type=so101_ros2)
    ;;
  real)
    TYPES=(--robot.type=so101_follower --teleop.type=so101_leader)
    ;;
  "")
    cat >&2 <<USAGE
usage: lerobot-record-mode.sh --mode sim|real [lerobot-record flags ...]

  sim:  --robot.type=so101_ros2    --teleop.type=so101_ros2
  real: --robot.type=so101_follower --teleop.type=so101_leader
USAGE
    exit 1
    ;;
  *)
    echo "lerobot-record-mode.sh: unknown mode '$MODE' (expected sim|real)" >&2
    exit 1
    ;;
esac

exec lerobot-record "${TYPES[@]}" "${PASSTHROUGH[@]}"
