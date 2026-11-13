#!/bin/bash
# stack_start.sh — bring up the Linux sim stack (no policy):
#   1. Isaac Sim with the soarm101-dt extension + omni-kit-mcp bridge (:8767)
#   2. soarm101.quick_start scene build (via the bridge)
#   3. ROS2 control stack (control_gui + MoveIt + RViz + ros2_control)
#
# Idempotent: if anything is already running, skips that step. For the full
# deployment (this + the SmolVLA policy, optional WebRTC streaming) use
# deploy_sim.sh, which runs the same steps from _lib.sh.
#
# Usage:
#   stack_start.sh                     # local Isaac window, RViz on
#   stack_start.sh no-rviz             # control stack without RViz
#   stack_start.sh [no-rviz] --stream  # headless Isaac + WebRTC livestream
#
# Logs:
#   Isaac Sim:  /tmp/isaacsim-soarm101.log (ISAACSIM_LOG)
#   Control:    /tmp/control_stack_<HHMMSS>.log
#
# Tear-down: bash stack_stop.sh

set -e
# NOTE: no `set -u` — ROS2's setup.bash references unbound AMENT/COLCON vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

RVIZ_FLAG="true"
ISAAC_ARGS=()
for arg in "$@"; do
    case "$arg" in
        no-rviz)  RVIZ_FLAG="false" ;;
        --stream) ISAAC_ARGS+=(--stream) ;;
        *) echo "Unknown argument: $arg (expected: no-rviz, --stream)" >&2; exit 2 ;;
    esac
done

stack_preflight || exit 1
stack_source_ros

echo "=== Linux sim stack: starting ==="

echo "[1/3] Isaac Sim ($ISAAC_EXT_ID, bridge $KIT_MCP_HOST:$KIT_MCP_PORT)..."
isaac_up "${ISAAC_ARGS[@]}"

echo "[2/3] Scene (soarm101.quick_start: scene + robot + action graphs + publishers)..."
scene_up

echo "[3/3] ROS2 control stack (rviz=$RVIZ_FLAG)..."
control_stack_up "$RVIZ_FLAG" true   # stack_start always opens control_gui

echo ""
echo "=== Stack ready ==="
echo "  Verify:    bash $SCRIPT_DIR/stack_status.sh"
echo "  Tear down: bash $SCRIPT_DIR/stack_stop.sh"
