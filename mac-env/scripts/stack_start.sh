#!/usr/bin/env bash
# stack_start.sh — launch the SO-ARM101 Gazebo + control stack (ONE instance).
#
# Usage:
#   bash mac-env/scripts/stack_start.sh [MODE]
#
# Modes (pick based on what you need to look at):
#   headless   default. No windows. Server-only. Best for recording / CI.
#              aliases: record, none
#   gz         Gazebo GUI only (see the sim scene; no RViz).
#              aliases: gazebo, sim
#   rviz       RViz only (see camera feeds, TF, planning; Gazebo server headless).
#   all        Gazebo GUI + RViz (full visual inspection).
#              aliases: full, visual, inspect
#   help       Show this help.
#
# No mode = headless. Visuals are opt-in — recording and automation don't need
# them and benefit from the extra CPU.
#
# Runs detached. View logs at /tmp/soarm_stack.log.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

stack_preflight

MODE="${1:-headless}"

case "$MODE" in
  headless|record|none|'')
    HEADLESS="true";  RVIZ_ARG="false"; MODE_PRETTY="headless" ;;
  gz|gazebo|sim)
    HEADLESS="false"; RVIZ_ARG="false"; MODE_PRETTY="gz (Gazebo GUI)" ;;
  rviz)
    HEADLESS="true";  RVIZ_ARG="true";  MODE_PRETTY="rviz (RViz only, Gazebo headless)" ;;
  all|full|visual|inspect)
    HEADLESS="false"; RVIZ_ARG="true";  MODE_PRETTY="all (Gazebo GUI + RViz)" ;;
  help|-h|--help)
    sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *)
    echo "ERROR: unknown mode '$MODE'. Use: headless | gz | rviz | all | help" >&2
    exit 1 ;;
esac

# Refuse to start if anything is already running.
N=$(stack_running_count)
if [ "$N" -gt 0 ]; then
  echo "ERROR: $N existing stack processes running. Stop them first:" >&2
  echo "       bash $SCRIPT_DIR/stack_stop.sh" >&2
  exit 1
fi

echo "== Starting SO-ARM101 stack: mode=$MODE_PRETTY =="

export PATH="$HOME/.pixi/bin:$PATH"
: > "$STACK_LOG"

# Launch detached. pixi run handles env activation; sourcing the ws setup
# inside the subshell extends AMENT_PREFIX_PATH with our built packages.
nohup pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  set -e
  source '$SOARM_WS_SETUP' 2>/dev/null
  export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
  export CYCLONEDDS_URI='$CYCLONEDDS_URI'
  export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
  export AMENT_PYTHON_EXECUTABLE='$AMENT_PYTHON_EXECUTABLE'
  exec ros2 launch so_arm101_control gazebo.launch.py headless:=$HEADLESS rviz:=$RVIZ_ARG
" > "$STACK_LOG" 2>&1 &

LPID=$!
echo "$LPID" > "$STACK_PIDFILE"
echo "launch pid: $LPID  log: $STACK_LOG  pidfile: $STACK_PIDFILE"

sleep 2
CHILDREN=$(pgrep -P "$LPID" | wc -l | tr -d ' ')
echo "launch has $CHILDREN direct children after 2s"

case "$MODE_PRETTY" in
  gz*|all*)
    echo "  Gazebo GUI should appear as a macOS window within 10-30s." ;;
esac
case "$MODE_PRETTY" in
  rviz*|all*)
    echo "  RViz window will appear after controllers spawn (~30-40s)." ;;
esac
echo "  Control GUI (tkinter) appears in all modes after controllers spawn."

echo ""
echo "Tail:    tail -f $STACK_LOG"
echo "Status:  bash $SCRIPT_DIR/stack_status.sh"
echo "Stop:    bash $SCRIPT_DIR/stack_stop.sh"
