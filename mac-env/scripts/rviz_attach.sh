#!/usr/bin/env bash
# rviz_attach.sh — launch RViz attached to an already-running headless stack.
#
# Use case: you started the stack with `stack_start.sh headless` (no GUI windows
# for fast boot / low CPU) and now want to inspect TF, planning scene, or topics
# in RViz without tearing down and re-launching.
#
# RViz reads /robot_description from move_group, /joint_states from
# controller_manager, /tf + /tf_static from robot_state_publisher — all of which
# the headless stack already publishes. So this is a pure attach: no new
# rosgraph nodes needed beyond rviz2 itself.
#
# Pidfile + log mirror the stack_start.sh pattern so a companion rviz_stop.sh
# can clean up without grepping ps.
#
# Usage:  bash mac-env/scripts/rviz_attach.sh
# Stop:   bash mac-env/scripts/rviz_stop.sh   (or pkill -SIGINT -f rviz2)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

stack_preflight

RVIZ_PIDFILE=/tmp/soarm_rviz.pid
RVIZ_LOG=/tmp/soarm_rviz.log
RVIZ_CONFIG="$SOARM_WS/install/so_arm101_moveit_config/share/so_arm101_moveit_config/config/moveit.rviz"

# Refuse to start if RViz is already running (one window is enough).
if [ -f "$RVIZ_PIDFILE" ] && kill -0 "$(cat "$RVIZ_PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "ERROR: RViz already running (pid $(cat "$RVIZ_PIDFILE")). Stop it first:" >&2
  echo "       bash $SCRIPT_DIR/rviz_stop.sh" >&2
  exit 1
fi
if pgrep -f "rviz2 -d.*moveit.rviz" >/dev/null 2>&1; then
  echo "ERROR: rviz2 is running but no pidfile exists. Kill it first:" >&2
  echo "       pkill -SIGINT -f rviz2" >&2
  exit 1
fi

# Confirm the stack is actually up — RViz against a dead stack just shows
# "Global Status: Error" and is more confusing than useful.
if ! pgrep -f "moveit_ros_move_group/move_group" >/dev/null 2>&1; then
  echo "ERROR: move_group not running. Start the stack first:" >&2
  echo "       bash $SCRIPT_DIR/stack_start.sh headless" >&2
  exit 1
fi
if [ ! -f "$RVIZ_CONFIG" ]; then
  echo "ERROR: RViz config missing at $RVIZ_CONFIG" >&2
  echo "       Did so_arm101_moveit_config build? Try:" >&2
  echo "         cd $SOARM_WS && pixi run --manifest-path $MAC_ENV_MANIFEST colcon build --packages-select so_arm101_moveit_config" >&2
  exit 1
fi

echo "== Launching RViz attached to running stack =="
echo "  config: $RVIZ_CONFIG"
echo "  log:    $RVIZ_LOG"

export PATH="$HOME/.pixi/bin:$PATH"
: > "$RVIZ_LOG"

# Use the project's own moveit_rviz.launch.py — it passes
# robot_description / robot_description_semantic / kinematics / planning_pipelines
# / joint_limits as RViz parameters via MoveItConfigsBuilder. Calling rviz2
# directly skips that wiring and leaves RViz unable to parse the SRDF
# (no other node in the stack publishes /robot_description_semantic), so the
# robot model never renders. The launch file even documents this in its
# comment block.
nohup pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  set -e
  source '$SOARM_WS_SETUP' 2>/dev/null
  export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
  export CYCLONEDDS_URI='$CYCLONEDDS_URI'
  export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
  exec ros2 launch so_arm101_moveit_config moveit_rviz.launch.py
" > "$RVIZ_LOG" 2>&1 &

PID=$!
echo "$PID" > "$RVIZ_PIDFILE"
echo "  pid: $PID"
echo ""
echo "  RViz window appears within ~5-10s as a macOS window."
echo ""
echo "Tail:    tail -f $RVIZ_LOG"
echo "Stop:    bash $SCRIPT_DIR/rviz_stop.sh"
