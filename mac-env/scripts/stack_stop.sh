#!/usr/bin/env bash
# stack_stop.sh — SIGINT the launch, then SIGKILL stragglers by exact name.
# SIGINT should propagate via the process group (per project CLAUDE.md KILL
# RULE); SIGKILL only targets specific known binaries — never broad wildcards.
#
# Usage:  bash mac-env/scripts/stack_stop.sh

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

echo "== 1. SIGINT the launch (propagates to child processes) =="
pkill -SIGINT -f "ros2 launch" 2>/dev/null
sleep 3

echo "== 2. kill stragglers by exact process name =="
for pat in \
  "gz sim" \
  "parameter_bridge" \
  "move_group" \
  "robot_state_publisher" \
  "control_gui" \
  "spawner.py" \
  "spawner " \
  "ros2 launch" \
  "smoke_publisher.py" \
  "smoke_l1_camera.py" \
  "smoke_l2_dataset.py"
do
  n=$(pgrep -f "$pat" | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    echo "  -> $n x $pat"
    pkill -9 -f "$pat" 2>/dev/null
  fi
done

sleep 1
rm -f "$STACK_PIDFILE"

REMAIN=$(stack_running_count)
if [ "$REMAIN" -eq 0 ]; then
  echo "== [CLEAN] no SO-ARM101 / Gazebo / smoke processes running =="
else
  echo "== [WARN] $REMAIN processes still running =="
  ps aux | grep -iE "ros2 launch|gz sim|parameter_bridge|move_group|robot_state_publisher|control_gui|smoke_l|smoke_p" | \
    grep -v grep | awk '{print "    pid=" $2, $11, $12}'
fi
