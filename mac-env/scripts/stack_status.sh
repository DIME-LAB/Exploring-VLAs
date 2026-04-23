#!/usr/bin/env bash
# stack_status.sh — show every SO-ARM101 / Gazebo / smoke process running.
# Exit 0 = clean, >0 = that many processes still up.
#
# Usage:  bash mac-env/scripts/stack_status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

ps aux | \
  grep -iE "ros2 launch|gz sim|parameter_bridge|move_group|robot_state_publisher|control_gui|rviz2|spawner|smoke_l|smoke_p" | \
  grep -v grep | \
  awk '{printf "%-7s %-8s %s %s %s\n", $2, $10, $11, $12, $13}'

echo ""
echo "total: $(stack_running_count)"
exit "$(stack_running_count)"
