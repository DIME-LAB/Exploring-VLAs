#!/usr/bin/env bash
# rviz_stop.sh — stop the RViz instance launched by rviz_attach.sh.
#
# SIGINT first (RViz cleans up its rcl context and exits), SIGKILL the
# leftover pid only if it's still alive after 2s. Mirrors stack_stop.sh's
# pattern: never broad pkill wildcards.
#
# Usage:  bash mac-env/scripts/rviz_stop.sh

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RVIZ_PIDFILE=/tmp/soarm_rviz.pid

if [ -f "$RVIZ_PIDFILE" ]; then
  PID=$(cat "$RVIZ_PIDFILE" 2>/dev/null)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "== SIGINT pid $PID =="
    kill -SIGINT "$PID" 2>/dev/null
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
      echo "  -> still alive, SIGKILL"
      kill -9 "$PID" 2>/dev/null
    fi
  fi
  rm -f "$RVIZ_PIDFILE"
fi

# Catch any rviz2 that wasn't started via this script.
n=$(pgrep -f "rviz2 -d.*moveit.rviz" | wc -l | tr -d ' ')
if [ "$n" -gt 0 ]; then
  echo "  -> $n stray rviz2 process(es), SIGKILL"
  pkill -9 -f "rviz2 -d.*moveit.rviz" 2>/dev/null
fi

if pgrep -f "rviz2 -d.*moveit.rviz" >/dev/null 2>&1; then
  echo "== [WARN] rviz2 still running =="
  ps aux | grep "rviz2" | grep -v grep
else
  echo "== [CLEAN] no RViz running =="
fi
