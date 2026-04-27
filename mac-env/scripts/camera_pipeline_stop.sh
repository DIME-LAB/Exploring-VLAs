#!/usr/bin/env bash
# camera_pipeline_stop.sh — kill the camera + aruco + yoloe pipeline.
#
# Mirrors stack_stop.sh's pattern: SIGINT the launch wrapper first (clean
# shutdown), then SIGKILL the per-binary children by exact name. The wrapper
# kill alone doesn't propagate to the python grandchildren on macOS — they
# get re-parented to launchd and survive — so the per-binary pass is required.
#
# Usage:
#   bash mac-env/scripts/camera_pipeline_stop.sh

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1. SIGINT ros2 launch wrapper =="
pkill -SIGINT -f "ros2 launch aruco_camera_localizer" 2>/dev/null
sleep 2

echo "== 2. SIGKILL per-binary stragglers =="
for pat in \
  "camera_publisher" \
  "localize_aruco" \
  "localize_yoloe"
do
  n=$(pgrep -f "$pat" | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    echo "  -> $n x $pat"
    pkill -9 -f "$pat" 2>/dev/null
  fi
done

sleep 1

REMAIN=$(pgrep -f "aruco_camera_localizer\|camera_publisher\|localize_aruco\|localize_yoloe" | wc -l | tr -d ' ')
if [ "$REMAIN" -eq 0 ]; then
  echo "== [CLEAN] no camera pipeline processes running =="
else
  echo "== [WARN] $REMAIN processes still running =="
  ps aux | grep -E "aruco_camera_localizer|camera_publisher|localize_aruco|localize_yoloe" | grep -v grep | awk '{print "    pid=" $2, $11, $12}'
fi
