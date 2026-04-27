#!/usr/bin/env bash
# Phase 2 runtime verification: launch SO-ARM101 Gazebo headless, then confirm
# the three required topics are publishing:
#   /wrist_camera  /top_camera  /joint_states
# Plus /clock, /camera_info, /top_camera/camera_info for context.
#
# Prereqs:
#   - mac-env pixi env installed (`pixi install` under mac-env/)
#   - SO-ARM101 packages colcon-built into mac-env/ws/
#
# Usage:
#   pixi run bash scripts/verify_sim_topics.sh

# Note: do NOT 'set -u'; colcon's install/setup.bash references many
# unset variables (COLCON_TRACE, AMENT_PREFIX_PATH_MODIFIED_BY_COLCON, ...).

# Runtime env
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/cyclonedds.xml"
export KMP_DUPLICATE_LIB_OK=TRUE
# colcon-generated local_setup.sh line 105 doesn't quote $_ament_python_executable,
# so bash splits it on the space in our pixi env path. Workaround: set
# AMENT_PYTHON_EXECUTABLE to a spaceless symlink (created once by this script).
if [ ! -L /tmp/mac-env-python ]; then
  ln -sf "$PWD/.pixi/envs/default/bin/python" /tmp/mac-env-python
fi
export AMENT_PYTHON_EXECUTABLE=/tmp/mac-env-python

# colcon workspace lives at /tmp/soarm-ws because colcon install scripts can't
# handle spaces in paths. src/ symlinks into Exploring-VLAs/vla_SO-ARM101/src/.
WS="/tmp/soarm-ws"
LAUNCH_LOG="/tmp/soarm_gazebo.log"
TOPIC_LOG="/tmp/soarm_topics.log"

echo "=== 1. source colcon workspace ==="
if [ ! -f "$WS/install/setup.bash" ]; then
  echo "ERROR: $WS/install/setup.bash missing. Run colcon build first:"
  echo "  cd $WS && pixi run colcon build --symlink-install"
  exit 1
fi
source "$WS/install/setup.bash"

echo "=== 2. launch gazebo in background (headless, no RViz) ==="
ros2 launch so_arm101_control gazebo.launch.py headless:=true rviz:=false \
  > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
echo "launch pid: $LAUNCH_PID  (log: $LAUNCH_LOG)"

# Wait for startup: Gazebo + bridge + controllers spawn sequentially
echo "=== 3. wait 45s for sim to come up (Gazebo + spawn + bridge warmup) ==="
sleep 45

echo "=== 4. ros2 topic list (--no-daemon — macOS daemon hangs) ==="
ros2 topic list --no-daemon 2>&1 | sort | tee "$TOPIC_LOG"

echo
echo "=== 5. required-topic check ==="
for t in /wrist_camera /top_camera /joint_states /clock; do
  if grep -qx "$t" "$TOPIC_LOG"; then
    echo "  [PASS] $t"
  else
    echo "  [MISS] $t"
  fi
done

echo
echo "=== 6. brief rate check (3s each) ==="
for t in /wrist_camera /top_camera /joint_states; do
  echo "--- $t ---"
  timeout 4 ros2 topic hz "$t" --no-daemon 2>&1 | head -3 || true
done

echo
echo "=== 7. cleanup ==="
kill -SIGINT "$LAUNCH_PID" 2>/dev/null
sleep 3
kill -9 "$LAUNCH_PID" 2>/dev/null
pkill -SIGINT -f "ros2.*launch" 2>/dev/null
pkill -SIGINT -f "gz sim" 2>/dev/null
echo "done. Launch log: $LAUNCH_LOG"
