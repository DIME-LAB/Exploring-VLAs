# _lib.sh — shared constants for stack_* scripts.
#
# RoboStack conda packages have shebangs and shell scripts that break when the
# env lives at a path with spaces. Since our repo lives under `.../untitled
# folder/...` we can't host the pixi env in-tree. The env lives at /tmp/mac-env
# (installed once via bootstrap.sh), and the colcon workspace at /tmp/soarm-ws.
#
# Scripts source this file to keep paths consistent.

# Source of truth for the Mac pixi environment (spaceless clone of mac-env/).
export MAC_ENV_DIR=/tmp/mac-env
export MAC_ENV_MANIFEST="$MAC_ENV_DIR/pixi.toml"
export MAC_ENV_PYTHON="$MAC_ENV_DIR/.pixi/envs/default/bin/python"

# Colcon workspace where SO-ARM101 packages are built.
export SOARM_WS=/tmp/soarm-ws
export SOARM_WS_SETUP="$SOARM_WS/install/setup.bash"

# Runtime env for anything that touches ROS 2 on macOS.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$MAC_ENV_DIR/cyclonedds.xml"
export KMP_DUPLICATE_LIB_OK=TRUE
export AMENT_PYTHON_EXECUTABLE="$MAC_ENV_PYTHON"

# Pidfile + log for the running stack (for start/stop coordination).
export STACK_PIDFILE=/tmp/soarm_stack.pid
export STACK_LOG=/tmp/soarm_stack.log

# Preflight: ensure /tmp/mac-env and /tmp/soarm-ws exist.
stack_preflight() {
  if [ ! -f "$MAC_ENV_MANIFEST" ]; then
    echo "ERROR: $MAC_ENV_MANIFEST missing. Run bootstrap.sh once:" >&2
    echo "       bash mac-env/scripts/bootstrap.sh" >&2
    return 1
  fi
  if [ ! -f "$SOARM_WS_SETUP" ]; then
    echo "ERROR: $SOARM_WS_SETUP missing. Run bootstrap.sh once:" >&2
    echo "       bash mac-env/scripts/bootstrap.sh" >&2
    return 1
  fi
  return 0
}

# Count running SO-ARM101 / Gazebo / smoke-test processes.
stack_running_count() {
  ps aux | \
    grep -iE "ros2 launch|gz sim|parameter_bridge|move_group|robot_state_publisher|control_gui|smoke_l|smoke_p" | \
    grep -v grep | wc -l | tr -d ' '
}
