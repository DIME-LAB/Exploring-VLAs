# _lib.sh — shared paths + env for the Linux stack.
#
# Sourced by stack_*.sh, bootstrap.sh, and any other linux-env script that needs
# canonical paths. Mirrors mac-env/scripts/_lib.sh but for the Isaac-Sim-backed
# Linux topology (system Humble producer side + pixi-Jazzy lerobot consumer).
#
# Resolves REPO_ROOT from this file's location, so colleagues cloning to
# arbitrary paths get correct values without editing anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LINUX_ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT="$(cd "$LINUX_ENV_DIR/.." && pwd)"

# Repo subtrees (some are submodules — bootstrap.sh inits them)
export VLA_PKG="$REPO_ROOT/vla_SO-ARM101"
export ISAAC_MCP="$REPO_ROOT/isaac-sim-mcp"
export LEROBOT_DIR="$REPO_ROOT/lerobot"

# System ROS2 (producer side: Isaac Sim, control_gui, MoveIt, mirror node)
export ROS2_DISTRO="${ROS2_DISTRO:-humble}"
export ROS2_SETUP="${ROS2_SETUP:-/opt/ros/$ROS2_DISTRO/setup.bash}"

# Colcon workspace lives in-tree on Linux (no spaceless-/tmp/ workaround needed —
# Linux paths are spaceless by convention, and in-tree keeps build artefacts
# tied to the source for symlink-install to resolve correctly).
export SOARM_WS="$VLA_PKG"
export SOARM_WS_SETUP="$SOARM_WS/install/setup.bash"

# Pixi env for the lerobot consumer side (Python 3.12 + RoboStack-Jazzy + lerobot deps).
# linux-env/pixi.toml is materialized in-tree under linux-env/.pixi/.
export PIXI_MANIFEST="$LINUX_ENV_DIR/pixi.toml"

# Cross-Python DDS discovery (multicast on lo — system default disables this,
# which prevents Jazzy consumers from seeing Humble producers across process trees).
# Force-set CYCLONEDDS_URI: any pre-existing value in the user's shell pointing
# at a different cyclonedds.xml will partition the discovery graph and make
# stack_status.sh report "node missing" even when nodes are healthy.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$LINUX_ENV_DIR/cyclonedds.xml"

# Isaac Sim / MCP
export ISAAC_LAUNCHER="$LINUX_ENV_DIR/scripts/isaac/isaacsim_launch.sh"
export MCP_HOST="${MCP_HOST:-localhost}"
export MCP_PORT="${MCP_PORT:-8767}"
export ISAAC_EXT_ID="${ISAAC_EXT_ID:-soarm101-dt}"

# Logs (single dir, prefixed names — easy to clean up with one rm)
export LOG_DIR="${LOG_DIR:-/tmp}"
export STACK_PIDFILE="${STACK_PIDFILE:-$LOG_DIR/soarm_linux_stack.pid}"

# ---------------------------------------------------------------------------
# Helpers used by stack_*.sh
# ---------------------------------------------------------------------------

# Verify the prerequisites a colleague needs before stack_start can succeed.
# Prints helpful errors pointing at bootstrap.sh when something is missing.
stack_preflight() {
    local missing=0
    if [ ! -f "$ROS2_SETUP" ]; then
        echo "ERROR: ROS2 setup not found at $ROS2_SETUP" >&2
        echo "       Install ros-$ROS2_DISTRO-desktop or set ROS2_SETUP env var." >&2
        missing=1
    fi
    if [ ! -d "$ISAAC_MCP/exts" ]; then
        echo "ERROR: isaac-sim-mcp submodule not initialized at $ISAAC_MCP" >&2
        echo "       Run from $REPO_ROOT: git submodule update --init --recursive" >&2
        missing=1
    fi
    if [ ! -f "$SOARM_WS_SETUP" ]; then
        echo "ERROR: ROS2 control workspace not built at $SOARM_WS_SETUP" >&2
        echo "       Run: bash $LINUX_ENV_DIR/scripts/bootstrap.sh" >&2
        missing=1
    fi
    if [ ! -x "$ISAAC_LAUNCHER" ]; then
        echo "ERROR: isaacsim_launch.sh not executable at $ISAAC_LAUNCHER" >&2
        echo "       chmod +x $ISAAC_LAUNCHER" >&2
        missing=1
    fi
    return $missing
}

# Source ROS2 + the colcon workspace into the current shell.
stack_source_ros() {
    # shellcheck disable=SC1090
    source "$ROS2_SETUP"
    # shellcheck disable=SC1090
    [ -f "$SOARM_WS_SETUP" ] && source "$SOARM_WS_SETUP"
}

# Count linux-stack processes (useful for stack_status / stop verification).
stack_running_count() {
    pgrep -f "ros2.*launch.*control.launch|move_group|control_gui|rviz2|isaacsim|joint_states_to_commands|robot_state_publisher" 2>/dev/null | wc -l | tr -d ' '
}
