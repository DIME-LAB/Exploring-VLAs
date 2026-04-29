#!/bin/bash
# stack_status.sh — print health of every layer of the Linux sim stack.
#
# Useful as a preflight before recording, or to debug "why isn't <thing> publishing".

# NOTE: no `set -u` — sourcing /opt/ros/humble/setup.bash trips on unbound
# AMENT/COLCON variables; we want this script to keep going regardless.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

stack_source_ros 2>/dev/null || true

echo "=== Linux sim stack status ==="
echo ""

# ---------------------------------------------------------------------------
# Isaac Sim
# ---------------------------------------------------------------------------

echo "[Isaac Sim]"
if pgrep -f "bin/isaacsim" >/dev/null; then
    echo "  Process: RUNNING"
else
    echo "  Process: not running"
fi
if ss -tlnp 2>/dev/null | grep -q ":$MCP_PORT "; then
    echo "  MCP socket :$MCP_PORT: LISTENING"
else
    echo "  MCP socket :$MCP_PORT: down"
fi
echo ""

# ---------------------------------------------------------------------------
# ROS2 control stack
# ---------------------------------------------------------------------------

echo "[Control stack]"
NODES="$(ros2 node list 2>/dev/null | sort)"
for n in /so_arm101_control_gui /move_group /controller_manager /robot_state_publisher /rviz; do
    if echo "$NODES" | grep -q "^$n\$"; then
        echo "  $n: alive"
    else
        echo "  $n: missing"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Topics (sim contract)
# ---------------------------------------------------------------------------

echo "[Sim topics]"
TOPICS="$(ros2 topic list 2>/dev/null | sort)"
for t in /clock /joint_states /joint_commands /drop_poses /objects_poses_sim /wrist_camera_rgb_sim /workspace_camera_sim; do
    if echo "$TOPICS" | grep -q "^$t\$"; then
        echo "  $t: present"
    else
        echo "  $t: missing"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# /clock actually ticking?
# ---------------------------------------------------------------------------

echo "[Sim time]"
CLOCK="$(timeout 2 ros2 topic echo /clock --once 2>/dev/null | grep "sec:" | head -1)"
if [ -n "$CLOCK" ]; then
    echo "  /clock ticking: $CLOCK"
else
    echo "  /clock not advancing (sim may be paused or down)"
fi
