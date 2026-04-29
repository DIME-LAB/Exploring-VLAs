#!/bin/bash
# stack_stop.sh — graceful tear-down of the Linux sim stack.
#
# Order matters: tear down the consumer (control stack) first, then Isaac Sim.
# Use SIGINT (not SIGKILL) on processes that own X11 windows to avoid KWin
# BadWindow cascades — see top-level CLAUDE.md "Linux gotchas".

# NOTE: no `set -u` — pkill exits non-zero when nothing matches; we treat
# that as "good, already stopped" rather than as a script error.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

echo "=== Linux sim stack: stopping ==="

# ---------------------------------------------------------------------------
# 1. ROS2 control stack (SIGINT propagates to children; SIGTERM does not)
# ---------------------------------------------------------------------------

echo "[1/2] ROS2 control stack..."
if pgrep -f "ros2.*launch.*control.launch" >/dev/null; then
    pkill -SIGINT -f "ros2.*launch.*control.launch" 2>/dev/null || true
    # Wait up to 10s for clean shutdown
    for i in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -f "ros2.*launch.*control.launch" >/dev/null || { echo "  Stopped (${i}s)"; break; }
        sleep 1
    done
    if pgrep -f "ros2.*launch.*control.launch" >/dev/null; then
        echo "  Still running after 10s — sending SIGTERM"
        pkill -SIGTERM -f "ros2.*launch.*control.launch" 2>/dev/null || true
    fi
else
    echo "  Not running"
fi

# Mirror node (if started by Record Sim tab and not auto-cleaned)
if pgrep -f "joint_states_to_commands" >/dev/null; then
    echo "  Stopping mirror node..."
    pkill -SIGINT -f "joint_states_to_commands" 2>/dev/null || true
fi

rm -f "$STACK_PIDFILE"

# ---------------------------------------------------------------------------
# 2. Isaac Sim (handles its own SIGTERM-then-SIGKILL escalation)
# ---------------------------------------------------------------------------

echo "[2/2] Isaac Sim..."
bash "$ISAAC_LAUNCHER" close

echo ""
echo "=== Stopped ==="
