#!/bin/bash
# stack_stop.sh — graceful tear-down of the Linux sim stack.
#
# Order matters: tear down the consumer (control stack) first, then Isaac Sim.
# Use SIGINT (not SIGKILL) on processes that own X11 windows to avoid KWin
# BadWindow cascades — see top-level CLAUDE.md "Linux gotchas".

# Stop only process trees recorded by these launchers; legacy PID-only records refuse.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

echo "=== Linux sim stack: stopping ==="

# ---------------------------------------------------------------------------
# 1. ROS2 control stack (SIGINT propagates to children; SIGTERM does not)
# ---------------------------------------------------------------------------

echo "[1/2] ROS2 control stack..."
if [ -f "$STACK_PIDFILE" ]; then
    stop_pid "$(cat "$STACK_PIDFILE")" "control stack" 15 "$STACK_PIDFILE" || exit 1
else
    echo "  No launcher-owned control stack recorded; leaving other processes alone"
fi
# Record Sim owns its mirror lifecycle. No name-based cleanup is permitted.

# ---------------------------------------------------------------------------
# 2. Isaac Sim (handles its own SIGTERM-then-SIGKILL escalation)
# ---------------------------------------------------------------------------

echo "[2/2] Isaac Sim..."
bash "$ISAAC_LAUNCHER" close

echo ""
echo "=== Stopped ==="
