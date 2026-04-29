#!/bin/bash
# stack_start.sh — bring up the full Linux sim stack:
#   1. Isaac Sim (with soarm101-dt extension, MCP socket on 8767)
#   2. quick_start scene build (via MCP)
#   3. ROS2 control stack (control_gui + MoveIt + RViz + ros2_control)
#
# Idempotent: if anything is already running, skips that step.
#
# Usage:
#   stack_start.sh             # default extension (soarm101-dt), RViz on
#   stack_start.sh no-rviz     # control stack without RViz (lighter for headless)
#
# Logs:
#   Isaac Sim:  /tmp/isaacsim.log
#   Control:    /tmp/control_stack_<HHMMSS>.log
#
# Tear-down: bash stack_stop.sh

set -e
# NOTE: no `set -u` — ROS2's setup.bash references unbound AMENT/COLCON vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

stack_preflight || exit 1
stack_source_ros

RVIZ_FLAG="true"
[ "${1:-}" = "no-rviz" ] && RVIZ_FLAG="false"

echo "=== Linux sim stack: starting ==="

# ---------------------------------------------------------------------------
# 1. Isaac Sim
# ---------------------------------------------------------------------------

echo "[1/3] Isaac Sim ($ISAAC_EXT_ID, port $MCP_PORT)..."
if ss -tlnp 2>/dev/null | grep -q ":$MCP_PORT"; then
    echo "  Already listening on $MCP_PORT — skipping launch"
else
    bash "$ISAAC_LAUNCHER" launch "$ISAAC_EXT_ID"
fi

# ---------------------------------------------------------------------------
# 2. quick_start (build scene + spawn publishers)
# ---------------------------------------------------------------------------

echo "[2/3] quick_start (scene + robot + action graphs + publishers)..."
python3 - <<PY
import socket, json
s = socket.socket()
s.settimeout(180)
s.connect(("$MCP_HOST", $MCP_PORT))
s.sendall(json.dumps({"type":"quick_start","params":{}}).encode())
data = b""
while True:
    chunk = s.recv(4096)
    if not chunk: break
    data += chunk
    try:
        r = json.loads(data.decode())
        print(f"  {r.get('result', r)}")
        break
    except json.JSONDecodeError:
        continue
s.close()
PY

# Wait for /clock (sim is actually ticking)
echo "  Waiting for /clock to tick..."
for i in 1 2 3 4 5 6 7 8 9 10; do
    if timeout 2 ros2 topic echo /clock --once 2>/dev/null | grep -q "sec:"; then
        echo "  /clock alive"
        break
    fi
    sleep 3
    if [ "$i" -eq 10 ]; then
        echo "  WARNING: /clock not ticking after 30s — sim may not be playing"
    fi
done

# ---------------------------------------------------------------------------
# 3. ROS2 control stack
# ---------------------------------------------------------------------------

echo "[3/3] ROS2 control stack (rviz=$RVIZ_FLAG)..."
if ros2 node list 2>/dev/null | grep -q so_arm101_control_gui; then
    echo "  control_gui already running — skipping"
else
    LOG="$LOG_DIR/control_stack_$(date +%H%M%S).log"
    nohup ros2 launch so_arm101_control control.launch.py rviz:="$RVIZ_FLAG" \
        > "$LOG" 2>&1 &
    echo "$!" > "$STACK_PIDFILE"
    disown
    echo "  Log: $LOG"
    # Block until control_gui registers
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        if ros2 node list 2>/dev/null | grep -q so_arm101_control_gui; then
            echo "  control_gui up after ~$((i*3))s"
            break
        fi
        sleep 3
        if [ "$i" -eq 15 ]; then
            echo "  TIMEOUT 45s waiting for control_gui — check $LOG"
            tail -20 "$LOG"
            exit 1
        fi
    done
fi

echo ""
echo "=== Stack ready ==="
echo "  Verify:    bash $SCRIPT_DIR/stack_status.sh"
echo "  Tear down: bash $SCRIPT_DIR/stack_stop.sh"
