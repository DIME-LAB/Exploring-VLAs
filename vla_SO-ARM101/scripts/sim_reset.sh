#!/usr/bin/env bash
# Reset the Isaac Sim scene + control_gui state to a known-clean starting
# point for a fresh QS cycle.
#
# Steps (in order):
#   1. Abort any running QS player                        → /qs_restart
#   2. Force-detach any held lego                          → /detach_lego
#   3. MCP update_cups   (resets cups to default poses)
#   4. MCP randomize_object_poses  (scatters legos)
#   5. /qs_refresh_all  (resync MoveIt collision scene)
#   6. /grasp_home      (move arm to home pose)
#
# Usage:
#   scripts/sim_reset.sh

set -euo pipefail

NODE=/so_arm101_control_gui
MCP_HOST=127.0.0.1
MCP_PORT=8767

mcp_call() {
    local tool="$1"
    python3 - "$tool" <<'PY'
import socket, json, sys
tool = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(30)
s.connect(('127.0.0.1', 8767))
s.sendall(json.dumps({'type': tool, 'params': {}}).encode() + b'\n')
buf = b''
while True:
    try:
        chunk = s.recv(8192)
        if not chunk: break
        buf += chunk
        if buf.endswith(b'}'): break
    except socket.timeout:
        break
print(buf.decode()[:200])
PY
}

log() { printf '[reset %(%H:%M:%S)T] %s\n' -1 "$*"; }

log "1/6 abort any QS player"
ros2 service call "$NODE/qs_restart" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 0.5

log "2/6 detach any held lego"
ros2 service call "$NODE/detach_lego" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 1.5  # detach is async; wait for force-cleanup to settle

log "3/6 MCP update_cups (reset cup poses)"
mcp_call update_cups
sleep 1

log "4/6 MCP randomize_object_poses (scatter legos)"
mcp_call randomize_object_poses
sleep 1.5

log "5/6 qs_refresh_all (resync MoveIt scene)"
ros2 service call "$NODE/qs_refresh_all" std_srvs/srv/Trigger >/dev/null
sleep 2.5  # _cmd_qs_refresh_all schedules listbox repop at +1.2s

log "6/6 grasp_home"
ros2 service call "$NODE/grasp_home" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 4

log "reset complete"
