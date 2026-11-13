#!/usr/bin/env bash
# Reset the Isaac Sim scene + control_gui state to a known-clean starting
# point for a fresh QS cycle. Cups land at the same poses they have on a
# fresh quick_start (CUP_LAYOUT_REAL_WORLD overrides), regardless of any
# slider tweaks made during the session.
#
# Steps (in order):
#   1. Abort any running QS player                        → /qs_restart
#   2. Force-detach any held lego                          → /detach_lego
#   3. soarm101.match_real_world  (reset CUP_LAYOUT to the quick_start defaults)
#   4. soarm101.update_cups       (re-snap existing cup prims to those poses)
#   5. soarm101.randomize_object_poses  (scatter legos)
#   6. /qs_refresh_all  (resync MoveIt collision scene)
#   7. /grasp_home      (move arm to home pose)
#
# Usage:
#   scripts/sim_reset.sh
#
# Env: SOARM_KIT_MCP_HOST / SOARM_KIT_MCP_PORT (bridge, default 127.0.0.1:8767),
#      SOARM_KIT_MCP_DIR (omni-kit-mcp checkout, default <repo>/omni-kit-mcp).

set -euo pipefail

NODE=/so_arm101_control_gui
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Scene tools live on the omni-kit-mcp bridge as soarm101.<name>; the client is
# the stdlib-only kit_mcp.client from the omni-kit-mcp submodule.
KIT_MCP_DIR="${SOARM_KIT_MCP_DIR:-$REPO_ROOT/omni-kit-mcp}"
KIT_MCP_HOST="${SOARM_KIT_MCP_HOST:-127.0.0.1}"
KIT_MCP_PORT="${SOARM_KIT_MCP_PORT:-8767}"

# mcp_call <tool> [json-params] — prints the first 200 chars of the reply.
# Exits non-zero (aborting the reset under set -e) when the bridge is down
# (client exit 2) or the tool raised (client exit 1).
mcp_call() {
    local tool="$1" params="${2:-}" out
    [ -n "$params" ] || params='{}'
    out="$(PYTHONPATH="$KIT_MCP_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m kit_mcp.client \
        "soarm101.$tool" "$params" \
        --host "$KIT_MCP_HOST" --port "$KIT_MCP_PORT" --timeout 30 2>&1)" || {
        echo "  soarm101.$tool failed: ${out:0:300}" >&2
        return 1
    }
    echo "${out:0:200}"
}

log() { printf '[reset %(%H:%M:%S)T] %s\n' -1 "$*"; }

# Steps 1, 2, 6 and 7 are control_gui services; without the node `ros2 service
# call` waits forever. Check up front, before touching the scene.
if ! ros2 node list --no-daemon 2>/dev/null | grep -qx "$NODE"; then
    echo "sim_reset.sh needs control_gui ($NODE is not running): start with \`deploy_sim.sh up --gui\` (or stack_start.sh)" >&2
    exit 1
fi

log "1/7 abort any QS player"
ros2 service call "$NODE/qs_restart" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 0.5

log "2/7 detach any held lego"
ros2 service call "$NODE/detach_lego" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 1.5  # detach is async; wait for force-cleanup to settle

log "3/7 MCP match_real_world (reset CUP_LAYOUT to quick_start defaults)"
mcp_call match_real_world
sleep 0.5

log "4/7 MCP update_cups (re-snap cups to those poses)"
mcp_call update_cups
sleep 1

log "5/7 MCP randomize_object_poses (scatter legos)"
mcp_call randomize_object_poses
sleep 1.5

log "6/7 qs_refresh_all (resync MoveIt scene)"
ros2 service call "$NODE/qs_refresh_all" std_srvs/srv/Trigger >/dev/null
sleep 2.5  # _cmd_qs_refresh_all schedules listbox repop at +1.2s

log "7/7 grasp_home"
ros2 service call "$NODE/grasp_home" std_srvs/srv/Trigger >/dev/null 2>&1 || true
sleep 4

log "reset complete"
