#!/usr/bin/env bash
# Trigger one Quickstart pick-and-drop cycle via ROS2 services and report
# where it halted. Matches the manual flow: select a lego in the QS tab,
# press Play, wait for terminal status.
#
# Usage:
#   scripts/test_qs_cycle.sh [lego_name] [log_out_path]
#
# Defaults: lego=blue_2x4, log=/tmp/qs_cycle_test.log
#
# Terminal sentinels in get_log:
#   "Quickstart: pick-and-drop cycle complete"   → SUCCESS
#   "Quickstart halted at: <step>"               → HALT (QS runner saw failure)
#   "Quickstart aborted"                         → user-initiated abort

set -euo pipefail

LEGO="${1:-blue_2x4}"
LOG_FILE="${2:-/tmp/qs_cycle_test.log}"
TIMEOUT_S="${TIMEOUT_S:-180}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-2}"

NODE=/so_arm101_control_gui

echo "=== QS cycle test ==="
echo "lego target: $LEGO"
echo "log file:    $LOG_FILE"
echo "timeout:     ${TIMEOUT_S}s"
echo "started:     $(date -Iseconds)"
echo

# 1) Set ik_target so qs_select picks the right lego (Trigger service has no args).
ros2 param set "$NODE" ik_target "$LEGO" >/dev/null
echo "[1/4] set ik_target=$LEGO"

# 2) Refresh both listboxes. _cmd_qs_refresh_all fires grasp_refresh + drop_refresh
#    and schedules the local listbox repopulate via root.after(1200). Sleep longer
#    to cover the 2 s drop_refresh budget.
ros2 service call "$NODE/qs_refresh_all" std_srvs/srv/Trigger >/dev/null
sleep 2.5
echo "[2/4] qs_refresh_all done"

# 3) Select the lego (reads ik_target, falls through to first entry if not found).
ros2 service call "$NODE/qs_select" std_srvs/srv/Trigger >/dev/null
sleep 0.5
echo "[3/4] qs_select done"

# Snapshot current log length (# of lines) — we only poll for sentinels in
# NEW content after qs_play. Without this, a prior halted cycle's sentinel
# in the persistent log history would match immediately on the first poll.
PRE_LOG=$(mktemp)
ros2 service call "$NODE/get_log" std_srvs/srv/Trigger 2>/dev/null \
    | sed -n "s/.*message='\(.*\)')/\1/p" \
    | sed 's/\\n/\n/g' > "$PRE_LOG"
PRE_LINES=$(wc -l < "$PRE_LOG")

# 4) Press Play. Service returns immediately; the QS runner thread drives the cycle.
ros2 service call "$NODE/qs_play" std_srvs/srv/Trigger >/dev/null
echo "[4/4] qs_play triggered — polling get_log every ${POLL_INTERVAL_S}s"
echo "     (only matching sentinels in lines after #${PRE_LINES})"
echo

T0=$(date +%s)
while :; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - T0))
    if [ "$ELAPSED" -ge "$TIMEOUT_S" ]; then
        echo "=== TIMEOUT after ${ELAPSED}s — aborting QS ==="
        ros2 service call "$NODE/qs_restart" std_srvs/srv/Trigger >/dev/null || true
        break
    fi

    ros2 service call "$NODE/get_log" std_srvs/srv/Trigger 2>/dev/null \
        | sed -n "s/.*message='\(.*\)')/\1/p" \
        | sed 's/\\n/\n/g' \
        > "$LOG_FILE"

    # Only scan new lines (post-qs_play) for terminal sentinels.
    if tail -n +$((PRE_LINES + 1)) "$LOG_FILE" | \
         grep -qE "pick-and-drop cycle complete|Quickstart halted at|Quickstart aborted"; then
        echo "=== cycle terminated after ${ELAPSED}s ==="
        break
    fi
    sleep "$POLL_INTERVAL_S"
done

# Slice log to new content only. We overwrite LOG_FILE in place with the
# per-cycle slice so downstream consumers (e.g. test_pick_all.sh) see only
# this cycle's lines, not the full session-long get_log history.
NEW_LOG=$(mktemp)
tail -n +$((PRE_LINES + 1)) "$LOG_FILE" > "$NEW_LOG"
cp "$NEW_LOG" "$LOG_FILE"
rm -f "$PRE_LOG"

echo
echo "=== terminal sentinel (this cycle only) ==="
grep -nE "Quickstart: pick-and-drop cycle complete|Quickstart halted at|Quickstart aborted|Quickstart: step failed" "$NEW_LOG" | tail -3

echo
echo "=== planner_used breakdown (this cycle only) ==="
count() { grep -cE "$1" "$NEW_LOG" || echo 0; }
printf "  tier1 linear successes:       %s\n" "$(count 'tier1 linear: [0-9]+ wps clean')"
printf "  tier2 retract-pan-settle:     %s\n" "$(count 'tier2 retract-pan-settle: [0-9]+ wps clean')"
printf "  OMPL fallback fired:          %s\n" "$(count 'falling back to OMPL')"
printf "  REFUSED (allow_ompl=False):   %s\n" "$(count '^\s*REFUSED:')"
printf "  OMPL post-check pass:         %s\n" "$(count 'OMPL post-check.*clear')"
printf "  already-at-target noops:      %s\n" "$(count 'already at target')"

echo
echo "=== last 40 lines of this cycle ==="
tail -40 "$NEW_LOG"

echo
echo "full log:      $LOG_FILE"
echo "this cycle:    $NEW_LOG"
