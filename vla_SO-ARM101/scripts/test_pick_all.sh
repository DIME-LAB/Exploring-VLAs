#!/usr/bin/env bash
# Pick-and-place every available (not-in-cup) lego sequentially. For each
# lego: run sim_reset, run one QS cycle, capture verdict.
#
# Usage:
#   scripts/test_pick_all.sh [out_dir]
#
# Out dir contains per-lego logs and a summary.csv with pass/fail and
# planner_used breakdown.

set -euo pipefail

OUT_DIR="${1:-/tmp/pick_all_$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"

NODE=/so_arm101_control_gui
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "lego,verdict,elapsed_s,tier1_n,tier2_n,ompl_n,refused_n,halt_step,halt_reason" > "$SUMMARY"

log() { printf '[pick-all %(%H:%M:%S)T] %s\n' -1 "$*"; }

# Discover available (not-in-cup) legos. update_cups + sort_into_cups before
# this would clear cup-bound ones; for the test we just enumerate all then
# the GUI's own _add_lego_collision_objects skip-list filters in-cup ones.
log "discovering legos from /objects_poses_sim"
LEGOS=$(ros2 topic echo /objects_poses_sim --once 2>/dev/null \
    | sed -n 's/  child_frame_id: \(.*\)/\1/p')
log "found legos: $(echo $LEGOS | tr '\n' ' ')"

CYCLE_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0
for LEGO in $LEGOS; do
    CYCLE_COUNT=$((CYCLE_COUNT + 1))
    log "--- cycle $CYCLE_COUNT: $LEGO ---"
    LOG_FILE="$OUT_DIR/${LEGO}.log"

    log "  resetting sim"
    "$SCRIPTS_DIR/sim_reset.sh" >> "$LOG_FILE" 2>&1 || true

    log "  running QS for $LEGO"
    T0=$(date +%s)
    if "$SCRIPTS_DIR/test_qs_cycle.sh" "$LEGO" "$OUT_DIR/${LEGO}.qslog" >> "$LOG_FILE" 2>&1; then
        ELAPSED=$(($(date +%s) - T0))
    else
        ELAPSED=$(($(date +%s) - T0))
    fi

    # Parse verdict from per-cycle log file
    QSLOG="$OUT_DIR/${LEGO}.qslog"
    if grep -q "pick-and-drop cycle complete" "$QSLOG" 2>/dev/null; then
        VERDICT=PASS
        PASS_COUNT=$((PASS_COUNT + 1))
        HALT_STEP=""
        HALT_REASON=""
    else
        VERDICT=FAIL
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HALT_STEP=$(grep "Quickstart halted at:" "$QSLOG" 2>/dev/null | tail -1 | sed 's/.*halted at: //')
        HALT_REASON=$(grep "Quickstart: step failed" "$QSLOG" 2>/dev/null | tail -1 | sed 's/.*step failed (\([^)]*\)).*/\1/')
    fi

    # Use awk: grep -c prints "0" AND exits 1 on no matches, so `|| echo 0`
    # produced "0\n0" which broke CSV columns. awk always exits 0 with one line.
    TIER1=$(awk '/tier1 linear: [0-9]+ wps clean/{n++}END{print n+0}' "$QSLOG")
    TIER2=$(awk '/tier2 retract-pan-settle: [0-9]+ wps clean/{n++}END{print n+0}' "$QSLOG")
    OMPL=$(awk '/falling back to OMPL/{n++}END{print n+0}' "$QSLOG")
    REFUSED=$(awk '/^[[:space:]]*REFUSED:/{n++}END{print n+0}' "$QSLOG")

    echo "$LEGO,$VERDICT,$ELAPSED,$TIER1,$TIER2,$OMPL,$REFUSED,\"$HALT_STEP\",\"$HALT_REASON\"" >> "$SUMMARY"
    log "  $LEGO: $VERDICT (${ELAPSED}s) tier1=$TIER1 tier2=$TIER2 ompl=$OMPL refused=$REFUSED ${HALT_STEP:+halt=$HALT_STEP}"
done

log "=== summary: $PASS_COUNT pass / $FAIL_COUNT fail / $CYCLE_COUNT total ==="
column -t -s, "$SUMMARY"
log "logs: $OUT_DIR"
