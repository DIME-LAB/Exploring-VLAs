#!/bin/bash
# rec_progress.sh — print a concise progress report for a Record Sim run.
#
# Usage:
#   bash linux-env/scripts/rec_progress.sh                    # auto-detect current run
#   bash linux-env/scripts/rec_progress.sh rec_215624         # specific dataset
#   watch -n 30 bash linux-env/scripts/rec_progress.sh        # poll every 30s
#
# Reports: saved/target episodes, retries/discards (scoped to THIS run only),
# lerobot health, ROS topic health, FOV-constraint hits, current rate, ETA.

DS_NAME="${1:-}"
if [ -z "$DS_NAME" ] && [ -f /tmp/current_rec.txt ]; then
    DS_NAME="$(cat /tmp/current_rec.txt)"
fi
if [ -z "$DS_NAME" ]; then
    echo "Usage: $0 <dataset_name>     # e.g. rec_215624"
    echo "       (or write the name to /tmp/current_rec.txt)"
    exit 1
fi

DATASET="$HOME/.cache/huggingface/lerobot/local/$DS_NAME"
LEROBOT_LOG="/tmp/$DS_NAME.log"

# Locate the control_stack log that contains this run's events.
CTRL_LOG=""
for f in $(ls -t /tmp/control_stack_*.log 2>/dev/null); do
    grep -q "Record: starting → local/$DS_NAME" "$f" 2>/dev/null && { CTRL_LOG="$f"; break; }
done

# Scope to lines AFTER "Record: starting → local/<DS_NAME>" so we don't pick
# up retries/discards from earlier runs that share the same control_stack log.
scoped() {
    [ -n "$CTRL_LOG" ] || return
    awk -v ds="$DS_NAME" '
        $0 ~ "Record: starting → local/"ds {p=1}
        p {print}
    ' "$CTRL_LOG"
}

echo "=================================================="
echo "  Record Sim progress — $DS_NAME"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# 1. Dataset on disk
EP=0
FRAMES=0
if [ -f "$DATASET/meta/info.json" ]; then
    EP=$(python3 -c "import json; print(json.load(open('$DATASET/meta/info.json')).get('total_episodes',0))" 2>/dev/null || echo 0)
    FRAMES=$(python3 -c "import json; print(json.load(open('$DATASET/meta/info.json')).get('total_frames',0))" 2>/dev/null || echo 0)
fi

# Target episodes — read from the spawned cmd if we can grab it
TARGET=$(ps -ef | grep -oE "num_episodes=[0-9]+" | head -1 | grep -oE "[0-9]+")
[ -z "$TARGET" ] && TARGET="?"

echo ""
echo "[Dataset]"
echo "  saved : $EP / $TARGET episodes  ($FRAMES frames)"
echo "  path  : $DATASET"

# 2. Process health
echo ""
echo "[Processes]"
LEROBOT_PID=$(pgrep -f "lerobot-record.*$DS_NAME" | tail -1)
if [ -n "$LEROBOT_PID" ]; then
    LEROBOT_ETIME=$(ps -p "$LEROBOT_PID" -o etime= 2>/dev/null | tr -d ' ')
    echo "  lerobot       : alive (PID $LEROBOT_PID, up $LEROBOT_ETIME)"
else
    echo "  lerobot       : NOT RUNNING"
fi
MIRROR_PID=$(pgrep -f "joint_states_to_commands" | head -1)
if [ -n "$MIRROR_PID" ]; then
    echo "  mirror node   : alive (PID $MIRROR_PID)"
else
    echo "  mirror node   : not running"
fi
ISAAC_PID=$(pgrep -f "bin/isaacsim" | head -1)
SOCK_OK=$(ss -tlnp 2>/dev/null | grep -q ':8767' && echo "ok" || echo "DOWN")
echo "  Isaac Sim     : ${ISAAC_PID:+alive (PID $ISAAC_PID)} ${ISAAC_PID:-DOWN}, MCP socket :8767 $SOCK_OK"

# 3. Per-run save/retry stats (scoped — won't double-count across runs)
SAVED=$(scoped | grep -c "Record: ✓ episode" 2>/dev/null)
DISCARDED=$(scoped | grep -c "✗ episode discarded" 2>/dev/null)
RETRIES=$(scoped | grep -c "re-randomizing" 2>/dev/null)

echo ""
echo "[Orchestrator (this run only)]"
echo "  saves          : $SAVED"
echo "  discards       : $DISCARDED"
echo "  retries        : $RETRIES"

# 4. Recording rate / ETA
echo ""
echo "[Rate / ETA]"
if [ -n "$CTRL_LOG" ]; then
    START_TS=$(grep "Record: starting → local/$DS_NAME" "$CTRL_LOG" | tail -1 \
        | grep -oE "\[1[67][0-9]{8}\.[0-9]+\]" | tr -d '[]')
    if [ -n "$START_TS" ]; then
        START_S=${START_TS%.*}
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_S))
        ELAPSED_MIN=$((ELAPSED / 60))
        echo "  elapsed         : ${ELAPSED_MIN} min  (${ELAPSED}s)"
        if [ "$EP" -gt 0 ]; then
            RATE=$((ELAPSED / EP))
            echo "  rate            : ${RATE}s/ep"
            if [ "$TARGET" != "?" ] && [ "$EP" -lt "$TARGET" ]; then
                REM=$(( (TARGET - EP) * RATE ))
                ETA_MIN=$(( REM / 60 ))
                ETA_TIME=$(date -d "+${REM} seconds" '+%H:%M' 2>/dev/null)
                echo "  ETA remaining   : ${ETA_MIN} min  (~${ETA_TIME})"
            fi
        fi
    fi
fi

# 5. lerobot framerate (current)
echo ""
echo "[lerobot framerate]"
if [ -f "$LEROBOT_LOG" ]; then
    LAST_HZ=$(tail -200 "$LEROBOT_LOG" 2>/dev/null \
        | grep -oE "running slower \(([0-9.]+) Hz\)" \
        | grep -oE "[0-9.]+" \
        | tail -10 \
        | awk '{s+=$1; n++} END {if (n>0) printf "%.1f Hz over last %d samples", s/n, n; else print "no rate warnings (likely at target 30 Hz)"}')
    echo "  $LAST_HZ"
else
    echo "  (lerobot log not found at $LEROBOT_LOG)"
fi

# 6. FOV constraint hits (visibility check fired N times)
echo ""
echo "[FOV constraint]"
FOV_HITS=$(grep -c "requiring.*lego.*in workspace.*FOV" /tmp/isaacsim.log 2>/dev/null)
echo "  randomize calls with FOV constraint: $FOV_HITS"

# 7. Last 3 record events
echo ""
echo "[Recent events]"
scoped | grep -E "Record: ✓|Record: ✗|Record: completed|Record: WARN|Record: lerobot exited|Quickstart halted" | tail -3 | sed 's/^/  /'

# 8. Copy job status (if we know where to look)
COPY_LOG="/tmp/${DS_NAME}_copy.log"
if [ -f "$COPY_LOG" ]; then
    echo ""
    echo "[Copy job]"
    if pgrep -f "${DS_NAME}_copy_and_summarize" >/dev/null 2>&1; then
        echo "  scheduled, log: $COPY_LOG"
        tail -3 "$COPY_LOG" | sed 's/^/    /'
    else
        echo "  finished/dead, log: $COPY_LOG"
        tail -5 "$COPY_LOG" | sed 's/^/    /'
    fi
fi

echo ""
echo "=================================================="
