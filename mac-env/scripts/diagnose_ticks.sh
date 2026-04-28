#!/usr/bin/env bash
# diagnose_ticks.sh — one-shot motion-quality capture + analysis.
#
# Workflow:
#   1. Starts stream_motion_recorder.py in the background, capturing
#      /joint_commands + /joint_states to a JSONL.
#   2. Prompts you to drive the arm via the GUI to reproduce whatever
#      motion-quality issue you're investigating.
#   3. On Enter, SIGINTs the recorder (file is already streamed to disk).
#   4. Runs tick_classifier.py to count during-motion vs idle-drift events.
#   5. Runs tick_plots_with_events.py to overlay shim trajectory boundaries
#      on cmd/state plots.
#   6. Opens the PNGs.
#
# Output:
#   /tmp/diagnose_ticks_<timestamp>.jsonl  (raw recording)
#   /tmp/tick_plot_session_v2.png          (cmd vs state per joint, events overlaid)
#   /tmp/tick_plot_zoom_v2.png             (shoulder_lift zoom on idle-drift cluster)
#   /tmp/tick_plot_with_events.png         (cmd vs state with shim trajectory boundaries)
#
# Prereqs: stack already up via stack_start.sh; trajectory_shim and
# jointstatereader publishing on the canonical topics.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/mac-env/scripts"
MAC_ENV_MANIFEST="${MAC_ENV_MANIFEST:-/tmp/mac-env/pixi.toml}"
SOARM_WS_SETUP="${SOARM_WS_SETUP:-/tmp/soarm-ws/install/setup.bash}"

if [[ ! -f "$MAC_ENV_MANIFEST" ]]; then
  echo "ERROR: pixi manifest not found at $MAC_ENV_MANIFEST"
  echo "Run mac-env/scripts/bootstrap.sh first."
  exit 1
fi
if [[ ! -f "$SOARM_WS_SETUP" ]]; then
  echo "ERROR: ROS2 workspace not found at $SOARM_WS_SETUP"
  echo "Run colcon build in /tmp/soarm-ws first."
  exit 1
fi

ts="$(date +%Y%m%dT%H%M%S)"
out="/tmp/diagnose_ticks_${ts}.jsonl"

echo "============================================================"
echo "  motion-quality tick diagnosis — capturing to $out"
echo "============================================================"
echo
echo "Recorder is starting in the background. Drive the arm via the"
echo "GUI now (Grasp Home / Grasp Move / etc.) to reproduce the ticks"
echo "or motion-quality issue you're investigating."
echo

pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  source '$SOARM_WS_SETUP'
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI='file:///tmp/mac-env/cyclonedds.xml'
  export KMP_DUPLICATE_LIB_OK=TRUE
  python3 '$SCRIPT_DIR/stream_motion_recorder.py' '$out' 1800
" &
REC_PID=$!

# give it a moment to set up subscriptions
sleep 2

if ! kill -0 "$REC_PID" 2>/dev/null; then
  echo "ERROR: recorder process died before we could record. Check stack."
  exit 1
fi

# Wait for user
read -p "Press Enter when you're done driving the arm... "

# SIGINT the recorder so it flushes/closes cleanly
kill -SIGINT "$REC_PID" 2>/dev/null || true
# In case it spawned children
pkill -SIGINT -f "stream_motion_recorder.py.*$out" 2>/dev/null || true
wait "$REC_PID" 2>/dev/null || true
sleep 1

if [[ ! -s "$out" ]]; then
  echo "ERROR: $out is empty — no data captured. Check that"
  echo "trajectory_shim and jointstatereader were publishing."
  exit 1
fi

echo
echo "============================================================"
echo "  captured $(wc -l < "$out") rows — analyzing..."
echo "============================================================"
echo

pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  source '$SOARM_WS_SETUP'
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI='file:///tmp/mac-env/cyclonedds.xml'
  export KMP_DUPLICATE_LIB_OK=TRUE

  echo '--- TICK CLASSIFIER ---'
  python3 '$SCRIPT_DIR/tick_classifier.py' '$out'

  echo
  echo '--- TICK PLOTS (overall) ---'
  python3 '$SCRIPT_DIR/tick_plots.py' '$out' || true

  echo
  echo '--- TICK PLOTS WITH SHIM EVENTS ---'
  # tick_plots_with_events.py reads /tmp/traj_shim.log and aligns
  # shim event timestamps with the recorder. Edit the path inside
  # the script if your shim log lives elsewhere.
  python3 '$SCRIPT_DIR/tick_plots_with_events.py' '$out' || true
"

echo
echo "============================================================"
echo "  done. PNGs in /tmp/:"
ls -la /tmp/tick_plot_*.png 2>/dev/null | tail -10
echo "============================================================"

# Open the most useful plot if `open` is available (macOS)
if command -v open >/dev/null 2>&1; then
  for p in /tmp/tick_plot_with_events.png /tmp/tick_plot_session_v2.png; do
    [[ -f "$p" ]] && open "$p"
  done
fi
