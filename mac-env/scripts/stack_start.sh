#!/usr/bin/env bash
# stack_start.sh — launch the SO-ARM101 stack via stack.launch.py.
#
# Usage:
#   bash mac-env/scripts/stack_start.sh [--mode MODE] [--rviz] [--gazebo-gui]
#   bash mac-env/scripts/stack_start.sh [LEGACY_POSITIONAL]    (back-compat)
#
# Modes (pick the BACKEND that produces /joint_states + /objects_* + /drop_*):
#   gazebo  — gz sim + bridge + controllers + sim_ground_truth   (default)
#   isaac   — Isaac Sim runs externally; this brings up only the core
#             (RSP, move_group, control_gui). Connect Isaac separately.
#   real    — jointstatereader (Feetech) + real_cameras + ArUco/YOLOE.
#             Motion-execution shim NOT YET IMPLEMENTED — planning works,
#             execution to the real arm doesn't, until that lands.
#
# Flags:
#   --rviz         Launch RViz too (works in any mode).
#   --gazebo-gui   In gazebo mode, also fork the Gazebo GUI client.
#                  Without this, gazebo runs headless (server only).
#
# Legacy positional args (kept for back-compat with scripts/test_*.sh and
# the cross-repo restart-control-stack.sh):
#   headless | record | none  →  --mode gazebo                     (no GUI, no RViz)
#   gz | gazebo | sim         →  --mode gazebo --gazebo-gui        (Gazebo GUI only)
#   rviz                      →  --mode gazebo --rviz              (RViz, gazebo headless)
#   all | full | visual       →  --mode gazebo --gazebo-gui --rviz (everything)
#
# Runs detached. View logs at /tmp/soarm_stack.log.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

stack_preflight

MODE="gazebo"
RVIZ_ARG="false"
GAZEBO_GUI="false"
EXTRA_LAUNCH_ARGS=()

# Argument parser — handles BOTH new flags and legacy positional arg.
# We do this manually (not getopt) to keep both styles working in one
# script without surprising users mid-migration.
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --mode=*)      MODE="${1#*=}"; shift ;;
    --rviz)        RVIZ_ARG="true"; shift ;;
    --gazebo-gui)  GAZEBO_GUI="true"; shift ;;
    --help|-h|help)
      sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    # ---- Legacy positional arg ----
    headless|record|none)
      MODE="gazebo"; GAZEBO_GUI="false"; RVIZ_ARG="false"; shift ;;
    gz|gazebo|sim)
      MODE="gazebo"; GAZEBO_GUI="true";  RVIZ_ARG="false"; shift ;;
    rviz)
      MODE="gazebo"; GAZEBO_GUI="false"; RVIZ_ARG="true";  shift ;;
    all|full|visual|inspect)
      MODE="gazebo"; GAZEBO_GUI="true";  RVIZ_ARG="true";  shift ;;
    # ---- Launch-arg passthrough (e.g. follower_port:=/dev/tty..., read_only:=true) ----
    *:=*)
      EXTRA_LAUNCH_ARGS+=("$1"); shift ;;
    *)
      echo "ERROR: unknown argument '$1'. Try '--help'." >&2
      exit 1 ;;
  esac
done

case "$MODE" in
  gazebo|isaac|real) ;;
  *)
    echo "ERROR: --mode must be one of: gazebo, isaac, real (got '$MODE')" >&2
    exit 1 ;;
esac

# stack.launch.py uses `headless` to mean "no Gazebo GUI client". Our
# user-facing flag is the inverse (--gazebo-gui = launch the client),
# so flip it here.
HEADLESS="true"
[ "$GAZEBO_GUI" = "true" ] && HEADLESS="false"

# Pretty mode label for the startup banner.
case "$MODE" in
  gazebo)
    case "$GAZEBO_GUI:$RVIZ_ARG" in
      true:true)   MODE_PRETTY="gazebo (Gazebo GUI + RViz)" ;;
      true:false)  MODE_PRETTY="gazebo (Gazebo GUI, no RViz)" ;;
      false:true)  MODE_PRETTY="gazebo (headless server + RViz)" ;;
      false:false) MODE_PRETTY="gazebo (headless server, no RViz)" ;;
    esac ;;
  isaac)
    if [ "$RVIZ_ARG" = "true" ]; then
      MODE_PRETTY="isaac (RViz + control_gui; Isaac Sim runs externally)"
    else
      MODE_PRETTY="isaac (control_gui only; Isaac Sim runs externally)"
    fi ;;
  real)
    if [ "$RVIZ_ARG" = "true" ]; then
      MODE_PRETTY="real (jointstatereader + cameras + aruco + RViz)"
    else
      MODE_PRETTY="real (jointstatereader + cameras + aruco)"
    fi ;;
esac

# Refuse to start if anything is already running.
N=$(stack_running_count)
if [ "$N" -gt 0 ]; then
  echo "ERROR: $N existing stack processes running. Stop them first:" >&2
  echo "       bash $SCRIPT_DIR/stack_stop.sh" >&2
  exit 1
fi

echo "== Starting SO-ARM101 stack: mode=$MODE_PRETTY =="

export PATH="$HOME/.pixi/bin:$PATH"
: > "$STACK_LOG"

# Launch detached. pixi run handles env activation; sourcing the ws setup
# inside the subshell extends AMENT_PREFIX_PATH with our built packages.
nohup pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  set -e
  source '$SOARM_WS_SETUP' 2>/dev/null
  export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'
  export CYCLONEDDS_URI='$CYCLONEDDS_URI'
  export KMP_DUPLICATE_LIB_OK='$KMP_DUPLICATE_LIB_OK'
  export AMENT_PYTHON_EXECUTABLE='$AMENT_PYTHON_EXECUTABLE'
  exec ros2 launch so_arm101_control stack.launch.py mode:=$MODE rviz:=$RVIZ_ARG headless:=$HEADLESS ${EXTRA_LAUNCH_ARGS[*]}
" > "$STACK_LOG" 2>&1 &

LPID=$!
echo "$LPID" > "$STACK_PIDFILE"
echo "launch pid: $LPID  log: $STACK_LOG  pidfile: $STACK_PIDFILE"

sleep 2
CHILDREN=$(pgrep -P "$LPID" | wc -l | tr -d ' ')
echo "launch has $CHILDREN direct children after 2s"

if [ "$MODE" = "gazebo" ] && [ "$GAZEBO_GUI" = "true" ]; then
  echo "  Gazebo GUI should appear as a macOS window within 10-30s."
fi
if [ "$RVIZ_ARG" = "true" ]; then
  echo "  RViz window will appear within ~10-15s."
fi
echo "  Control GUI (tkinter) appears in all modes after init."

if [ "$MODE" = "real" ]; then
  echo ""
  echo "  Real mode: perception + state ingest + motion execution via"
  echo "  trajectory_shim (50 Hz, time-interpolates FollowJointTrajectory"
  echo "  goals to /joint_commands). control_gui auto-Connect spawns the"
  echo "  live jsr+shim ~3s after launch — backend_real's own jsr+shim die"
  echo "  with exit 1 (cosmetic; expected; see vla_SO-ARM101/CLAUDE.md)."
fi
if [ "$MODE" = "isaac" ]; then
  echo ""
  echo "  NOTE: isaac mode launches no backend processes here. Make sure"
  echo "        Isaac Sim is running externally and publishing /joint_states,"
  echo "        /objects_poses_sim, /drop_poses on the same DDS domain."
fi

echo ""
echo "Tail:    tail -f $STACK_LOG"
echo "Status:  bash $SCRIPT_DIR/stack_status.sh"
echo "Stop:    bash $SCRIPT_DIR/stack_stop.sh"
echo "RViz:    bash $SCRIPT_DIR/rviz_attach.sh   # attach RViz post-launch"
