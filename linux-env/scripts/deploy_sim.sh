#!/bin/bash
# deploy_sim.sh — one command for the SO-ARM101 digital twin + SmolVLA policy.
#
#   up [--rate-hz HZ] [--stream] [--gui|--no-gui] [--rviz|--no-rviz] [--model <preset|hf_id|path>] [--task "<text>"] [--no-policy] [--fps Hz] [--action-duration-s seconds] [--no-rtc]
#       1. Isaac Sim + soarm101-dt + omni-kit-mcp bridge     (isaac/isaacsim_launch.sh)
#       2. Scene: soarm101.quick_start over the bridge        (skipped if already live)
#       3. ROS 2 control stack: MoveIt + ros2_control (+ control_gui, + RViz)
#       4. SmolVLA async policy server + robot client on GPU 1 (RTC prefix guidance)
#          (inference_async_lerobot.sh → run_async_inference_sim.py)
#   down [--policy]
#            Stop what `up` started, in reverse order. Things that were already
#            running before `up` (another Isaac, a control stack you started by
#            hand) are left alone. --policy stops only the policy (then
#            `up --model <other>` swaps models without restarting Isaac).
#   status   One-screen summary + log paths.
#
# --stream  Isaac runs headless with the WebRTC livestream on; watch it from the
#           Isaac Sim WebRTC Streaming Client on another machine (see README).
#           Without it Isaac opens a local window.
# --gui     control_gui (Tk window) and RViz are opt-in. Default: on in local
# --rviz    mode, off with --stream. Headless, the control stack is ros2_control
#           + MoveIt only and is ready when its controllers are active. When a
#           window is asked for, the box's X display must be usable (a logged-in
#           desktop, or DISPLAY/XAUTHORITY you set); otherwise `up` stops with
#           that message. No virtual-display fallback.
#
# Model presets (or pass any Hugging Face repo id / local pretrained_model dir):
#   sim          anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200   (default)
#   mix-75s25r   anirudhrani/smolvla_sim_100ep_75S_25R_fft__10ksteps_h200
#   mix-25s75r   anirudhrani/smolvla_sim_100ep_25S_75R_fft__10ksteps_h200
#   mix-50s50r   anirudhrani/smolvla_blue_sort_50sim_50real_10k
#
# Env: SOARM_KIT_MCP_PORT (bridge, default 8767) · SOARM_POLICY_GPU (default 1) ·
#      LOG_DIR (default /tmp) · ISAACSIM_BIN / ISAACSIM_LOG (see isaacsim_launch.sh)
#
# Idempotent: `up` twice is safe — each step skips when its piece is already up.

set -e
# NOTE: no `set -u` — ROS2's setup.bash references unbound AMENT/COLCON vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

STATE_DIR="$LOG_DIR/soarm_deploy"
ISAAC_MARK="$STATE_DIR/isaac.started"       # present → `up` launched Isaac
CONTROL_PIDFILE="$STATE_DIR/control.pid"    # ros2 launch PID, when `up` started it
CONTROL_LOGFILE="$STATE_DIR/control.logpath" # its log
POLICY_PIDFILE="$STATE_DIR/policy.pid"      # process-group id of the policy launcher
POLICY_LOGFILE="$STATE_DIR/policy.logpath"  # path of the current policy log
POLICY_MODELFILE="$STATE_DIR/policy.model"  # model the running policy serves
export SOARM_POLICY_GPU="${SOARM_POLICY_GPU:-1}"   # read by inference_async_lerobot.sh
POLICY_PORT=8080                            # run_async_inference_sim.py default
DEFAULT_TASK="Pick a blue lego and place it in blue cup"
mkdir -p "$STATE_DIR"

resolve_model() {
    case "$1" in
        ""|sim)     echo "anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200" ;;
        mix-75s25r) echo "anirudhrani/smolvla_sim_100ep_75S_25R_fft__10ksteps_h200" ;;
        mix-25s75r) echo "anirudhrani/smolvla_sim_100ep_25S_75R_fft__10ksteps_h200" ;;
        mix-50s50r) echo "anirudhrani/smolvla_blue_sort_50sim_50real_10k" ;;
        */*)        echo "$1" ;;   # HF repo id (owner/name) or a local path
        *) echo "Unknown model preset '$1' (presets: sim, mix-75s25r, mix-25s75r, mix-50s50r; or an HF repo id)" >&2
           return 1 ;;
    esac
}

policy_pgid() {
    local pgid
    pgid="$(cat "$POLICY_PIDFILE" 2>/dev/null)" || return 1
    [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null && echo "$pgid"
}

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------------------
cmd_up() {
    local stream="" rviz="" gui="" model_arg="" task="$DEFAULT_TASK" policy=1 model
    local policy_args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --fps|--action-duration-s) policy_args+=("$1" "${2:?option needs a value}"); shift ;;
            --fps=*|--action-duration-s=*|--no-rtc) policy_args+=("$1") ;;
            --stream)    stream="--stream" ;;
            --rate-hz) export SOARM_SIM_RATE_HZ="${2:?--rate-hz needs a value}"; shift ;;
            --rate-hz=*) export SOARM_SIM_RATE_HZ="${1#--rate-hz=}" ;;
            --model)     model_arg="${2:?--model needs a value}"; shift ;;
            --model=*)   model_arg="${1#--model=}" ;;
            --task)      task="${2:?--task needs a value}"; shift ;;
            --task=*)    task="${1#--task=}" ;;
            --gui)       gui="true" ;;
            --no-gui)    gui="false" ;;
            --rviz)      rviz="true" ;;
            --no-rviz)   rviz="false" ;;
            --no-policy) policy=0 ;;
            -h|--help)   usage; exit 0 ;;
            *) echo "Unknown option for up: $1" >&2; usage >&2; exit 2 ;;
        esac
        shift
    done
    [ -n "$rviz" ] || { [ -n "$stream" ] && rviz="false" || rviz="true"; }
    [ -n "$gui" ] || { [ -n "$stream" ] && gui="false" || gui="true"; }
    model="$(resolve_model "$model_arg")" || exit 2

    stack_preflight || exit 1
    stack_source_ros

    echo "=== deploy_sim up ==="
    echo "[1/4] Isaac Sim ($ISAAC_EXT_ID${stream:+, streaming}, bridge $KIT_MCP_HOST:$KIT_MCP_PORT)..."
    local rc=0
    if [ -n "$stream" ]; then isaac_up --stream || rc=$?; else isaac_up || rc=$?; fi
    # Mark even a failed launch, so `down` cleans up the half-started Isaac.
    [ "$ISAAC_STARTED" = 1 ] && touch "$ISAAC_MARK"
    [ "$rc" = 0 ] || { echo "  Isaac did not come up — see ${ISAACSIM_LOG:-/tmp/isaacsim-soarm101.log}" >&2; exit 1; }

    echo "[2/4] Scene (soarm101.quick_start)..."
    scene_up

    echo "[3/4] ROS 2 control stack (gui=$gui, rviz=$rviz)..."
    rc=0
    control_stack_up "$rviz" "$gui" || rc=$?
    [ "$CONTROL_STARTED" = 1 ] && { cp "$STACK_PIDFILE" "$CONTROL_PIDFILE"; echo "$CONTROL_LOG" > "$CONTROL_LOGFILE"; }
    [ "$rc" = 0 ] || exit 1

    echo "[4/4] SmolVLA policy (async server + robot client, GPU $SOARM_POLICY_GPU)..."
    if [ "$policy" = 0 ]; then
        echo "  skipped (--no-policy)"
    elif pgid="$(policy_pgid)"; then
        echo "  already running (pgid $pgid, $(cat "$POLICY_MODELFILE" 2>/dev/null))"
        echo "  to switch models: $0 down --policy && $0 up --model <preset|hf_id>"
        echo "  Log: $(cat "$POLICY_LOGFILE" 2>/dev/null)"
    else
        local log pid i
        log="$LOG_DIR/policy_$(date +%H%M%S).log"
        nvidia-smi -i "$SOARM_POLICY_GPU" >/dev/null 2>&1 || \
            { echo "  ERROR: no GPU at index $SOARM_POLICY_GPU (nvidia-smi); set SOARM_POLICY_GPU" >&2; exit 1; }
        # The wrapper pins CUDA_VISIBLE_DEVICES to SOARM_POLICY_GPU's UUID (same
        # path as control_gui's Inference tab). setsid: own process group, so
        # `down` can SIGINT server + client together. env --default-signal=INT:
        # bash starts `&` jobs with SIGINT ignored, and a Python started that way
        # never raises KeyboardInterrupt, so `down` could only SIGTERM it.
        setsid env --default-signal=INT bash "$SCRIPT_DIR/inference_async_lerobot.sh" \
            --pretrained="$model" --task="$task" "${policy_args[@]}" > "$log" 2>&1 < /dev/null &
        pid=$!
        echo "$pid" > "$POLICY_PIDFILE"
        echo "$log" > "$POLICY_LOGFILE"
        echo "$model" > "$POLICY_MODELFILE"
        echo "  Model: $model"
        echo "  Task:  $task"
        echo "  Log:   $log"
        echo "  Waiting for the policy server to load (first run downloads the model)..."
        for i in $(seq 1 120); do
            if grep -q "Starting robot client" "$log" 2>/dev/null; then
                echo "  policy server up, robot client started (~$((i*3))s)"
                break
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "  ERROR: policy launcher exited — last lines of $log:" >&2
                tail -20 "$log" >&2
                rm -f "$POLICY_PIDFILE"
                exit 1
            fi
            sleep 3
        done
        grep -q "Starting robot client" "$log" 2>/dev/null || \
            echo "  still loading after 6 min — keep watching: tail -F $log"
    fi

    echo ""
    echo "=== Up ==="
    print_logs
    echo "  Status:    bash $0 status"
    echo "  Stop:      bash $0 down"
}

# ---------------------------------------------------------------------------
cmd_down() {
    local pgid i policy_only=0
    case "${1:-}" in
        "") ;;
        --policy) policy_only=1 ;;
        *) echo "Unknown option for down: $1 (expected: --policy)" >&2; exit 2 ;;
    esac
    echo "=== deploy_sim down ==="

    echo "[1/3] SmolVLA policy..."
    if pgid="$(policy_pgid)"; then
        # SIGINT → run_async_inference_sim.py stops the client, then the server.
        kill -INT -- "-$pgid" 2>/dev/null || true
        for i in $(seq 1 20); do
            kill -0 -- "-$pgid" 2>/dev/null || { echo "  stopped (${i}s)"; break; }
            sleep 1
        done
        if kill -0 -- "-$pgid" 2>/dev/null; then
            echo "  still running after 20s — sending SIGTERM"
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    else
        echo "  not running (or not started by deploy_sim)"
    fi
    rm -f "$POLICY_PIDFILE" "$POLICY_MODELFILE" "$POLICY_LOGFILE"
    [ "$policy_only" = 1 ] && { echo "=== Policy down (Isaac + control stack left running) ==="; return 0; }

    echo "[2/3] ROS 2 control stack..."
    if [ -f "$CONTROL_PIDFILE" ]; then
        stop_pid "$(cat "$CONTROL_PIDFILE")" "control stack" 15
        rm -f "$CONTROL_PIDFILE" "$CONTROL_LOGFILE" "$STACK_PIDFILE"
    else
        echo "  not started by deploy_sim — leaving it (stack_stop.sh stops it)"
    fi

    echo "[3/3] Isaac Sim..."
    if [ -f "$ISAAC_MARK" ]; then
        bash "$ISAAC_LAUNCHER" close
        rm -f "$ISAAC_MARK"
    else
        echo "  not started by deploy_sim — leaving it (isaacsim_launch.sh close stops it)"
    fi
    echo "=== Down ==="
}

# ---------------------------------------------------------------------------
# Logs of what is running now and was started by this repo's scripts. Pieces
# that are down (or someone else's) are left out rather than pointing at an old
# file from an earlier run.
print_logs() {
    local pid
    echo "  Logs:"
    if bash "$ISAAC_LAUNCHER" status >/dev/null 2>&1; then
        echo "    Isaac Sim:     ${ISAACSIM_LOG:-/tmp/isaacsim-soarm101.log}"
    fi
    pid="$(cat "$CONTROL_PIDFILE" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && [ -s "$CONTROL_LOGFILE" ]; then
        echo "    Control stack: $(cat "$CONTROL_LOGFILE")"
    fi
    if policy_pgid >/dev/null && [ -s "$POLICY_LOGFILE" ]; then
        echo "    Policy:        $(cat "$POLICY_LOGFILE")"
    fi
}

cmd_status() {
    local pgid
    bash "$SCRIPT_DIR/stack_status.sh"
    echo ""
    echo "[SmolVLA policy]"
    if pgid="$(policy_pgid)"; then
        echo "  launcher: running (pgid $pgid, GPU $SOARM_POLICY_GPU)"
        if ss -tln 2>/dev/null | grep -q ":$POLICY_PORT "; then
            echo "  policy server :$POLICY_PORT: LISTENING"
        else
            echo "  policy server :$POLICY_PORT: not listening yet (loading?)"
        fi
        echo "  model: $(cat "$POLICY_MODELFILE" 2>/dev/null)"
    else
        echo "  not running"
    fi
    echo ""
    echo "[Started by deploy_sim]"
    echo "  Isaac Sim:     $([ -f "$ISAAC_MARK" ] && echo yes || echo no)"
    echo "  Control stack: $([ -f "$CONTROL_PIDFILE" ] && echo "yes (pid $(cat "$CONTROL_PIDFILE"))" || echo no)"
    echo "  Policy:        $([ -n "$(policy_pgid)" ] && echo yes || echo no)"
    print_logs
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    down)   shift; cmd_down "$@" ;;
    status) cmd_status ;;
    -h|--help|help|"") usage ;;
    *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
