#!/bin/bash
# Launch, close and check Isaac Sim 6.1 with the soarm101-dt extension and the omni-kit-mcp bridge.
#
# Usage:
#   isaacsim_launch.sh launch [ext-id] [--stream|--headless] [--instance N] [--rate-hz HZ]  # launch, then block until the bridge answers
#   isaacsim_launch.sh rate [hz]                   # apply/read pacing after World exists; 0 = unlimited
#   isaacsim_launch.sh wait [window]               # block until the bridge answers; first-run RTX compile extends it
#   isaacsim_launch.sh quick-start [json]          # soarm101.quick_start while watching Kit (wedge detection, no retry)
#   isaacsim_launch.sh status                      # process, bridge, stream port
#   isaacsim_launch.sh close                       # SIGTERM, then SIGKILL, the Isaac this script started
#   isaacsim_launch.sh kill                        # verified tree shutdown (same ownership policy as close)
#   isaacsim_launch.sh restart [ext-id] [--stream] # close, then launch
# Exit codes: launch/wait 0 ready, 1 error, 2 usage; quick-start = the client's (0 ok, 1 tool
# error, 2 unreachable); all: 3 GPU wedge (no bridge and no Kit log progress for
# ISAACSIM_WEDGE_S=120 s). A wedge is reported, never retried: our Isaac is left running for
# inspection; stop it with `close`.
#
# --stream: headless with the WebRTC livestream (isaacsim.exp.full.streaming), for the
#   "Isaac Sim WebRTC Streaming Client" on another machine: connect it to this box's address
#   (tailnet IP by default), signalling TCP 49100, media UDP 47998. Without --stream: a local window.
#
# Commands reach the sim through the omni-kit-mcp bridge (TCP on localhost:$OMNI_KIT_MCP_PORT):
#   PYTHONPATH=$REPO_ROOT/omni-kit-mcp python3 -m kit_mcp.client soarm101.quick_start
# The bridge and its panel are enabled for this launch only (--ext-folder/--enable) and the
# soarm101_tools package is loaded through KIT_MCP_TOOL_PATHS/KIT_MCP_TOOL_MODULES, so nothing is
# written to the Isaac install's user config (shared with other projects on the box).
#
# Env overrides (all optional):
#   ISAACSIM_BIN       isaacsim launcher            (default $HOME/env_isaacsim61/bin/isaacsim)
#   ISAACSIM_GPU       GPU index for Isaac          (default 0)
#   OMNI_KIT_MCP_PORT  bridge port, else SOARM_KIT_MCP_PORT, else 8767
#   ISAACSIM_STREAM_IP public address for --stream  (default this box's tailnet IPv4)
#   ROS_DOMAIN_ID      DDS domain                   (default 0)
#   ISAACSIM_LOG       log path                     (default /tmp/isaacsim-soarm101.log)
#   MCP_CLIENT_OUTPUT_DIR  scene_states/ and videos/ (default /tmp/soarm101-resources)
#   ISAACSIM_WAIT_CAP  overall cap for launch/wait/quick-start during the first-run RTX compile (default 900)
#   ISAACSIM_WEDGE_S   Kit-log silence (no bridge) that counts as a GPU wedge (default 120)
#   ISAACSIM_EXTRA_ARGS extra Kit args appended verbatim
#
# Default caches (persistent/shared); ISAACSIM_EXTRA_ARGS="--portable-root PATH" isolates data/cache/logs:
#   RTX shader cache   ~/.cache/ov/shaders/nv_shadercache/<kit-version>/<build>  (first run fills it)
#   DerivedDataCache   ~/.cache/ov/DerivedDataCache  (cooked collision/geometry, UJITSO)

set -u

# An index scopes every resource; default launch/close behavior remains the manual instance.
INSTANCE_INDEX="${SOARM_INSTANCE_INDEX:-}"
launch_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --instance) INSTANCE_INDEX="${2:?--instance requires an index}"; shift 2 ;;
        *) launch_args+=("$1"); shift ;;
    esac
done
set -- "${launch_args[@]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ -n "$INSTANCE_INDEX" ]; then
    [[ "$INSTANCE_INDEX" =~ ^[0-9]+$ ]] && [ "$INSTANCE_INDEX" -le 9 ] || {
        echo '--instance must be an integer from 0 to 9' >&2; exit 2;
    }
    export SOARM_INSTANCE_INDEX="$INSTANCE_INDEX"
    export ROS_DOMAIN_ID="$((90 + INSTANCE_INDEX))"
    export OMNI_KIT_MCP_PORT="$((8777 + 2 * INSTANCE_INDEX))"
    export SOARM_KIT_MCP_PORT="$OMNI_KIT_MCP_PORT"
    INSTANCE_ROOT="${SOARM_INSTANCE_ROOT:-/mnt/a4500-backup/exploring-vlas/runtime/instance-$INSTANCE_INDEX}"
    export SOARM_INSTANCE_ROOT="$INSTANCE_ROOT"
    export ISAACSIM_LOG="$INSTANCE_ROOT/isaac.log"
    python3 "$SCRIPT_DIR/../collect/storage.py" "$INSTANCE_ROOT" "$ISAACSIM_LOG"
    mkdir -p "$INSTANCE_ROOT"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    if [ -n "${ISAACSIM_EXTRA_ARGS:-}" ]; then
        echo 'Indexed instances use pinned settings; ISAACSIM_EXTRA_ARGS is not permitted' >&2; exit 2
    fi
    export TMPDIR="$INSTANCE_ROOT/tmp"
    export TEMP="$TMPDIR" TMP="$TMPDIR"
    export XDG_CACHE_HOME="$INSTANCE_ROOT/cache"
    export OPTIX_CACHE_PATH="$XDG_CACHE_HOME/optix"
    export ROS_LOG_DIR="$INSTANCE_ROOT/ros-logs"
    mkdir -p "$TMPDIR" "$OPTIX_CACHE_PATH" "$ROS_LOG_DIR"
fi


ISAACSIM_BIN="${ISAACSIM_BIN:-$HOME/env_isaacsim61/bin/isaacsim}"
EXT_FOLDER="$REPO_ROOT/isaac/exts"
KIT_MCP_ROOT="$REPO_ROOT/omni-kit-mcp"
TOOLS_PATH="$REPO_ROOT/isaac"
TOOLS_MODULE="soarm101_tools"
CYCLONEDDS_XML="$REPO_ROOT/linux-env/cyclonedds.xml"
GPU="${ISAACSIM_GPU:-0}"
# Default recording cadence; update this one value when dataset timing changes.
DEFAULT_SIM_RATE_HZ=17.4
RATE_HZ="${SOARM_SIM_RATE_HZ:-$DEFAULT_SIM_RATE_HZ}"
MCP_PORT="${OMNI_KIT_MCP_PORT:-${SOARM_KIT_MCP_PORT:-8767}}"
STREAM_SIGNAL_PORT=49100
LOG="${ISAACSIM_LOG:-/tmp/isaacsim-soarm101.log}"
PIDFILE="${INSTANCE_ROOT:-/tmp}/isaacsim-soarm101.pid"
WEDGE_S="${ISAACSIM_WEDGE_S:-120}"

client() {  # client <tool> [json] — the omni-kit-mcp reference client, pinned to our bridge
    PYTHONPATH="$KIT_MCP_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m kit_mcp.client "$@" --port "$MCP_PORT"
}

our_pid() {  # Only a live identity recorded by this launcher is reusable.
    python3 "$SCRIPT_DIR/../collect/managed_process.py" status --pidfile "$PIDFILE"
}

claim_launcher_operation() {
    if [ -n "${INSTANCE_ROOT:-}" ]; then
        source "$SCRIPT_DIR/../collect/instance_claim.sh"
        claim_collection_instance "$INSTANCE_ROOT"
    fi
}

bridge_ready() {  # the bridge answers AND the extension behind soarm101.* is up
    client soarm101.get_recording_status --timeout 10 > /dev/null 2>&1
}

stream_listening() {
    ss -ltn 2>/dev/null | grep -q ":$STREAM_SIGNAL_PORT\b"
}

strip_ros_paths() {  # drop /opt/ros entries: Isaac must load its bundled ROS, not the system one
    local out="" p IFS=:
    for p in $1; do
        case "$p" in /opt/ros/*|"") ;; *) out="${out:+$out:}$p" ;; esac
    done
    echo "$out"
}

kit_log() {  # Read only a Kit log held by a verified managed descendant.
    python3 "$SCRIPT_DIR/../collect/managed_process.py" kit-log --pidfile "$PIDFILE"
}

rtpso_lines() {  # count of RTX pipeline-compile progress lines in the Kit log so far
    local kl
    kl="$(kit_log)" && [ -n "$kl" ] || { echo 0; return; }
    local n
    n="$(grep -c "Waiting for RtPso async group async compilation" "$kl" 2>/dev/null)"
    echo "${n:-0}"
}

rtpso_last() {  # "N" seconds from the latest RTX pipeline-compile progress line, if any
    local kl
    kl="$(kit_log)" && [ -n "$kl" ] || return 1
    grep "Waiting for RtPso async group async compilation" "$kl" 2>/dev/null | tail -1 \
        | grep -o '[0-9]* seconds so far' | grep -o '^[0-9]*'
}

kit_log_size() {
    local kl
    kl="$(kit_log)" && [ -n "$kl" ] && stat -c %s "$kl" 2>/dev/null || echo 0
}

wedge_msg() {
    local drv pid
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU" 2>/dev/null | head -1)"
    echo "GPU wedge (no Kit progress for ${1}s, bridge not answering). Known on NVIDIA driver <580 (this box: ${drv:-unknown}) on cold-cache launches."
    echo "  Stop it with \`isaacsim_launch.sh close\` and rerun \`deploy_sim.sh up\` (idempotent; caches are warmer now), or upgrade to driver ≥580."
    if pid="$(our_pid)"; then
        echo "  Isaac (pid $pid) is left running for inspection; Kit log: $(kit_log || echo "$LOG")"
    fi
}

# Ready = the bridge answers AND soarm101.* answers. The first run on a new GPU/driver/Isaac version
# also compiles RTX ray-tracing pipelines ("Waiting for RtPso async group async compilation: N
# seconds so far" in the Kit log, typically 100-150 s here); Kit's main loop serves no requests
# meanwhile, so it looks hung. Any new Kit log output counts as progress and extends the deadline,
# up to ISAACSIM_WAIT_CAP (default 900 s). No bridge AND no new Kit log output for the wedge window
# (arg, default ISAACSIM_WEDGE_S=120 s) = GPU wedge: return 3.
do_wait() {
    local window="${1:-$WEDGE_S}" cap="${ISAACSIM_WAIT_CAP:-900}" start progress now elapsed seen n last size lsize=-1
    [ "$cap" -lt "$window" ] && cap="$window"
    seen="$(rtpso_lines)"; start="$(date +%s)"; progress="$start"
    echo "Waiting for the omni-kit-mcp bridge on :$MCP_PORT (wedge after ${window}s without Kit log output, cap ${cap}s)..."
    while :; do
        now="$(date +%s)"; elapsed=$((now - start))
        if bridge_ready; then
            echo "READY ($(( $(date +%s) - start ))s)"
            return 0
        fi
        if [ -f "$PIDFILE" ] && ! our_pid > /dev/null; then
            echo "ERROR: Isaac Sim exited. Last log lines ($LOG):"
            tail -15 "$LOG" | sed 's/^/  /'
            return 1
        fi
        size="$(kit_log_size)"
        if [ "$size" != "$lsize" ] || [ "$size" = 0 ]; then lsize="$size"; progress="$(date +%s)"; fi
        n="$(rtpso_lines)"
        if [ "$n" -gt "$seen" ]; then
            seen="$n"; last="$(rtpso_last)"
            echo "  first-run RTX shader compile, ${last:-?} s so far (cached under this launch's cache root)"
        fi
        if [ $(( $(date +%s) - progress )) -ge "$window" ]; then
            wedge_msg "$window"
            return 3
        fi
        if [ "$elapsed" -ge "$cap" ]; then
            echo "TIMEOUT: ${cap}s cap reached (ISAACSIM_WAIT_CAP); Kit log: $(kit_log || echo '?')"
            return 1
        fi
        sleep 5
    done
}

# quick-start [json]: run soarm101.quick_start while watching Kit, so a GPU wedge during the
# first scene build reports in ~ISAACSIM_WEDGE_S instead of after the whole client timeout.
# New Kit log output (incl. RtPso compile progress) counts as progress; no progress for the
# wedge window AND no answer from the bridge = wedge: stop waiting for the call (the tool is not
# cancelled), print the wedge message, exit 3, leave Isaac running. Detection only, no retry.
# stdout = the tool's payload (or the client's error); progress and the wedge message -> stderr.
do_quick_start() {
    local params="${1:-}" out cpid size lsize=-1 progress seen n last rc
    [ -n "$params" ] || params='{}'
    out="$(mktemp)"
    cpid="$(PYTHONPATH="$KIT_MCP_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$SCRIPT_DIR/../collect/managed_process.py" start \
        --pidfile "$out.pid" --log "$out" -- python3 -m kit_mcp.client soarm101.quick_start "$params" \
        --timeout "${ISAACSIM_WAIT_CAP:-900}" --port "$MCP_PORT")" || return 1
    progress="$(date +%s)"; seen="$(rtpso_lines)"
    while python3 "$SCRIPT_DIR/../collect/managed_process.py" status --pidfile "$out.pid" >/dev/null; do
        size="$(kit_log_size)"
        # 0 = no Kit log to watch (not our Isaac): never call that a wedge.
        if [ "$size" != "$lsize" ] || [ "$size" = 0 ]; then lsize="$size"; progress="$(date +%s)"; fi
        n="$(rtpso_lines)"
        if [ "$n" -gt "$seen" ]; then
            seen="$n"; last="$(rtpso_last)"
            echo "  first-run RTX shader compile, ${last:-?} s so far (cached under this launch's cache root)" >&2
        fi
        if [ $(( $(date +%s) - progress )) -ge "$WEDGE_S" ] && ! client status --timeout 3 > /dev/null 2>&1; then
            python3 "$SCRIPT_DIR/../collect/managed_process.py" stop --pidfile "$out.pid" || return 1
            rm -f "$out"
            wedge_msg "$WEDGE_S" >&2
            return 3
        fi
        sleep 5
    done
    rc="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["workload_returncode"])' "$out.pid.identity.json")"
    cat "$out"; rm -f "$out"
    return "$rc"
}

validate_rate() {
    python3 - "$RATE_HZ" <<'PY'
import math, sys
try:
    hz = float(sys.argv[1])
    assert math.isfinite(hz) and hz >= 0
except (ValueError, AssertionError):
    sys.exit("--rate-hz must be finite and nonnegative (0 = unthrottled)")
PY
}

do_rate() {
    RATE_HZ="${1:-$RATE_HZ}"
    validate_rate || return 2
    local params
    params="$(python3 - "$RATE_HZ" <<'PY'
import json, sys
print(json.dumps({"code": "from so_arm101_dt.extension import get_instance\nimport carb\ns=carb.settings.get_settings()\ns.set_bool('/app/runLoops/main/syncToPresent',False)\ns.set_bool('/app/runLoops/main/rateLimitUseBusyLoop',False)\nresult=get_instance()._apply_sim_rate(" + repr(float(sys.argv[1])) + ")"}))
PY
)"
    client run_python "$params" --timeout 30
}

do_launch() {
    if [ -n "${INSTANCE_ROOT:-}" ]; then
        source "$SCRIPT_DIR/../collect/instance_claim.sh"
        claim_collection_instance "$INSTANCE_ROOT"
    fi
    local ext="soarm101-dt" stream=0 headless=0 arg
    while [ $# -gt 0 ]; do
        arg="$1"
        case "$arg" in
            --rate-hz) RATE_HZ="${2:?--rate-hz needs a value}"; shift ;;
            --rate-hz=*) RATE_HZ="${arg#--rate-hz=}" ;;
            --stream) stream=1 ;;
            --headless) headless=1 ;;
            -*) echo "unknown option: $arg" >&2; return 2 ;;
            *) ext="$arg" ;;
        esac
        shift
    done
    validate_rate || return 2
    if [ "$stream" = 1 ] && [ "$headless" = 1 ]; then
        echo '--stream and --headless are mutually exclusive' >&2; return 2
    fi
    if [ -n "${INSTANCE_ROOT:-}" ]; then
        available="$(df --output=avail -B1 / | tail -1)"
        [ "$available" -ge 21474836480 ] || { echo 'Root disk below 20 GiB safety floor' >&2; return 1; }
        bulk_available="$(df --output=avail -B1 "$INSTANCE_ROOT" | tail -1)"
        [ "$bulk_available" -ge 53687091200 ] || { echo 'Output disk below 50 GiB free' >&2; return 1; }
    fi

    if [ ! -x "$ISAACSIM_BIN" ]; then
        echo "ERROR: ISAACSIM_BIN not executable: $ISAACSIM_BIN" >&2
        return 1
    fi
    if [ ! -d "$EXT_FOLDER/$ext" ]; then
        echo "ERROR: extension not found: $EXT_FOLDER/$ext" >&2
        return 1
    fi
    if [ ! -d "$KIT_MCP_ROOT/exts/omni.kit.mcp" ]; then
        echo "ERROR: omni-kit-mcp missing at $KIT_MCP_ROOT; run: git submodule update --init omni-kit-mcp" >&2
        return 1
    fi
    if pid="$(our_pid)"; then
        echo "Isaac Sim already running (pid $pid)"
        do_wait
        return $?
    fi
    if client status --timeout 3 > /dev/null 2>&1; then
        echo "ERROR: something already answers on bridge port $MCP_PORT (not started by this script)." >&2
        echo "       Stop it, or set OMNI_KIT_MCP_PORT to a free port." >&2
        return 1
    fi

    # GPU pin: Kit's renderer and PhysX take the index; CUDA_VISIBLE_DEVICES (by UUID) keeps
    # Warp/torch contexts off the other card, which a Kit argument alone does not.
    local gpu_uuid
    gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null | head -1)"
    if [ -z "$gpu_uuid" ]; then
        echo "ERROR: no GPU at index $GPU (nvidia-smi)" >&2
        return 1
    fi

    local args=()
    if [ "$stream" = 1 ]; then
        local ip="${ISAACSIM_STREAM_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
        if [ -z "$ip" ]; then
            echo "ERROR: --stream needs a public address; set ISAACSIM_STREAM_IP" >&2
            return 1
        fi
        if stream_listening; then
            echo "ERROR: a stream already listens on :$STREAM_SIGNAL_PORT" >&2
            return 1
        fi
        # isaacsim.exp.full.streaming = full app + omni.kit.livestream.app (WebRTC, signal 49100,
        # media 47998). publicIp is required for a remote client: without it the SDP offer
        # carries no usable address.
        args+=(isaacsim.exp.full.streaming --no-window
               "--/exts/omni.kit.livestream.app/primaryStream/publicIp=$ip")
    elif [ "$headless" = 1 ]; then
        args+=(isaacsim.exp.full --no-window
               --disable omni.kit.livestream.app
               --disable omni.kit.livestream.webrtc
               --disable omni.kit.livestream.native)
    else
        args+=(isaacsim.exp.full)
        export DISPLAY="${DISPLAY:-:0}"
    fi
    physics_gpu=0
    [ -z "${INSTANCE_ROOT:-}" ] || physics_gpu="$GPU"
    args+=(--ext-folder "$EXT_FOLDER"
           --ext-folder "$KIT_MCP_ROOT/exts"
           --enable omni.kit.mcp
           --enable "$ext"
           --/renderer/multiGpu/enabled=false "--/renderer/activeGpu=$GPU" "--/physics/cudaDevice=$physics_gpu"
           # isaacsim.exp.full turns on the async-rendering toggle (isaacsim.core.throttling); with it,
           # reset()/play() can deadlock compiling an MDL material. Ported from isaac-sim-mcp
           # 83dcf46 / e2a2247 (ur5e-dt, aic-dt launchers).
           --/exts/isaacsim.core.throttling/enable_async=false
           # syncToPresent bypasses the main loop limiter in Kit 110.3.
           --/app/runLoopsGlobal/syncToPresent=false
           --/app/runLoops/main/syncToPresent=false
           --/app/runLoops/main/rateLimitUseBusyLoop=false
           # Never save this session's /persistent settings into the shared install's config.
           --/app/player/useFixedTimeStepping=true
           --/exts/omni.kit.renderer.core/present/enabled=false
           --/app/renderer/skipWhileMinimized=false
           --/app/renderer/sleepMsOnFocus=0
           --/app/renderer/sleepMsOutOfFocus=0
           --/persistent/omni/replicator/captureOnPlay=1
           --/app/settings/persistent=false
           "--/soarm101/rateHz=$RATE_HZ")
    [ "$headless" = 1 ] || args+=(--enable omni.kit.mcp.panel)
    if [ -n "${INSTANCE_ROOT:-}" ]; then
        args+=(--portable-root "$INSTANCE_ROOT/portable"
               "--/exts/omni.services.transport.server.http/port=$((8778 + 2 * INSTANCE_INDEX))"
               --/exts/omni.services.transport.server.http/allow_port_range=false
               "--/persistent/physics/numThreads=${SOARM_PHYSICS_THREADS:-4}"
               --/plugins/carb.tasking.plugin/threadCount=6
               --/exts/omni.replicator.srtx/enabled=false
               "--/rtx/shaderDb/shaderCachePath=$HOME/.cache/ov/shaders")
    fi
    # shellcheck disable=SC2206
    [ -n "${ISAACSIM_EXTRA_ARGS:-}" ] && args+=(${ISAACSIM_EXTRA_ARGS})

    echo "Launching Isaac Sim 6.1: $ext ($([ "$stream" = 1 ] && echo "WebRTC stream on $ip:$STREAM_SIGNAL_PORT" || { [ "$headless" = 1 ] && echo "headless, no stream" || echo "local window"; }))"
    echo "  Bin:    $ISAACSIM_BIN"
    echo "  GPU:    $GPU ($gpu_uuid)"
    echo "  Bridge: localhost:$MCP_PORT, tools $TOOLS_MODULE"
    echo "  Log:    $LOG"
    (
        # Isaac-only environment, scoped to this launch.
        unset ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH \
              COLCON_PREFIX_PATH ROS_PACKAGE_PATH ROS_LOCALHOST_ONLY
        venv="$(dirname "$(dirname "$ISAACSIM_BIN")")"
        # Isaac's bundled ROS 2 Humble (same distro as the host's /opt/ros/humble) over
        # CycloneDDS. rcutils dlopens the RMW through LD_LIBRARY_PATH, so the bundled lib dir
        # must be on it before start (NVIDIA's documented setup for the internal libraries).
        ros_lib="$(ls -d "$venv"/lib/python3.*/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib 2>/dev/null | head -1)"
        ld="$(strip_ros_paths "${LD_LIBRARY_PATH:-}")"
        export LD_LIBRARY_PATH="${ld:+$ld:}$ros_lib"
        export PYTHONPATH="$(strip_ros_paths "${PYTHONPATH:-}")"
        [ -z "$PYTHONPATH" ] && unset PYTHONPATH
        export ROS_DISTRO=humble
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export CYCLONEDDS_URI="file://$CYCLONEDDS_XML"
        export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
        if [ -n "${INSTANCE_ROOT:-}" ]; then
            unset CUDA_VISIBLE_DEVICES
        else
            export CUDA_VISIBLE_DEVICES="$gpu_uuid"
        fi
        export OMNI_KIT_ACCEPT_EULA=YES
        export PYTHONDONTWRITEBYTECODE=1
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        export OMNI_KIT_MCP_PORT="$MCP_PORT"
        export KIT_MCP_TOOL_PATHS="$TOOLS_PATH"
        export KIT_MCP_TOOL_MODULES="$TOOLS_MODULE"
        # Runtime outputs (scene_states/, videos/) go outside the source tree, not Isaac's cwd.
        default_output=/tmp/soarm101-resources
        [ -n "${INSTANCE_ROOT:-}" ] && default_output="$INSTANCE_ROOT/resources"
        if [ -n "${INSTANCE_ROOT:-}" ]; then
            # Indexed instances never inherit another session's output path.
            export MCP_CLIENT_OUTPUT_DIR="$default_output"
            python3 "$SCRIPT_DIR/../collect/storage.py" "$MCP_CLIENT_OUTPUT_DIR" || exit 2
        else
            export MCP_CLIENT_OUTPUT_DIR="${MCP_CLIENT_OUTPUT_DIR:-$default_output}"
        fi
        mkdir -p "$MCP_CLIENT_OUTPUT_DIR"
        # The ROS 2 bridge's dlopens exhaust glibc's static-TLS surplus on top of Isaac+torch, and
        # torch needs the venv's nvJitLink resolved first. ld.so reads both before Python starts,
        # so they live here, on this launch only.
        export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.optional_static_tls=2097152"
        nvjitlink="$(ls "$venv"/lib/python3.*/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 2>/dev/null | head -1)"
        [ -f "$nvjitlink" ] && export LD_PRELOAD="$nvjitlink${LD_PRELOAD:+:$LD_PRELOAD}"
        launch_nice="${SOARM_SIM_NICE:-0}"
        [ -n "${INSTANCE_ROOT:-}" ] && launch_nice="${SOARM_SIM_NICE:-10}"
        python3 "$SCRIPT_DIR/../collect/managed_process.py" start --pidfile "$PIDFILE" --log "$LOG" -- \
            nice -n "$launch_nice" "$ISAACSIM_BIN" "${args[@]}" || exit 1
    ) || return 1
    echo "  PID:    $(our_pid)"
    sleep 3
    if ! our_pid > /dev/null; then
        echo "ERROR: exited immediately; last log lines:"
        tail -15 "$LOG" | sed 's/^/  /'
        return 1
    fi
    do_wait || return $?      # 3 = GPU wedge: reported by do_wait, Isaac left running
    if [ "$stream" = 1 ]; then
        local i
        for i in $(seq 1 30); do stream_listening && break; sleep 2; done
        if stream_listening; then
            echo "Stream: connect the WebRTC Streaming Client to $ip (signal :$STREAM_SIGNAL_PORT)"
        else
            echo "WARNING: bridge is up but nothing listens on :$STREAM_SIGNAL_PORT; see $LOG"
        fi
    fi
}

do_close() {
    claim_launcher_operation
    python3 "$SCRIPT_DIR/../collect/managed_process.py" stop --pidfile "$PIDFILE"
}

do_status() {
    local pid
    if pid="$(our_pid)"; then
        echo "Isaac Sim: RUNNING (pid $pid, log $LOG)"
    else
        echo "Isaac Sim: NOT RUNNING (no live process from this script)"
    fi
    if bridge_ready; then
        echo "  omni-kit-mcp bridge :$MCP_PORT: READY (soarm101.* answering)"
    elif client status --timeout 3 > /dev/null 2>&1; then
        echo "  omni-kit-mcp bridge :$MCP_PORT: UP, soarm101 extension not answering yet"
    else
        echo "  omni-kit-mcp bridge :$MCP_PORT: not answering (down, or Kit's main loop is busy)"
    fi
    stream_listening && echo "  WebRTC stream: listening on :$STREAM_SIGNAL_PORT"
    local kl last age
    if kl="$(kit_log)" && [ -n "$kl" ] && last="$(rtpso_last)"; then
        age=$(( $(date +%s) - $(date -r "$kl" +%s) ))
        [ "$age" -lt 15 ] && grep -q "Waiting for RtPso" <(tail -3 "$kl") \
            && echo "  RTX shader compile in progress (${last}s so far; first run only, main loop busy)"
    fi
    [ -n "${kl:-}" ] && echo "  Kit log: $kl"
    [ -n "${pid:-}" ]
}

case "${1:-help}" in
    launch)  shift; do_launch "$@" ;;
    collection-rate)
        fps="${2:?collection-rate requires 20 or 30}"
        params="$(python3 - "$REPO_ROOT" "$fps" "${3:-collect}" <<'PYCODE'
import json, pathlib, sys
source = (pathlib.Path(sys.argv[1]) / 'linux-env/scripts/collect/configure_clock.py').read_text()
print(json.dumps({'code': source + '\nresult = configure(' + repr(int(sys.argv[2])) + ', preset=' + repr(sys.argv[3]) + ')'}))
PYCODE
)"
        client run_python "$params" --timeout 30 ;;
    rate)    do_rate "${2:-$RATE_HZ}" ;;
    wait)    do_wait "${2:-$WEDGE_S}" ;;
    quick-start) do_quick_start "${2:-}" ;;
    status)  do_status ;;
    close)   do_close 20 ;;
    kill)    do_close 0 ;;
    restart) shift; claim_launcher_operation; do_close 20 && do_launch "$@" ;;
    *)
        sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
