# shellcheck shell=bash
# _lib.sh — shared paths + env + bring-up steps for the Linux stack.
#
# Sourced by stack_*.sh, deploy_sim.sh, bootstrap.sh, and any other linux-env
# script that needs canonical paths. Mirrors mac-env/scripts/_lib.sh but for the
# Isaac-Sim-backed Linux topology (system Humble producer side + pixi-Jazzy
# lerobot consumer).
#
# Resolves REPO_ROOT from this file's location, so colleagues cloning to
# arbitrary paths get correct values without editing anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LINUX_ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT="$(cd "$LINUX_ENV_DIR/.." && pwd)"
# control_gui resolves helper scripts (sim_reset.sh, record/inference wrappers)
# and the omni-kit-mcp submodule from this. Force-set so a stale value from the
# user's shell can't point the GUI at a different checkout.
export SOARM_REPO_ROOT="$REPO_ROOT"

# Repo subtrees (some are submodules — bootstrap.sh inits them)
export VLA_PKG="$REPO_ROOT/vla_SO-ARM101"
export LEROBOT_DIR="$REPO_ROOT/lerobot"
export ISAAC_EXTS="$REPO_ROOT/isaac/exts"           # vendored soarm101-dt extension
export KIT_MCP_DIR="$REPO_ROOT/omni-kit-mcp"        # omni-kit-mcp submodule (bridge + client)

# System ROS2 (producer side: Isaac Sim, control_gui, MoveIt, mirror node)
export ROS2_DISTRO="${ROS2_DISTRO:-humble}"
export ROS2_SETUP="${ROS2_SETUP:-/opt/ros/$ROS2_DISTRO/setup.bash}"

# Colcon workspace lives in-tree on Linux (no spaceless-/tmp/ workaround needed —
# Linux paths are spaceless by convention, and in-tree keeps build artefacts
# tied to the source for symlink-install to resolve correctly).
export SOARM_WS="$VLA_PKG"
export SOARM_WS_SETUP="$SOARM_WS/install/setup.bash"

# Pixi env for the lerobot consumer side (Python 3.12 + RoboStack-Jazzy + lerobot deps).
# linux-env/pixi.toml is materialized in-tree under linux-env/.pixi/.
export PIXI_MANIFEST="$LINUX_ENV_DIR/pixi.toml"

# Cross-Python DDS discovery (multicast on lo — system default disables this,
# which prevents Jazzy consumers from seeing Humble producers across process trees).
# Force-set CYCLONEDDS_URI: any pre-existing value in the user's shell pointing
# at a different cyclonedds.xml will partition the discovery graph and make
# stack_status.sh report "node missing" even when nodes are healthy.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$LINUX_ENV_DIR/cyclonedds.xml"

# Isaac Sim + the omni-kit-mcp bridge. The soarm101-dt scene tools are served on
# the bridge as `soarm101.<name>`. SOARM_KIT_MCP_HOST/PORT are the one knob for
# every caller (these scripts, control_gui, sim_reset.sh); OMNI_KIT_MCP_PORT is
# what the launcher's Kit process binds, so the two always agree.
export ISAAC_LAUNCHER="$LINUX_ENV_DIR/scripts/isaac/isaacsim_launch.sh"
export ISAAC_EXT_ID="${ISAAC_EXT_ID:-soarm101-dt}"
export SOARM_KIT_MCP_HOST="${SOARM_KIT_MCP_HOST:-127.0.0.1}"
export SOARM_KIT_MCP_PORT="${SOARM_KIT_MCP_PORT:-8767}"
export OMNI_KIT_MCP_PORT="$SOARM_KIT_MCP_PORT"
KIT_MCP_HOST="$SOARM_KIT_MCP_HOST"
KIT_MCP_PORT="$SOARM_KIT_MCP_PORT"

# Logs (single dir, prefixed names — easy to clean up with one rm)
export LOG_DIR="${LOG_DIR:-/tmp}"
export STACK_PIDFILE="${STACK_PIDFILE:-$LOG_DIR/soarm_linux_stack.pid}"

# ---------------------------------------------------------------------------
# Helpers used by stack_*.sh and deploy_sim.sh
# ---------------------------------------------------------------------------

# Verify the prerequisites a colleague needs before stack_start can succeed.
# Prints helpful errors pointing at bootstrap.sh when something is missing.
stack_preflight() {
    local missing=0
    if [ ! -f "$ROS2_SETUP" ]; then
        echo "ERROR: ROS2 setup not found at $ROS2_SETUP" >&2
        echo "       Install ros-$ROS2_DISTRO-desktop or set ROS2_SETUP env var." >&2
        missing=1
    fi
    if [ ! -f "$KIT_MCP_DIR/kit_mcp/client.py" ]; then
        echo "ERROR: omni-kit-mcp submodule not initialized at $KIT_MCP_DIR" >&2
        echo "       Run from $REPO_ROOT: git submodule update --init --recursive" >&2
        missing=1
    fi
    if [ ! -d "$ISAAC_EXTS/$ISAAC_EXT_ID" ]; then
        echo "ERROR: Isaac extension $ISAAC_EXT_ID not found under $ISAAC_EXTS" >&2
        missing=1
    fi
    if [ ! -f "$SOARM_WS_SETUP" ]; then
        echo "ERROR: ROS2 control workspace not built at $SOARM_WS_SETUP" >&2
        echo "       Run: bash $LINUX_ENV_DIR/scripts/bootstrap.sh" >&2
        missing=1
    fi
    if [ ! -x "$ISAAC_LAUNCHER" ]; then
        echo "ERROR: isaacsim_launch.sh not executable at $ISAAC_LAUNCHER" >&2
        echo "       chmod +x $ISAAC_LAUNCHER" >&2
        missing=1
    fi
    return $missing
}

# Source ROS2 + the colcon workspace into the current shell.
stack_source_ros() {
    # shellcheck disable=SC1090
    source "$ROS2_SETUP"
    # shellcheck disable=SC1090
    [ -f "$SOARM_WS_SETUP" ] && source "$SOARM_WS_SETUP"
}

# Count linux-stack processes (useful for stack_status / stop verification).
stack_running_count() {
    pgrep -f "ros2.*launch.*control.launch|move_group|control_gui|rviz2|isaacsim|joint_states_to_commands|robot_state_publisher" 2>/dev/null | wc -l | tr -d ' '
}

# kit_mcp_call <tool> [json-params] [timeout_s]
# One call to the omni-kit-mcp bridge via its stdlib-only reference client.
# Prints the JSON payload. Exit: 0 ok · 1 the tool/bridge reported an error ·
# 2 bridge unreachable (or timed out).
kit_mcp_call() {
    local tool="$1" params="${2:-}" t="${3:-120}"
    [ -n "$params" ] || params='{}'
    PYTHONPATH="$KIT_MCP_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m kit_mcp.client \
        "$tool" "$params" --host "$KIT_MCP_HOST" --port "$KIT_MCP_PORT" --timeout "$t"
}

# True when the bridge answers at all (exit 1 = answered with an error, e.g.
# busy running quick_start — still up).
kit_mcp_ready() {
    local rc=0
    kit_mcp_call status '{}' 5 >/dev/null 2>&1 || rc=$?
    [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]
}

# True when the SO-ARM101 scene is already built and the timeline is playing
# (so a re-run of `up`/stack_start doesn't rebuild a scene someone is using).
scene_is_live() {
    local out
    out="$(kit_mcp_call run_python '{"code": "import omni.usd, omni.timeline\nst = omni.usd.get_context().get_stage()\nresult = bool(st and st.GetPrimAtPath(\"/World/SO_ARM101\").IsValid() and omni.timeline.get_timeline_interface().is_playing())"}' 15 2>/dev/null)" || return 1
    printf '%s' "$out" | grep -q '"result": true'
}

# isaac_up [--stream]   Launch Isaac Sim + soarm101-dt and block until the
# soarm101.* tools answer on the bridge. Sets ISAAC_STARTED=1 when this call
# launched Isaac (even if it then failed to come up, so the caller can still
# clean it up), 0 when a bridge was already answering on $KIT_MCP_PORT.
isaac_up() {
    ISAAC_STARTED=0
    # Already up: the bridge answers, or the launcher's Isaac is alive but busy
    # (e.g. mid RTX shader compile, when Kit's main loop serves no requests).
    if kit_mcp_ready || bash "$ISAAC_LAUNCHER" status >/dev/null 2>&1; then
        echo "  Isaac already running (bridge $KIT_MCP_HOST:$KIT_MCP_PORT) — skipping launch"
        [ "${1:-}" = "--stream" ] && echo "  NOTE: --stream only applies to a fresh launch (stop Isaac first to relaunch streaming)"
        # soarm101.* registered too? Launcher-default timeout: it knows about the
        # first-launch RTX shader compile (minutes) and prints its progress.
        bash "$ISAAC_LAUNCHER" wait || return 1
        return 0
    fi
    ISAAC_STARTED=1
    # `launch` blocks until the soarm101.* tools answer (or fails with the log tail).
    bash "$ISAAC_LAUNCHER" launch "$ISAAC_EXT_ID" "$@" || return 1
}

# scene_up   Build the scene (soarm101.quick_start) unless it's already live,
# then wait for /clock. Needs ROS sourced (stack_source_ros).
scene_up() {
    local out rc=0 wrc=0
    if scene_is_live; then
        echo "  Scene already built and playing — skipping quick_start"
    else
        # Warm runs take ~13 s. A first run on a new GPU/driver/Isaac version
        # compiles RTX shaders during quick_start and Kit answers nothing until
        # it's done. The launcher's watched `quick-start` keeps waiting while Kit
        # makes progress (up to SOARM_QUICK_START_TIMEOUT, default 900 s) and
        # exits 3 within ~ISAACSIM_WEDGE_S when Kit goes silent (GPU wedge).
        # If the call still gets no answer (exit 2), the launcher's `wait`
        # decides. One attempt, no retry.
        echo "  (first run on this GPU/driver: RTX shader compile can add minutes — don't interrupt)"
        out="$(ISAACSIM_WAIT_CAP="${SOARM_QUICK_START_TIMEOUT:-900}" bash "$ISAAC_LAUNCHER" quick-start '{}')" || rc=$?
        if [ "$rc" -eq 3 ]; then
            echo "  Scene FAILED: GPU wedge during quick_start (see the launcher message above)." >&2
            return 1
        elif [ "$rc" -eq 2 ]; then
            echo "  quick_start got no answer; checking Isaac (launcher wait: first-run compile vs GPU wedge)..."
            bash "$ISAAC_LAUNCHER" wait || wrc=$?
            if [ "$wrc" -eq 3 ]; then
                echo "  Scene FAILED: GPU wedge (see the launcher message above)." >&2
                return 1
            elif [ "$wrc" -ne 0 ]; then
                echo "  Scene FAILED: Isaac did not become ready (launcher wait exit $wrc) — see ${ISAACSIM_LOG:-/tmp/isaacsim-soarm101.log}" >&2
                return 1
            elif scene_is_live; then
                rc=0
                out='  "message": "scene live after wait"'
            else
                echo "  Scene FAILED: Isaac answers again but the scene is not live — see ${ISAACSIM_LOG:-/tmp/isaacsim-soarm101.log}" >&2
                return 1
            fi
        fi
        if [ "$rc" -ne 0 ] || printf '%s' "$out" | grep -q '"status": "error"'; then
            echo "  quick_start FAILED (exit $rc):" >&2
            printf '%s\n' "$out" | head -20 | sed 's/^/    /' >&2
            return 1
        fi
        printf '%s\n' "$out" | grep -m1 '"message"' | sed 's/^ */  /' || true
    fi
    # World initialization resets Kit pacing. Apply and read it after scene creation,
    # including an idempotent up against an already-live scene.
    bash "$ISAAC_LAUNCHER" rate || return $?
    echo "  Waiting for /clock to tick..."
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if timeout 2 ros2 topic echo /clock --once 2>/dev/null | grep -q "sec:"; then
            echo "  /clock alive"
            return 0
        fi
        sleep 3
    done
    echo "  WARNING: /clock not ticking after 30s — sim may not be playing"
}

# stack_detect_display   When DISPLAY is unset (an ssh shell), point at the
# box's X session: the first /tmp/.X11-unix/X<n> socket, plus GDM's Xauthority
# when present. An existing DISPLAY / XAUTHORITY always wins. Detection only;
# stack_require_display decides whether the display is usable.
stack_detect_display() {
    local sock
    if [ -z "${DISPLAY:-}" ]; then
        for sock in /tmp/.X11-unix/X*; do
            [ -S "$sock" ] || continue
            export DISPLAY=":${sock##*/X}"
            break
        done
    fi
    [ -n "${DISPLAY:-}" ] && export DISPLAY      # a plain shell var is invisible to X clients
    if [ -n "${DISPLAY:-}" ] && [ -z "${XAUTHORITY:-}" ] && \
            [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    fi
    return 0
}

# stack_require_display <what>   For windows we start on purpose (control_gui,
# RViz): detect the display, then check it is usable (xdpyinfo). After a reboot
# with nobody logged in, :0 is only the GDM login screen and isn't. There is
# deliberately no virtual-display fallback: fail with the fix instead.
stack_require_display() {
    local what="$1"
    stack_detect_display
    if [ -z "${DISPLAY:-}" ]; then
        echo "  ERROR: $what needs an X display and there is none (no /tmp/.X11-unix/X*)." >&2
    elif ! command -v xdpyinfo >/dev/null 2>&1; then
        echo "  ERROR: can't check DISPLAY=$DISPLAY for $what: xdpyinfo missing (sudo apt install x11-utils)." >&2
        return 1
    elif ! xdpyinfo >/dev/null 2>&1; then
        echo "  ERROR: $what needs an X display, but DISPLAY=$DISPLAY${XAUTHORITY:+ (XAUTHORITY=$XAUTHORITY)} isn't usable" >&2
        echo "         (nobody logged in to the box's desktop, or no permission for that session)." >&2
    else
        echo "  Display: $DISPLAY${XAUTHORITY:+ (XAUTHORITY=$XAUTHORITY)}"
        return 0
    fi
    echo "         Fix: log in to the box's desktop, or export DISPLAY (and XAUTHORITY) for a session you own." >&2
    echo "         Or run without windows: deploy_sim.sh up --stream (control_gui and RViz off)." >&2
    return 1
}

# controllers_active   True when the controller manager reports
# joint_state_broadcaster, arm_controller and gripper_controller all `active`.
controllers_active() {
    local out c
    out="$(timeout 15 ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')" || return 1
    for c in joint_state_broadcaster arm_controller gripper_controller; do
        printf '%s\n' "$out" | grep -qE "^${c}[[:space:]].*[[:space:]]active([[:space:]]|\$)" || return 1
    done
}

# control_stack_up <rviz:true|false> [gui:true|false]   Launch control.launch.py
# (ros2_control + MoveIt + pose publishers, + control_gui / RViz when asked)
# unless a controller manager is already up, and block until its controllers
# are active (and control_gui registered, when asked). Windows need a usable
# display (stack_require_display). Sets CONTROL_STARTED=1 and writes
# $STACK_PIDFILE (the `ros2 launch` PID) when this call launched it. Needs ROS
# sourced.
control_stack_up() {
    local rviz="${1:-true}" gui="${2:-true}" log pid i nodes windows=""
    CONTROL_STARTED=0
    nodes="$(ros2 node list --no-daemon 2>/dev/null)"
    if printf '%s\n' "$nodes" | grep -qx /controller_manager; then
        if [ "$gui" = true ] && ! printf '%s\n' "$nodes" | grep -qx /so_arm101_control_gui; then
            echo "  ERROR: a control stack is already running without control_gui." >&2
            echo "         Stop it first (deploy_sim.sh down, or stack_stop.sh), then start again with the GUI." >&2
            return 1
        fi
        echo "  control stack already running — skipping"
        return 0
    fi
    # Windows are opt-in; when asked for, the display must be usable (no fallback).
    [ "$gui" = true ] && windows="control_gui"
    [ "$rviz" = true ] && windows="${windows:+$windows + }RViz"
    if [ -n "$windows" ]; then
        stack_require_display "$windows" || return 1
    fi
    log="$LOG_DIR/control_stack_$(date +%H%M%S).log"
    nohup ros2 launch so_arm101_control control.launch.py rviz:="$rviz" gui:="$gui" \
        > "$log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$STACK_PIDFILE"
    disown
    CONTROL_STARTED=1
    CONTROL_LOG="$log"
    echo "  Log: $log"
    # Ready = the three controllers active (what the policy and MoveIt need),
    # plus control_gui registered when it was asked for. Spawners wait up to
    # 60 s for the controller manager, hence 90 s here.
    for i in $(seq 1 30); do
        if controllers_active && { [ "$gui" != true ] || \
                ros2 node list --no-daemon 2>/dev/null | grep -qx /so_arm101_control_gui; }; then
            echo "  controllers active${windows:+, $windows up} after ~$((i*3))s"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  control stack exited — last lines of $log:" >&2
            tail -20 "$log" >&2
            return 1
        fi
        sleep 3
    done
    echo "  TIMEOUT 90s waiting for controllers${windows:+ / $windows} — check $log" >&2
    tail -20 "$log" >&2
    return 1
}

# stop_pid <pid> <name> [grace_s]   SIGINT (lets ros2 launch / lerobot shut
# children down cleanly), then SIGTERM after the grace period. Never SIGKILL:
# these processes own X11 windows (see top-level CLAUDE.md "Linux gotchas").
stop_pid() {
    local pid="$1" name="$2" grace="${3:-15}" i
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "  $name: not running"
        return 0
    fi
    kill -INT "$pid" 2>/dev/null || true
    for i in $(seq 1 "$grace"); do
        kill -0 "$pid" 2>/dev/null || { echo "  $name: stopped (${i}s)"; return 0; }
        sleep 1
    done
    echo "  $name: still running after ${grace}s — sending SIGTERM"
    kill -TERM "$pid" 2>/dev/null || true
}
