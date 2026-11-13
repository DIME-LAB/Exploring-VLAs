#!/usr/bin/env bash
# Isolated headless ROS collection: collect_sim.sh up|run|export|deploy-clock|down INDEX [args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
command="${1:?up|run|export|deploy-clock|down}"; shift
index="${1:?instance index 0..9}"; shift
[[ "$index" =~ ^[0-9]$ ]] || { echo 'Instance index must be 0..9' >&2; exit 2; }
export SOARM_INSTANCE_INDEX="$index"
export ROS_DOMAIN_ID="$((90 + index))"
export SOARM_KIT_MCP_PORT="$((8777 + 2 * index))"
export OMNI_KIT_MCP_PORT="$SOARM_KIT_MCP_PORT"
unset ROS_LOCALHOST_ONLY
export CYCLONEDDS_URI="file://$REPO_ROOT/linux-env/cyclonedds.xml"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SOARM_COLLECTION_ROOT="${SOARM_COLLECTION_ROOT:-/mnt/a4500-backup/exploring-vlas}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SOARM_COLLECTION_ROOT/datasets}"
export SOARM_INSTANCE_ROOT="${SOARM_INSTANCE_ROOT:-$SOARM_COLLECTION_ROOT/runtime/instance-$index}"
mkdir -p "$SOARM_INSTANCE_ROOT" "$HF_LEROBOT_HOME"
launcher="$SCRIPT_DIR/isaac/isaacsim_launch.sh"
case "$command" in
    up)
        bash "$launcher" launch --headless --instance "$index" --rate-hz 0
        scene_live="$(PYTHONPATH="$REPO_ROOT/omni-kit-mcp" python3 -m kit_mcp.client run_python \
            '{"code":"import omni.usd\nresult=bool(omni.usd.get_context().get_stage().GetPrimAtPath(\"/World/SO_ARM101\").IsValid())"}' \
            --port "$SOARM_KIT_MCP_PORT" --timeout 30)"
        if ! python3 -c 'import json,sys; sys.exit(not json.load(sys.stdin).get("result"))' <<< "$scene_live"; then
            # Construct the singleton before quick_start: later World() calls preserve its dt.
            PYTHONPATH="$REPO_ROOT/omni-kit-mcp" python3 - "$SOARM_KIT_MCP_PORT" <<'PYWORLD'
import json, sys
from kit_mcp.client import call
code = "from isaacsim.core.api import World\nfrom so_arm101_dt.extension import get_instance\nget_instance()._min_frame_rate=30\nassert World.instance() is None, 'A fresh World is required'\nw=World(physics_dt=1/120, rendering_dt=1/30)\nresult={'physics_dt': 1/120, 'rendering_dt': 1/30}"
print(json.dumps(call('run_python', {'code': code}, port=int(sys.argv[1]), timeout=30)))
PYWORLD
            PYTHONPATH="$REPO_ROOT/omni-kit-mcp" python3 -m kit_mcp.client soarm101.quick_start \
                --port "$SOARM_KIT_MCP_PORT" --timeout 900
        fi
        bash "$launcher" collection-rate 30 --instance "$index"
        set +u
        source /opt/ros/humble/setup.bash
        source "$REPO_ROOT/vla_SO-ARM101/install/setup.bash"
        set -u
        pidfile="$SOARM_INSTANCE_ROOT/control.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            echo 'This instance already has a control stack'; exit 1
        fi
        python3 - "$REPO_ROOT" "$SOARM_INSTANCE_ROOT/controllers.yaml" <<'PYCONFIG'
from pathlib import Path
import sys, yaml
source = Path(sys.argv[1]) / 'vla_SO-ARM101/src/so_arm101_moveit_config/config/ros2_controllers.yaml'
config = yaml.safe_load(source.read_text())
config['controller_manager']['ros__parameters']['update_rate'] = 120
Path(sys.argv[2]).write_text(yaml.safe_dump(config))
PYCONFIG
        setsid ros2 launch so_arm101_control control.launch.py gui:=false rviz:=false use_sim:=true use_sim_time:=true controllers_config:="$SOARM_INSTANCE_ROOT/controllers.yaml" \
            > "$SOARM_INSTANCE_ROOT/control.log" 2>&1 < /dev/null &
        echo "$!" > "$pidfile"
        echo "Instance $index ready: domain=$ROS_DOMAIN_ID bridge=$SOARM_KIT_MCP_PORT"
        ;;
    run)
        set +u
        source /opt/ros/humble/setup.bash
        source "$REPO_ROOT/vla_SO-ARM101/install/setup.bash"
        set -u
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        exec python3 -u "$SCRIPT_DIR/collect/collect.py" --instance "$index" "$@"
        ;;
    export)
        unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
        unset LD_LIBRARY_PATH
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        exec "$HOME/.pixi/bin/pixi" run --manifest-path "$REPO_ROOT/linux-env/pixi.toml" \
            python "$SCRIPT_DIR/collect/export.py" "$@"
        ;;
    deploy-clock)
        bash "$launcher" collection-rate 30 deploy --instance "$index"
        ;;
    down)
        pidfile="$SOARM_INSTANCE_ROOT/control.pid"
        if [ -f "$pidfile" ]; then
            control_pid="$(cat "$pidfile")"
            if kill -0 "$control_pid" 2>/dev/null; then
                kill -INT -- "-$control_pid"
            fi
        fi
        bash "$launcher" close --instance "$index"
        ;;
    *) echo 'Expected up|run|export|deploy-clock|down' >&2; exit 2 ;;
esac
