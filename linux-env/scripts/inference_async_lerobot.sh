#!/usr/bin/env bash
# inference_async_lerobot.sh — wrapper around run_async_inference_sim.py.
#
# Same env-scrubbing pattern as inference_smolvla.sh (avoid PYTHONPATH leak
# from system Humble Python 3.10 into pixi-Jazzy Python 3.12).
#
# Usage:
#   bash inference_async_lerobot.sh \
#     --pretrained=anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200 \
#     --task='Pick a blue lego and place it in blue cup'
#
# Dry run (print commands, don't launch):
#   bash inference_async_lerobot.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/.." && pwd)"

# Same scrub as smolvla_inference.sh — see comments there for why.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROSCONSOLE_FORMAT \
      ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -vE '/(opt/ros|ros2_ws|Desktop/ros2_ws)' \
        | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

# Cyclone DDS — multicast on lo so the pixi-Jazzy client discovers Isaac
# Sim's Humble-side topics (joint_states + cameras + arm_controller actions).
export CYCLONEDDS_URI="file://$ENV_DIR/cyclonedds.xml"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# Idempotent dep top-up. Async inference needs lerobot[async] (grpcio) on
# top of [smolvla]. Distinct marker from smolvla_inference.sh's so the
# initial install of [async] happens even on a system where the smolvla
# marker already exists.
DEP_MARKER="$ENV_DIR/.pixi/.async_inference_deps_installed"
if [ ! -f "$DEP_MARKER" ]; then
    echo "[inference_async] installing SmolVLA + async + peft extras into pixi env (one-time)…" >&2
    pixi run --manifest-path "$ENV_DIR/pixi.toml" pip install \
        -e "$REPO_ROOT/lerobot[smolvla,async]" \
        "peft>=0.18.0,<1.0.0" \
        || { echo "[inference_async] dep install failed" >&2; exit 1; }
    mkdir -p "$(dirname "$DEP_MARKER")"
    touch "$DEP_MARKER"
fi

exec pixi run --manifest-path "$ENV_DIR/pixi.toml" \
    python3 -u "$SCRIPT_DIR/run_async_inference_sim.py" "$@"
