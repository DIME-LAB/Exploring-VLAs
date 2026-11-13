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
# `|| true`: grep -v exits 1 when EVERY entry is a ROS path (any shell that
# sourced only /opt/ros/humble, e.g. deploy_sim.sh or control_gui). Under
# `set -euo pipefail` that silently killed this script before its first output
# line (empty log, no process).
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | { grep -vE '/(opt/ros|ros2_ws|Desktop/ros2_ws)' || true; } \
        | paste -sd: -)"
    if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
fi

# GPU — the policy runs on SOARM_POLICY_GPU (nvidia-smi index, default 1) so a
# server started from here, control_gui's Inference tab or deploy_sim.sh never
# lands on Isaac's GPU 0. Pinned by UUID: CUDA's default device order need not
# match nvidia-smi's. An explicit CUDA_VISIBLE_DEVICES wins.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${SOARM_POLICY_GPU:-1}" 2>/dev/null | head -1 || true)"
    if [[ "$_gpu_uuid" == GPU-* ]]; then   # nvidia-smi prints errors on stdout
        export CUDA_VISIBLE_DEVICES="$_gpu_uuid"
    else
        echo "[$(basename "$0")] WARNING: no GPU at index ${SOARM_POLICY_GPU:-1}; leaving CUDA_VISIBLE_DEVICES unset (set SOARM_POLICY_GPU)" >&2
    fi
fi
echo "[$(basename "$0")] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}" >&2

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
