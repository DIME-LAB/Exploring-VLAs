#!/usr/bin/env bash
# inference_smolvla.sh — wrapper around smolvla_inference.py.
#
# Mirrors record_sim_isaac.sh's env setup:
#   - sources cyclonedds.xml so cross-Python DDS discovery works
#     (system Humble producers ↔ pixi Jazzy consumer here)
#   - runs the Python entry inside the pixi-Jazzy env
#
# First-run idempotent install: SmolVLA + LoRA need lerobot[smolvla] and
# peft. The bootstrap installs lerobot with [viz,dataset,feetech] only, so
# we top up here. Re-running is a no-op.
#
# Usage:
#   bash inference_smolvla.sh \
#     --model.path=anirudhrani/smolvla_blue_sort_ven_50k \
#     --model.checkpoint=050000 \
#     --task="Pick a blue lego and place it in blue cup"
#
# Smoke-test (no ROS, just verify model loads + runs one inference):
#   bash inference_smolvla.sh \
#     --model.path=anirudhrani/smolvla_blue_sort_ven_50k \
#     --model.checkpoint=050000 \
#     --task="..." \
#     --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/.." && pwd)"

# Scrub system-Humble ROS env vars before invoking pixi — otherwise the
# pixi-Jazzy Python 3.12 process inherits PYTHONPATH that points at
# /opt/ros/humble's Python 3.10 site-packages, and importing things like
# control_msgs.action.FollowJointTrajectory tries to load the 3.10 C type-
# support extensions inside a 3.12 interpreter (UnsupportedTypeSupport).
#
# This script runs as a child of control_gui (system Humble, sourced) OR
# from a shell that ran `colcon build` (also Humble-sourced), so scrubbing
# is mandatory. record_sim_isaac.sh gets away without scrubbing because
# its plugins import only sensor_msgs (whose wire format is identical
# across distros and which doesn't need the C type-support module loaded
# at import time the way action types do).
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROSCONSOLE_FORMAT \
      ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE
# Filter LD_LIBRARY_PATH to keep CUDA bits (needed by torch) and drop ROS bits.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -vE '/(opt/ros|ros2_ws|Desktop/ros2_ws)' \
        | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

# DDS — same config as record_sim_isaac.sh (multicast on lo).
export CYCLONEDDS_URI="file://$ENV_DIR/cyclonedds.xml"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# Idempotent dep top-up. Marker file avoids running pip on every launch
# (pip resolution is slow). Touch the marker to skip; remove to force.
DEP_MARKER="$ENV_DIR/.pixi/.smolvla_deps_installed"
if [ ! -f "$DEP_MARKER" ]; then
    echo "[inference_smolvla] installing SmolVLA + peft extras into pixi env (one-time)…" >&2
    pixi run --manifest-path "$ENV_DIR/pixi.toml" pip install \
        -e "$REPO_ROOT/lerobot[smolvla]" \
        "peft>=0.18.0,<1.0.0" \
        || { echo "[inference_smolvla] dep install failed" >&2; exit 1; }
    mkdir -p "$(dirname "$DEP_MARKER")"
    touch "$DEP_MARKER"
fi

# Hand off to the Python entry. Any unmatched flags are passed through to
# argparse — same convention as record_sim_isaac.sh.
exec pixi run --manifest-path "$ENV_DIR/pixi.toml" \
    python3 -u "$SCRIPT_DIR/smolvla_inference.py" "$@"
