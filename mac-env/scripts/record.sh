#!/usr/bin/env bash
# record.sh — one-liner wrapper around lerobot-record-mode.sh (defaults: sim).
#
# Sets the right env vars (CycloneDDS, RMW, KMP) and the two-camera config,
# then dispatches to lerobot-record-mode.sh --mode sim with sensible defaults.
# Any extra flags you pass override the defaults (draccus takes the last value).
# For real hardware, pass `--mode real` (or set --robot.type / --teleop.type
# explicitly) — the so101_ros2 plugin contract is identical, only the
# producers change.
#
# Usage:
#   bash record.sh --dataset.repo_id=<user>/<repo> --dataset.num_episodes=5 [other flags]
#
# Back-compat: `record_sim.sh` is a symlink to this file — existing docs and
# scripts that invoke `record_sim.sh` keep working.
#
# Before running (sim): `bash stack_start.sh headless` (or gz/rviz/all) so the
# sim topics are live. During recording, drive the arm via control_gui
# (or a script publishing to /joint_commands).
#
# Before running (real): see vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md —
# camera_publisher + jointstatereader with publish_source=follower replace
# Gazebo as the topic producers.
#
# Defaults (override via explicit flags):
#   --robot.robot_type=so_follower      # dataset robot_type parity
#   --robot.use_degrees=true            # joint values in degrees
#   --robot.cameras=<wrist+top @ 640x480>
#   --dataset.fps=30
#   --dataset.episode_time_s=30
#   --dataset.reset_time_s=5
#   --dataset.push_to_hub=false
#   --display_data=true                 # rerun live viewer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.pixi/bin:${PATH:-}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///tmp/mac-env/cyclonedds.xml}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

DEFAULT_CAMERAS='{wrist: {type: ros2, topic: /wrist_camera, width: 640, height: 480, fps: 30}, top: {type: ros2, topic: /top_camera, width: 640, height: 480, fps: 30}}'

pixi run --manifest-path /tmp/mac-env/pixi.toml \
  bash "$SCRIPT_DIR/lerobot-record-mode.sh" \
    --mode sim \
    --robot.robot_type=so_follower \
    --robot.use_degrees=true \
    --robot.cameras="$DEFAULT_CAMERAS" \
    --dataset.fps=30 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=false \
    --display_data=true \
    "$@"
