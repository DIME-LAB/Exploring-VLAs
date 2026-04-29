#!/usr/bin/env bash
# record_sim_isaac.sh — Linux flavor of mac-env/scripts/record_sim.sh, but
# subscribes to Isaac Sim's existing _sim-suffixed camera topics instead of
# Gazebo's canonical `/wrist_camera` + `/top_camera`.
#
# Why we don't rename the Isaac Sim topics: the BYOH `cameras` config maps
# arbitrary ROS topics to dataset feature keys (`observation.images.<name>`),
# so we point `topic:` at whatever Isaac Sim already publishes and let
# lerobot do the mapping at config time. Result: byte-identical schema to
# the real parity target dataset, zero code changes anywhere on the wire.
#
# Prereqs (do once — easiest path is the bootstrap):
#   bash linux-env/scripts/bootstrap.sh
#
# Or each step manually:
#   1. Pixi env installed:  cd linux-env && pixi install
#   2. Lerobot installed:   pixi run pip install -e "../lerobot[viz,dataset]"
#   3. Isaac Sim + control stack running:
#        bash linux-env/scripts/stack_start.sh   # easiest
#      Or manually:
#        bash linux-env/scripts/isaac/isaacsim_launch.sh launch soarm101-dt
#        # then send {"type": "quick_start"} on socket 8767
#        ros2 launch so_arm101_control control.launch.py rviz:=true
#   4. Verify topics live:
#        ros2 topic hz /wrist_camera_rgb_sim
#        ros2 topic hz /workspace_camera_sim
#        ros2 topic hz /joint_states
#
# During recording, drive the arm via control_gui (button-driven pick-place)
# or a script publishing to /joint_commands. The recorder captures whatever
# motion is happening — the recorder is passive.
#
# Usage:
#   bash record_sim_isaac.sh \
#     --dataset.repo_id=<user>/<dataset_name> \
#     --dataset.num_episodes=5 \
#     --dataset.single_task="Sort lego blocks by color into matching cups"
#
# Any flag not specified here passes through to lerobot-record (draccus
# takes the last value, so user-supplied flags override defaults).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# DDS config — match Mac (Cyclonedds, lo interface, multicast loopback).
# Force-override CYCLONEDDS_URI: the system default disables multicast,
# which prevents cross-process-tree discovery (system Humble publishers
# and pixi Jazzy subscribers can't see each other). Our config enables
# multicast on `lo`, which is what makes the Humble↔Jazzy bridge work.
export CYCLONEDDS_URI="file://$ENV_DIR/cyclonedds.xml"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# Disable lerobot's global pynput keyboard listener. Sim recording is fully
# script-driven (no human teleop), so the listener has no purpose — and it
# captures arrow keys / Esc system-wide, so a stray keypress in any other
# window aborts the record loop with rc=1 mid-episode. Honored by our
# patched init_keyboard_listener() in lerobot/common/control_utils.py.
# Override by exporting LEROBOT_DISABLE_KEYBOARD= (empty) before invoking.
export LEROBOT_DISABLE_KEYBOARD="${LEROBOT_DISABLE_KEYBOARD:-1}"

# Camera config — pointed at Isaac Sim's existing _sim-suffixed topics.
# Feature keys (`wrist`, `top`) become `observation.images.{wrist,top}` in
# the v3 dataset, matching the real parity target's schema.
# Both cameras publish at 640x480 rgba8 (alpha stripped on the lerobot
# consumer side via the ROS2Camera _msg_to_ndarray patch in this fork —
# stripping in the Isaac Sim publisher's render-thread callback throttles
# RTF to ~1 Hz; consumer-side strip leaves Isaac Sim at 18 Hz).
# Matches real-target parity dataset resolution.
DEFAULT_CAMERAS='{wrist: {type: ros2, topic: /wrist_camera_rgb_sim, encoding: rgba8, width: 640, height: 480, fps: 30}, top: {type: ros2, topic: /workspace_camera_sim, encoding: rgba8, width: 640, height: 480, fps: 30}}'

cd "$ENV_DIR" && exec pixi run lerobot-record \
  --robot.type=so101_ros2 \
  --robot.robot_type=so_follower \
  --robot.use_degrees=true \
  --robot.cameras="$DEFAULT_CAMERAS" \
  --robot.joint_states_topic=/joint_states \
  --robot.image_timeout_ms=10000 \
  --teleop.type=so101_ros2 \
  --teleop.action_topic=/joint_commands_lerobot \
  --teleop.use_degrees=true \
  --teleop.action_stale_ms=10000 \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --display_data=true \
  "$@"
