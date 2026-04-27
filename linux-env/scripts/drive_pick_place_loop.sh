#!/usr/bin/env bash
# Drive sequential pick-and-place cycles via the direct-publish driver.
#
# Reference: ../mac-env/scripts/record_one_shot.sh (script topology)
#
# Mac runs `drive_base_yaw_sweep.py` alongside `record.sh` — both share the
# same pixi env, both publish/subscribe to ROS2 topics directly. We mirror
# that pattern: this loop just sources ROS2 + invokes drive_pick_place.py,
# which itself owns the cycle iteration via geometric IK + direct
# JointTrajectory publishing. control_gui is NOT in the critical path —
# previous service-call-based version timed out under recording load.
#
# Usage:
#   bash drive_pick_place_loop.sh                    # default 5 legos
#   bash drive_pick_place_loop.sh red_2x2 blue_2x3
#   bash drive_pick_place_loop.sh --mode sweep       # Mac-equivalent sine sweep

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source ROS2 (caller may not have done it). System Humble suffices —
# rclpy + message types are wire-compatible with the Jazzy publishers
# Isaac Sim and lerobot use.
source /opt/ros/humble/setup.bash
source ~/Projects/Exploring-VLAs/vla_SO-ARM101/install/setup.bash 2>/dev/null || true

# DDS config: when this driver runs in the same Humble process tree as the
# control stack (no cross-boundary), the system default config works fine —
# they share a daemon and discovery is local. Only override CYCLONEDDS_URI
# when explicitly needed (e.g. simultaneous pixi-Jazzy lerobot consumer):
#
#   export CYCLONEDDS_URI="file://$ENV_DIR/cyclonedds.xml"
#
# Forcing it unconditionally triggers "lo: the same interface may not be
# selected twice" when the existing env has both system default and our
# override active. The default $RMW_IMPLEMENTATION inherits from caller.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

exec python3 "$SCRIPT_DIR/drive_pick_place.py" "$@"
