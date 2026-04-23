#!/usr/bin/env bash
# stack_restart.sh — stop any running stack, then start a fresh one.
# Forwards its arg to stack_start.sh (which picks the mode).
#
# Usage:  bash mac-env/scripts/stack_restart.sh [headless|gz|rviz|all]
#   default: headless

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/stack_stop.sh"
sleep 1
bash "$SCRIPT_DIR/stack_start.sh" "$@"
