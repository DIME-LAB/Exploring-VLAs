#!/usr/bin/env bash
# Run the episode harness against an already-running Isaac scene + controllers.
set -e
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS_DIR/../_lib.sh"
stack_source_ros
exec /usr/bin/python3 -u "$HARNESS_DIR/run_episodes.py" "$@"
