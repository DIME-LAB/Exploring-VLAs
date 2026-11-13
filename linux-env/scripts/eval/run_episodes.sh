#!/usr/bin/env bash
# Run the episode harness against an already-running Isaac scene + controllers.
set -e
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HARNESS_DIR/../inference_async_lerobot.sh" --episode-harness "$@"
