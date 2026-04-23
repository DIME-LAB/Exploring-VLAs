#!/usr/bin/env bash
# bootstrap.sh — one-time Mac setup.
#
# RoboStack conda packages can't live at a spaced path (shebangs, setup
# scripts break). This script materializes a spaceless clone of
# Exploring-VLAs/mac-env/ under /tmp/mac-env, installs the pixi env there,
# and creates a /tmp/soarm-ws colcon workspace with symlinks to
# vla_SO-ARM101/src/. Run once after cloning the repo; idempotent.
#
# Usage:  bash Exploring-VLAs/mac-env/scripts/bootstrap.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_ENV_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MAC_ENV_SRC/.." && pwd)"
VLA_SRC="$REPO_ROOT/vla_SO-ARM101/src"

echo "== mac-env source:    $MAC_ENV_SRC"
echo "== vla_SO-ARM101 src: $VLA_SRC"

if [ ! -d "$VLA_SRC/so_arm101_description" ]; then
  echo "ERROR: $VLA_SRC/so_arm101_description not found." >&2
  echo "       Are you running from Exploring-VLAs? Submodules initialized?" >&2
  exit 1
fi

echo ""
echo "== 1. copy mac-env config to /tmp/mac-env =="
mkdir -p /tmp/mac-env
cp "$MAC_ENV_SRC/pixi.toml"       /tmp/mac-env/
cp "$MAC_ENV_SRC/pixi.lock"       /tmp/mac-env/
cp "$MAC_ENV_SRC/cyclonedds.xml"  /tmp/mac-env/
# Scripts are invoked from the committed location — don't copy them.

echo ""
echo "== 2. pixi install (first run: ~10-15 min; subsequent: fast) =="
export PATH="$HOME/.pixi/bin:$PATH"
( cd /tmp/mac-env && pixi install )

echo ""
echo "== 3. set up colcon workspace at /tmp/soarm-ws =="
mkdir -p /tmp/soarm-ws/src
cd /tmp/soarm-ws/src
for pkg in so_arm101_description so_arm101_moveit_config so_arm101_control jointstatereader; do
  if [ ! -e "$pkg" ]; then
    ln -sf "$VLA_SRC/$pkg" .
    echo "  linked $pkg"
  fi
done
cd /tmp/soarm-ws

echo ""
echo "== 4. colcon build =="
LIBRARY_PATH="/tmp/mac-env/.pixi/envs/default/lib" \
  "$HOME/.pixi/bin/pixi" run --manifest-path /tmp/mac-env/pixi.toml colcon build

echo ""
echo "== 5. install lerobot editable (for recording — Phase 4+) =="
"$HOME/.pixi/bin/pixi" run --manifest-path /tmp/mac-env/pixi.toml pip install -e "$REPO_ROOT/lerobot" || {
  echo "WARN: lerobot editable install failed. Not fatal for sim-only work."
}

echo ""
echo "== done =="
echo "Next: bash $SCRIPT_DIR/stack_start.sh        # headed, no rviz"
echo "      bash $SCRIPT_DIR/stack_start.sh headless no-rviz"
