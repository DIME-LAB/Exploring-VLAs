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

# aruco_camera_localizer is managed outside Exploring-VLAs (submodule-add
# into a spaced path hits a known git pack-index bug on macOS). We clone
# it into the spaceless /tmp side by URL below and symlink into the
# workspace, keeping bootstrap self-contained without needing the user to
# `git clone` manually.
ARUCO_URL="${ARUCO_URL:-https://github.com/inbarajaldrin/aruco_camera_localizer.git}"
ARUCO_BRANCH="${ARUCO_BRANCH:-robosort}"
ARUCO_SRC="/tmp/aruco_camera_localizer"

echo "== mac-env source:    $MAC_ENV_SRC"
echo "== vla_SO-ARM101 src: $VLA_SRC"
echo "== aruco_localizer:   $ARUCO_SRC (branch: $ARUCO_BRANCH)"

if [ ! -d "$VLA_SRC/so_arm101_description" ]; then
  echo "ERROR: $VLA_SRC/so_arm101_description not found." >&2
  echo "       Are you running from Exploring-VLAs? Submodules initialized?" >&2
  echo "       Try: git submodule update --init --recursive" >&2
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
echo "== 2b. clone aruco_camera_localizer to $ARUCO_SRC =="
if [ ! -d "$ARUCO_SRC/aruco_camera_localizer" ]; then
  git clone --depth 1 -b "$ARUCO_BRANCH" "$ARUCO_URL" "$ARUCO_SRC" || {
    echo "WARN: aruco_camera_localizer clone failed. Real-hardware vision" >&2
    echo "      won't build. Retry: git clone -b $ARUCO_BRANCH $ARUCO_URL $ARUCO_SRC" >&2
  }
else
  ( cd "$ARUCO_SRC" && git pull --ff-only 2>&1 | tail -3 ) || echo "  (fetch skipped)"
fi

echo ""
echo "== 3. set up colcon workspace at /tmp/soarm-ws =="
mkdir -p /tmp/soarm-ws/src
cd /tmp/soarm-ws/src

# vla_SO-ARM101 packages (sim + real-hw bringup + sim ground truth).
for pkg in so_arm101_description so_arm101_moveit_config so_arm101_control \
           jointstatereader so_arm101_bringup sim_ground_truth; do
  if [ -d "$VLA_SRC/$pkg" ] && [ ! -e "$pkg" ]; then
    ln -sf "$VLA_SRC/$pkg" .
    echo "  linked $pkg"
  fi
done

# aruco_camera_localizer (real-side vision — ArUco + YOLO pose detection).
# Lives at the top level of Exploring-VLAs as a submodule.
if [ -d "$ARUCO_SRC/aruco_camera_localizer" ] && [ ! -e "aruco_camera_localizer" ]; then
  ln -sf "$ARUCO_SRC" aruco_camera_localizer
  echo "  linked aruco_camera_localizer"
fi

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
