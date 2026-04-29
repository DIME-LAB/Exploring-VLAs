#!/bin/bash
# bootstrap.sh — one-shot Linux setup for the Exploring-VLAs stack.
#
# What this does:
#   1. Preflight (Humble installed, pixi available, Isaac Sim findable)
#   2. Initialize git submodules (lerobot, isaac-sim-mcp)
#   3. Install the linux-env pixi env (Jazzy + lerobot deps)
#   4. Build the colcon workspace at vla_SO-ARM101/
#   5. Print a ready summary
#
# Idempotent: re-running picks up where you left off. Re-run after editing
# pixi.toml or pulling new submodule SHAs.
#
# Companion to mac-env/scripts/bootstrap.sh — different topology (Linux uses
# Isaac Sim; Mac uses Gazebo) but the same one-command-to-ready ergonomics.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_lib.sh"

echo "=== Exploring-VLAs Linux bootstrap ==="
echo "REPO_ROOT: $REPO_ROOT"
echo ""

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------

echo "[1/5] Preflight..."

# ROS2 Humble (system install for the producer side)
if [ ! -f "$ROS2_SETUP" ]; then
    echo "  ERROR: ROS2 Humble not at $ROS2_SETUP"
    echo "         Install via:"
    echo "             https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html"
    echo "         Or set ROS2_SETUP env var to a different distro's setup.bash."
    exit 1
fi
echo "  ✓ ROS2 $ROS2_DISTRO at $ROS2_SETUP"

# pixi (for the lerobot-record consumer side)
if ! command -v pixi >/dev/null 2>&1; then
    if [ -x "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        echo "  ERROR: pixi not on \$PATH"
        echo "         Install: curl -fsSL https://pixi.sh/install.sh | bash"
        echo "         Then: export PATH=\"\$HOME/.pixi/bin:\$PATH\""
        exit 1
    fi
fi
echo "  ✓ pixi $(pixi --version 2>/dev/null | awk '{print $2}')"

# Isaac Sim launcher (best-effort — colleague may install elsewhere)
ISAACSIM_BIN_DEFAULT="${ISAACSIM_BIN:-$HOME/env_isaaclab/bin/isaacsim}"
if [ ! -x "$ISAACSIM_BIN_DEFAULT" ]; then
    echo "  ⚠ Isaac Sim launcher not found at $ISAACSIM_BIN_DEFAULT"
    echo "    Install Isaac Sim 5.x and either symlink to that path or"
    echo "    export ISAACSIM_BIN=/your/path/to/isaacsim"
    echo "    (continuing — bootstrap doesn't actually require Isaac Sim,"
    echo "     but stack_start.sh will fail until this is resolved)"
else
    echo "  ✓ Isaac Sim at $ISAACSIM_BIN_DEFAULT"
fi

# colcon (comes with ros-humble-desktop)
if ! command -v colcon >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$ROS2_SETUP"
fi
if ! command -v colcon >/dev/null 2>&1; then
    echo "  ERROR: colcon not found even after sourcing $ROS2_SETUP"
    echo "         Install: sudo apt install python3-colcon-common-extensions"
    exit 1
fi
echo "  ✓ colcon"
echo ""

# ---------------------------------------------------------------------------
# 2. Submodules
# ---------------------------------------------------------------------------

echo "[2/5] Initializing submodules..."
cd "$REPO_ROOT"
git submodule update --init --recursive
echo "  ✓ lerobot @ $(git -C "$LEROBOT_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  ✓ isaac-sim-mcp @ $(git -C "$ISAAC_MCP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

# ---------------------------------------------------------------------------
# 3. Pixi env (lerobot consumer side)
# ---------------------------------------------------------------------------

echo "[3/5] Installing pixi env (linux-env)..."
if [ ! -f "$PIXI_MANIFEST" ]; then
    echo "  ERROR: $PIXI_MANIFEST missing"
    exit 1
fi
pixi install --manifest-path "$PIXI_MANIFEST"
echo "  ✓ pixi env materialized at $LINUX_ENV_DIR/.pixi/"
echo ""

# ---------------------------------------------------------------------------
# 4. Colcon build (producer side)
# ---------------------------------------------------------------------------

echo "[4/5] Building colcon workspace..."
cd "$SOARM_WS"
# shellcheck disable=SC1090
source "$ROS2_SETUP"
colcon build --symlink-install
echo "  ✓ Built into $SOARM_WS/install/"
echo ""

# ---------------------------------------------------------------------------
# 5. Ready
# ---------------------------------------------------------------------------

echo "[5/5] Ready."
echo ""
echo "Next steps:"
echo "  bash $LINUX_ENV_DIR/scripts/stack_start.sh    # Isaac Sim + control stack + RViz + tkinter GUI"
echo "  bash $LINUX_ENV_DIR/scripts/stack_status.sh   # verify"
echo "  # … record episodes via control_gui's Record Sim tab …"
echo "  bash $LINUX_ENV_DIR/scripts/stack_stop.sh"
echo ""
echo "Read $VLA_PKG/docs/ROS2_LINUX_SETUP.md for the full from-zero walkthrough."
