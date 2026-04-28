#!/usr/bin/env bash
# dataset_viz.sh — launch lerobot-dataset-viz (rerun-backed) with sensible
# defaults for the SO-101 reference dataset, plus an env-recovery path
# for when the macOS pixi env drifts (datasets/av/rerun-sdk missing,
# pyarrow `_iconv` linkage bug, numpy bumped above 2).
#
# Usage:
#   bash mac-env/scripts/dataset_viz.sh                 # episode 0 of default repo
#   bash mac-env/scripts/dataset_viz.sh 5               # episode 5 of default repo
#   bash mac-env/scripts/dataset_viz.sh 5 my-org/my-ds  # episode 5 of any repo
#   bash mac-env/scripts/dataset_viz.sh --restore       # only run env restore, don't launch
#   bash mac-env/scripts/dataset_viz.sh --check         # only check imports, don't launch
#
# Default repo is the colleague's real SO-101 pick-and-place reference
# (`arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep`) used as
# the schema-parity target for our sim recordings (REC-04, VERIFY-01).
#
# What this wraps:
#   1. Sanity-check that pyarrow/datasets/rerun/av/lerobot all import OK.
#   2. If anything is missing or pyarrow has the macOS `_iconv` linkage
#      bug, run the deterministic recovery sequence documented in
#      MEMORY.md ("Dataset viewer setup" section).
#   3. Launch `lerobot-dataset-viz` with rerun in local mode.
#
# The recovery section auto-runs only when imports fail; if everything
# is healthy it just launches the viewer.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAC_ENV_MANIFEST="${MAC_ENV_MANIFEST:-/tmp/mac-env/pixi.toml}"
SOARM_WS_SETUP="${SOARM_WS_SETUP:-/tmp/soarm-ws/install/setup.bash}"
DEFAULT_REPO_ID='arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep'

# --- Arg parsing ------------------------------------------------------------
ONLY_RESTORE=0
ONLY_CHECK=0
EPISODE=0
REPO_ID="$DEFAULT_REPO_ID"

while (( $# )); do
  case "$1" in
    --restore) ONLY_RESTORE=1; shift ;;
    --check)   ONLY_CHECK=1;   shift ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      # First positional = episode, second = repo_id
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        EPISODE="$1"
      else
        REPO_ID="$1"
      fi
      shift
      ;;
  esac
done

# --- Sanity checks ----------------------------------------------------------
if [[ ! -f "$MAC_ENV_MANIFEST" ]]; then
  echo "ERROR: pixi manifest not found at $MAC_ENV_MANIFEST"
  echo "Run mac-env/scripts/bootstrap.sh first."
  exit 1
fi
if [[ ! -f "$SOARM_WS_SETUP" ]]; then
  echo "ERROR: ROS2 workspace not found at $SOARM_WS_SETUP"
  exit 1
fi

# --- Health check (returns 0 if all good, 1 if env needs restore) ----------
check_imports() {
  pixi run --manifest-path "$MAC_ENV_MANIFEST" python3 -c \
    "import pyarrow, datasets, rerun, av, lerobot; \
     v = pyarrow.__version__.split('.'); \
     assert int(v[0]) >= 23, f'pyarrow {pyarrow.__version__} < 23'; \
     dv = datasets.__version__.split('.'); \
     assert int(dv[0]) >= 4, f'datasets {datasets.__version__} < 4'; \
     print('ok:', pyarrow.__version__, datasets.__version__, rerun.__version__)" \
    2>&1
}

# --- Recovery sequence (mirrors MEMORY.md "Dataset viewer setup") ----------
restore_env() {
  echo "============================================================"
  echo "  restoring dataset-viewer env (this takes ~1 min)"
  echo "============================================================"

  # 1. pyarrow 23 — must NOT be the pip-wheel macOS arm64 (broken libiconv).
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip uninstall -y pyarrow >/dev/null 2>&1 || true
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install --force-reinstall \
    --no-deps "pyarrow==23.0.0"

  # 2. datasets >= 4.0 (older versions lack ParquetConfig.filters)
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install --force-reinstall \
    --no-deps "datasets==4.0.*"

  # 3. PyAV for video decoding
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install --force-reinstall \
    --no-deps "av"

  # 4. rerun-sdk + dependencies. This pulls pyarrow as a transitive — we'll
  #    swap it back to 23.0.0 below.
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install rerun-sdk

  # 5. Restore pyarrow 23 (rerun's transitive overwrote it)
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip uninstall -y pyarrow >/dev/null 2>&1 || true
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install --force-reinstall \
    --no-deps "pyarrow==23.0.0"

  # 6. Restore numpy < 2 (any pip install without --no-deps bumps it)
  pixi run --manifest-path "$MAC_ENV_MANIFEST" pip install --force-reinstall \
    --no-deps "numpy<2"

  echo
  echo "--- post-restore import check ---"
  check_imports
}

# --- Dispatch ---------------------------------------------------------------
if (( ONLY_RESTORE )); then
  restore_env
  exit 0
fi

echo "--- import check ---"
if ! check_imports; then
  echo
  echo "imports failed; running env restore..."
  restore_env
fi

if (( ONLY_CHECK )); then
  echo "imports OK; --check requested, not launching viewer."
  exit 0
fi

# --- Launch ---------------------------------------------------------------
echo
echo "============================================================"
echo "  launching lerobot-dataset-viz"
echo "    repo_id      = $REPO_ID"
echo "    episode      = $EPISODE"
echo "    mode         = local (rerun spawns its own viewer window)"
echo "============================================================"
echo

cd /tmp/soarm-ws
pixi run --manifest-path "$MAC_ENV_MANIFEST" bash -c "
  source '$SOARM_WS_SETUP'
  export KMP_DUPLICATE_LIB_OK=TRUE
  lerobot-dataset-viz \
    --repo-id '$REPO_ID' \
    --episode-index '$EPISODE' \
    --mode local
"
