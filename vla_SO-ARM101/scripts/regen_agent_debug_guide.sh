#!/usr/bin/env bash
# Regenerate the "Button ↔ Service Mapping" AND "Widgets" sections of
# docs/AGENT_DEBUG_GUIDE.md from live dump_services + list_widgets output.
#
# Usage:
#   1. Launch the control stack: ros2 launch so_arm101_control control.launch.py
#   2. Run this script: bash scripts/regen_agent_debug_guide.sh
#
# Fails fast if either service is unreachable. Emits a diff summary so you can
# see what changed. Idempotent — running twice on a clean tree produces no diff.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GUIDE="${REPO_ROOT}/docs/AGENT_DEBUG_GUIDE.md"
BUTTON_SVC="/so_arm101_control_gui/dump_services"
WIDGET_SVC="/so_arm101_control_gui/list_widgets"
TMP_DUMP_RAW="$(mktemp -t dump_services.raw.XXXXXX)"
TMP_DUMP_MD="$(mktemp -t dump_services.md.XXXXXX)"
TMP_LIST_RAW="$(mktemp -t list_widgets.raw.XXXXXX)"
TMP_LIST_MD="$(mktemp -t list_widgets.md.XXXXXX)"
trap 'rm -f "$TMP_DUMP_RAW" "$TMP_DUMP_MD" "$TMP_LIST_RAW" "$TMP_LIST_MD"' EXIT

echo "[1/5] Checking service availability"
for svc in "$BUTTON_SVC" "$WIDGET_SVC"; do
  if ! ros2 service list 2>/dev/null | grep -q "^${svc}$"; then
    echo "ERROR: service $svc not found on the ROS graph."
    echo "  Launch the control stack first:"
    echo "    ros2 launch so_arm101_control control.launch.py"
    exit 1
  fi
done

echo "[2/5] Calling $BUTTON_SVC"
ros2 service call "$BUTTON_SVC" std_srvs/srv/Trigger "{}" > "$TMP_DUMP_RAW"

echo "[3/5] Calling $WIDGET_SVC"
ros2 service call "$WIDGET_SVC" std_srvs/srv/Trigger "{}" > "$TMP_LIST_RAW"

echo "[4/5] Extracting response messages → markdown"
python3 - "$TMP_DUMP_RAW" "$TMP_DUMP_MD" "$TMP_LIST_RAW" "$TMP_LIST_MD" <<'PY'
import re, sys
def extract(in_path, out_path):
    raw = open(in_path, encoding='utf-8').read()
    m = re.search(r"message='(.*)'\)\s*$", raw, flags=re.DOTALL)
    if not m:
        m = re.search(r"message='(.*?)'\)", raw, flags=re.DOTALL)
    if not m:
        sys.exit(f"ERROR: could not parse response in {in_path}")
    s = m.group(1)
    msg = (s.replace("\\\\", "\x00")
             .replace("\\'", "'")
             .replace("\\n", "\n")
             .replace("\\t", "\t")
             .replace("\x00", "\\"))
    open(out_path, 'w', encoding='utf-8').write(msg)

extract(sys.argv[1], sys.argv[2])
extract(sys.argv[3], sys.argv[4])
PY

echo "[5/5] Splicing into $GUIDE"
python3 - "$GUIDE" "$TMP_DUMP_MD" "$TMP_LIST_MD" <<'PY'
import sys
guide_path = sys.argv[1]
dump = open(sys.argv[2], encoding='utf-8').read().rstrip() + '\n'
widgets = open(sys.argv[3], encoding='utf-8').read().rstrip() + '\n'
guide = open(guide_path, encoding='utf-8').read()

BUTTON_MARKER = '# Button ↔ Service Mapping'
WIDGET_MARKER = '# Widgets (auto-generated from list_widgets)'

combined = dump.rstrip() + '\n\n---\n\n' + widgets

# Find the button section start. If found, everything from there forward
# (including any existing Widgets section) is replaced with the new combined
# content. This keeps both sections together and idempotent across re-runs.
idx = guide.find(BUTTON_MARKER)
if idx == -1:
    # First-time install — append at EOF with separator
    if not guide.endswith('\n'):
        guide += '\n'
    new = guide + '\n---\n\n' + combined
else:
    new = guide[:idx] + combined

open(guide_path, 'w', encoding='utf-8').write(new)
PY

echo ""
echo "Diff summary:"
git -C "$REPO_ROOT" --no-pager diff --stat -- docs/AGENT_DEBUG_GUIDE.md || true
echo ""
echo "✓ Regen complete: $GUIDE"
