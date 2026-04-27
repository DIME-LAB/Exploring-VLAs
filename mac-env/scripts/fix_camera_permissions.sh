#!/usr/bin/env bash
# fix_camera_permissions.sh — open the macOS Camera privacy pane and force a
# fresh "Allow / Deny" prompt for Terminal.app, so the camera pipeline can
# actually open /dev/video* devices.
#
# Why this exists: macOS TCC grants are per-app-bundle. Our camera pipeline
# launcher spawns processes under Terminal.app; that bundle needs the grant.
# When TCC is in a bad state (Sequoia point-update reset, prompt dismissed,
# or simply never-asked), `cv2.VideoCapture` fails silently with
#   "OpenCV: not authorized to capture video (status 0), requesting..."
#
# This script:
#   1. Resets Terminal.app's Camera TCC entry (no admin needed).
#   2. Opens System Settings → Privacy & Security → Camera so the user can
#      verify the toggle.
#   3. Triggers a one-shot cv2 open from inside Terminal.app to force the
#      OS dialog to appear. User clicks Allow once; future launches just work.
#
# Usage:
#   bash mac-env/scripts/fix_camera_permissions.sh
#
# Idempotent — safe to run multiple times.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

stack_preflight

echo "== 1. Reset Terminal.app's Camera TCC entry =="
# tccutil only resets — it doesn't grant. The user must still click Allow
# in the dialog when it pops. But resetting is needed because once a TCC
# entry exists in 'denied' state, the OS won't re-prompt; reset clears it
# back to 'undetermined' so the next access triggers the dialog.
tccutil reset Camera com.apple.Terminal 2>&1 || true

echo ""
echo "== 2. Open System Settings → Privacy & Security → Camera =="
# x-apple.systempreferences URL scheme works on Ventura+; older macOS uses
# a slightly different pane id but this usually still works as a fallback.
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera" 2>/dev/null || \
  open "/System/Library/PreferencePanes/Security.prefPane"

echo ""
echo "== 3. Trigger a one-shot cv2 open inside a fresh Terminal window =="
echo "   When the Allow / Deny dialog appears, click Allow."
echo ""

# Generate a tiny .command that opens cv2.VideoCapture(0), reads one frame,
# closes. The actual frame is irrelevant — we just need the OS to register
# that Terminal.app is asking for camera, which forces the dialog.
PROMPT_CMD=/tmp/soarm_camera_permission_prompt.command
cat > "$PROMPT_CMD" <<EOF
#!/bin/bash
echo "Triggering camera-access prompt..."
echo "When macOS shows the Allow dialog, click Allow, then close this window."
exec /Users/aldrininbaraj/.pixi/bin/pixi run --manifest-path "$MAC_ENV_MANIFEST" python -c '
import cv2, time
cap = cv2.VideoCapture(0)   # idx 0 is enough — we just need TCC to fire
print(f"  cap.isOpened() = {cap.isOpened()}")
ok, _ = cap.read()
print(f"  cap.read() ok  = {ok}")
cap.release()
print()
print("Done. If you saw and accepted the dialog, Terminal now has Camera permission.")
print("Re-run: bash mac-env/scripts/camera_pipeline_start.sh")
'
echo ""
echo "Press any key to close this window..."
read -n 1 -s
EOF
chmod +x "$PROMPT_CMD"

open -a Terminal "$PROMPT_CMD"

echo "  Terminal opened — accept the dialog when it appears, then come back."
echo ""
echo "After clicking Allow, retry the camera pipeline:"
echo "  bash $SCRIPT_DIR/camera_pipeline_start.sh"
