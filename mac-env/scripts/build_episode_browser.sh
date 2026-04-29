#!/usr/bin/env bash
# build_episode_browser.sh — compile EpisodeBrowser.swift into a minimal
# .app bundle, ad-hoc sign it (no Apple developer account / notarization
# needed since this only runs locally), and launch it.
#
# Re-run any time after editing EpisodeBrowser.swift; the script is
# idempotent and rebuilds in-place.
#
# Output: /tmp/EpisodeBrowser.app  (regenerable; wiped on reboot)
#
# Optional flags:
#   --install   After building, also rsync the .app into /Applications/
#               (or ~/Applications/ if /Applications isn't writable) so
#               it shows up in Launchpad and survives reboots. Re-run
#               with --install whenever you ship a new version.
#   --no-launch Skip the auto-launch at the end (useful in CI).
#
# Why /tmp for the working bundle: same convention as soarm-ws / mac-env
# — built artefacts live at spaceless paths and can be wiped on reboot.
# Source stays in mac-env/scripts/ where it's version-controlled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/EpisodeBrowser.swift"
# Display name in Finder / Dock / Launchpad. Bundle identifier
# (local.lerobot.episodebrowser) is intentionally NOT renamed here
# so @AppStorage persistence carries across rebuilds.
APP_NAME="ReRunEpisodeBrowser"
APP="/tmp/${APP_NAME}.app"
BIN="$APP/Contents/MacOS/${APP_NAME}"

INSTALL_TO_APPLICATIONS=0
NO_LAUNCH=0
while [ $# -gt 0 ]; do
    case "$1" in
        --install)   INSTALL_TO_APPLICATIONS=1; shift ;;
        --no-launch) NO_LAUNCH=1; shift ;;
        --help|-h)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC missing" >&2
    exit 1
fi

if ! command -v swiftc >/dev/null 2>&1; then
    echo "ERROR: swiftc not on PATH. Install Xcode Command Line Tools:" >&2
    echo "       xcode-select --install" >&2
    exit 1
fi

echo "== 1. Compile Swift -> binary =="
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# -parse-as-library: required when the module has @main (otherwise swiftc
#   treats the file as a top-level script and ignores @main).
# -O: release-mode optimisations.
# Frameworks are auto-linked by SwiftUI imports on macOS 14+; explicit
#   -framework lines aren't required as of Swift 6.0 / Xcode 16.
swiftc -O -parse-as-library -o "$BIN" "$SRC"

echo "== 2. Write Info.plist =="
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>                <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>         <string>ReRun Episode Browser</string>
  <key>CFBundleExecutable</key>          <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>          <string>local.lerobot.episodebrowser</string>
  <key>CFBundleVersion</key>             <string>1</string>
  <key>CFBundleShortVersionString</key>  <string>1.0</string>
  <key>CFBundlePackageType</key>         <string>APPL</string>
  <key>LSMinimumSystemVersion</key>      <string>14.0</string>
  <key>NSHighResolutionCapable</key>     <true/>
</dict>
</plist>
PLIST

echo "== 3. Ad-hoc code sign =="
# Local-only ad-hoc sign so Gatekeeper doesn't block first launch with the
# "damaged / cannot be opened" dialog. No team ID, no notarization.
codesign --force --sign - "$APP" 2>/dev/null || true

echo "== 4. Done =="
echo "  bundle: $APP"
echo "  binary: $BIN  ($(wc -c <"$BIN") bytes)"
echo ""

# --- Optional: install to /Applications so it sticks around ----------------
if [ "$INSTALL_TO_APPLICATIONS" -eq 1 ]; then
    INSTALL_TARGET="/Applications/${APP_NAME}.app"
    if [ ! -w "/Applications" ]; then
        # /Applications usually requires admin to write to. Fall back to
        # ~/Applications/ which is per-user, no sudo needed. macOS treats
        # both identically for Launchpad / Spotlight.
        mkdir -p "$HOME/Applications"
        INSTALL_TARGET="$HOME/Applications/${APP_NAME}.app"
        echo "== 5. Install -> $INSTALL_TARGET (per-user, /Applications not writable) =="
    else
        echo "== 5. Install -> $INSTALL_TARGET =="
    fi
    rm -rf "$INSTALL_TARGET"
    cp -R "$APP" "$INSTALL_TARGET"
    # Re-sign at the new location — the bundle's identifier is now bound
    # to a different absolute path; ad-hoc resign keeps Gatekeeper happy.
    codesign --force --sign - "$INSTALL_TARGET" 2>/dev/null || true
    echo "  installed: $INSTALL_TARGET"
    echo ""
fi

if [ "$NO_LAUNCH" -eq 0 ]; then
    LAUNCH_TARGET="$APP"
    if [ "$INSTALL_TO_APPLICATIONS" -eq 1 ]; then
        LAUNCH_TARGET="$INSTALL_TARGET"
    fi
    echo "Launching: open '$LAUNCH_TARGET'"
    open "$LAUNCH_TARGET"
else
    echo "Launch:"
    echo "  open $APP"
fi
echo ""
echo "First run: macOS may prompt 'cannot verify developer'."
echo "  Right-click the .app in Finder → Open → Open. Allowed permanently after."
