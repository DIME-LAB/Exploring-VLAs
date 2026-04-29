#!/usr/bin/env bash
# build_episode_browser.sh — compile EpisodeBrowser.swift into a minimal
# .app bundle, ad-hoc sign it (no Apple developer account / notarization
# needed since this only runs locally), and launch it.
#
# Re-run any time after editing EpisodeBrowser.swift; the script is
# idempotent and rebuilds in-place.
#
# Output: /tmp/EpisodeBrowser.app  (regenerable; not committed anywhere)
#
# Why /tmp: same convention as soarm-ws / mac-env — built artefacts live
# at spaceless paths and can be wiped on reboot. Source stays in
# mac-env/scripts/ where it's version-controlled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/EpisodeBrowser.swift"
APP="/tmp/EpisodeBrowser.app"
BIN="$APP/Contents/MacOS/EpisodeBrowser"

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
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>                <string>EpisodeBrowser</string>
  <key>CFBundleExecutable</key>          <string>EpisodeBrowser</string>
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
echo "Launch:"
echo "  open $APP"
echo ""
echo "First run: macOS may prompt 'cannot verify developer'."
echo "  Right-click the .app in Finder → Open → Open. Allowed permanently after."
