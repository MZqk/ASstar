#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="Starun"
BUNDLE_ID="StarunC"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
APP_BINARY="$APP_BUNDLE/Contents/MacOS/$APP_NAME"

case "$MODE" in
  run|--debug|debug|--logs|logs|--telemetry|telemetry|--verify|verify) ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify]" >&2
    exit 2
    ;;
esac

pkill -x "$APP_NAME" >/dev/null 2>&1 || true

"$PROJECT_ROOT/build/build_macos_app.sh" --output-dir "$DIST_DIR"

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

case "$MODE" in
  run)
    open_app
    ;;
  --debug|debug)
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    open_app
    /usr/bin/log stream --info --style compact \
      --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    open_app
    /usr/bin/log stream --info --style compact \
      --predicate "process == \"$APP_NAME\" OR subsystem == \"$BUNDLE_ID\""
    ;;
  --verify|verify)
    open_app
    for _attempt in {1..20}; do
      if pgrep -x "$APP_NAME" >/dev/null; then
        sleep 1
        if pgrep -x "$APP_NAME" >/dev/null; then
          exit 0
        fi
      fi
      sleep 0.5
    done
    echo "$APP_NAME did not remain running after launch" >&2
    exit 1
    ;;
esac
