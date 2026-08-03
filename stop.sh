#!/usr/bin/env bash
# Stop the project-local WhisperDock service started by start.sh.
set -Eeuo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_ROOT/workspace/whisperdock.pid"

if [[ ! -f "$PID_FILE" ]]; then
  printf 'No WhisperDock PID file found; no managed service is running.\n'
  exit 0
fi

pid="$(<"$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  rm -f "$PID_FILE"
  printf 'Removed an invalid WhisperDock PID file.\n'
  exit 0
fi

command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
if [[ "$command" != *"$APP_ROOT/.venv/bin/python"* ]] || [[ "$command" != *"backend.main:app"* ]]; then
  rm -f "$PID_FILE"
  printf 'The recorded PID is no longer a WhisperDock process; removed the stale PID file.\n'
  exit 0
fi

kill -TERM "$pid" 2>/dev/null || true
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done

if kill -0 "$pid" 2>/dev/null; then
  printf 'WhisperDock did not stop within 5 seconds; PID %s is still running.\n' "$pid" >&2
  exit 1
fi

rm -f "$PID_FILE"
printf 'WhisperDock stopped.\n'
