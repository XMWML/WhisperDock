#!/usr/bin/env bash
# Finder-friendly macOS launcher for WhisperDock.
set -Eeuo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${WHISPERDOCK_PORT:-8848}"
URL="http://127.0.0.1:${PORT}"
PID_FILE="$APP_ROOT/workspace/whisperdock.pid"
LOG_FILE="$APP_ROOT/logs/whisperdock.log"

if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'This launcher is for macOS. Use ./run.sh on Linux.\n' >&2
  exit 1
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'Invalid WHISPERDOCK_PORT: %s\n' "$PORT" >&2
  exit 1
fi

mkdir -p "$APP_ROOT/workspace/tmp" "$APP_ROOT/logs" "$APP_ROOT/config" "$APP_ROOT/models" "$APP_ROOT/outputs"

is_project_process() {
  local pid="$1"
  local command
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"$APP_ROOT/.venv/bin/python"* ]] && [[ "$command" == *"backend.main:app"* ]]
}

health_response() {
  curl -fsS --max-time 1 "$URL/api/health" 2>/dev/null || true
}

is_whisperdock_service() {
  local response="$1"
  [[ "$response" == *'"service":"WhisperDock"'* || "$response" == *'"service": "WhisperDock"'* ]]
}

# Reuse a running instance started by this launcher or by ./run.sh.
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && is_project_process "$existing_pid"; then
    response="$(health_response)"
    if is_whisperdock_service "$response"; then
      [[ "${WHISPERDOCK_OPEN_BROWSER:-1}" == "0" ]] || open "$URL"
      printf 'WhisperDock is already running at %s (PID %s).\n' "$URL" "$existing_pid"
      exit 0
    fi
  fi
  rm -f "$PID_FILE"
fi

# If the port belongs to an existing WhisperDock process without our PID file,
# reuse it; never terminate an unrelated process automatically.
response="$(health_response)"
if is_whisperdock_service "$response"; then
  [[ "${WHISPERDOCK_OPEN_BROWSER:-1}" == "0" ]] || open "$URL"
  printf 'WhisperDock is already running at %s.\n' "$URL"
  exit 0
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  printf 'Port %s is already in use. Set WHISPERDOCK_PORT to another port and retry.\n' "$PORT" >&2
  exit 1
fi

if [[ ! -x "$APP_ROOT/.venv/bin/python" ]]; then
  printf 'First launch: installing WhisperDock inside the project folder...\n'
  bash "$APP_ROOT/bootstrap.sh"
fi

printf 'Starting WhisperDock...\n'
nohup env WHISPERDOCK_PORT="$PORT" bash "$APP_ROOT/run.sh" >>"$LOG_FILE" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"

ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'WhisperDock stopped during startup. See %s\n' "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    tail -n 40 "$LOG_FILE" >&2 || true
    exit 1
  fi
  response="$(health_response)"
  if is_whisperdock_service "$response"; then
    ready=1
    break
  fi
  sleep 0.5
done

if (( ready == 0 )); then
  printf 'WhisperDock did not become ready within 30 seconds. See %s\n' "$LOG_FILE" >&2
  kill -TERM "$server_pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi

if [[ "${WHISPERDOCK_OPEN_BROWSER:-1}" != "0" ]]; then
  open "$URL"
fi
printf 'WhisperDock is ready at %s (PID %s).\n' "$URL" "$server_pid"
printf 'Double-click stop-macos.command to stop it.\n'
