#!/usr/bin/env bash
# Foreground launcher for the local WebUI. Nothing is written outside this project.
set -Eeuo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${WHISPERDOCK_PORT:-8848}"
URL="http://127.0.0.1:${PORT}"
PID_FILE="$APP_ROOT/workspace/whisperdock.pid"
LOG_FILE="$APP_ROOT/logs/whisperdock.log"
server_pid=""

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'Invalid WHISPERDOCK_PORT: %s\n' "$PORT" >&2
  exit 1
fi

mkdir -p "$APP_ROOT/workspace/tmp" "$APP_ROOT/logs" "$APP_ROOT/config" "$APP_ROOT/models" "$APP_ROOT/outputs"

health_response() {
  curl -fsS --max-time 1 "$URL/api/health" 2>/dev/null || true
}

is_whisperdock_service() {
  local response="$1"
  [[ "$response" == *'"service":"WhisperDock"'* || "$response" == *'"service": "WhisperDock"'* ]]
}

port_is_busy() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v port=":$PORT" '$4 ~ port "$" { found=1 } END { exit !found }'
  else
    return 1
  fi
}

open_browser() {
  [[ "${WHISPERDOCK_OPEN_BROWSER:-1}" == "0" ]] && return 0
  case "$(uname -s)" in
    Darwin)
      command -v open >/dev/null 2>&1 && open "$URL" || true
      ;;
    Linux)
      if command -v xdg-open >/dev/null 2>&1; then
        nohup xdg-open "$URL" >/dev/null 2>&1 < /dev/null &
      else
        printf 'No browser opener found; open %s manually.\n' "$URL"
      fi
      ;;
    *)
      printf 'Open %s manually.\n' "$URL"
      ;;
  esac
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
  exit "$exit_code"
}

handle_signal() {
  printf '\nStopping WhisperDock...\n'
  exit 130
}

trap cleanup EXIT
trap handle_signal INT TERM HUP

if is_whisperdock_service "$(health_response)"; then
  printf 'WhisperDock is already running at %s.\n' "$URL"
  open_browser
  exit 0
fi

if port_is_busy; then
  printf 'Port %s is already in use. Set WHISPERDOCK_PORT to another port and retry.\n' "$PORT" >&2
  exit 1
fi

if [[ ! -x "$APP_ROOT/.venv/bin/python" ]]; then
  printf 'First launch: installing WhisperDock inside the project folder...\n'
  bash "$APP_ROOT/bootstrap.sh"
fi

mkdir -p "$APP_ROOT"/{cache/pip,cache/huggingface,cache/torch,cache/xdg,workspace/tmp,outputs,models,logs,config,.home}
export PIP_CACHE_DIR="$APP_ROOT/cache/pip"
export HF_HOME="$APP_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$APP_ROOT/cache/huggingface/hub"
export TORCH_HOME="$APP_ROOT/cache/torch"
export XDG_CACHE_HOME="$APP_ROOT/cache/xdg"
export TMPDIR="$APP_ROOT/workspace/tmp"
export HOME="$APP_ROOT/.home"
export PYTHONPYCACHEPREFIX="$APP_ROOT/cache/pycache"
export WHISPERDOCK_ROOT="$APP_ROOT"

cd "$APP_ROOT"
printf 'Starting WhisperDock in the foreground...\n'
printf 'Live logs: %s\n' "$LOG_FILE"
env WHISPERDOCK_PORT="$PORT" "$APP_ROOT/.venv/bin/python" -m uvicorn backend.main:app \
  --host 127.0.0.1 --port "$PORT" > >(tee -a "$LOG_FILE") 2>&1 &
server_pid=$!
printf '%s\n' "$$" > "$PID_FILE"

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf 'WhisperDock stopped during startup. Recent logs:\n' >&2
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
  printf 'WhisperDock did not become ready within 30 seconds. Recent logs:\n' >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi

open_browser
printf 'WhisperDock is ready at %s. Press Ctrl+C to stop it.\n' "$URL"
wait "$server_pid"
