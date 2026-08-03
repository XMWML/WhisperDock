#!/usr/bin/env bash
# Start the local WebUI. Nothing is written outside this project directory.
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${WHISPERDOCK_PORT:-8848}"

if [ ! -x "$APP_ROOT/.venv/bin/python" ]; then
  "$APP_ROOT/bootstrap.sh"
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

exec "$APP_ROOT/.venv/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
