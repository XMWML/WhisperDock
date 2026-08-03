#!/usr/bin/env bash
# Install WhisperDock into a project-local virtual environment.
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

is_supported_python() {
  command -v "$1" >/dev/null 2>&1 || return 1
  case "$($1 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" in
    3.10|3.11|3.12|3.13) return 0 ;;
    *) return 1 ;;
  esac
}

if ! is_supported_python "$PYTHON_BIN"; then
  # uv's managed interpreter is intentionally placed inside WhisperDock so a
  # copied project retains its interpreter as well as its virtual environment.
  if command -v uv >/dev/null 2>&1; then
    mkdir -p "$APP_ROOT/runtime/python" "$APP_ROOT/cache/uv"
    UV_CACHE_DIR="$APP_ROOT/cache/uv" uv python install 3.12 --install-dir "$APP_ROOT/runtime/python"
    PYTHON_BIN="$(find "$APP_ROOT/runtime/python" -type f -path '*/bin/python3*' ! -name '*config' | head -n 1)"
  fi
fi

if [ -z "${PYTHON_BIN:-}" ] || ! is_supported_python "$PYTHON_BIN"; then
  echo "WhisperDock requires Python 3.10-3.13. Install one, set PYTHON_BIN, or install uv and retry."
  exit 1
fi

mkdir -p "$APP_ROOT"/{cache/pip,cache/huggingface,cache/torch,cache/xdg,workspace/tmp,outputs,models,logs,config,.home}
export PIP_CACHE_DIR="$APP_ROOT/cache/pip"
export HF_HOME="$APP_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$APP_ROOT/cache/huggingface/hub"
export TORCH_HOME="$APP_ROOT/cache/torch"
export XDG_CACHE_HOME="$APP_ROOT/cache/xdg"
# Keep installer unpack files project-local and remove them when bootstrap exits,
# so a cancelled or repeated install cannot bloat the portable workspace.
BOOTSTRAP_TMPDIR="$(mktemp -d "$APP_ROOT/workspace/tmp/bootstrap.XXXXXX")"
trap 'rm -rf "$BOOTSTRAP_TMPDIR"' EXIT
export TMPDIR="$BOOTSTRAP_TMPDIR"
export HOME="$APP_ROOT/.home"

if [ ! -x "$APP_ROOT/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$APP_ROOT/.venv"
fi

"$APP_ROOT/.venv/bin/python" -m pip install --upgrade pip wheel
"$APP_ROOT/.venv/bin/python" -m pip install -r "$APP_ROOT/requirements.txt"

echo "Ready. Start WhisperDock in the foreground with: ./run.sh"
