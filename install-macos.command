#!/usr/bin/env bash
# WhisperDock macOS one-click installer / WhisperDock macOS 一键安装脚本
set -Eeuo pipefail

TARGET_DIR_DEFAULT="$HOME/WhisperDock"
TARGET_DIR="${1:-${WHISPERDOCK_INSTALL_DIR:-$TARGET_DIR_DEFAULT}}"
RELEASE_REF="${WHISPERDOCK_REF:-v0.1.0}"
REPO_URL="${WHISPERDOCK_REPO_URL:-https://github.com/XMWML/WhisperDock.git}"

show_help() {
  cat <<'EOF'
WhisperDock macOS one-click installer
WhisperDock macOS 一键安装脚本

Usage / 用法:
  bash install-macos.command [target_dir]

What it does / 脚本会做什么:
  1. Create a new local folder for WhisperDock.
     新建一个本地 WhisperDock 文件夹。
  2. Clone the GitHub repository at the selected release/tag.
     按指定 release/tag 克隆 GitHub 仓库。
  3. Create a project-local virtual environment and install dependencies.
     创建项目内虚拟环境并安装依赖。
  4. Start WhisperDock in the foreground with live logs.
     以前台模式启动 WhisperDock，并实时显示日志。

Defaults / 默认值:
  target_dir: ~/WhisperDock
  WHISPERDOCK_REF: v0.1.0
  WHISPERDOCK_REPO_URL: https://github.com/XMWML/WhisperDock.git

Notes / 说明:
  - Existing target directories are not overwritten.
    已存在的目标目录不会被覆盖。
  - Press Ctrl+C in the terminal to stop the service.
    在终端按 Ctrl+C 即可停止服务。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n缺少必需命令：%s\n' "$1" "$1" >&2
    exit 1
  fi
}

printf 'WhisperDock installer for macOS\n'
printf 'WhisperDock macOS 安装器\n\n'

require_command git
require_command curl

if [[ -e "$TARGET_DIR" ]]; then
  printf 'Target directory already exists: %s\n' "$TARGET_DIR" >&2
  printf '目标目录已存在：%s\n' "$TARGET_DIR" >&2
  printf 'Choose another path or remove it first, then retry.\n' >&2
  printf '请更换目录，或先删除该目录后再重试。\n' >&2
  exit 1
fi

printf 'Repository / 仓库: %s\n' "$REPO_URL"
printf 'Release / 版本: %s\n' "$RELEASE_REF"
printf 'Install dir / 安装目录: %s\n\n' "$TARGET_DIR"

mkdir -p "$(dirname "$TARGET_DIR")"

printf 'Cloning repository...\n'
printf '正在克隆仓库...\n'
git clone --depth 1 --single-branch --branch "$RELEASE_REF" "$REPO_URL" "$TARGET_DIR"

printf '\nBootstrapping project environment...\n'
printf '正在初始化项目环境...\n'
bash "$TARGET_DIR/bootstrap.sh"

printf '\nStarting WhisperDock...\n'
printf '正在启动 WhisperDock...\n'
printf 'A browser window should open automatically when ready.\n'
printf '服务就绪后会自动打开浏览器。\n'
printf 'Press Ctrl+C in this terminal to stop the service.\n'
printf '在这个终端窗口按 Ctrl+C 即可停止服务。\n\n'

exec bash "$TARGET_DIR/run.sh"
