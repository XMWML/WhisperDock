#!/usr/bin/env bash
# Linux one-click/background entry point.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh"
