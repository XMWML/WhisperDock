#!/usr/bin/env bash
# Linux foreground entry point.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.sh"
