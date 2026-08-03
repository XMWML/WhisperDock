#!/usr/bin/env bash
# Finder-friendly macOS foreground entry point.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.sh"
