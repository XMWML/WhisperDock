#!/usr/bin/env bash
# Linux stop entry point.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stop.sh"
