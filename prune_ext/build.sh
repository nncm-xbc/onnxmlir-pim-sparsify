#!/usr/bin/env bash
# Build the prune_ext C++/OpenMP extension.
# Usage:  bash prune_ext/build.sh
# Output: prune_ext.*.so in the repo root (importable as `import prune_ext`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pass the active interpreter so CMake doesn't pick up python.app on macOS.
PYTHON_EXE="$(python -c 'import sys; print(sys.executable)')"

cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON_EXE"

cmake --build "$SCRIPT_DIR/build" \
    -- -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"
