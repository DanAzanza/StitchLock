#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_EXE="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
fi

if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
    echo "[ERROR] Python was not found in PATH or .venv/bin/python."
    echo "Please run ./install.sh first or install Python 3.11+."
    exit 1
fi

echo "=============================================================================="
echo "  StitchLocker - Launching Artist Studio UI (http://127.0.0.1:7860)"
echo "  Closing the browser tab will automatically shut down ComfyUI and the UI."
echo "=============================================================================="
exec "${PYTHON_EXE}" -u app.py "$@"
