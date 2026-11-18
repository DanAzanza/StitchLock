#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==========================================================================="
echo "  StitchLocker - Automated Environment Setup and Pixel-Lock Self-Test"
echo "==========================================================================="
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed or not in PATH (Python 3.11+ required)."
    exit 1
fi

PY_VER="$(python3 --version)"
echo "[OK] Detected ${PY_VER}"

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment in .venv..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created."
fi

VENV_PYTHON=".venv/bin/python"
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

echo "[INFO] Upgrading pip and installing dependencies..."
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r requirements.txt

echo "[INFO] Checking test_cases directory..."
"${VENV_PYTHON}" setup_test_cases.py

echo ""
echo "[INFO] Running automated Quality Gate and Pixel-Lock test suite..."
"${VENV_PYTHON}" scripts/run_ci_locally.py

echo ""
echo "==========================================================================="
echo "  [SUCCESS] StitchLocker Setup and Verification Completed!"
echo "  Run './run_pipeline.sh' to launch the Artist Studio UI,"
echo "  or run '${VENV_PYTHON} run_batch.py' for headless batch processing."
echo "==========================================================================="
