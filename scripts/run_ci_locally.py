"""Forwarding CLI entrypoint for running the full local CI Quality Gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ci import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
