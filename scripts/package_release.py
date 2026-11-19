"""Forwarding CLI entrypoint for building StitchLocker release distribution archives."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.packaging as _pkg  # noqa: E402

build_release_package = _pkg.build_release_package
collect_distribution_files = _pkg.collect_distribution_files
compute_sha256 = _pkg.compute_sha256
main = _pkg.main
validate_package = _pkg.validate_package

__all__ = [
    "build_release_package",
    "collect_distribution_files",
    "compute_sha256",
    "main",
    "validate_package",
]

if __name__ == "__main__":
    sys.exit(main())
