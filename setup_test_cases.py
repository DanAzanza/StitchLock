"""Test Case Setup CLI Wrapper for the Garment Detail Correction Pipeline.

Populates `test_cases/` with flat image pairs using the `<name>.png` +
`<name>_reference.png` convention (zero subfolders or JSON manifests required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.fixtures import setup_all_test_cases

__all__ = ["setup_all_test_cases", "main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Initialize flat `<name>.png` + `<name>_reference.png` test cases."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing test case image pairs with procedural fixtures.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    created = setup_all_test_cases(root, force=args.force)
    for path in created:
        print(f"[READY] Test case pair initialized: {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
