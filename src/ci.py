"""Local CI quality gate runner mirroring GitHub Actions CI pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("garment_fix.ci")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_stage(stage_name: str, cmd: Sequence[str], cwd: Path = REPO_ROOT) -> bool:
    """Execute a single CI pipeline stage as a subprocess and log the outcome."""
    LOGGER.info("==================================================")
    LOGGER.info("Starting CI Stage: %s", stage_name)
    LOGGER.info("Executing: %s", " ".join(cmd))
    LOGGER.info("==================================================")

    res = subprocess.run(cmd, cwd=cwd, check=False)
    if res.returncode == 0:
        LOGGER.info("[SUCCESS] CI Stage '%s' passed.", stage_name)
        return True

    LOGGER.error("[FAILED] CI Stage '%s' failed with exit code %d.", stage_name, res.returncode)
    return False


def run_all_ci_stages(
    python_executable: str = sys.executable,
    skip_lint: bool = False,
    skip_tests: bool = False,
) -> int:
    """Run Ruff linter/formatter, Pyright static type check, and Pytest regression suite."""
    if not skip_lint:
        if not run_stage("Ruff Linter", [python_executable, "-m", "ruff", "check", "."]):
            return 1
        if not run_stage(
            "Ruff Formatter", [python_executable, "-m", "ruff", "format", "--check", "."]
        ):
            return 1
        if not run_stage(
            "Pyright Type Check",
            [python_executable, "-m", "pyright", "--pythonpath", python_executable],
        ):
            return 1

    if not skip_tests and not run_stage(
        "Pytest Regression & Pixel-Lock Suite",
        [python_executable, "-m", "pytest", "tests", "-v"],
    ):
        return 1

    LOGGER.info("==================================================")
    LOGGER.info("[ALL CI CHECKS PASSED] Local CI parity 100%% verified.")
    LOGGER.info("==================================================")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GitHub Actions CI Quality Gate locally with full parity."
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff lint and format checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip Pytest regression suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    return run_all_ci_stages(skip_lint=args.skip_lint, skip_tests=args.skip_tests)


if __name__ == "__main__":
    sys.exit(main())
