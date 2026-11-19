"""Deterministic release packaging tool for StitchLocker distribution archives."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import re
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from src.models import resolve_safe_path

LOGGER = logging.getLogger("garment_fix.packaging")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_SEMVER_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:-[A-Za-z0-9.]+)?$")
_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]+$")

MANDATORY_ROOT_FILES: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements.txt",
    "config.yaml",
    "README.md",
    "LICENSE",
    "app.py",
    "run_batch.py",
    "setup_test_cases.py",
    "install.bat",
    "install.sh",
    "run_pipeline.bat",
    "run_pipeline.sh",
)

INCLUDED_DIRECTORIES: Final[tuple[str, ...]] = (
    "src",
    "scripts",
    "workflows",
    "docs",
    "test_cases",
    "tests",
)

_IGNORED_PARTS: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        ".ci_local_staging",
        "dist",
        "build",
    }
)

_IGNORED_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        ".coverage",
        "coverage.xml",
        ".comfyui_server.pid",
    }
)

_IGNORED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo", ".tmp", ".bak", ".part")


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file with streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_ignore(path: Path) -> bool:
    """Return True if a path matches any cache, build, or transient artifact filter."""
    for part in path.parts:
        if part in _IGNORED_PARTS or part.endswith((".egg-info", ".tmp", ".bak")):
            return True
    name = path.name
    if name in _IGNORED_NAMES:
        return True
    return name.endswith(_IGNORED_SUFFIXES)


def _is_safe_source_file(path: Path, resolved_root: Path) -> bool:
    """Reject symlinks and files resolving outside repository root."""
    if not path.is_file() or path.is_symlink():
        return False
    return path.resolve().is_relative_to(resolved_root)


def collect_distribution_files(repo_root: Path) -> list[tuple[Path, str]]:
    """Collect all source, config, workflow, asset, and test files for distribution."""
    resolved_root = repo_root.resolve()
    collected: list[tuple[Path, str]] = []

    for filename in MANDATORY_ROOT_FILES:
        file_path = resolve_safe_path(resolved_root, filename)
        if _is_safe_source_file(file_path, resolved_root):
            collected.append((file_path, filename))
        else:
            LOGGER.warning("Mandatory root file not found or unsafe: %s", filename)

    for directory_name in INCLUDED_DIRECTORIES:
        dir_path = resolve_safe_path(resolved_root, directory_name)
        if not dir_path.is_dir() or dir_path.is_symlink():
            LOGGER.warning("Included directory not found or symlinked: %s", directory_name)
            continue

        for path in sorted(dir_path.rglob("*")):
            if not _is_safe_source_file(path, resolved_root):
                continue
            rel = path.relative_to(dir_path.parent)
            if should_ignore(rel):
                continue
            collected.append((path, rel.as_posix()))

    return collected


def _validate_release_names(version: str, archive_prefix: str | None) -> tuple[str, str]:
    """Validate semantic version and archive prefix against path-traversal characters."""
    clean_version = version.strip().lstrip("v")
    if not clean_version or not _SEMVER_RE.fullmatch(clean_version):
        raise ValueError(
            f"Invalid release version '{version}': expected semantic version like '1.0.0'."
        )
    prefix = (archive_prefix or f"StitchLocker-v{clean_version}").strip()
    if not prefix or prefix in (".", "..") or not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(
            f"Invalid archive prefix '{prefix}': only [A-Za-z0-9._-] without '/' or '..' allowed."
        )
    return clean_version, prefix


def build_release_package(
    version: str,
    repo_root: Path = REPO_ROOT,
    output_dir: Path | None = None,
    archive_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Build a standalone StitchLocker zip archive and companion SHA256SUMS.txt atomically."""
    _, prefix = _validate_release_names(version, archive_prefix)
    target_dir = (output_dir or (repo_root / "dist")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = f"{prefix}.zip"
    zip_path = resolve_safe_path(target_dir, zip_filename)
    sha256_path = resolve_safe_path(target_dir, "SHA256SUMS.txt")
    stamp = f"{os.getpid()}_{time.time_ns()}"
    zip_temp = target_dir / f"{zip_filename}.{stamp}.part"
    sha_temp = target_dir / f"SHA256SUMS.txt.{stamp}.part"

    LOGGER.info("Building release archive: %s", zip_path)
    files_to_pack = collect_distribution_files(repo_root)
    executable_mode = 0o100755 << 16
    regular_mode = 0o100644 << 16

    try:
        with zipfile.ZipFile(zip_temp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for scaffold_dir in ("outputs", "outputs/_reports"):
                dir_info = zipfile.ZipInfo(f"{prefix}/{scaffold_dir}/")
                dir_info.external_attr = 0o40755 << 16
                zf.writestr(dir_info, "")

            for src_path, arc_subpath in files_to_pack:
                zinfo = zipfile.ZipInfo(f"{prefix}/{arc_subpath}")
                zinfo.date_time = time.gmtime(src_path.stat().st_mtime)[:6]
                zinfo.external_attr = executable_mode if src_path.suffix == ".sh" else regular_mode
                zf.writestr(zinfo, src_path.read_bytes())

        LOGGER.info("Verifying archive integrity for: %s", zip_path.name)
        validation_errors = validate_package(zip_temp, prefix)
        if validation_errors:
            raise RuntimeError(
                f"Package validation failed for {zip_path.name}: {'; '.join(validation_errors)}"
            )

        checksum = compute_sha256(zip_temp)
        sha_temp.write_text(f"{checksum}  {zip_filename}\n", encoding="utf-8")
        zip_temp.replace(zip_path)
        sha_temp.replace(sha256_path)
        LOGGER.info("SHA-256: %s (%s)", checksum, zip_filename)
        return zip_path, sha256_path
    finally:
        for leftover in (zip_temp, sha_temp):
            if leftover.exists():
                with contextlib.suppress(OSError):
                    leftover.unlink()


def validate_package(zip_path: Path, prefix: str) -> list[str]:
    """Verify zip CRC integrity, mandatory runtime entries, and absence of dev caches."""
    if not zip_path.is_file():
        return [f"Archive does not exist: {zip_path}"]

    errors: list[str] = []
    try:
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            corrupt = zf.testzip()
            if corrupt:
                errors.append(f"Archive testzip failed on entry: {corrupt}")

            entries = set(zf.namelist())
            required_entries = (
                "pyproject.toml",
                "requirements.txt",
                "config.yaml",
                "app.py",
                "run_batch.py",
                "setup_test_cases.py",
                "install.bat",
                "install.sh",
                "run_pipeline.bat",
                "run_pipeline.sh",
                "src/__init__.py",
                "workflows/qwen_2_1_edit_api.json",
                "workflows/qwen3_vl_inspector_api.json",
            )
            for expected_file in required_entries:
                expected_arcname = f"{prefix}/{expected_file}"
                if expected_arcname not in entries:
                    errors.append(f"Mandatory entry missing: {expected_arcname}")

            for entry in entries:
                if any(bad in entry for bad in ("__pycache__", ".venv", ".pytest_cache")):
                    errors.append(f"Blacklisted transient file found in package: {entry}")
    except (zipfile.BadZipFile, OSError) as exc:
        errors.append(f"Failed to inspect zip file: {exc}")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for release packaging."""
    parser = argparse.ArgumentParser(
        description="Build deterministic StitchLocker distribution package."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Release version string (e.g. 1.0.0 or v1.0.0). Defaults to pyproject.toml version.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target output directory for the archive. Defaults to 'dist/'.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory. Defaults to detected repo root.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def resolve_default_version(repo_root: Path) -> str:
    """Read version from pyproject.toml as fallback."""
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.is_file():
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.1"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for StitchLocker release packaging."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    version = args.version.strip() if args.version else resolve_default_version(args.repo_root)
    try:
        zip_path, sha_path = build_release_package(
            version=version,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
        )
        print(f"[SUCCESS] Release package created: {zip_path}")
        print(f"[SUCCESS] Checksum manifest created: {sha_path}")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.exception("Failed to build release package: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
