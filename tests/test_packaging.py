"""Test suite for release packaging, version alignment, and GitHub release workflow parity."""

from __future__ import annotations

import sys
import tomllib
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import scripts.package_release as pkg_cli
import src
import yaml
from src.packaging import (
    REPO_ROOT,
    build_release_package,
    collect_distribution_files,
    compute_sha256,
    parse_args,
    resolve_default_version,
    should_ignore,
    validate_package,
)
from src.packaging import (
    main as pkg_main,
)


def _create_mock_repo(root: Path) -> None:
    for fname in (
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
    ):
        (root / fname).write_text(f"content of {fname}", encoding="utf-8")

    (root / "src").mkdir()
    (root / "src" / "__init__.py").write_text("__version__ = '1.2.3'\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "package_release.py").write_text("print('pkg')", encoding="utf-8")
    (root / "workflows").mkdir()
    (root / "workflows" / "qwen_2_1_edit_api.json").write_text("{}", encoding="utf-8")
    (root / "workflows" / "qwen3_vl_inspector_api.json").write_text("{}", encoding="utf-8")
    (root / "docs" / "assets").mkdir(parents=True)
    (root / "docs" / "assets" / "ui.png").write_bytes(b"png")
    (root / "test_cases").mkdir()
    (root / "test_cases" / "kova_zip_jacket.png").write_bytes(b"m")
    (root / "test_cases" / "kova_zip_jacket_reference.png").write_bytes(b"r")
    (root / "tests").mkdir()
    (root / "tests" / "test_sample.py").write_text("def test_ok(): pass", encoding="utf-8")
    (root / "outputs" / "_reports").mkdir(parents=True)
    (root / "outputs" / "kova_zip_jacket.png").write_bytes(b"generated")


def test_compute_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 checksum matches known digest."""
    test_file = tmp_path / "hello.txt"
    test_file.write_bytes(b"hello world\n")
    expected = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    assert compute_sha256(test_file) == expected


def test_should_ignore() -> None:
    """Verify cache, build, and runtime artifact filtering."""
    assert should_ignore(Path("src/__pycache__/module.pyc")) is True
    assert should_ignore(Path(".pytest_cache/v/cache/nodeids")) is True
    assert should_ignore(Path(".ruff_cache/0.1/abc")) is True
    assert should_ignore(Path(".venv/lib/python.exe")) is True
    assert should_ignore(Path("dist/StitchLocker-v1.0.0.zip")) is True
    assert should_ignore(Path("outputs/test.png.part")) is True
    assert should_ignore(Path(".comfyui_server.pid")) is True
    assert should_ignore(Path("Thumbs.db")) is True
    assert should_ignore(Path("coverage.xml")) is True

    assert should_ignore(Path("src/packaging.py")) is False
    assert should_ignore(Path("workflows/qwen_2_1_edit_api.json")) is False
    assert should_ignore(Path("test_cases/kova_zip_jacket.png")) is False
    assert should_ignore(Path("install.sh")) is False


def test_collect_distribution_files_from_real_repo() -> None:
    """Verify real repository collection includes runtime/workflows/test_cases and skips outputs."""
    collected = collect_distribution_files(REPO_ROOT)
    arcnames = {arc for _, arc in collected}

    for expected in (
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
        "src/packaging.py",
        "scripts/package_release.py",
        "scripts/run_ci_locally.py",
        "workflows/qwen_2_1_edit_api.json",
        "workflows/qwen3_vl_inspector_api.json",
        "test_cases/kova_zip_jacket.png",
        "test_cases/kova_zip_jacket_reference.png",
    ):
        assert expected in arcnames

    assert not any(arc.startswith("outputs/") for arc in arcnames)


def test_build_release_package_mock_repo(tmp_path: Path) -> None:
    """Verify release archive structure, checksums, POSIX permissions, and excluded outputs."""
    fake_repo = tmp_path / "build" / "fake_repo"
    fake_repo.mkdir(parents=True)
    _create_mock_repo(fake_repo)

    zip_path, sha_path = build_release_package(
        version="1.2.3",
        repo_root=fake_repo,
        output_dir=tmp_path / "dist",
    )

    assert zip_path.is_file()
    assert sha_path.is_file()
    assert zip_path.name == "StitchLocker-v1.2.3.zip"

    sha_text = sha_path.read_text(encoding="utf-8")
    assert "StitchLocker-v1.2.3.zip" in sha_text
    assert compute_sha256(zip_path) in sha_text

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        names = set(zf.namelist())
        assert "StitchLocker-v1.2.3/config.yaml" in names
        assert "StitchLocker-v1.2.3/app.py" in names
        assert "StitchLocker-v1.2.3/install.sh" in names
        assert "StitchLocker-v1.2.3/run_pipeline.sh" in names
        assert "StitchLocker-v1.2.3/outputs/" in names
        assert "StitchLocker-v1.2.3/outputs/_reports/" in names
        assert "StitchLocker-v1.2.3/outputs/kova_zip_jacket.png" not in names

        sh_mode = (zf.getinfo("StitchLocker-v1.2.3/install.sh").external_attr >> 16) & 0o777
        assert sh_mode == 0o755
        py_mode = (zf.getinfo("StitchLocker-v1.2.3/src/__init__.py").external_attr >> 16) & 0o777
        assert py_mode == 0o644


def test_validate_package_negative_and_boundary_cases(tmp_path: Path) -> None:
    """Verify validate_package rejects missing, corrupt, incomplete, and polluted zips."""
    missing_errs = validate_package(tmp_path / "missing.zip", "StitchLocker-v1.0.0")
    assert any("does not exist" in err for err in missing_errs)

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not-a-zip")
    corrupt_errs = validate_package(corrupt, "StitchLocker-v1.0.0")
    assert any("Failed to inspect zip file" in err for err in corrupt_errs)

    incomplete = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete, "w") as zf:
        zf.writestr("prefix/dummy.txt", "data")
    assert any("Mandatory entry missing" in err for err in validate_package(incomplete, "prefix"))

    polluted = tmp_path / "polluted.zip"
    with zipfile.ZipFile(polluted, "w") as zf:
        for req in (
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
        ):
            zf.writestr(f"prefix/{req}", "ok")
        zf.writestr("prefix/src/__pycache__/bad.pyc", "bad")
    assert any("Blacklisted transient file" in err for err in validate_package(polluted, "prefix"))


def test_build_release_package_raises_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_release_package fails closed and leaves zero partial/unvalidated artifacts."""
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    _create_mock_repo(fake_repo)
    dist_dir = tmp_path / "dist"

    monkeypatch.setattr("src.packaging.validate_package", lambda *_a, **_kw: ["Mock error"])
    with pytest.raises(RuntimeError, match="Package validation failed"):
        build_release_package(version="1.0.0", repo_root=fake_repo, output_dir=dist_dir)
    assert list(dist_dir.iterdir()) == []


def test_release_package_rejects_traversal_invalid_version_and_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify invalid versions, path-traversal prefixes, and external symlinks are rejected."""
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    _create_mock_repo(fake_repo)
    dist_dir = tmp_path / "dist"

    for bad_ver in ("", "   ", "v", "../evil", "1.0.0/../../bad", "1.0\\evil"):
        with pytest.raises(ValueError, match="Invalid release version"):
            build_release_package(version=bad_ver, repo_root=fake_repo, output_dir=dist_dir)

    for bad_prefix in ("../evil", "a/b", "..", "."):
        with pytest.raises(ValueError, match="Invalid archive prefix"):
            build_release_package(
                version="1.0.0",
                repo_root=fake_repo,
                output_dir=dist_dir,
                archive_prefix=bad_prefix,
            )

    secret_file = tmp_path / "outside_secret.txt"
    secret_file.write_text("secret", encoding="utf-8")
    escape_link = fake_repo / "src" / "escape_link.txt"
    try:
        escape_link.symlink_to(secret_file)
    except OSError:
        (fake_repo / "src" / "escape_link.txt").write_text("dummy", encoding="utf-8")
        orig_resolve = Path.resolve

        def _fake_resolve(self: Path, strict: bool = False) -> Path:
            if self.name == "escape_link.txt":
                return secret_file.resolve(strict=strict)
            return orig_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _fake_resolve)

    collected_arcs = {arc for _, arc in collect_distribution_files(fake_repo)}
    assert "src/escape_link.txt" not in collected_arcs


def test_version_alignment_cli_and_release_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pyproject.toml == src.__version__, CLI entrypoint, and release workflow YAML."""
    pyproject_ver = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    assert src.__version__ == pyproject_ver
    assert resolve_default_version(REPO_ROOT) == pyproject_ver
    assert resolve_default_version(tmp_path) == "1.0.0"

    args = parse_args(["--version", "v1.0.0", "--verbose", "--output-dir", "custom_dist"])
    assert args.version == "v1.0.0"
    assert args.verbose is True
    assert args.output_dir == Path("custom_dist")
    assert hasattr(pkg_cli, "build_release_package")

    monkeypatch.setattr(
        "src.packaging.build_release_package",
        lambda **_kw: (tmp_path / "out.zip", tmp_path / "SHA256SUMS.txt"),
    )
    assert pkg_main(["--version", "1.0.0"]) == 0

    mock_fail = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr("src.packaging.build_release_package", mock_fail)
    assert pkg_main(["--version", "1.0.0"]) == 1

    release_yaml = REPO_ROOT / ".github" / "workflows" / "release.yml"
    if not release_yaml.is_file():
        pytest.skip("Release workflow .github/workflows/release.yml not present in archive")
    yaml_text = release_yaml.read_text(encoding="utf-8")
    parsed_wf = yaml.safe_load(yaml_text)
    assert isinstance(parsed_wf, dict)
    assert "quality-gate" in parsed_wf.get("jobs", {})
    assert "publish-release" in parsed_wf.get("jobs", {})
    assert "scripts/package_release.py" in yaml_text
    assert "scripts/run_ci_locally.py" in yaml_text
    assert "gh release create" in yaml_text
    assert "contents: write" in yaml_text


def test_release_workflow_check_skips_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify workflow YAML check skips cleanly when .github/ is omitted from release zips."""
    this_mod = sys.modules[__name__]
    monkeypatch.setattr(this_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{src.__version__}"\n', encoding="utf-8"
    )
    with pytest.raises(pytest.skip.Exception, match="Release workflow"):
        test_version_alignment_cli_and_release_workflow(tmp_path, monkeypatch)
