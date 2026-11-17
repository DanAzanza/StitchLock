"""Automated ComfyUI Server Detection, Healthcheck, and Headless Launcher."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from src.models import ComfyConfig, ComfyExecutionError


@dataclass(frozen=True)
class ComfyInstance:
    """Metadata describing a locally detected ComfyUI installation."""

    name: str
    mode: str  # "headless" or "gui"
    python_exe: Path | None = None
    main_py: Path | None = None
    working_dir: Path | None = None
    base_dir: Path | None = None
    user_dir: Path | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    extra_model_paths: Path | None = None
    extra_args: tuple[str, ...] = ()
    gui_exe: Path | None = None


_SPAWNED_PROCESS: subprocess.Popen[Any] | None = None
_SPAWNED_SERVER_URL: str | None = None
_ATEXIT_REGISTERED = False


def get_desktop_appdata_dir() -> Path | None:
    """Return the platform-specific configuration directory for Comfy Desktop."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "Comfy Desktop" if appdata else None
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Comfy Desktop"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "Comfy Desktop"
    return Path.home() / ".config" / "Comfy Desktop"


def _resolve_main_py(install_path: Path) -> tuple[Path | None, Path]:
    nested = install_path / "ComfyUI" / "main.py"
    if nested.is_file():
        return nested, nested.parent
    direct = install_path / "main.py"
    if direct.is_file():
        return direct, install_path
    return None, install_path


def _parse_desktop_entry(inst: dict[str, Any], config_dir: Path) -> ComfyInstance | None:
    if inst.get("status") != "installed":
        return None
    py_raw = inst.get("adoptedPythonPath")
    path_raw = inst.get("installPath")
    if not py_raw or not path_raw:
        return None

    python_exe = Path(str(py_raw))
    install_path = Path(str(path_raw))
    if not python_exe.is_file() or not install_path.is_dir():
        return None

    main_py, working_dir = _resolve_main_py(install_path)
    if main_py is None:
        return None

    base_raw = inst.get("adoptedBaseDir")
    base_dir = Path(str(base_raw)) if base_raw and Path(str(base_raw)).is_dir() else None
    user_dir = base_dir / "user" if base_dir else None

    in_raw = inst.get("inputDir")
    input_dir = Path(str(in_raw)) if in_raw and Path(str(in_raw)).is_dir() else None
    out_raw = inst.get("outputDir")
    output_dir = Path(str(out_raw)) if out_raw and Path(str(out_raw)).is_dir() else None

    inst_id = str(inst.get("id", ""))
    inst_yaml = config_dir / "instance-model-paths" / f"{inst_id}.yaml"
    shared_yaml = config_dir / "shared_model_paths.yaml"
    extra_model_paths = (
        inst_yaml if inst_yaml.is_file() else (shared_yaml if shared_yaml.is_file() else None)
    )
    return ComfyInstance(
        name=str(inst.get("name", "ComfyUI Desktop Backend")),
        mode="headless",
        python_exe=python_exe,
        main_py=main_py,
        working_dir=working_dir,
        base_dir=base_dir,
        user_dir=user_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        extra_model_paths=extra_model_paths,
        extra_args=("--enable-manager", "--enable-assets"),
    )


def detect_comfy_desktop_instance() -> ComfyInstance | None:
    """Detect local ComfyUI backend configured via Comfy Desktop `installations.json`."""
    config_dir = get_desktop_appdata_dir()
    if config_dir is None or not config_dir.is_dir():
        return None
    installations_file = config_dir / "installations.json"
    if not installations_file.is_file():
        return None
    try:
        data = json.loads(installations_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_desktop_entry(entry, config_dir)
        if parsed is not None:
            return parsed
    return None


def _resolve_standalone_python(cand: Path) -> Path:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    py_name = "python.exe" if sys.platform == "win32" else "python"
    for env_name in (".venv", "venv"):
        venv_py = cand / env_name / bin_dir / py_name
        if venv_py.is_file():
            return venv_py
        if sys.platform != "win32":
            py3 = cand / env_name / bin_dir / "python3"
            if py3.is_file():
                return py3
    if sys.platform == "win32":
        embedded = cand.parent / "python_embeded" / "python.exe"
        if embedded.is_file():
            return embedded
        return Path(sys.executable)
    which_py3 = shutil.which("python3")
    return Path(which_py3) if which_py3 else Path(sys.executable)


def detect_standalone_instance() -> ComfyInstance | None:
    """Detect standalone git clones, portable distributions, or environment overrides."""
    env_dir = os.environ.get("COMFYUI_DIR")
    env_py = os.environ.get("COMFYUI_PYTHON")
    if env_dir:
        cand_dir = Path(env_dir)
        main_py, work_dir = _resolve_main_py(cand_dir)
        if main_py is not None:
            py_exe = (
                Path(env_py)
                if env_py and Path(env_py).is_file()
                else _resolve_standalone_python(cand_dir)
            )
            return ComfyInstance(
                name="Environment ComfyUI",
                mode="headless",
                python_exe=py_exe,
                main_py=main_py,
                working_dir=work_dir,
                base_dir=cand_dir,
            )

    if sys.platform == "win32":
        candidates = [
            Path.home() / "Documents" / "ComfyUI",
            Path.home() / "ComfyUI",
            Path("C:/ComfyUI_windows_portable/ComfyUI"),
            Path("D:/ComfyUI_windows_portable/ComfyUI"),
        ]
    else:
        candidates = [
            Path.home() / "ComfyUI",
            Path.home() / "comfy" / "ComfyUI",
            Path.home() / "Documents" / "ComfyUI",
            Path("/workspace/ComfyUI"),
            Path("/opt/ComfyUI"),
        ]
    for cand in candidates:
        if not cand.is_dir():
            continue
        main_py, work_dir = _resolve_main_py(cand)
        if main_py is None:
            continue
        return ComfyInstance(
            name=f"Standalone ({cand.name})",
            mode="headless",
            python_exe=_resolve_standalone_python(cand),
            main_py=main_py,
            working_dir=work_dir,
            base_dir=cand,
        )
    return None


def detect_desktop_gui_exe() -> ComfyInstance | None:
    """Detect Comfy Desktop Electron executable as GUI fallback."""
    gui_paths: list[Path] = []
    if sys.platform == "win32":
        prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        local_app = os.environ.get("LOCALAPPDATA", "")
        gui_paths.extend(
            [
                Path(prog_files) / "Comfy Desktop" / "Comfy Desktop.exe",
                Path(local_app) / "Programs" / "Comfy Desktop" / "Comfy Desktop.exe",
                Path(prog_files) / "ComfyUI" / "ComfyUI.exe",
                Path(local_app) / "Programs" / "ComfyUI" / "ComfyUI.exe",
            ]
        )
    elif sys.platform == "darwin":
        gui_paths.append(Path("/Applications/Comfy Desktop.app/Contents/MacOS/Comfy Desktop"))

    for path in gui_paths:
        if path.is_file():
            return ComfyInstance(name="Comfy Desktop GUI", mode="gui", gui_exe=path)
    return None


def detect_comfyui() -> ComfyInstance | None:
    """Detect the highest-priority local ComfyUI installation."""
    return (
        detect_comfy_desktop_instance() or detect_standalone_instance() or detect_desktop_gui_exe()
    )


def build_headless_command(instance: ComfyInstance, host: str, port: str) -> list[str]:
    """Build subprocess argv vector to start ComfyUI headlessly."""
    if instance.python_exe is None or instance.main_py is None:
        raise ComfyExecutionError(
            f"ComfyUI instance '{instance.name}' is missing python_exe or main_py."
        )
    cmd = [
        str(instance.python_exe),
        str(instance.main_py),
        "--listen",
        host,
        "--port",
        port,
    ]
    optional_flags = [
        ("--base-directory", instance.base_dir),
        ("--user-directory", instance.user_dir),
        ("--input-directory", instance.input_dir),
        ("--output-directory", instance.output_dir),
        ("--extra-model-paths-config", instance.extra_model_paths),
    ]
    for flag, path_val in optional_flags:
        if path_val is not None:
            cmd.extend([flag, str(path_val)])
    cmd.extend(instance.extra_args)
    return cmd


def is_comfy_server_online(server_url: str, timeout: float = 1.5) -> bool:
    """Return True if ComfyUI responds to `/system_stats` with system/device metadata."""
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/system_stats", timeout=timeout)
        if resp.status_code != 200:
            return False
        payload = resp.json()
        return isinstance(payload, dict) and ("devices" in payload or "system" in payload)
    except (requests.RequestException, ValueError, OSError):
        return False


def probe_candidate_servers(config: ComfyConfig) -> str | None:
    """Return the first responsive ComfyUI server URL from `config`, or None if offline."""
    parsed = urlparse(config.server_url)
    host = parsed.hostname or "127.0.0.1"
    scheme = parsed.scheme or "http"
    candidates = [config.server_url.rstrip("/")]
    for port in config.fallback_ports:
        url = f"{scheme}://{host}:{port}"
        if url not in candidates:
            candidates.append(url)
    for base_url in candidates:
        if is_comfy_server_online(base_url, timeout=1.5):
            return base_url
    return None


def _spawn_instance(instance: ComfyInstance, host: str, port: str) -> subprocess.Popen[Any]:
    if instance.mode == "headless":
        cmd = build_headless_command(instance, host, port)
        cwd = str(instance.working_dir or instance.main_py.parent)  # type: ignore[union-attr]
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 6  # SW_MINIMIZE
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            return subprocess.Popen(cmd, cwd=cwd, startupinfo=startupinfo, creationflags=flags)
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if instance.mode == "gui" and instance.gui_exe is not None:
        if sys.platform == "win32":
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            return subprocess.Popen([str(instance.gui_exe)], creationflags=detached)
        return subprocess.Popen([str(instance.gui_exe)], start_new_session=True)

    raise ComfyExecutionError(f"Unsupported ComfyUI instance mode: '{instance.mode}'.")


def stop_auto_started_server() -> None:
    """Cleanly unload VRAM and stop the ComfyUI server spawned by this process."""
    global _SPAWNED_PROCESS, _SPAWNED_SERVER_URL
    proc = _SPAWNED_PROCESS
    server_url = _SPAWNED_SERVER_URL
    _SPAWNED_PROCESS = None
    _SPAWNED_SERVER_URL = None

    if server_url and is_comfy_server_online(server_url, timeout=1.0):
        with contextlib.suppress(requests.RequestException, OSError):
            requests.post(f"{server_url}/interrupt", timeout=2.0)
        with contextlib.suppress(requests.RequestException, OSError):
            requests.post(
                f"{server_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=3.0,
            )

    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        return
    killpg = getattr(os, "killpg", None)
    if callable(killpg):
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=4.0)
            return
        with contextlib.suppress(OSError):
            sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            killpg(proc.pid, sigkill)
        return
    with contextlib.suppress(OSError):
        proc.terminate()
        proc.wait(timeout=4.0)


def ensure_comfyui_server(config: ComfyConfig, startup_timeout: float = 90.0) -> str:
    """Return a live ComfyUI URL, auto-launching a detected installation if configured."""
    global _SPAWNED_PROCESS, _SPAWNED_SERVER_URL, _ATEXIT_REGISTERED

    active = probe_candidate_servers(config)
    if active is not None:
        return active

    if not config.auto_launch:
        raise ComfyExecutionError(
            "ComfyUI server is offline and auto_launch is disabled. "
            f"Checked {config.server_url} and ports {list(config.fallback_ports)}. "
            "Recovery: Start ComfyUI manually or set comfyui.auto_launch: true in config.yaml."
        )

    instance = detect_comfyui()
    if instance is None:
        raise ComfyExecutionError(
            "ComfyUI server is offline and no local ComfyUI Desktop or standalone "
            "installation was detected. Recovery: Install/start ComfyUI or set COMFYUI_DIR."
        )

    target_url = config.server_url.rstrip("/")
    parsed = urlparse(target_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8000)

    print(f"[AUTO-START] Launching ComfyUI ({instance.name}) on {target_url}...")
    proc = _spawn_instance(instance, host, port)
    _SPAWNED_PROCESS = proc
    _SPAWNED_SERVER_URL = target_url
    if not _ATEXIT_REGISTERED:
        atexit.register(stop_auto_started_server)
        _ATEXIT_REGISTERED = True

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_comfy_server_online(target_url, timeout=1.5):
            print(f"[OK] ComfyUI server is online at {target_url}.")
            return target_url
        if proc.poll() is not None and instance.mode == "headless":
            raise ComfyExecutionError(
                f"Auto-started ComfyUI ({instance.name}) exited prematurely with code "
                f"{proc.returncode}."
            )
        time.sleep(1.0)

    raise ComfyExecutionError(
        f"ComfyUI ({instance.name}) did not respond at {target_url} within {int(startup_timeout)}s."
    )
