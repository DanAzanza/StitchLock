"""Smart Local Housekeeping for intermediate ComfyUI input/output assets.

Safely purges temporary ROI uploads and intermediate files in local ComfyUI directories
with canonical path containment guards, pre-existing asset protection, and bounded
Windows NTFS file-lock micro-retries.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

LOGGER = logging.getLogger("garment_fix.housekeeper")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def is_loopback_host(server_url: str) -> bool:
    """Return True if the server URL points to a local loopback interface."""
    raw = server_url.strip().lower()
    if raw in LOOPBACK_HOSTS or raw.startswith("::1"):
        return True
    if not raw.startswith(("http://", "https://", "ws://", "wss://")):
        raw = f"http://{raw}"
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        return host in LOOPBACK_HOSTS or host.startswith("127.") or host.startswith("::1")
    except ValueError:
        return False


class ComfyUIHousekeeper:
    """Manages safe purging of temporary ComfyUI assets on local filesystems."""

    def __init__(
        self,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        enabled: bool = True,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.05,
    ) -> None:
        self.input_dir = input_dir.resolve() if input_dir and input_dir.exists() else None
        self.output_dir = output_dir.resolve() if output_dir and output_dir.exists() else None
        self.enabled = enabled
        self.max_retries = max(1, max_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._pre_existing_inputs: set[Path] = set()
        self._tracked_inputs: set[Path] = set()

    def record_pre_upload(self, asset_name: str) -> None:
        """Record whether a filename already existed in ComfyUI input prior to upload."""
        if not self.enabled or self.input_dir is None:
            return
        candidate = (self.input_dir / asset_name).resolve()
        if not candidate.is_relative_to(self.input_dir):
            return
        if candidate.exists():
            self._pre_existing_inputs.add(candidate)
        else:
            self._tracked_inputs.add(candidate)

    def record_uploaded_name(self, server_asset_name: str) -> None:
        """Track the actual server-assigned filename returned by POST /upload/image."""
        if not self.enabled or self.input_dir is None:
            return
        candidate = (self.input_dir / server_asset_name).resolve()
        if not candidate.is_relative_to(self.input_dir):
            LOGGER.warning("Rejected path traversal in record_uploaded_name: %s", server_asset_name)
            return
        if candidate not in self._pre_existing_inputs:
            self._tracked_inputs.add(candidate)

    def cleanup_session_inputs(self) -> int:
        """Purge all tracked temporary inputs uploaded during the current session."""
        if not self.enabled or self.input_dir is None:
            return 0
        purged = 0
        for target in list(self._tracked_inputs):
            if target in self._pre_existing_inputs:
                continue
            if not target.is_relative_to(self.input_dir):
                continue
            if self._safe_delete(target):
                self._tracked_inputs.discard(target)
                purged += 1
        return purged

    def _safe_delete(self, path: Path) -> bool:
        """Delete a file with bounded micro-retries for Windows NTFS handle resilience."""
        for attempt in range(self.max_retries):
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return True
            except OSError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds)
        return False
