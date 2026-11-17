"""EU AI Act Article 50 provenance metadata & zero-recompression PNG chunk processor.

Directly parses and rewrites PNG `tEXt` chunks with CRC32 validation without
decompressing or re-encoding `IDAT` pixel streams, guaranteeing zero pixel mutation
while stripping proprietary ComfyUI workflow graphs and embedding Pixel-Lock audit tags.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import uuid
import zlib
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("garment_fix.metadata")

DEFAULT_SOFTWARE_TAG: Final[str] = "StitchLocker (garment-fix-pipeline)"
DEFAULT_AI_GENERATED_TAG: Final[str] = "true"
DEFAULT_AI_SYSTEM_TAG: Final[str] = "StitchLocker / Qwen-Image-2.1"
DEFAULT_AI_COMPLIANCE_TAG: Final[str] = "EU-AI-Act-Article-50-Synthetic-Media"

PROPRIETARY_CHUNKS_TO_STRIP: Final[frozenset[str]] = frozenset({"prompt", "workflow"})
PROVENANCE_TAG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "software",
        "ai-generated",
        "ai-system",
        "ai-compliance",
        "case-id",
        "pixel-lock-verified",
        "unmasked-max-diff",
        "unmasked-ssim",
    }
)

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


def _build_text_chunk(keyword: str, text: str) -> bytes:
    """Build a standard PNG tEXt chunk with valid CRC32."""
    kw_bytes = keyword.encode("latin-1", errors="replace")
    txt_bytes = text.encode("latin-1", errors="replace")
    chunk_data = kw_bytes + b"\x00" + txt_bytes
    chunk_type = b"tEXt"
    length = struct.pack(">I", len(chunk_data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    return length + chunk_type + chunk_data + crc


def embed_png_provenance_metadata(
    image_path: Path,
    case_id: str = "unknown",
    pixel_lock_verified: bool = True,
    unmasked_max_diff: int = 0,
    unmasked_ssim: float = 1.0,
    keep_workflow: bool = False,
) -> bool:
    """Embed EU AI Act Article 50 & Pixel-Lock audit tags into PNG without touching IDAT pixels."""
    if not image_path.is_file():
        return False

    tmp_target: Path | None = None
    try:
        raw_bytes = image_path.read_bytes()
        if not raw_bytes.startswith(PNG_SIGNATURE):
            return False

        tags_to_inject = [
            ("Software", DEFAULT_SOFTWARE_TAG),
            ("AI-Generated", DEFAULT_AI_GENERATED_TAG),
            ("AI-System", DEFAULT_AI_SYSTEM_TAG),
            ("AI-Compliance", DEFAULT_AI_COMPLIANCE_TAG),
            ("Case-ID", case_id),
            ("Pixel-Lock-Verified", "true" if pixel_lock_verified else "false"),
            ("Unmasked-Max-Diff", str(unmasked_max_diff)),
            ("Unmasked-SSIM", f"{unmasked_ssim:.6f}"),
        ]

        output_chunks: list[bytes] = [PNG_SIGNATURE]
        offset = len(PNG_SIGNATURE)
        total_len = len(raw_bytes)
        has_ihdr = False
        has_iend = False

        while offset + 8 <= total_len:
            length = struct.unpack(">I", raw_bytes[offset : offset + 4])[0]
            chunk_type = raw_bytes[offset + 4 : offset + 8]
            chunk_total_len = 12 + length

            if offset + chunk_total_len > total_len:
                return False

            chunk_data = raw_bytes[offset + 8 : offset + 8 + length]
            chunk_crc = struct.unpack(
                ">I", raw_bytes[offset + 8 + length : offset + chunk_total_len]
            )[0]

            calc_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if chunk_crc != calc_crc:
                return False

            if chunk_type == b"IHDR":
                has_ihdr = True

            if chunk_type in (b"tEXt", b"iTXt", b"zTXt"):
                parts = chunk_data.split(b"\x00", 1)
                keyword = parts[0].decode("latin-1", errors="ignore").lower() if parts else ""
                if not keep_workflow and keyword in PROPRIETARY_CHUNKS_TO_STRIP:
                    offset += chunk_total_len
                    continue
                if keyword in PROVENANCE_TAG_KEYS:
                    offset += chunk_total_len
                    continue

            if chunk_type == b"IEND":
                has_iend = True
                for kw, val in tags_to_inject:
                    output_chunks.append(_build_text_chunk(kw, val))

            output_chunks.append(raw_bytes[offset : offset + chunk_total_len])
            offset += chunk_total_len

        if not has_ihdr or not has_iend:
            return False

        tmp_target = image_path.with_name(f"{image_path.stem}_{uuid.uuid4().hex[:8]}.meta.part")
        tmp_target.write_bytes(b"".join(output_chunks))
        tmp_target.replace(image_path)
        return True

    except (OSError, ValueError, struct.error, zlib.error) as err:
        LOGGER.debug("Skipping PNG metadata processing on %s: %s", image_path.name, err)
        return False
    finally:
        if tmp_target is not None and tmp_target.exists():
            with contextlib.suppress(OSError):
                tmp_target.unlink()


def read_png_text_chunks(image_path: Path) -> dict[str, str]:
    """Read all tEXt key-value metadata pairs from a PNG file."""
    if not image_path.is_file():
        return {}
    raw_bytes = image_path.read_bytes()
    if not raw_bytes.startswith(PNG_SIGNATURE):
        return {}

    result: dict[str, str] = {}
    offset = len(PNG_SIGNATURE)
    total_len = len(raw_bytes)
    while offset + 8 <= total_len:
        length = struct.unpack(">I", raw_bytes[offset : offset + 4])[0]
        chunk_type = raw_bytes[offset + 4 : offset + 8]
        chunk_total_len = 12 + length
        if offset + chunk_total_len > total_len:
            break
        if chunk_type == b"tEXt":
            chunk_data = raw_bytes[offset + 8 : offset + 8 + length]
            parts = chunk_data.split(b"\x00", 1)
            if len(parts) == 2:
                key = parts[0].decode("latin-1", errors="ignore")
                val = parts[1].decode("latin-1", errors="ignore")
                result[key] = val
        offset += chunk_total_len
    return result
