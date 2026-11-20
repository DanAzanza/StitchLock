"""Pure UI Inbox, Batch & Session Lifecycle Handlers for StitchLocker Studio."""

from __future__ import annotations

import html
import json
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.auto_masker import (
    discover_case_paths,
    load_case_manifest,
    resolve_case_path,
)
from src.models import (
    MaskingStrategy,
    PipelineConfig,
    PipelineError,
    resolve_safe_path,
)
from src.pipeline import (
    REPORTS_SUBDIR,
    GarmentCorrectionPipeline,
    convert_reference_image_to_rgb,
    diff_heatmap_filename,
    save_rgb_atomic,
)
from src.ui_custom_handlers import (
    CONFIG_PATH,
    ROOT_DIR,
    UI_CSS_PATH,
    VIEW_COMPARE,
    VIEW_HEATMAP,
    ImageInput,
    coerce_image_and_stem,
    coerce_reference_image_and_stem,
    detect_custom_ui,
    ensure_environment_ready,
    preview_custom_boxes_ui,
    render_box_overlay,
    render_multi_box_overlay,
    run_custom_upload_ui,
)
from src.verifier import PixelLockVerifier

__all__ = [
    "CONFIG_PATH",
    "ROOT_DIR",
    "UI_CSS_PATH",
    "VIEW_COMPARE",
    "VIEW_HEATMAP",
    "BrowserSessionLifecycle",
    "ImageInput",
    "add_pair_to_inbox",
    "build_add_look_tile",
    "build_artist_heatmap",
    "coerce_image_and_stem",
    "detect_custom_ui",
    "ensure_environment_ready",
    "execute_batch_ui",
    "format_look_bar",
    "gallery_with_add_tile",
    "get_inbox_overview_data",
    "inspect_single_case_ui",
    "is_case_repaired",
    "load_single_case_view_ui",
    "preview_custom_boxes_ui",
    "read_reference_rgb_optional",
    "read_rgb_optional",
    "render_box_overlay",
    "render_multi_box_overlay",
    "run_custom_upload_ui",
    "slider_pair",
]


class BrowserSessionLifecycle:
    """Tracks connected browser tabs and fires shutdown cleanly after last disconnect."""

    def __init__(self, on_shutdown: Callable[[], None], grace_seconds: float = 2.5) -> None:
        self._on_shutdown = on_shutdown
        self._grace_seconds = grace_seconds
        self._lock = threading.Lock()
        self._active_clients = 0
        self._timer: threading.Timer | None = None

    @property
    def active_clients(self) -> int:
        with self._lock:
            return self._active_clients

    def on_connect(self) -> None:
        """Register an active browser tab and cancel any pending shutdown timer."""
        with self._lock:
            self._active_clients += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def on_disconnect(self) -> None:
        """Schedule graceful shutdown once all browser tabs have closed."""
        with self._lock:
            self._active_clients = max(0, self._active_clients - 1)
            if self._active_clients > 0:
                return
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._grace_seconds, self._fire_if_idle)
            timer.daemon = True
            timer.start()
            self._timer = timer

    def cancel(self) -> None:
        """Cancel any pending shutdown timer cleanly."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _fire_if_idle(self) -> None:
        with self._lock:
            if self._active_clients > 0:
                return
            self._timer = None
        self._on_shutdown()


def read_rgb_optional(path: Path) -> NDArray[np.uint8] | None:
    """Read an optional image file as RGB uint8 numpy array, returning None if absent."""
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_reference_rgb_optional(path: Path) -> NDArray[np.uint8] | None:
    """Read an optional reference image as RGB, matting transparent logo PNGs safely."""
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return convert_reference_image_to_rgb(im)


def slider_pair(
    orig_rgb: NDArray[np.uint8] | None,
    final_rgb: NDArray[np.uint8] | None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
    """Build Gradio ImageSlider input tuple when both images are available."""
    if orig_rgb is not None and final_rgb is not None:
        return (orig_rgb, final_rgb)
    return None


def format_look_bar(case_id: str | None) -> str:
    """Render HTML header bar showing the currently selected look ID."""
    label = case_id or "No look selected"
    return f"<div class='stitch-look-bar'>{html.escape(label)}</div>"


def get_inbox_overview_data(
    config_override: PipelineConfig | None = None,
) -> tuple[list[list[str]], list[tuple[str, str]], list[str]]:
    """Scan `test_cases/` and return status rows, labeled thumbnails, and sorted case IDs."""
    cfg = config_override or ensure_environment_ready().config
    case_paths = discover_case_paths(cfg.test_cases_dir, allow_empty=True)

    rows: list[list[str]] = []
    thumbnails: list[tuple[str, str]] = []
    case_ids: list[str] = []

    for case_path in case_paths:
        manifest = load_case_manifest(case_path)
        case_root = case_path if case_path.is_dir() else case_path.parent
        model_file = resolve_safe_path(case_root, manifest.model_image_filename)
        resolve_safe_path(case_root, manifest.reference_image_filename)
        with Image.open(model_file) as im:
            w, h = im.size
        out_file = resolve_safe_path(cfg.outputs_dir, f"{manifest.case_id}.png")
        is_ready = out_file.is_file()
        status = f"Ready in outputs/{out_file.name}" if is_ready else "Pending"
        thumb_label = f"✓ {manifest.case_id}" if is_ready else manifest.case_id

        rows.append(
            [
                manifest.case_id,
                manifest.model_image_filename,
                manifest.reference_image_filename,
                f"{w}x{h}",
                status,
            ]
        )
        thumbnails.append((str(model_file), thumb_label))
        case_ids.append(manifest.case_id)

    return rows, thumbnails, case_ids


def load_single_case_view_ui(
    case_id: str | None,
    config_override: PipelineConfig | None = None,
) -> tuple[
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    dict[str, Any] | None,
]:
    """Load input pair plus any saved output/heatmap from `outputs/` for a single case."""
    if not case_id:
        return None, None, None, None, None, None

    cfg = config_override or ensure_environment_ready().config
    case_path = resolve_case_path(cfg.test_cases_dir, case_id)
    manifest = load_case_manifest(case_path)
    case_root = case_path if case_path.is_dir() else case_path.parent

    orig_rgb = read_rgb_optional(resolve_safe_path(case_root, manifest.model_image_filename))
    ref_rgb = read_reference_rgb_optional(
        resolve_safe_path(case_root, manifest.reference_image_filename)
    )

    reports_dir = resolve_safe_path(cfg.outputs_dir, REPORTS_SUBDIR)
    final_rgb = read_rgb_optional(resolve_safe_path(cfg.outputs_dir, f"{manifest.case_id}.png"))
    report_rgb = read_rgb_optional(reports_dir / f"{manifest.case_id}_verification_report.png")
    heatmap_name = diff_heatmap_filename(
        manifest.case_id, cfg.verification.diff_amplification_factor
    )
    heatmap_rgb = read_rgb_optional(reports_dir / heatmap_name)
    if heatmap_rgb is None:
        heatmap_rgb = read_rgb_optional(reports_dir / f"{manifest.case_id}_diff_heatmap_x50.png")

    metrics_path = reports_dir / f"{manifest.case_id}_metrics.json"
    metrics_data: dict[str, Any] | None = None
    if metrics_path.is_file():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    return orig_rgb, ref_rgb, final_rgb, heatmap_rgb, report_rgb, metrics_data


def build_add_look_tile(size: int = 320) -> NDArray[np.uint8]:
    """Render a clean `+` thumbnail card for adding a new look into the Inbox."""
    canvas = Image.new("RGB", (size, size), color=(20, 30, 48))
    draw = ImageDraw.Draw(canvas)
    pad = max(14, size // 16)
    border_col = (51, 65, 85)
    dash_len = max(10, size // 24)
    gap_len = max(8, size // 32)
    step = dash_len + gap_len
    x0, y0, x1, y1 = pad, pad, size - pad, size - pad
    for pos in range(x0, x1, step):
        seg_end = min(pos + dash_len, x1)
        draw.line([(pos, y0), (seg_end, y0)], fill=border_col, width=3)
        draw.line([(pos, y1), (seg_end, y1)], fill=border_col, width=3)
    for pos in range(y0, y1, step):
        seg_end = min(pos + dash_len, y1)
        draw.line([(x0, pos), (x0, seg_end)], fill=border_col, width=3)
        draw.line([(x1, pos), (x1, seg_end)], fill=border_col, width=3)

    cx, cy = size // 2, size // 2
    arm = max(28, size // 7)
    stroke = max(6, size // 40)
    plus_col = (16, 185, 129)
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=plus_col, width=stroke)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=plus_col, width=stroke)
    return np.asarray(canvas, dtype=np.uint8)


def gallery_with_add_tile(thumbnails: list[tuple[str, str]]) -> list[tuple[Any, str]]:
    """Append the `+ Add Look` tile at the end of existing Inbox look thumbnails."""
    return [*thumbnails, (build_add_look_tile(), "Add Look")]


def is_case_repaired(
    orig_rgb: NDArray[np.uint8] | None,
    final_rgb: NDArray[np.uint8] | None,
    metrics_data: dict[str, Any] | None,
) -> bool:
    """Return True only when an output image actually differs from the original."""
    if orig_rgb is None or final_rgb is None:
        return False
    if metrics_data is not None:
        return int(metrics_data.get("masked_changed_pixels_count", 0)) > 0
    return not np.array_equal(orig_rgb, final_rgb)


def build_artist_heatmap(
    orig_rgb: NDArray[np.uint8] | None,
    final_rgb: NDArray[np.uint8] | None,
    config_override: PipelineConfig | None = None,
) -> NDArray[np.uint8] | None:
    """Compute a balanced change heatmap using the configured amplification factor."""
    if orig_rgb is None or final_rgb is None:
        return None
    abs_diff = np.abs(orig_rgb.astype(np.int16) - final_rgb.astype(np.int16))
    changed = (np.max(abs_diff, axis=2) > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(changed)) == 0:
        return None
    closed_mask = np.asarray(
        cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8)),
        dtype=np.uint8,
    )
    cfg = config_override or ensure_environment_ready().config
    return PixelLockVerifier(cfg.verification).build_diff_heatmap(orig_rgb, final_rgb, closed_mask)


def _resolve_unique_case_stem(test_cases_dir: Path, raw_stem: str) -> str:
    """Return a clean, collision-free case ID for a newly uploaded look pair."""
    base = re.sub(r"_reference$", "", raw_stem, flags=re.IGNORECASE).strip("._-")
    base = base or "uploaded_look"
    existing_ids = {p.stem.lower() for p in discover_case_paths(test_cases_dir, allow_empty=True)}
    if base.lower() not in existing_ids:
        return base
    idx = 2
    while f"{base}_{idx}".lower() in existing_ids:
        idx += 1
    return f"{base}_{idx}"


def add_pair_to_inbox(
    model_img: ImageInput,
    ref_img: ImageInput,
    config_override: PipelineConfig | None = None,
) -> str:
    """Persist an uploaded (Model Photo, Product Reference) pair directly into `test_cases/`."""
    model_rgb, model_stem = coerce_image_and_stem(model_img, "uploaded_look")
    ref_rgb, _ = coerce_reference_image_and_stem(ref_img, "uploaded_ref")
    if model_rgb is None or ref_rgb is None:
        raise PipelineError("Please upload both a Model Photo and a Product Reference.")

    cfg = config_override or ensure_environment_ready().config
    cfg.test_cases_dir.mkdir(parents=True, exist_ok=True)
    case_id = _resolve_unique_case_stem(cfg.test_cases_dir, model_stem)

    model_dest = resolve_safe_path(cfg.test_cases_dir, f"{case_id}.png")
    ref_dest = resolve_safe_path(cfg.test_cases_dir, f"{case_id}_reference.png")
    save_rgb_atomic(model_dest, model_rgb)
    save_rgb_atomic(ref_dest, ref_rgb)
    return case_id


def execute_batch_ui(
    strategy_val: str = MaskingStrategy.AUTO_VISION.value,
) -> tuple[
    str,
    list[tuple[NDArray[np.uint8], str]],
    dict[str, Any],
    list[tuple[str, str]],
]:
    """Run all test cases in batch mode and return status summary + updated thumbnails."""
    base_pipeline = ensure_environment_ready()
    cfg = replace(
        base_pipeline.config,
        masking=replace(base_pipeline.config.masking, strategy=MaskingStrategy(strategy_val)),
    )
    pipeline = GarmentCorrectionPipeline(cfg)
    artifacts = pipeline.run_all_cases()

    passed_count = sum(1 for a in artifacts if a.metrics.passed)
    summary_text = (
        f"**✓ Batch Complete** · **{passed_count} / {len(artifacts)}** looks fixed and saved to "
        "`outputs/` · Click any thumbnail on the left to compare Before/After or Heatmap."
    )
    gallery_items = [(a.final_corrected_rgb, f"{a.manifest.case_id}") for a in artifacts]
    _, thumbs, _ = get_inbox_overview_data(cfg)
    summary_json = {
        "all_passed": passed_count == len(artifacts),
        "cases": [a.metrics.to_dict() for a in artifacts],
    }
    return summary_text, gallery_items, summary_json, thumbs


def inspect_single_case_ui(
    case_id: str,
    strategy_val: str = MaskingStrategy.AUTO_VISION.value,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    list[tuple[str, str]],
]:
    """Fix a single selected look and return updated images plus refreshed thumbnails."""
    if not case_id:
        raise PipelineError("Please select a look from the Inbox first.")
    base_pipeline = ensure_environment_ready()
    cfg = replace(
        base_pipeline.config,
        masking=replace(base_pipeline.config.masking, strategy=MaskingStrategy(strategy_val)),
    )
    pipeline = GarmentCorrectionPipeline(cfg)
    case_path = resolve_case_path(cfg.test_cases_dir, case_id)
    artifact = pipeline.run_case(case_path)
    _, thumbs, _ = get_inbox_overview_data(cfg)

    return (
        artifact.original_model_rgb,
        artifact.reference_product_rgb,
        artifact.final_corrected_rgb,
        artifact.diff_heatmap_rgb,
        thumbs,
    )
