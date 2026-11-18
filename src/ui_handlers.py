"""Pure UI Service Handlers & Session Lifecycle for StitchLocker Studio."""

from __future__ import annotations

import html
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.auto_masker import (
    AutoMasker,
    discover_case_paths,
    load_case_manifest,
    resolve_case_path,
)
from src.comfy_client import ComfyQwenBridge
from src.compositor import PixelLockCompositor
from src.fixtures import setup_all_test_cases
from src.models import (
    CaseManifest,
    ComfyExecutionError,
    DefectTarget,
    MaskingStrategy,
    NormalizedBox,
    PipelineConfig,
    PipelineError,
    VerificationMetrics,
    load_pipeline_config,
    resolve_safe_path,
)
from src.pipeline import (
    REPORTS_SUBDIR,
    GarmentCorrectionPipeline,
    diff_heatmap_filename,
    execute_multi_defect_restoration,
    extract_reference_detail_crop,
    persist_correction_outputs,
    save_rgb_atomic,
)
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

ImageInput = NDArray[np.uint8] | str | Path | None

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
UI_CSS_PATH = ROOT_DIR / "src" / "ui_style.css"
VIEW_COMPARE = "Compare (Before / After)"
VIEW_HEATMAP = "Change Heatmap"


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


def ensure_environment_ready() -> GarmentCorrectionPipeline:
    """Load pipeline config, initialize sample fixtures if empty, and build pipeline."""
    config = load_pipeline_config(CONFIG_PATH, base_dir=ROOT_DIR)
    if not discover_case_paths(config.test_cases_dir, allow_empty=True):
        setup_all_test_cases(ROOT_DIR)
    return GarmentCorrectionPipeline(config)


def read_rgb_optional(path: Path) -> NDArray[np.uint8] | None:
    """Read an optional image file as RGB uint8 numpy array, returning None if absent."""
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


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
    ref_rgb = read_rgb_optional(resolve_safe_path(case_root, manifest.reference_image_filename))

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
    model_rgb, model_stem = _coerce_image_and_stem(model_img, "uploaded_look")
    ref_rgb, _ = _coerce_image_and_stem(ref_img, "uploaded_ref")
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


def render_multi_box_overlay(
    image: NDArray[np.uint8] | None,
    boxes: list[tuple[NormalizedBox, tuple[int, int, int], str]],
) -> NDArray[np.uint8] | None:
    """Draw one or more interactive normalized bounding box overlays onto an image."""
    if image is None:
        return None
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for norm_box, color, label in boxes:
        px = norm_box.to_pixel_bbox(w, h)
        cv2.rectangle(canvas, (px.x1, px.y1), (px.x2, px.y2), color, 3)
        cv2.putText(
            canvas,
            label,
            (px.x1, max(18, px.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def render_box_overlay(
    image: NDArray[np.uint8] | None,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    color: tuple[int, int, int],
    label: str,
) -> NDArray[np.uint8] | None:
    """Draw a live interactive bounding box overlay on an uploaded image."""
    if image is None:
        return None
    if x_max <= x_min or y_max <= y_min:
        return image.copy()
    clamped = NormalizedBox(
        max(0.0, min(0.98, float(x_min))),
        max(0.0, min(0.98, float(y_min))),
        max(0.02, min(1.0, float(x_max))),
        max(0.02, min(1.0, float(y_max))),
    )
    return render_multi_box_overlay(image, [(clamped, color, label)])


def _coerce_image_and_stem(
    image_input: ImageInput,
    fallback_stem: str,
) -> tuple[NDArray[np.uint8] | None, str]:
    """Load filepath or numpy image into RGB array while preserving a safe filename stem."""
    if image_input is None:
        return None, fallback_stem
    if isinstance(image_input, np.ndarray):
        return image_input, fallback_stem
    path = Path(image_input)
    if not path.is_file():
        return None, fallback_stem
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return rgb, (clean_stem or fallback_stem)


def preview_custom_boxes_ui(
    model_input: ImageInput,
    ref_input: ImageInput,
    m_xmin: float,
    m_ymin: float,
    m_xmax: float,
    m_ymax: float,
    r_xmin: float,
    r_ymin: float,
    r_xmax: float,
    r_ymax: float,
) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8] | None]:
    """Update both bounding-box preview images whenever sliders or uploads change."""
    model_img, _ = _coerce_image_and_stem(model_input, "custom_model")
    ref_img, _ = _coerce_image_and_stem(ref_input, "custom_ref")
    model_preview = render_box_overlay(
        model_img, m_xmin, m_ymin, m_xmax, m_ymax, (255, 65, 65), "TARGET MASK ROI"
    )
    ref_preview = render_box_overlay(
        ref_img, r_xmin, r_ymin, r_xmax, r_ymax, (0, 225, 160), "AUTHENTIC REF CROP"
    )
    return model_preview, ref_preview


def _run_vlm_inspection_for_upload(
    model_img: NDArray[np.uint8],
    ref_img: NDArray[np.uint8],
    cfg: PipelineConfig,
    bridge: ComfyQwenBridge,
    inspector: Qwen3VLInspector | None = None,
) -> VLMInspectionResult:
    """Run Qwen3-VL inspection on an uploaded image pair (fail-fast if ComfyUI offline)."""
    active_inspector = inspector
    if active_inspector is None:
        inspector_wf = cfg.resolved_inspector_workflow_path
        if not inspector_wf.is_file():
            raise ComfyExecutionError(f"Inspector workflow file not found: {inspector_wf}")
        active_inspector = Qwen3VLInspector(cfg.comfyui, inspector_wf, heuristics=cfg.heuristics)

    active_server = bridge.probe_live_server()
    return active_inspector.inspect_garment_pair(
        model_rgb=model_img,
        ref_rgb=ref_img,
        case_id="adhoc_custom_upload",
        server_url=active_server,
    )


def detect_custom_ui(
    model_img: ImageInput,
    ref_img: ImageInput,
    *,
    config_override: PipelineConfig | None = None,
    inspector: Qwen3VLInspector | None = None,
    bridge: ComfyQwenBridge | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
    NDArray[np.uint8] | None,
    NDArray[np.uint8] | None,
    dict[str, Any],
]:
    """Run Qwen3-VL auto-detection on uploaded images to inspect detected regions."""
    model_rgb, _ = _coerce_image_and_stem(model_img, "custom_model")
    ref_rgb, _ = _coerce_image_and_stem(ref_img, "custom_ref")
    if model_rgb is None or ref_rgb is None:
        raise PipelineError("Please upload both a Model Photo and a Product Reference.")

    cfg = config_override or ensure_environment_ready().config
    active_bridge = bridge or ComfyQwenBridge(cfg.comfyui, cfg.workflow_path)
    vlm_result = _run_vlm_inspection_for_upload(
        model_img=model_rgb,
        ref_img=ref_rgb,
        cfg=cfg,
        bridge=active_bridge,
        inspector=inspector,
    )
    mb = vlm_result.model_target_box
    rb = vlm_result.reference_detail_box
    defects = vlm_result.resolved_defects
    if defects:
        m_prev = render_multi_box_overlay(
            model_rgb,
            [
                (d.model_target_box, (255, 65, 65), f"TARGET #{idx}")
                for idx, d in enumerate(defects, start=1)
            ],
        )
        r_prev = render_multi_box_overlay(
            ref_rgb,
            [
                (d.reference_detail_box, (0, 225, 160), f"REF #{idx}")
                for idx, d in enumerate(defects, start=1)
            ],
        )
    else:
        m_prev, r_prev = preview_custom_boxes_ui(
            model_rgb,
            ref_rgb,
            mb.x_min,
            mb.y_min,
            mb.x_max,
            mb.y_max,
            rb.x_min,
            rb.y_min,
            rb.x_max,
            rb.y_max,
        )
    return (
        mb.x_min,
        mb.y_min,
        mb.x_max,
        mb.y_max,
        rb.x_min,
        rb.y_min,
        rb.x_max,
        rb.y_max,
        vlm_result.edit_prompt,
        m_prev,
        r_prev,
        vlm_result.to_dict(),
    )


def run_custom_upload_ui(
    model_img: ImageInput,
    ref_img: ImageInput,
    m_xmin: float = 0.25,
    m_ymin: float = 0.25,
    m_xmax: float = 0.75,
    m_ymax: float = 0.75,
    r_xmin: float = 0.25,
    r_ymin: float = 0.25,
    r_xmax: float = 0.75,
    r_ymax: float = 0.75,
    prompt: str = "",
    auto_vlm: bool = True,
    *,
    config_override: PipelineConfig | None = None,
    inspector: Qwen3VLInspector | None = None,
    bridge: ComfyQwenBridge | None = None,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    dict[str, Any],
]:
    """Execute live ComfyUI correction on an uploaded pair and save outputs to `outputs/`."""
    default_stem = f"custom_upload_{time.strftime('%Y%m%d_%H%M%S')}"
    model_rgb, output_stem = _coerce_image_and_stem(model_img, default_stem)
    ref_rgb, _ = _coerce_image_and_stem(ref_img, f"{output_stem}_reference")
    if model_rgb is None or ref_rgb is None:
        raise PipelineError("Please upload both a Model Photo and a Product Reference.")

    cfg = config_override or ensure_environment_ready().config
    active_bridge = bridge or ComfyQwenBridge(cfg.comfyui, cfg.workflow_path)
    compositor = PixelLockCompositor()
    verifier = PixelLockVerifier(cfg.verification)
    amp_factor = cfg.verification.diff_amplification_factor
    heatmap_name = diff_heatmap_filename(output_stem, amp_factor)

    vlm_result: VLMInspectionResult | None = None
    model_box = NormalizedBox(m_xmin, m_ymin, m_xmax, m_ymax)
    ref_box = NormalizedBox(r_xmin, r_ymin, r_xmax, r_ymax)
    effective_prompt = prompt

    defects: tuple[DefectTarget, ...]
    detection_prefix = ""
    if auto_vlm:
        vlm_result = _run_vlm_inspection_for_upload(
            model_img=model_rgb,
            ref_img=ref_rgb,
            cfg=cfg,
            bridge=active_bridge,
            inspector=inspector,
        )
        if not vlm_result.has_defect:
            h, w = model_rgb.shape[:2]
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            total_px = int(h * w)
            no_defect_metrics = VerificationMetrics(
                case_id=output_stem,
                passed=True,
                unmasked_max_abs_diff=0,
                unmasked_changed_pixels_count=0,
                unmasked_changed_pixels_pct=0.0,
                unmasked_psnr=100.0,
                unmasked_ssim=1.0,
                masked_mean_abs_diff=0.0,
                masked_changed_pixels_count=0,
                total_pixels=total_px,
                unmasked_pixels=total_px,
                masked_pixels=0,
                backend_used="qwen3_vl_no_defect_short_circuit",
                detection_method=vlm_result.inspector_source,
            )
            diff_heatmap = verifier.build_diff_heatmap(model_rgb, model_rgb, empty_mask)
            report_img = verifier.render_four_panel_report(
                model_rgb, model_rgb, empty_mask, ref_rgb, no_defect_metrics
            )
            reports_dir, saved_out, _, _ = persist_correction_outputs(
                outputs_dir=cfg.outputs_dir,
                case_id=output_stem,
                final_corrected_rgb=model_rgb,
                four_panel_report_rgb=report_img,
                diff_heatmap_rgb=diff_heatmap,
                metrics=no_defect_metrics,
                vlm_result=vlm_result,
                diff_amplification_factor=amp_factor,
            )
            payload = no_defect_metrics.to_dict()
            payload["vlm_inspection"] = vlm_result.to_dict()
            payload["saved_output_path"] = str(saved_out)
            payload["saved_heatmap_path"] = str(reports_dir / heatmap_name)
            return model_rgb.copy(), model_rgb.copy(), report_img, payload

        model_box = vlm_result.model_target_box
        ref_box = vlm_result.reference_detail_box
        effective_prompt = vlm_result.edit_prompt or prompt
        defects = vlm_result.resolved_defects
        detection_prefix = vlm_result.inspector_source
    else:
        defects = (
            DefectTarget(
                defect_summary="Custom uploaded garment detail",
                model_target_box=model_box,
                reference_detail_box=ref_box,
                edit_prompt=effective_prompt,
            ),
        )

    manifest = CaseManifest(
        case_id=output_stem,
        title="Custom Live Upload",
        garment_category="custom",
        target_detail="Custom uploaded garment detail",
        prompt=effective_prompt,
        model_image_filename=f"{output_stem}.png",
        reference_image_filename=f"{output_stem}_reference.png",
        model_target_box=model_box,
        reference_detail_box=ref_box,
    )

    mask_strategy = MaskingStrategy.AUTO_VISION if auto_vlm else MaskingStrategy.MANIFEST_ROI
    masker_cfg = replace(cfg.masking, strategy=mask_strategy)
    masker = AutoMasker(masker_cfg)

    restoration = execute_multi_defect_restoration(
        original_model_rgb=model_rgb,
        ref_rgb=ref_rgb,
        base_manifest=manifest,
        defects=defects,
        masker=masker,
        bridge=active_bridge,
        compositor=compositor,
        detection_prefix=detection_prefix,
    )
    loc = restoration.localization
    final_img = restoration.final_corrected_rgb
    metrics = verifier.verify(
        manifest.case_id,
        model_rgb,
        final_img,
        loc.binary_mask,
        backend_used=restoration.backend_used,
        detection_method=loc.detection_method,
        reference_crop=extract_reference_detail_crop(ref_rgb, loc.ref_detail_bbox),
    )
    diff_heatmap = verifier.build_diff_heatmap(model_rgb, final_img, loc.binary_mask)
    report_img = verifier.render_four_panel_report(
        model_rgb, final_img, loc.binary_mask, loc.cropped_product_ref, metrics
    )
    reports_dir, saved_out, _, _ = persist_correction_outputs(
        outputs_dir=cfg.outputs_dir,
        case_id=output_stem,
        final_corrected_rgb=final_img,
        four_panel_report_rgb=report_img,
        diff_heatmap_rgb=diff_heatmap,
        metrics=metrics,
        vlm_result=vlm_result,
        diff_amplification_factor=amp_factor,
    )
    metrics_payload = metrics.to_dict()
    metrics_payload["saved_output_path"] = str(saved_out)
    metrics_payload["saved_heatmap_path"] = str(reports_dir / heatmap_name)
    if vlm_result is not None:
        metrics_payload["vlm_inspection"] = vlm_result.to_dict()
    return restoration.restored_roi_rgb, final_img, report_img, metrics_payload
