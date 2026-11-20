"""Custom Upload, VLM Box Preview & Interactive ROI Handlers for StitchLocker Studio."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.auto_masker import AutoMasker, discover_case_paths
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
    ReferenceType,
    VerificationMetrics,
    load_pipeline_config,
)
from src.pipeline import (
    GarmentCorrectionPipeline,
    LocalizationResult,
    convert_reference_image_to_rgb,
    diff_heatmap_filename,
    execute_multi_defect_restoration,
    extract_reference_detail_crop,
    persist_correction_outputs,
    run_vlm_post_check,
)
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

ImageInput = NDArray[np.uint8] | str | Path | None

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
UI_CSS_PATH = ROOT_DIR / "src" / "ui_style.css"
VIEW_COMPARE = "Compare (Before / After)"
VIEW_HEATMAP = "Change Heatmap"


def ensure_environment_ready() -> GarmentCorrectionPipeline:
    """Load pipeline config, initialize sample fixtures if empty, and build pipeline."""
    config = load_pipeline_config(CONFIG_PATH, base_dir=ROOT_DIR)
    if not discover_case_paths(config.test_cases_dir, allow_empty=True):
        setup_all_test_cases(ROOT_DIR)
    return GarmentCorrectionPipeline(config)


def _resolve_clean_stem(path: Path, fallback_stem: str) -> str:
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return clean_stem or fallback_stem


def coerce_image_and_stem(
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
    return rgb, _resolve_clean_stem(path, fallback_stem)


def coerce_reference_image_and_stem(
    image_input: ImageInput,
    fallback_stem: str,
) -> tuple[NDArray[np.uint8] | None, str]:
    """Load reference filepath or numpy image into RGB, matting transparent logo assets."""
    if image_input is None:
        return None, fallback_stem
    if isinstance(image_input, np.ndarray):
        return image_input, fallback_stem
    path = Path(image_input)
    if not path.is_file():
        return None, fallback_stem
    with Image.open(path) as im:
        rgb = convert_reference_image_to_rgb(im)
    return rgb, _resolve_clean_stem(path, fallback_stem)


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
    model_img, _ = coerce_image_and_stem(model_input, "custom_model")
    ref_img, _ = coerce_reference_image_and_stem(ref_input, "custom_ref")
    model_preview = render_box_overlay(
        model_img, m_xmin, m_ymin, m_xmax, m_ymax, (255, 65, 65), "TARGET MASK ROI"
    )
    ref_preview = render_box_overlay(
        ref_img, r_xmin, r_ymin, r_xmax, r_ymax, (0, 225, 160), "AUTHENTIC REF CROP"
    )
    return model_preview, ref_preview


def _resolve_inspector_and_server(
    cfg: PipelineConfig,
    bridge: ComfyQwenBridge,
    inspector: Qwen3VLInspector | None = None,
) -> tuple[Qwen3VLInspector, str | None]:
    """Resolve Qwen3-VL inspector and probe a live ComfyUI server (fail-fast if unavailable)."""
    active_inspector = inspector
    if active_inspector is None:
        inspector_wf = cfg.resolved_inspector_workflow_path
        if not inspector_wf.is_file():
            raise ComfyExecutionError(f"Inspector workflow file not found: {inspector_wf}")
        active_inspector = Qwen3VLInspector(cfg.comfyui, inspector_wf, heuristics=cfg.heuristics)
    active_server = bridge.probe_live_server()
    return active_inspector, active_server


def _run_vlm_inspection_for_upload(
    model_img: NDArray[np.uint8],
    ref_img: NDArray[np.uint8],
    cfg: PipelineConfig,
    bridge: ComfyQwenBridge,
    inspector: Qwen3VLInspector | None = None,
) -> VLMInspectionResult:
    """Run Qwen3-VL inspection on an uploaded image pair (fail-fast if ComfyUI offline)."""
    active_inspector, active_server = _resolve_inspector_and_server(cfg, bridge, inspector)
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
    model_rgb, _ = coerce_image_and_stem(model_img, "custom_model")
    ref_rgb, _ = coerce_reference_image_and_stem(ref_img, "custom_ref")
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


def _build_no_defect_upload_response(
    *,
    cfg: PipelineConfig,
    output_stem: str,
    model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    vlm_result: VLMInspectionResult,
    verifier: PixelLockVerifier,
    amp_factor: float,
    heatmap_name: str,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    dict[str, Any],
]:
    """Persist and return clean short-circuit outputs when VLM detects no garment defect."""
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


def _maybe_run_upload_post_check(
    *,
    cfg: PipelineConfig,
    bridge: ComfyQwenBridge,
    inspector: Qwen3VLInspector | None,
    active_server: str | None,
    metrics: VerificationMetrics,
    manifest: CaseManifest,
    final_img: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    localization: LocalizationResult,
    defect_count: int,
) -> VerificationMetrics:
    """Run opt-in semantic/OCR VLM post-check on custom upload outputs when configured."""
    if not cfg.verification.enable_vlm_post_check:
        return metrics
    resolved_inspector = inspector
    if resolved_inspector is None and cfg.resolved_inspector_workflow_path.is_file():
        resolved_inspector = Qwen3VLInspector(
            cfg.comfyui, cfg.resolved_inspector_workflow_path, heuristics=cfg.heuristics
        )
    resolved_server = active_server
    if resolved_server is None:
        try:
            resolved_server = bridge.probe_live_server()
        except ComfyExecutionError:
            resolved_server = None
    return run_vlm_post_check(
        inspector=resolved_inspector,
        active_server=resolved_server,
        metrics=metrics,
        manifest=manifest,
        final_corrected_rgb=final_img,
        reference_product_rgb=ref_rgb,
        localization=localization,
        defect_count=defect_count,
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
    model_rgb, output_stem = coerce_image_and_stem(model_img, default_stem)
    ref_rgb, _ = coerce_reference_image_and_stem(ref_img, f"{output_stem}_reference")
    if model_rgb is None or ref_rgb is None:
        raise PipelineError("Please upload both a Model Photo and a Product Reference.")

    cfg = config_override or ensure_environment_ready().config
    active_bridge = bridge or ComfyQwenBridge(cfg.comfyui, cfg.workflow_path)
    compositor = PixelLockCompositor()
    verifier = PixelLockVerifier(cfg.verification)
    amp_factor = cfg.verification.diff_amplification_factor
    heatmap_name = diff_heatmap_filename(output_stem, amp_factor)

    vlm_result: VLMInspectionResult | None = None
    active_inspector: Qwen3VLInspector | None = inspector
    active_server: str | None = None
    model_box = NormalizedBox(m_xmin, m_ymin, m_xmax, m_ymax)
    ref_box = NormalizedBox(r_xmin, r_ymin, r_xmax, r_ymax)
    effective_prompt = prompt

    defects: tuple[DefectTarget, ...]
    detection_prefix = ""
    if auto_vlm:
        active_inspector, active_server = _resolve_inspector_and_server(
            cfg, active_bridge, inspector
        )
        vlm_result = active_inspector.inspect_garment_pair(
            model_rgb=model_rgb,
            ref_rgb=ref_rgb,
            case_id="adhoc_custom_upload",
            server_url=active_server,
        )
        if not vlm_result.has_defect:
            return _build_no_defect_upload_response(
                cfg=cfg,
                output_stem=output_stem,
                model_rgb=model_rgb,
                ref_rgb=ref_rgb,
                vlm_result=vlm_result,
                verifier=verifier,
                amp_factor=amp_factor,
                heatmap_name=heatmap_name,
            )

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

    target_detail = (
        vlm_result.defect_summary
        if vlm_result is not None and vlm_result.defect_summary
        else "Custom uploaded garment detail"
    )
    manifest = CaseManifest(
        case_id=output_stem,
        title="Custom Live Upload",
        garment_category="custom",
        target_detail=target_detail,
        prompt=effective_prompt,
        model_image_filename=f"{output_stem}.png",
        reference_image_filename=f"{output_stem}_reference.png",
        model_target_box=model_box,
        reference_detail_box=ref_box,
        reference_type=(
            vlm_result.reference_type if vlm_result is not None else ReferenceType.GARMENT
        ),
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
    metrics = _maybe_run_upload_post_check(
        cfg=cfg,
        bridge=active_bridge,
        inspector=active_inspector,
        active_server=active_server,
        metrics=metrics,
        manifest=manifest,
        final_img=final_img,
        ref_rgb=ref_rgb,
        localization=loc,
        defect_count=len(defects),
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
