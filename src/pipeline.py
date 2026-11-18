"""End-to-End Garment Detail Correction Pipeline Orchestrator."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.auto_masker import (
    AutoMasker,
    discover_case_paths,
    has_manifest_roi,
    load_case_manifest,
)
from src.comfy_client import ComfyQwenBridge
from src.compositor import PixelLockCompositor
from src.metadata import embed_png_provenance_metadata
from src.models import (
    CaseManifest,
    ConfigurationError,
    DefectTarget,
    LocalizationResult,
    MaskingStrategy,
    PipelineConfig,
    PixelBBox,
    VerificationMetrics,
    resolve_safe_path,
)
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

REPORTS_SUBDIR = "_reports"
LOGGER = logging.getLogger("garment_fix.pipeline")


def extract_padded_roi(
    image_rgb: NDArray[np.uint8], bbox: PixelBBox, pad_ratio: float = 0.08
) -> NDArray[np.uint8]:
    """Extract a context-padded detail crop around bbox for high-resolution VLM inspection."""
    h, w = image_rgb.shape[:2]
    pad_x = max(8, int(round(w * pad_ratio)))
    pad_y = max(8, int(round(h * pad_ratio)))
    x1 = max(0, bbox.x1 - pad_x)
    y1 = max(0, bbox.y1 - pad_y)
    x2 = min(w, bbox.x2 + pad_x)
    y2 = min(h, bbox.y2 + pad_y)
    return image_rgb[y1:y2, x1:x2].copy()


def diff_heatmap_filename(case_id: str, amplification_factor: float = 4.0) -> str:
    """Build deterministic heatmap filename encoding the amplification factor."""
    factor_tag = max(1, int(round(amplification_factor)))
    return f"{case_id}_diff_heatmap_x{factor_tag}.png"


def extract_reference_detail_crop(
    ref_rgb: NDArray[np.uint8], ref_detail_bbox: PixelBBox
) -> NDArray[np.uint8]:
    """Extract the localized reference detail patch defined by ref_detail_bbox."""
    return ref_rgb[
        ref_detail_bbox.y1 : ref_detail_bbox.y2,
        ref_detail_bbox.x1 : ref_detail_bbox.x2,
    ].copy()


def save_rgb_atomic(
    path: Path,
    array_rgb: NDArray[np.uint8],
    *,
    alpha_channel: NDArray[np.uint8] | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Write RGB/RGBA PNG image atomically via a temporary .part staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = path.with_suffix(f"{path.suffix}.{os.getpid()}_{time.time_ns()}.part")
    try:
        if alpha_channel is not None and alpha_channel.shape == array_rgb.shape[:2]:
            rgba = np.dstack([array_rgb, alpha_channel])
            pil_out = Image.fromarray(rgba, mode="RGBA")
        else:
            pil_out = Image.fromarray(array_rgb, mode="RGB")
        if icc_profile is not None:
            pil_out.save(temp_dest, format="PNG", icc_profile=icc_profile)
        else:
            pil_out.save(temp_dest, format="PNG")
        temp_dest.replace(path)
    finally:
        if temp_dest.exists():
            with contextlib.suppress(OSError):
                temp_dest.unlink()


def save_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON payload atomically via a temporary .part staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = path.with_suffix(f"{path.suffix}.{os.getpid()}_{time.time_ns()}.part")
    try:
        with temp_dest.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        temp_dest.replace(path)
    finally:
        if temp_dest.exists():
            with contextlib.suppress(OSError):
                temp_dest.unlink()


def persist_correction_outputs(
    *,
    outputs_dir: Path,
    case_id: str,
    final_corrected_rgb: NDArray[np.uint8],
    four_panel_report_rgb: NDArray[np.uint8],
    diff_heatmap_rgb: NDArray[np.uint8],
    metrics: VerificationMetrics,
    vlm_result: VLMInspectionResult | None = None,
    diff_amplification_factor: float = 4.0,
    alpha_channel: NDArray[np.uint8] | None = None,
    icc_profile: bytes | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Persist corrected image directly in `outputs/<case_id>.png` and reports in `_reports/`."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = resolve_safe_path(outputs_dir, REPORTS_SUBDIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    corrected_path = resolve_safe_path(outputs_dir, f"{case_id}.png")
    report_path = resolve_safe_path(reports_dir, f"{case_id}_verification_report.png")
    heatmap_path = resolve_safe_path(
        reports_dir, diff_heatmap_filename(case_id, diff_amplification_factor)
    )
    metrics_json_path = resolve_safe_path(reports_dir, f"{case_id}_metrics.json")

    save_rgb_atomic(
        corrected_path,
        final_corrected_rgb,
        alpha_channel=alpha_channel,
        icc_profile=icc_profile,
    )
    embed_png_provenance_metadata(
        corrected_path,
        case_id=case_id,
        pixel_lock_verified=metrics.passed,
        unmasked_max_diff=metrics.unmasked_max_abs_diff,
        unmasked_ssim=metrics.unmasked_ssim,
    )

    save_rgb_atomic(report_path, four_panel_report_rgb)
    embed_png_provenance_metadata(
        report_path,
        case_id=case_id,
        pixel_lock_verified=metrics.passed,
        unmasked_max_diff=metrics.unmasked_max_abs_diff,
        unmasked_ssim=metrics.unmasked_ssim,
    )

    save_rgb_atomic(heatmap_path, diff_heatmap_rgb)
    save_json_atomic(metrics_json_path, metrics.to_dict())
    if vlm_result is not None:
        vlm_json_path = resolve_safe_path(reports_dir, f"{case_id}_vlm_inspection.json")
        save_json_atomic(vlm_json_path, vlm_result.to_dict())

    return reports_dir, corrected_path, report_path, metrics_json_path


def run_vlm_post_check(
    *,
    inspector: Qwen3VLInspector | None,
    active_server: str | None,
    metrics: VerificationMetrics,
    manifest: CaseManifest,
    final_corrected_rgb: NDArray[np.uint8],
    reference_product_rgb: NDArray[np.uint8],
    localization: LocalizationResult,
    defect_count: int,
) -> VerificationMetrics:
    """Run fail-closed semantic/OCR VLM post-check when enabled in configuration."""
    if inspector is None or active_server is None:
        LOGGER.warning(
            "VLM post-check enabled for '%s' but inspector or ComfyUI server is unavailable; "
            "failing closed.",
            manifest.case_id,
        )
        return replace(metrics, passed=False, vlm_post_check_passed=False)

    if defect_count == 1:
        repaired_view = extract_padded_roi(final_corrected_rgb, localization.target_bbox)
        ref_view = extract_padded_roi(reference_product_rgb, localization.ref_detail_bbox)
    else:
        repaired_view = final_corrected_rgb
        ref_view = reference_product_rgb

    vlm_ok, vlm_reason = inspector.verify_repaired_detail(
        repaired_rgb=repaired_view,
        ref_rgb=ref_view,
        target_detail=manifest.target_detail,
        case_id=manifest.case_id,
        server_url=active_server,
    )
    if vlm_ok:
        LOGGER.info("VLM post-check passed for '%s': %s", manifest.case_id, vlm_reason)
    else:
        LOGGER.warning("VLM post-check failed for '%s': %s", manifest.case_id, vlm_reason)
    return replace(metrics, passed=metrics.passed and vlm_ok, vlm_post_check_passed=vlm_ok)


@dataclass(frozen=True)
class RestorationPassOutput:
    """Aggregated output from sequential single- or multi-defect restoration."""

    localization: LocalizationResult
    restored_roi_rgb: NDArray[np.uint8]
    final_corrected_rgb: NDArray[np.uint8]
    backend_used: str


def execute_multi_defect_restoration(
    *,
    original_model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    base_manifest: CaseManifest,
    defects: Sequence[DefectTarget],
    masker: AutoMasker,
    bridge: ComfyQwenBridge,
    compositor: PixelLockCompositor,
    detection_prefix: str = "",
) -> RestorationPassOutput:
    """Sequentially localize, repair, and composite one or more separated defects."""
    if not defects:
        raise ConfigurationError(
            f"Cannot execute restoration for '{base_manifest.case_id}': "
            "zero defect targets supplied."
        )

    height, width = original_model_rgb.shape[:2]
    current_rgb = original_model_rgb.copy()
    union_binary_mask = np.zeros((height, width), dtype=np.uint8)
    union_feathered_mask = np.zeros((height, width), dtype=np.float32)

    last_loc: LocalizationResult | None = None
    last_restored_roi = original_model_rgb.copy()
    backend_used = ""
    total = len(defects)

    for idx, defect in enumerate(defects, start=1):
        step_manifest = replace(
            base_manifest,
            target_detail=defect.defect_summary,
            model_target_box=defect.model_target_box,
            reference_detail_box=defect.reference_detail_box,
            prompt=defect.edit_prompt,
        )
        loc = masker.localize_and_mask(
            model_rgb=current_rgb,
            ref_rgb=ref_rgb,
            manifest=step_manifest,
        )
        step_case_id = f"{base_manifest.case_id}_d{idx}" if total > 1 else base_manifest.case_id
        restored_roi_rgb, backend_used = bridge.restore_detail_roi(
            cropped_model_roi=loc.cropped_model_roi,
            cropped_product_ref=loc.cropped_product_ref,
            cropped_mask_greyscale=loc.cropped_mask_greyscale,
            prompt=defect.edit_prompt,
            case_id=step_case_id,
        )
        current_rgb = compositor.composite(
            original_image=current_rgb,
            restored_roi=restored_roi_rgb,
            binary_mask=loc.binary_mask,
            feathered_mask=loc.feathered_mask,
            context_bbox=loc.context_bbox,
        )
        union_binary_mask = np.maximum(union_binary_mask, loc.binary_mask)
        union_feathered_mask = np.maximum(union_feathered_mask, loc.feathered_mask)
        last_loc = loc
        last_restored_roi = restored_roi_rgb

    if last_loc is None:
        raise ConfigurationError(
            f"Restoration loop produced no localization for '{base_manifest.case_id}'."
        )

    # Enforce global hard pixel-lock relative to original_model_rgb outside union_binary_mask
    final_corrected_rgb = np.where(
        union_binary_mask[..., None] > 0, current_rgb, original_model_rgb
    )
    cleaned_feather = np.where(union_binary_mask > 0, union_feathered_mask, np.float32(0.0))
    base_method = (
        f"{detection_prefix} + {last_loc.detection_method}"
        if detection_prefix
        else last_loc.detection_method
    )
    detection_method = f"{base_method} ({total} defects)" if total > 1 else base_method
    aggregated_loc = LocalizationResult(
        binary_mask=union_binary_mask,
        feathered_mask=cleaned_feather,
        cropped_model_roi=original_model_rgb.copy(),
        cropped_product_ref=ref_rgb.copy() if total > 1 else last_loc.cropped_product_ref,
        cropped_mask_greyscale=np.clip(np.round(cleaned_feather * 255.0), 0, 255).astype(np.uint8),
        context_bbox=last_loc.context_bbox,
        target_bbox=last_loc.target_bbox,
        ref_detail_bbox=last_loc.ref_detail_bbox,
        detection_method=detection_method,
    )
    return RestorationPassOutput(
        localization=aggregated_loc,
        restored_roi_rgb=last_restored_roi,
        final_corrected_rgb=final_corrected_rgb,
        backend_used=backend_used,
    )


@dataclass(frozen=True)
class CaseExecutionArtifact:
    """In-memory and on-disk artifacts produced for a single garment correction case."""

    manifest: CaseManifest
    original_model_rgb: NDArray[np.uint8]
    reference_product_rgb: NDArray[np.uint8]
    localization: LocalizationResult
    restored_roi_rgb: NDArray[np.uint8]
    final_corrected_rgb: NDArray[np.uint8]
    diff_heatmap_rgb: NDArray[np.uint8]
    four_panel_report_rgb: NDArray[np.uint8]
    metrics: VerificationMetrics
    output_case_dir: Path
    corrected_image_path: Path
    report_image_path: Path
    metrics_json_path: Path


class GarmentCorrectionPipeline:
    """Coordinates auto-masking, Qwen 2.1 restoration, pixel-lock compositing, and verification."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        inspector: Qwen3VLInspector | None = None,
        masker: AutoMasker | None = None,
        bridge: ComfyQwenBridge | None = None,
        compositor: PixelLockCompositor | None = None,
        verifier: PixelLockVerifier | None = None,
    ) -> None:
        self._config = config
        self._masker = masker or AutoMasker(config.masking)
        self._bridge = bridge or ComfyQwenBridge(config.comfyui, config.workflow_path)
        self._compositor = compositor or PixelLockCompositor()
        self._verifier = verifier or PixelLockVerifier(config.verification)
        inspector_wf = config.resolved_inspector_workflow_path
        self._inspector = (
            inspector
            if inspector is not None
            else (
                Qwen3VLInspector(config.comfyui, inspector_wf, heuristics=config.heuristics)
                if inspector_wf.is_file()
                else None
            )
        )

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def discover_cases(self) -> list[Path]:
        """Discover flat `<name>.*` + `<name>_reference.*` pairs and legacy manifest folders."""
        return discover_case_paths(self._config.test_cases_dir)

    def run_case(self, case_path: Path) -> CaseExecutionArtifact:
        """Execute all pipeline stages on a single test case (flat pair or folder) and persist."""
        manifest = load_case_manifest(case_path)
        case_root = case_path if case_path.is_dir() else case_path.parent
        model_path = resolve_safe_path(case_root, manifest.model_image_filename)
        ref_path = resolve_safe_path(case_root, manifest.reference_image_filename)

        original_model_rgb, model_alpha, model_icc = self._load_rgb_with_metadata(model_path)
        reference_product_rgb = self._load_rgb(ref_path)

        # Stage 0: Qwen3-VL-8B Automated Defect Inspection (when auto_vision strategy active)
        vlm_result: VLMInspectionResult | None = None
        active_server = self._bridge.probe_live_server()
        defects: Sequence[DefectTarget]
        detection_prefix = ""
        if (
            self._config.masking.strategy == MaskingStrategy.AUTO_VISION
            and self._inspector is not None
        ):
            vlm_result = self._inspector.inspect_garment_pair(
                original_model_rgb,
                reference_product_rgb,
                manifest.case_id,
                active_server,
            )
            if not vlm_result.has_defect:
                return self._finalize_no_defect_case(
                    manifest=manifest,
                    original_model_rgb=original_model_rgb,
                    reference_product_rgb=reference_product_rgb,
                    vlm_result=vlm_result,
                    alpha_channel=model_alpha,
                    icc_profile=model_icc,
                )
            defects = vlm_result.resolved_defects
            detection_prefix = vlm_result.inspector_source
            manifest = replace(
                manifest,
                target_detail=vlm_result.defect_summary or manifest.target_detail,
                model_target_box=vlm_result.model_target_box,
                reference_detail_box=vlm_result.reference_detail_box,
                prompt=vlm_result.edit_prompt or manifest.prompt,
            )
        else:
            if (
                self._config.masking.strategy == MaskingStrategy.MANIFEST_ROI
                and not has_manifest_roi(case_path)
            ):
                raise ConfigurationError(
                    f"Case '{manifest.case_id}' has no manifest ROI boxes for 'manifest_roi' mode. "
                    f"Add a '{manifest.case_id}.json' sidecar or use '--strategy auto_vision'."
                )
            defects = (
                DefectTarget(
                    defect_summary=manifest.target_detail,
                    model_target_box=manifest.model_target_box,
                    reference_detail_box=manifest.reference_detail_box,
                    edit_prompt=manifest.prompt,
                ),
            )

        # Stage 1-3: Sequential Multi-Defect Localization, Diffusion Restoration & Pixel-Lock
        restoration = execute_multi_defect_restoration(
            original_model_rgb=original_model_rgb,
            ref_rgb=reference_product_rgb,
            base_manifest=manifest,
            defects=defects,
            masker=self._masker,
            bridge=self._bridge,
            compositor=self._compositor,
            detection_prefix=detection_prefix,
        )
        localization = restoration.localization
        restored_roi_rgb = restoration.restored_roi_rgb
        final_corrected_rgb = restoration.final_corrected_rgb
        backend_used = restoration.backend_used

        # Stage 4: Automated Verification Gate
        metrics = self._verifier.verify(
            case_id=manifest.case_id,
            original_image=original_model_rgb,
            final_image=final_corrected_rgb,
            binary_mask=localization.binary_mask,
            backend_used=backend_used,
            detection_method=localization.detection_method,
            reference_crop=extract_reference_detail_crop(
                reference_product_rgb, localization.ref_detail_bbox
            ),
        )
        if self._config.verification.enable_vlm_post_check:
            metrics = self._run_vlm_post_check(
                metrics=metrics,
                manifest=manifest,
                final_corrected_rgb=final_corrected_rgb,
                reference_product_rgb=reference_product_rgb,
                localization=localization,
                defect_count=len(defects),
                active_server=active_server,
            )

        diff_heatmap_rgb = self._verifier.build_diff_heatmap(
            original_image=original_model_rgb,
            final_image=final_corrected_rgb,
            binary_mask=localization.binary_mask,
        )
        four_panel_report_rgb = self._verifier.render_four_panel_report(
            original_image=original_model_rgb,
            final_image=final_corrected_rgb,
            binary_mask=localization.binary_mask,
            cropped_product_ref=localization.cropped_product_ref,
            metrics=metrics,
        )

        return self._persist_case_artifacts(
            manifest=manifest,
            original_model_rgb=original_model_rgb,
            reference_product_rgb=reference_product_rgb,
            localization=localization,
            restored_roi_rgb=restored_roi_rgb,
            final_corrected_rgb=final_corrected_rgb,
            diff_heatmap_rgb=diff_heatmap_rgb,
            four_panel_report_rgb=four_panel_report_rgb,
            metrics=metrics,
            vlm_result=vlm_result,
            alpha_channel=model_alpha,
            icc_profile=model_icc,
        )

    def _run_vlm_post_check(
        self,
        *,
        metrics: VerificationMetrics,
        manifest: CaseManifest,
        final_corrected_rgb: NDArray[np.uint8],
        reference_product_rgb: NDArray[np.uint8],
        localization: LocalizationResult,
        defect_count: int,
        active_server: str | None,
    ) -> VerificationMetrics:
        """Delegate post-repair semantic/OCR VLM check to shared fail-closed helper."""
        return run_vlm_post_check(
            inspector=self._inspector,
            active_server=active_server,
            metrics=metrics,
            manifest=manifest,
            final_corrected_rgb=final_corrected_rgb,
            reference_product_rgb=reference_product_rgb,
            localization=localization,
            defect_count=defect_count,
        )

    def _finalize_no_defect_case(
        self,
        manifest: CaseManifest,
        original_model_rgb: NDArray[np.uint8],
        reference_product_rgb: NDArray[np.uint8],
        vlm_result: VLMInspectionResult,
        *,
        alpha_channel: NDArray[np.uint8] | None = None,
        icc_profile: bytes | None = None,
    ) -> CaseExecutionArtifact:
        """Short-circuit diffusion when Qwen3-VL detects no defect and keep original untouched."""
        height, width = original_model_rgb.shape[:2]
        ref_h, ref_w = reference_product_rgb.shape[:2]

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        feathered_mask = np.zeros((height, width), dtype=np.float32)
        target_bbox = vlm_result.model_target_box.to_pixel_bbox(width, height)
        ref_detail_bbox = vlm_result.reference_detail_box.to_pixel_bbox(ref_w, ref_h)
        context_bbox = PixelBBox(x1=0, y1=0, x2=width, y2=height)
        cropped_model_roi = original_model_rgb.copy()
        cropped_product_ref = reference_product_rgb.copy()
        cropped_mask_greyscale = np.zeros((height, width), dtype=np.uint8)

        localization = LocalizationResult(
            binary_mask=binary_mask,
            feathered_mask=feathered_mask,
            cropped_model_roi=cropped_model_roi,
            cropped_product_ref=cropped_product_ref,
            cropped_mask_greyscale=cropped_mask_greyscale,
            context_bbox=context_bbox,
            target_bbox=target_bbox,
            ref_detail_bbox=ref_detail_bbox,
            detection_method=f"{vlm_result.inspector_source} (no_defect)",
        )
        final_corrected_rgb = original_model_rgb.copy()
        restored_roi_rgb = cropped_model_roi.copy()
        total_px = int(height * width)
        metrics = VerificationMetrics(
            case_id=manifest.case_id,
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
            detection_method=localization.detection_method,
        )
        diff_heatmap_rgb = self._verifier.build_diff_heatmap(
            original_model_rgb, final_corrected_rgb, binary_mask
        )
        four_panel_report_rgb = self._verifier.render_four_panel_report(
            original_model_rgb, final_corrected_rgb, binary_mask, cropped_product_ref, metrics
        )
        return self._persist_case_artifacts(
            manifest=manifest,
            original_model_rgb=original_model_rgb,
            reference_product_rgb=reference_product_rgb,
            localization=localization,
            restored_roi_rgb=restored_roi_rgb,
            final_corrected_rgb=final_corrected_rgb,
            diff_heatmap_rgb=diff_heatmap_rgb,
            four_panel_report_rgb=four_panel_report_rgb,
            metrics=metrics,
            vlm_result=vlm_result,
            alpha_channel=alpha_channel,
            icc_profile=icc_profile,
        )

    def _persist_case_artifacts(
        self,
        manifest: CaseManifest,
        original_model_rgb: NDArray[np.uint8],
        reference_product_rgb: NDArray[np.uint8],
        localization: LocalizationResult,
        restored_roi_rgb: NDArray[np.uint8],
        final_corrected_rgb: NDArray[np.uint8],
        diff_heatmap_rgb: NDArray[np.uint8],
        four_panel_report_rgb: NDArray[np.uint8],
        metrics: VerificationMetrics,
        vlm_result: VLMInspectionResult | None,
        *,
        alpha_channel: NDArray[np.uint8] | None = None,
        icc_profile: bytes | None = None,
    ) -> CaseExecutionArtifact:
        """Persist corrected image flatly in `outputs/` and reports/metrics under `_reports/`."""
        reports_dir, corrected_path, report_path, metrics_json_path = persist_correction_outputs(
            outputs_dir=self._config.outputs_dir,
            case_id=manifest.case_id,
            final_corrected_rgb=final_corrected_rgb,
            four_panel_report_rgb=four_panel_report_rgb,
            diff_heatmap_rgb=diff_heatmap_rgb,
            metrics=metrics,
            vlm_result=vlm_result,
            diff_amplification_factor=self._config.verification.diff_amplification_factor,
            alpha_channel=alpha_channel,
            icc_profile=icc_profile,
        )

        return CaseExecutionArtifact(
            manifest=manifest,
            original_model_rgb=original_model_rgb,
            reference_product_rgb=reference_product_rgb,
            localization=localization,
            restored_roi_rgb=restored_roi_rgb,
            final_corrected_rgb=final_corrected_rgb,
            diff_heatmap_rgb=diff_heatmap_rgb,
            four_panel_report_rgb=four_panel_report_rgb,
            metrics=metrics,
            output_case_dir=reports_dir,
            corrected_image_path=corrected_path,
            report_image_path=report_path,
            metrics_json_path=metrics_json_path,
        )

    def run_all_cases(self) -> list[CaseExecutionArtifact]:
        """Run all discovered test cases sequentially and write a batch summary report."""
        case_dirs = self.discover_cases()
        artifacts: list[CaseExecutionArtifact] = []
        for case_dir in case_dirs:
            artifacts.append(self.run_case(case_dir))

        reports_dir = resolve_safe_path(self._config.outputs_dir, REPORTS_SUBDIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / "batch_verification_summary.json"
        summary_payload = {
            "total_cases": len(artifacts),
            "all_passed": all(a.metrics.passed for a in artifacts),
            "cases": [a.metrics.to_dict() for a in artifacts],
        }
        save_json_atomic(summary_path, summary_payload)

        return artifacts

    @staticmethod
    def _load_rgb_with_metadata(
        path: Path,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None, bytes | None]:
        if not path.is_file():
            raise ConfigurationError(f"Image file not found: {path}")
        with Image.open(path) as img:
            raw_icc = img.info.get("icc_profile")
            icc_profile = (
                raw_icc
                if (
                    img.mode in ("RGB", "RGBA")
                    and isinstance(raw_icc, bytes)
                    and len(raw_icc) >= 20
                    and raw_icc[16:20] == b"RGB "
                )
                else None
            )
            alpha_channel = (
                np.asarray(img.getchannel("A"), dtype=np.uint8).copy()
                if "A" in img.getbands()
                else None
            )
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            return rgb, alpha_channel, icc_profile

    @classmethod
    def _load_rgb(cls, path: Path) -> NDArray[np.uint8]:
        return cls._load_rgb_with_metadata(path)[0]
