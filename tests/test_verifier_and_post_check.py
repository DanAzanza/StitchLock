"""Tests for Calibrated Masked Similarity Verification and Opt-In VLM Post-Check."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from src.comfy_client import ComfyQwenBridge
from src.models import (
    CaseManifest,
    ComfyConfig,
    LocalizationResult,
    NormalizedBox,
    PipelineConfig,
    PixelBBox,
    ReferenceType,
    VerificationConfig,
    VerificationMetrics,
    load_pipeline_config,
)
from src.pipeline import GarmentCorrectionPipeline, run_vlm_post_check
from src.ui_custom_handlers import run_custom_upload_ui
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector

ROOT_DIR = Path(__file__).resolve().parent.parent


def _draw_centered_wordmark(text: str) -> NDArray[np.uint8]:
    """Create a 256x256 navy garment image with crisp centered OpenCV wordmark typography."""
    img = np.full((256, 256, 3), (24, 34, 58), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (56, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (246, 246, 246),
        3,
        cv2.LINE_AA,
    )
    return img


def test_verifier_distinguishes_misspelled_typography_from_authentic_logo() -> None:
    model_krovx = _draw_centered_wordmark("KROVX")
    final_kova = _draw_centered_wordmark("KOVA")
    ref_kova_crop = final_kova[92:148, 52:204].copy()

    tight_mask = np.zeros((256, 256), dtype=np.uint8)
    tight_mask[92:148, 52:204] = 1
    dilated_mask = np.zeros((256, 256), dtype=np.uint8)
    dilated_mask[72:168, 32:224] = 1

    # Preserve outside-mask pixels accurately for synthetic repair candidates
    repaired_kova_tight = np.where(tight_mask[..., None] > 0, final_kova, model_krovx)
    repaired_kova_dilated = np.where(dilated_mask[..., None] > 0, final_kova, model_krovx)
    repaired_krovx_tight = model_krovx.copy()
    # Slight color tweak inside tight KROVX mask so changed pixels > 0 without changing letters
    repaired_krovx_tight[96:144, 58:198] = np.clip(
        repaired_krovx_tight[96:144, 58:198].astype(np.int16) + 2, 0, 255
    ).astype(np.uint8)

    verifier = PixelLockVerifier(VerificationConfig(min_masked_similarity=0.55))

    pass_tight = verifier.verify(
        "typo_kova_tight",
        model_krovx,
        repaired_kova_tight,
        tight_mask,
        reference_crop=ref_kova_crop,
    )
    assert pass_tight.passed is True
    assert pass_tight.masked_reference_similarity >= 0.90

    pass_dilated = verifier.verify(
        "typo_kova_dilated",
        model_krovx,
        repaired_kova_dilated,
        dilated_mask,
        reference_crop=ref_kova_crop,
    )
    assert pass_dilated.passed is True
    assert pass_dilated.masked_reference_similarity >= 0.85

    fail_tight = verifier.verify(
        "typo_krovx_tight",
        _draw_centered_wordmark("XXXXX"),
        repaired_krovx_tight,
        tight_mask,
        reference_crop=ref_kova_crop,
    )
    assert fail_tight.passed is False
    assert fail_tight.masked_reference_similarity < 0.55

    fail_dilated = verifier.verify(
        "typo_krovx_dilated",
        np.where(dilated_mask[..., None] > 0, _draw_centered_wordmark("XXXXX"), model_krovx),
        np.where(dilated_mask[..., None] > 0, repaired_krovx_tight, model_krovx),
        dilated_mask,
        reference_crop=ref_kova_crop,
    )
    assert fail_dilated.passed is False
    assert fail_dilated.masked_reference_similarity < 0.55

    # Pure ERASE on shaded sage fabric vs un-erased orange remnant
    grad = np.linspace(112, 128, 256, dtype=np.uint8)[:, None]
    sage_clean = np.empty((256, 256, 3), dtype=np.uint8)
    sage_clean[..., 0] = grad
    sage_clean[..., 1] = np.clip(grad.astype(np.int16) + 16, 0, 255).astype(np.uint8)
    sage_clean[..., 2] = np.clip(grad.astype(np.int16) + 6, 0, 255).astype(np.uint8)

    sage_dirty = sage_clean.copy()
    sage_dirty[95:145, 85:175] = (225, 110, 42)
    erase_mask = np.zeros((256, 256), dtype=np.uint8)
    erase_mask[85:155, 75:185] = 1
    ref_sage_patch = sage_clean[85:155, 75:185].copy()

    clean_erase = verifier.verify(
        "erase_clean_sage",
        sage_dirty,
        sage_clean,
        erase_mask,
        reference_crop=ref_sage_patch,
    )
    assert clean_erase.passed is True
    assert clean_erase.masked_reference_similarity >= 0.85

    repaired_with_remnant = sage_clean.copy()
    repaired_with_remnant[102:136, 98:162] = (225, 110, 42)
    orig_altered = sage_dirty.copy()
    orig_altered[102:136, 98:162] = (210, 95, 35)
    dirty_erase = verifier.verify(
        "erase_dirty_remnant",
        orig_altered,
        repaired_with_remnant,
        erase_mask,
        reference_crop=ref_sage_patch,
    )
    assert dirty_erase.masked_reference_similarity <= (
        clean_erase.masked_reference_similarity - 0.05
    )


class _ScriptedPostCheckInspector(Qwen3VLInspector):
    """Scripted inspector returning deterministic dual inspection and post-check JSON responses."""

    def __init__(self, comfy_config: ComfyConfig, workflow_path: Path, post_json: str) -> None:
        super().__init__(comfy_config, workflow_path)
        self.post_json = post_json
        self.post_check_calls = 0

    def _classify_reference_type(
        self,
        server_url: str,
        ref_rgb: NDArray[np.uint8],
        case_id: str,
    ) -> ReferenceType:
        _ = (server_url, ref_rgb, case_id)
        return ReferenceType.GARMENT

    def _run_comfy_vlm(
        self,
        server_url: str,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        case_id: str,
    ) -> str:
        _ = (server_url, model_rgb, ref_rgb, case_id)
        return json.dumps(
            {
                "has_defect": True,
                "defect_summary": "Garbled chest typography KROVX instead of KOVA",
                "defects": [
                    {
                        "defect_summary": "Misspelled chest typography",
                        "model_bbox_2d": [200, 350, 800, 600],
                        "ref_bbox_2d": [200, 350, 800, 600],
                        "edit_prompt": "Edit <image1>: Replace chest wordmark matching <image2>.",
                    }
                ],
            }
        )

    def _execute_vlm_workflow(self, server_url: str, workflow: dict[str, Any], case_id: str) -> str:
        _ = (server_url, workflow, case_id)
        self.post_check_calls += 1
        return self.post_json


class _StubPostCheckBridge(ComfyQwenBridge):
    """Stub edit bridge restoring the authentic reference pixels without network ComfyUI."""

    def _probe_live_server(self) -> str:
        return "http://127.0.0.1:8000"

    def restore_detail_roi(
        self,
        cropped_model_roi: NDArray[np.uint8],
        cropped_product_ref: NDArray[np.uint8],
        cropped_mask_greyscale: NDArray[np.uint8],
        prompt: str,
        case_id: str,
        *,
        denoise_override: float | None = None,
    ) -> tuple[NDArray[np.uint8], str]:
        _ = (prompt, case_id, denoise_override)
        self._validate_inputs(cropped_model_roi, cropped_product_ref, cropped_mask_greyscale)
        return cropped_product_ref.copy(), "comfyui_qwen_2_1 (post_check_stub)"


def test_vlm_post_check_parser_pipeline_and_custom_upload_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Qwen3VLInspector.parse_post_check_json(
        '{"passed": true, "reason": "Wordmark KOVA matches ref"}'
    ) == (True, "Wordmark KOVA matches ref")
    assert Qwen3VLInspector.parse_post_check_json(
        '{"passed": false, "reason": "Letter R remains"}'
    ) == (False, "Letter R remains")

    for bad_raw in [
        "No JSON payload here",
        "{not: valid json",
        "[true, false]",
        '{"passed": "true", "reason": "string bool"}',
        '{"reason": "missing passed key"}',
    ]:
        ok, _reason = Qwen3VLInspector.parse_post_check_json(bad_raw)
        assert ok is False

    monkeypatch.setattr("src.vlm_inspector.upload_png_bytes", lambda _url, name, _png: name)

    model_rgb = _draw_centered_wordmark("KROVX")
    ref_rgb = _draw_centered_wordmark("KOVA")
    tc_dir = tmp_path / "test_cases"
    tc_dir.mkdir()
    Image.fromarray(model_rgb, mode="RGB").save(tc_dir / "postcheck_case.png")
    Image.fromarray(ref_rgb, mode="RGB").save(tc_dir / "postcheck_case_reference.png")

    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    cfg = PipelineConfig(
        test_cases_dir=tc_dir,
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=replace(base_cfg.verification, enable_vlm_post_check=True),
        inspector_workflow_path=base_cfg.resolved_inspector_workflow_path,
        heuristics=base_cfg.heuristics,
    )
    bridge = _StubPostCheckBridge(cfg.comfyui, cfg.workflow_path)

    passing_inspector = _ScriptedPostCheckInspector(
        cfg.comfyui,
        cfg.resolved_inspector_workflow_path,
        '{"passed": true, "reason": "Exact KOVA"}',
    )
    pass_artifact = GarmentCorrectionPipeline(
        cfg, inspector=passing_inspector, bridge=bridge
    ).run_case(tc_dir / "postcheck_case.png")
    assert passing_inspector.post_check_calls == 1
    assert pass_artifact.metrics.passed is True
    assert pass_artifact.metrics.vlm_post_check_passed is True
    assert pass_artifact.metrics.to_dict()["vlm_post_check_passed"] is True

    failing_inspector = _ScriptedPostCheckInspector(
        cfg.comfyui,
        cfg.resolved_inspector_workflow_path,
        '{"passed": false, "reason": "Wrong text"}',
    )
    fail_artifact = GarmentCorrectionPipeline(
        cfg, inspector=failing_inspector, bridge=bridge
    ).run_case(tc_dir / "postcheck_case.png")
    assert failing_inspector.post_check_calls == 1
    assert fail_artifact.metrics.passed is False
    assert fail_artifact.metrics.vlm_post_check_passed is False
    assert fail_artifact.metrics.to_dict()["vlm_post_check_passed"] is False

    malformed_inspector = _ScriptedPostCheckInspector(
        cfg.comfyui, cfg.resolved_inspector_workflow_path, "INVALID_NOT_JSON"
    )
    malformed_artifact = GarmentCorrectionPipeline(
        cfg, inspector=malformed_inspector, bridge=bridge
    ).run_case(tc_dir / "postcheck_case.png")
    assert malformed_artifact.metrics.passed is False
    assert malformed_artifact.metrics.vlm_post_check_passed is False

    # Fail-closed direct helper tests when inspector or server is unavailable
    sample_manifest = CaseManifest(
        case_id="helper_fc",
        title="Helper",
        garment_category="top",
        target_detail="chest wordmark",
        prompt="Replace",
        model_image_filename="m.png",
        reference_image_filename="r.png",
        model_target_box=NormalizedBox(0.2, 0.35, 0.8, 0.6),
        reference_detail_box=NormalizedBox(0.2, 0.35, 0.8, 0.6),
    )
    dummy_loc = LocalizationResult(
        binary_mask=np.zeros((256, 256), dtype=np.uint8),
        feathered_mask=np.zeros((256, 256), dtype=np.float32),
        cropped_model_roi=model_rgb,
        cropped_product_ref=ref_rgb,
        cropped_mask_greyscale=np.zeros((256, 256), dtype=np.uint8),
        context_bbox=PixelBBox(0, 0, 256, 256),
        target_bbox=PixelBBox(52, 92, 204, 148),
        ref_detail_bbox=PixelBBox(52, 92, 204, 148),
        detection_method="stub",
    )
    base_metrics = VerificationMetrics(
        case_id="helper_fc",
        passed=True,
        unmasked_max_abs_diff=0,
        unmasked_changed_pixels_count=0,
        unmasked_changed_pixels_pct=0.0,
        unmasked_psnr=100.0,
        unmasked_ssim=1.0,
        masked_mean_abs_diff=12.0,
        masked_changed_pixels_count=1000,
        total_pixels=256 * 256,
        unmasked_pixels=256 * 256 - 1000,
        masked_pixels=1000,
        backend_used="stub",
        detection_method="stub",
    )
    no_insp = run_vlm_post_check(
        inspector=None,
        active_server="http://127.0.0.1:8000",
        metrics=base_metrics,
        manifest=sample_manifest,
        final_corrected_rgb=ref_rgb,
        reference_product_rgb=ref_rgb,
        localization=dummy_loc,
        defect_count=1,
    )
    assert no_insp.passed is False and no_insp.vlm_post_check_passed is False

    no_srv = run_vlm_post_check(
        inspector=passing_inspector,
        active_server=None,
        metrics=base_metrics,
        manifest=sample_manifest,
        final_corrected_rgb=ref_rgb,
        reference_product_rgb=ref_rgb,
        localization=dummy_loc,
        defect_count=1,
    )
    assert no_srv.passed is False and no_srv.vlm_post_check_passed is False

    # Custom upload UI parity check
    _, _, _, ui_pass_payload = run_custom_upload_ui(
        str(tc_dir / "postcheck_case.png"),
        str(tc_dir / "postcheck_case_reference.png"),
        auto_vlm=True,
        config_override=cfg,
        inspector=passing_inspector,
        bridge=bridge,
    )
    assert ui_pass_payload["passed"] is True
    assert ui_pass_payload["vlm_post_check_passed"] is True

    _, _, _, ui_fail_payload = run_custom_upload_ui(
        str(tc_dir / "postcheck_case.png"),
        str(tc_dir / "postcheck_case_reference.png"),
        auto_vlm=True,
        config_override=cfg,
        inspector=failing_inspector,
        bridge=bridge,
    )
    assert ui_fail_payload["passed"] is False
    assert ui_fail_payload["vlm_post_check_passed"] is False
