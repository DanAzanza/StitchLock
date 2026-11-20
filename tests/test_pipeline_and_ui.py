"""Unit, Boundary, and Negative Tests for Compositor, Verifier, Metadata, Pipeline & Studio UI."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from setup_test_cases import setup_all_test_cases
from src.auto_masker import AutoMasker
from src.comfy_client import ComfyQwenBridge
from src.compositor import PixelLockCompositor
from src.metadata import embed_png_provenance_metadata, read_png_text_chunks
from src.models import (
    CaseManifest,
    ComfyConfig,
    ComfyExecutionError,
    ConfigurationError,
    DefectTarget,
    MaskingConfig,
    MaskingStrategy,
    NormalizedBox,
    PipelineConfig,
    PipelineError,
    PixelBBox,
    ReferenceType,
    VerificationConfig,
    VerificationError,
    load_pipeline_config,
)
from src.pipeline import GarmentCorrectionPipeline, execute_multi_defect_restoration
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

ROOT_DIR = Path(__file__).resolve().parent.parent


class _StubVLMInspector(Qwen3VLInspector):
    """Local test double simulating Qwen3-VL-8B inspection without network calls."""

    def __init__(
        self, comfy_config: ComfyConfig, workflow_path: Path, *, has_defect: bool = True
    ) -> None:
        super().__init__(comfy_config, workflow_path)
        self._has_defect = has_defect

    def inspect_garment_pair(
        self,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        case_id: str,
        server_url: str | None = None,
    ) -> VLMInspectionResult:
        _ = (model_rgb, ref_rgb, case_id, server_url)
        return VLMInspectionResult(
            has_defect=self._has_defect,
            defect_summary="Stub defect check" if self._has_defect else "No defect found",
            model_target_box=NormalizedBox(0.30, 0.36, 0.68, 0.60),
            reference_detail_box=NormalizedBox(0.32, 0.36, 0.68, 0.62),
            edit_prompt="Edit <image1>: Restore authentic detail from <image2>.",
            inspector_source="qwen3_vl_8b (stub)",
        )


class _StubComfyBridge(ComfyQwenBridge):
    """Local test double simulating ComfyUI Qwen-Image 2.1 Edit without network calls."""

    def __init__(self, config: ComfyConfig, workflow_path: Path) -> None:
        super().__init__(config, workflow_path)
        self.restore_calls = 0

    def _probe_live_server(self) -> str | None:
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
        self.restore_calls += 1
        return cropped_product_ref.copy(), "comfyui_qwen_2_1 (stub)"


def test_compositor_and_verifier_detect_single_pixel_tampering() -> None:
    rng = np.random.default_rng(42)
    orig = rng.integers(10, 240, size=(256, 256, 3), dtype=np.uint8)
    restored_roi = rng.integers(10, 240, size=(128, 128, 3), dtype=np.uint8)

    binary_mask = np.zeros((256, 256), dtype=np.uint8)
    binary_mask[80:150, 80:150] = 1
    feathered_mask = binary_mask.astype(np.float32)
    context_bbox = PixelBBox(x1=60, y1=60, x2=188, y2=188)

    compositor = PixelLockCompositor()
    final_img = compositor.composite(orig, restored_roi, binary_mask, feathered_mask, context_bbox)

    verifier = PixelLockVerifier(VerificationConfig())
    metrics = verifier.verify("unit_test_case", orig, final_img, binary_mask)
    assert metrics.passed is True
    assert metrics.unmasked_max_abs_diff == 0
    assert metrics.unmasked_changed_pixels_count == 0
    assert metrics.unmasked_ssim == 1.0

    tampered = final_img.copy()
    tampered[10, 10, 0] = np.uint8((int(tampered[10, 10, 0]) + 1) % 256)
    failed_metrics = verifier.verify("tampered_case", orig, tampered, binary_mask)
    assert failed_metrics.passed is False
    assert failed_metrics.unmasked_max_abs_diff == 1
    assert failed_metrics.unmasked_changed_pixels_count == 1
    assert failed_metrics.unmasked_ssim < 1.0

    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(VerificationError):
        verifier.verify("empty_mask_case", orig, final_img, empty_mask)


def test_verifier_2d_ssim_mask_isolation_and_non_strict_mode() -> None:
    rng = np.random.default_rng(11)
    orig = rng.integers(40, 200, size=(128, 128, 3), dtype=np.uint8)
    binary_mask = np.zeros((128, 128), dtype=np.uint8)
    binary_mask[40:88, 40:88] = 1

    # Huge masked change adjacent to unmasked boundary must not bleed into 2D unmasked SSIM
    edited = orig.copy()
    edited[40:88, 40:88] = 240
    edited[45:75, 45:75] = 20

    relaxed_verifier = PixelLockVerifier(VerificationConfig(strict_zero_outside_mask=False))
    clean_res = relaxed_verifier.verify("isolated", orig, edited, binary_mask)
    assert clean_res.unmasked_ssim == 1.0
    assert clean_res.passed is True

    # Small 1-level drift outside mask passes relaxed mode, fails strict mode; large drift fails
    tiny_drift = edited.copy()
    tiny_drift[5, 5, 0] = np.uint8(int(tiny_drift[5, 5, 0]) ^ 1)
    assert relaxed_verifier.verify("tiny_ok", orig, tiny_drift, binary_mask).passed is True
    strict_verifier = PixelLockVerifier(VerificationConfig(strict_zero_outside_mask=True))
    assert strict_verifier.verify("tiny_strict_fail", orig, tiny_drift, binary_mask).passed is False

    large_drift = edited.copy()
    large_drift[5, 5, :] = np.uint8((int(orig[5, 5, 0]) + 40) % 255)
    assert relaxed_verifier.verify("large_fail", orig, large_drift, binary_mask).passed is False


def test_verifier_rejects_collapsed_masked_edit_and_low_similarity() -> None:
    orig = np.full((128, 128, 3), 120, dtype=np.uint8)
    ref_crop = np.full((64, 64, 3), 120, dtype=np.uint8)
    ref_crop[16:48, 16:48] = np.array([230, 80, 60], dtype=np.uint8)

    binary_mask = np.zeros((128, 128), dtype=np.uint8)
    binary_mask[40:88, 40:88] = 1

    # Negative: all-black collapsed masked edit fails immediately
    blackout = orig.copy()
    blackout[40:88, 40:88] = 0
    verifier = PixelLockVerifier(VerificationConfig(min_masked_similarity=0.15))
    collapsed_metrics = verifier.verify(
        "blackout", orig, blackout, binary_mask, reference_crop=ref_crop
    )
    assert collapsed_metrics.masked_reference_similarity == 0.0
    assert collapsed_metrics.passed is False

    # Negative: inverted mismatched edit fails high similarity threshold
    wrong_edit = orig.copy()
    wrong_edit[40:88, 40:88] = 255 - np.array([230, 80, 60], dtype=np.uint8)
    strict_sim_verifier = PixelLockVerifier(VerificationConfig(min_masked_similarity=0.75))
    low_sim_metrics = strict_sim_verifier.verify(
        "wrong_color", orig, wrong_edit, binary_mask, reference_crop=ref_crop
    )
    assert low_sim_metrics.masked_reference_similarity < 0.75
    assert low_sim_metrics.passed is False

    # Positive: clean single-defect ERASE pass restoring plain cloth matches cloth reference crop
    sage_cloth = np.array([130, 140, 115], dtype=np.uint8)
    erase_orig = np.full((128, 128, 3), sage_cloth, dtype=np.uint8)
    erase_orig[48:80, 48:80] = np.array([195, 95, 55], dtype=np.uint8)
    erased_final = np.full((128, 128, 3), sage_cloth, dtype=np.uint8)
    plain_ref_crop = np.full((48, 48, 3), sage_cloth, dtype=np.uint8)
    erase_metrics = verifier.verify(
        "clean_erase", erase_orig, erased_final, binary_mask, reference_crop=plain_ref_crop
    )
    assert erase_metrics.masked_reference_similarity >= 0.99
    assert erase_metrics.passed is True


def test_end_to_end_batch_pipeline_both_cases(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    test_cfg = PipelineConfig(
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=base_cfg.verification,
        inspector_workflow_path=base_cfg.resolved_inspector_workflow_path,
    )

    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    stub_inspector = _StubVLMInspector(test_cfg.comfyui, inspector_wf, has_defect=True)
    stub_bridge = _StubComfyBridge(test_cfg.comfyui, test_cfg.workflow_path)
    pipeline = GarmentCorrectionPipeline(test_cfg, inspector=stub_inspector, bridge=stub_bridge)
    artifacts = pipeline.run_all_cases()
    assert len(artifacts) == 2
    assert stub_bridge.restore_calls == 2

    for artifact in artifacts:
        assert artifact.metrics.passed is True
        assert artifact.metrics.unmasked_max_abs_diff == 0
        assert artifact.metrics.unmasked_changed_pixels_count == 0
        assert artifact.metrics.unmasked_ssim == 1.0
        assert artifact.metrics.masked_changed_pixels_count > 100
        assert artifact.metrics.masked_reference_similarity >= 0.15
        assert (
            artifact.corrected_image_path
            == test_cfg.outputs_dir / f"{artifact.manifest.case_id}.png"
        )
        assert (
            artifact.output_case_dir / f"{artifact.manifest.case_id}_diff_heatmap_x4.png"
        ).is_file()
        assert artifact.report_image_path.is_file()
        assert artifact.metrics_json_path.is_file()

        chunks = read_png_text_chunks(artifact.corrected_image_path)
        assert chunks.get("AI-Compliance") == "EU-AI-Act-Article-50-Synthetic-Media"
        assert chunks.get("Pixel-Lock-Verified") == "true"
        assert chunks.get("Unmasked-Max-Diff") == "0"


def test_vlm_no_defect_short_circuits_diffusion_and_preserves_pixels(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    test_cfg = PipelineConfig(
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=base_cfg.verification,
    )

    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    no_defect_inspector = _StubVLMInspector(test_cfg.comfyui, inspector_wf, has_defect=False)
    stub_bridge = _StubComfyBridge(test_cfg.comfyui, test_cfg.workflow_path)
    pipeline = GarmentCorrectionPipeline(
        test_cfg, inspector=no_defect_inspector, bridge=stub_bridge
    )
    case_dir = pipeline.discover_cases()[0]
    artifact = pipeline.run_case(case_dir)

    assert stub_bridge.restore_calls == 0
    assert artifact.metrics.passed is True
    assert artifact.metrics.unmasked_max_abs_diff == 0
    assert artifact.metrics.masked_pixels == 0
    assert artifact.metrics.backend_used == "qwen3_vl_no_defect_short_circuit"
    assert np.array_equal(artifact.original_model_rgb, artifact.final_corrected_rgb)


def test_zero_recompression_metadata_preserves_exact_pixels_and_rejects_corrupt(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    png_path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="RGB").save(png_path, format="PNG")

    ok = embed_png_provenance_metadata(
        png_path,
        case_id="case_meta_test",
        pixel_lock_verified=True,
        unmasked_max_diff=0,
        unmasked_ssim=1.0,
    )
    assert ok is True

    with Image.open(png_path) as reloaded_img:
        reloaded = np.asarray(reloaded_img.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(arr, reloaded)

    tags = read_png_text_chunks(png_path)
    assert tags["Case-ID"] == "case_meta_test"
    assert tags["Pixel-Lock-Verified"] == "true"

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not_a_valid_png_stream")
    assert embed_png_provenance_metadata(corrupt_path) is False


def test_multi_defect_sequential_loop_pipeline_and_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app as ui_app

    setup_all_test_cases(tmp_path)
    cfg = replace(
        load_pipeline_config(Path("config.yaml"), base_dir=Path.cwd()),
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        masking=MaskingConfig(
            strategy=MaskingStrategy.AUTO_VISION,
            mask_dilation_ratio=0.015,
            feather_sigma_ratio=0.005,
        ),
    )
    case_dir = tmp_path / "test_cases" / "case_01_kova_zip_jacket.png"
    defect_chest = DefectTarget(
        defect_summary="Chest emblem mismatch",
        model_target_box=NormalizedBox(0.15, 0.15, 0.35, 0.35),
        reference_detail_box=NormalizedBox(0.15, 0.15, 0.35, 0.35),
        edit_prompt="Edit <image1>: Replace chest emblem with <image2>.",
    )
    defect_thigh = DefectTarget(
        defect_summary="Lower graphic mismatch",
        model_target_box=NormalizedBox(0.65, 0.65, 0.85, 0.85),
        reference_detail_box=NormalizedBox(0.65, 0.65, 0.85, 0.85),
        edit_prompt="Edit <image1>: Replace lower graphic with <image2>.",
    )
    multi_vlm = VLMInspectionResult(
        has_defect=True,
        defect_summary="Two separated garment defects",
        model_target_box=defect_chest.model_target_box,
        reference_detail_box=defect_chest.reference_detail_box,
        edit_prompt=defect_chest.edit_prompt,
        inspector_source="qwen3_vl_8b (mock)",
        defects=(defect_chest, defect_thigh),
    )

    recorded_calls: list[tuple[str, str]] = []

    class _MultiBridge(ComfyQwenBridge):
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
            _ = (cropped_product_ref, cropped_mask_greyscale, denoise_override)
            recorded_calls.append((case_id, prompt))
            edited = cropped_model_roi.copy()
            color = (
                np.array([210, 60, 40], dtype=np.uint8)
                if "_d1" in case_id
                else np.array([40, 200, 130], dtype=np.uint8)
            )
            edited[:, :] = color
            return edited, "comfyui_qwen_2_1 (multi_mock)"

    class _StubInspector:
        def inspect_garment_pair(
            self,
            model_rgb: NDArray[np.uint8],
            ref_rgb: NDArray[np.uint8],
            case_id: str,
            server_url: str | None = None,
        ) -> VLMInspectionResult:
            _ = (model_rgb, ref_rgb, case_id, server_url)
            return multi_vlm

    monkeypatch.setattr("src.pipeline.Qwen3VLInspector", lambda *_a, **_kw: _StubInspector())
    pipeline = GarmentCorrectionPipeline(
        cfg,
        inspector=_StubInspector(),  # type: ignore[arg-type]
        bridge=_MultiBridge(cfg.comfyui, cfg.workflow_path),
    )
    artifact = pipeline.run_case(case_dir)

    assert len(recorded_calls) == 2
    assert recorded_calls[0][0].endswith("_d1")
    assert recorded_calls[1][0].endswith("_d2")
    assert "(2 defects)" in artifact.metrics.detection_method
    assert artifact.metrics.passed is True
    assert artifact.metrics.unmasked_max_abs_diff == 0
    assert artifact.metrics.unmasked_changed_pixels_count == 0
    assert artifact.metrics.unmasked_ssim == 1.0

    h, w = artifact.original_model_rgb.shape[:2]
    assert not np.array_equal(
        artifact.final_corrected_rgb[int(0.25 * h), int(0.25 * w)],
        artifact.original_model_rgb[int(0.25 * h), int(0.25 * w)],
    )
    assert not np.array_equal(
        artifact.final_corrected_rgb[int(0.75 * h), int(0.75 * w)],
        artifact.original_model_rgb[int(0.75 * h), int(0.75 * w)],
    )

    vlm_json = json.loads(
        (artifact.output_case_dir / f"{artifact.manifest.case_id}_vlm_inspection.json").read_text(
            "utf-8"
        )
    )
    assert vlm_json["defect_count"] == 2

    with pytest.raises(ConfigurationError, match="zero defect targets"):
        execute_multi_defect_restoration(
            original_model_rgb=artifact.original_model_rgb,
            ref_rgb=artifact.reference_product_rgb,
            base_manifest=artifact.manifest,
            defects=(),
            masker=AutoMasker(cfg.masking),
            bridge=_MultiBridge(cfg.comfyui, cfg.workflow_path),
            compositor=PixelLockCompositor(),
        )

    recorded_calls.clear()
    det = ui_app.detect_custom_ui(
        artifact.original_model_rgb,
        artifact.reference_product_rgb,
        config_override=cfg,
        inspector=_StubInspector(),  # type: ignore[arg-type]
        bridge=_MultiBridge(cfg.comfyui, cfg.workflow_path),
    )
    assert det[11]["defect_count"] == 2
    _, ui_final, _, ui_metrics = ui_app.run_custom_upload_ui(
        artifact.original_model_rgb,
        artifact.reference_product_rgb,
        0.1,
        0.1,
        0.4,
        0.4,
        0.1,
        0.1,
        0.4,
        0.4,
        "fallback prompt",
        auto_vlm=True,
        config_override=cfg,
        inspector=_StubInspector(),  # type: ignore[arg-type]
        bridge=_MultiBridge(cfg.comfyui, cfg.workflow_path),
    )
    assert len(recorded_calls) == 2
    assert ui_metrics["passed"] is True
    assert ui_metrics["unmasked_max_abs_diff"] == 0
    assert ui_final.shape == artifact.original_model_rgb.shape


def test_gradio_custom_upload_and_box_preview_handlers(tmp_path: Path) -> None:
    from app import (
        build_gradio_app,
        detect_custom_ui,
        preview_custom_boxes_ui,
        run_custom_upload_ui,
    )

    rng = np.random.default_rng(99)
    model_img = rng.integers(20, 40, size=(320, 240, 3), dtype=np.uint8)
    ref_img = rng.integers(20, 40, size=(256, 256, 3), dtype=np.uint8)
    ref_img[90:160, 90:160] = 240

    m_prev, r_prev = preview_custom_boxes_ui(
        model_img, ref_img, 0.25, 0.25, 0.75, 0.75, 0.30, 0.30, 0.70, 0.70
    )
    assert m_prev is not None and m_prev.shape == (320, 240, 3)
    assert r_prev is not None and r_prev.shape == (256, 256, 3)

    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    offline_cfg = PipelineConfig(
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=base_cfg.verification,
    )

    with pytest.raises(ComfyExecutionError, match="ComfyUI server is offline"):
        run_custom_upload_ui(
            model_img=model_img,
            ref_img=ref_img,
            m_xmin=0.25,
            m_ymin=0.25,
            m_xmax=0.75,
            m_ymax=0.75,
            r_xmin=0.30,
            r_ymin=0.30,
            r_xmax=0.70,
            r_ymax=0.70,
            prompt="Restore authentic detail",
            auto_vlm=True,
            config_override=offline_cfg,
        )

    with pytest.raises(ComfyExecutionError, match="ComfyUI server is unreachable"):
        run_custom_upload_ui(
            model_img=model_img,
            ref_img=ref_img,
            m_xmin=0.25,
            m_ymin=0.25,
            m_xmax=0.75,
            m_ymax=0.75,
            r_xmin=0.30,
            r_ymin=0.30,
            r_xmax=0.70,
            r_ymax=0.70,
            prompt="Restore authentic detail",
            auto_vlm=False,
            config_override=offline_cfg,
        )

    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    stub_inspector = _StubVLMInspector(offline_cfg.comfyui, inspector_wf, has_defect=True)
    stub_bridge = _StubComfyBridge(offline_cfg.comfyui, offline_cfg.workflow_path)

    det_tuple = detect_custom_ui(
        model_img=model_img,
        ref_img=ref_img,
        config_override=offline_cfg,
        inspector=stub_inspector,
        bridge=stub_bridge,
    )
    assert len(det_tuple) == 12
    assert det_tuple[8].startswith("Edit <image1>:")

    restored_roi, final_img, report_img, metrics_dict = run_custom_upload_ui(
        model_img=model_img,
        ref_img=ref_img,
        m_xmin=0.25,
        m_ymin=0.25,
        m_xmax=0.75,
        m_ymax=0.75,
        r_xmin=0.30,
        r_ymin=0.30,
        r_xmax=0.70,
        r_ymax=0.70,
        prompt="Restore authentic detail",
        auto_vlm=True,
        config_override=offline_cfg,
        inspector=stub_inspector,
        bridge=stub_bridge,
    )
    assert restored_roi.shape == (256, 256, 3)
    assert final_img.shape == (320, 240, 3)
    assert report_img.ndim == 3
    assert metrics_dict["passed"] is True
    assert metrics_dict["unmasked_max_abs_diff"] == 0
    assert "vlm_inspection" in metrics_dict

    with pytest.raises(PipelineError):
        run_custom_upload_ui(
            None,
            ref_img,
            0.2,
            0.2,
            0.8,
            0.8,
            0.2,
            0.2,
            0.8,
            0.8,
            "p",
            config_override=offline_cfg,
        )

    demo = build_gradio_app()
    assert demo is not None


def test_ui_custom_upload_saves_matching_filename_and_dynamic_heatmap(tmp_path: Path) -> None:
    from app import run_custom_upload_ui

    tc = tmp_path / "inputs"
    tc.mkdir()
    model_file = tc / "hoodie_campaign_shot.jpg"
    ref_file = tc / "hoodie_campaign_shot_reference.png"
    Image.new("RGB", (192, 256), color=(25, 30, 38)).save(model_file)
    ref_arr = np.full((192, 192, 3), (25, 30, 38), dtype=np.uint8)
    ref_arr[60:120, 60:120] = 220
    Image.fromarray(ref_arr).save(ref_file)

    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    ui_cfg = replace(
        base_cfg,
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        comfyui=ComfyConfig(
            server_url="http://127.0.0.1:59999", fallback_ports=(), auto_launch=False
        ),
    )
    stub_inspector = _StubVLMInspector(
        ui_cfg.comfyui, ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json", has_defect=True
    )
    stub_bridge = _StubComfyBridge(ui_cfg.comfyui, ui_cfg.workflow_path)

    _, final_img, _, payload = run_custom_upload_ui(
        model_img=str(model_file),
        ref_img=str(ref_file),
        m_xmin=0.2,
        m_ymin=0.2,
        m_xmax=0.8,
        m_ymax=0.8,
        r_xmin=0.2,
        r_ymin=0.2,
        r_xmax=0.8,
        r_ymax=0.8,
        prompt="Edit <image1>: Replace emblem from <image2>.",
        auto_vlm=True,
        config_override=ui_cfg,
        inspector=stub_inspector,
        bridge=stub_bridge,
    )
    expected_out = ui_cfg.outputs_dir / "hoodie_campaign_shot.png"
    expected_heat = ui_cfg.outputs_dir / "_reports" / "hoodie_campaign_shot_diff_heatmap_x4.png"
    assert expected_out.is_file()
    assert expected_heat.is_file()
    assert Path(payload["saved_output_path"]) == expected_out
    assert Path(payload["saved_heatmap_path"]) == expected_heat
    assert final_img.shape == (256, 192, 3)


def test_ui_inbox_single_view_and_browser_lifecycle(tmp_path: Path) -> None:
    from app import BrowserSessionLifecycle, get_inbox_overview_data, load_single_case_view_ui

    setup_all_test_cases(tmp_path)
    cfg = replace(
        load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR),
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
    )

    rows, thumbs, ids = get_inbox_overview_data(cfg)
    assert len(rows) == 2
    assert len(thumbs) == 2
    assert ids[0] == "case_01_kova_zip_jacket"
    assert rows[0][4] == "Pending"

    orig, ref, final_img, heatmap, rep, met = load_single_case_view_ui(ids[0], config_override=cfg)
    assert orig is not None and ref is not None
    assert final_img is None and heatmap is None and rep is None and met is None

    shutdown_called = threading.Event()
    lifecycle = BrowserSessionLifecycle(on_shutdown=shutdown_called.set, grace_seconds=0.15)
    lifecycle.on_connect()
    lifecycle.on_disconnect()
    lifecycle.on_connect()
    time.sleep(0.22)
    assert shutdown_called.is_set() is False
    assert lifecycle.active_clients == 1

    lifecycle.on_disconnect()
    assert shutdown_called.wait(timeout=1.0) is True

    # Negative: legacy case_manifest.json with path traversal is rejected in UI handlers
    evil_dir = cfg.test_cases_dir / "z_evil_traversal"
    evil_dir.mkdir()
    (evil_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "z_evil_traversal",
                "title": "Evil",
                "garment_category": "top",
                "target_detail": "logo",
                "prompt": "Edit",
                "model_image_filename": "../escape.png",
                "reference_image_filename": "../escape_ref.png",
                "model_target_box": [0.2, 0.2, 0.5, 0.5],
                "reference_detail_box": [0.2, 0.2, 0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Path traversal rejected"):
        load_single_case_view_ui("z_evil_traversal", config_override=cfg)
    with pytest.raises(ConfigurationError, match="Path traversal rejected"):
        get_inbox_overview_data(cfg)


def test_ui_add_pair_to_inbox_tile_and_collision_handling(tmp_path: Path) -> None:
    from app import (
        add_pair_to_inbox,
        build_add_look_tile,
        build_artist_heatmap,
        gallery_with_add_tile,
        get_inbox_overview_data,
        is_case_repaired,
    )

    cfg = replace(
        load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR),
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
    )
    rng = np.random.default_rng(7)
    model_rgb = rng.integers(30, 80, size=(128, 128, 3), dtype=np.uint8)
    ref_rgb = rng.integers(80, 140, size=(128, 128, 3), dtype=np.uint8)
    model_file = tmp_path / "street_look.png"
    ref_file = tmp_path / "street_look_ref_src.png"
    Image.fromarray(model_rgb).save(model_file)
    Image.fromarray(ref_rgb).save(ref_file)

    first_id = add_pair_to_inbox(model_file, ref_file, config_override=cfg)
    assert first_id == "street_look"
    assert (cfg.test_cases_dir / "street_look.png").is_file()
    assert (cfg.test_cases_dir / "street_look_reference.png").is_file()

    second_id = add_pair_to_inbox(model_file, ref_file, config_override=cfg)
    assert second_id == "street_look_2"

    _, thumbs, ids = get_inbox_overview_data(cfg)
    assert ids == ["street_look", "street_look_2"]
    gallery = gallery_with_add_tile(thumbs)
    assert len(gallery) == 3
    assert gallery[-1][1] == "Add Look"
    assert build_add_look_tile().shape == (320, 320, 3)

    assert (
        is_case_repaired(model_rgb, model_rgb.copy(), {"masked_changed_pixels_count": 0}) is False
    )
    assert is_case_repaired(model_rgb, model_rgb.copy(), None) is False
    assert build_artist_heatmap(model_rgb, model_rgb.copy(), config_override=cfg) is None

    changed_rgb = model_rgb.copy()
    changed_rgb[40:70, 40:70] = 200
    assert is_case_repaired(model_rgb, changed_rgb, {"masked_changed_pixels_count": 900}) is True
    assert build_artist_heatmap(model_rgb, changed_rgb, config_override=cfg) is not None

    with pytest.raises(PipelineError):
        add_pair_to_inbox(None, ref_file, config_override=cfg)
    with pytest.raises(PipelineError):
        add_pair_to_inbox(model_file, None, config_override=cfg)


def test_logo_two_pass_surface_precomposite_and_pixel_lock() -> None:
    """Verify ReferenceType.LOGO runs ERASE (1.0) -> surface precomposite -> HARMONIZE (0.48)."""

    class _RecordingLogoBridge(_StubComfyBridge):
        def __init__(self, config: ComfyConfig, workflow_path: Path) -> None:
            super().__init__(config, workflow_path)
            self.calls: list[tuple[str, float | None, NDArray[np.uint8]]] = []

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
            _ = prompt
            self._validate_inputs(cropped_model_roi, cropped_product_ref, cropped_mask_greyscale)
            self.calls.append((case_id, denoise_override, cropped_model_roi.copy()))
            self.restore_calls += 1
            if case_id.endswith("_erase"):
                # Return plain shaded cloth matching outer garment gradient
                clean = np.full_like(cropped_model_roi, (70, 72, 76), dtype=np.uint8)
                return clean, "comfyui_qwen_2_1 (stub)"
            return cropped_model_roi.copy(), "comfyui_qwen_2_1 (stub)"

    cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    model_rgb = np.full((320, 320, 3), 240, dtype=np.uint8)
    model_rgb[40:300, 60:240] = (70, 72, 76)
    model_rgb[130:170, 75:175] = (230, 230, 225)  # Existing print to erase
    ref_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    ref_rgb[90:166, 20:236] = (245, 195, 55)

    manifest = CaseManifest(
        case_id="logo_two_pass_unit",
        title="Logo Two Pass",
        garment_category="hoodie",
        target_detail="Replace print with NEXORA logo",
        prompt="Edit <image1>: Replace print with NEXORA logo from <image2>.",
        model_image_filename="m.png",
        reference_image_filename="r.png",
        model_target_box=NormalizedBox(0.22, 0.38, 0.56, 0.56),
        reference_detail_box=NormalizedBox(0.08, 0.35, 0.92, 0.65),
        reference_type=ReferenceType.LOGO,
    )
    defect = DefectTarget(
        defect_summary=manifest.target_detail,
        model_target_box=manifest.model_target_box,
        reference_detail_box=manifest.reference_detail_box,
        edit_prompt=manifest.prompt,
    )
    bridge = _RecordingLogoBridge(cfg.comfyui, cfg.workflow_path)
    out = execute_multi_defect_restoration(
        original_model_rgb=model_rgb,
        ref_rgb=ref_rgb,
        base_manifest=manifest,
        defects=[defect],
        masker=AutoMasker(cfg.masking),
        bridge=bridge,
        compositor=PixelLockCompositor(),
    )
    assert len(bridge.calls) == 2
    assert bridge.calls[0][0] == "logo_two_pass_unit_erase"
    assert bridge.calls[0][1] == 1.0
    assert bridge.calls[1][0] == "logo_two_pass_unit_harmonize"
    assert bridge.calls[1][1] == pytest.approx(0.48)
    # Harmonize input contains projected golden logo pixels on clean cloth
    harm_roi = bridge.calls[1][2]
    assert int(np.count_nonzero((harm_roi[..., 0] > 180) & (harm_roi[..., 2] < 100))) > 50
    # Strict pixel-lock outside binary_mask
    unmasked = out.localization.binary_mask == 0
    assert np.array_equal(out.final_corrected_rgb[unmasked], model_rgb[unmasked])
