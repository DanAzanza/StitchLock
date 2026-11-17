"""Automated Unit, Boundary, and Negative Tests for the Garment Detail Correction Pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from setup_test_cases import setup_all_test_cases
from src.auto_masker import AutoMasker, load_case_manifest
from src.comfy_client import ComfyQwenBridge
from src.compositor import PixelLockCompositor
from src.housekeeper import ComfyUIHousekeeper, is_loopback_host
from src.metadata import embed_png_provenance_metadata, read_png_text_chunks
from src.models import (
    ComfyConfig,
    ComfyExecutionError,
    ConfigurationError,
    LocalizationError,
    MaskingConfig,
    MaskingStrategy,
    NormalizedBox,
    PipelineConfig,
    PipelineError,
    PixelBBox,
    VerificationConfig,
    VerificationError,
    load_pipeline_config,
    resolve_safe_path,
)
from src.pipeline import GarmentCorrectionPipeline
from src.verifier import PixelLockVerifier
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_normalized_box_validation_and_conversion() -> None:
    box = NormalizedBox(0.25, 0.50, 0.75, 0.80)
    px = box.to_pixel_bbox(width=800, height=1000)
    assert px == PixelBBox(x1=200, y1=500, x2=600, y2=800)
    assert px.width == 400
    assert px.height == 300

    # Negative tests: out-of-bounds or inverted coordinates must fail fast
    with pytest.raises(ConfigurationError):
        NormalizedBox(-0.1, 0.2, 0.5, 0.5)
    with pytest.raises(ConfigurationError):
        NormalizedBox(0.6, 0.2, 0.4, 0.5)
    with pytest.raises(ConfigurationError):
        box.to_pixel_bbox(width=0, height=100)


def test_path_traversal_guard_rejects_escape(tmp_path: Path) -> None:
    safe = resolve_safe_path(tmp_path, "subdir/file.png")
    assert safe.is_relative_to(tmp_path.resolve())

    with pytest.raises(ConfigurationError, match="Path traversal rejected"):
        resolve_safe_path(tmp_path, "../../etc/passwd")


def test_auto_masker_enforces_strict_zero_outside_binary_mask(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    case_dir = tmp_path / "test_cases" / "case_01_kova_zip_jacket.png"
    manifest = load_case_manifest(case_dir)

    rng = np.random.default_rng(123)
    model_rgb = rng.integers(0, 256, size=(600, 400, 3), dtype=np.uint8)
    ref_rgb = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    masker = AutoMasker(MaskingConfig(strategy=MaskingStrategy.AUTO_VISION))
    res = masker.localize_and_mask(model_rgb, ref_rgb, manifest)

    # Native non-square aspect ratio must be preserved without square stretching
    assert res.binary_mask.shape == (600, 400)
    assert res.feathered_mask.shape == (600, 400)
    assert res.cropped_model_roi.shape == (600, 400, 3)
    assert res.cropped_product_ref.shape == (512, 512, 3)
    assert res.cropped_mask_greyscale.shape == (600, 400)
    assert res.cropped_mask_greyscale.ndim == 2

    # Relative scaling: larger image produces proportionally wider mask dilation
    large_model = np.zeros((1600, 1200, 3), dtype=np.uint8)
    res_large = masker.localize_and_mask(large_model, ref_rgb, manifest)
    assert np.count_nonzero(res_large.binary_mask) > np.count_nonzero(res.binary_mask) * 3

    # Invariant: feathered_mask must be strictly 0.0 wherever binary_mask == 0
    unmasked_feather = res.feathered_mask[res.binary_mask == 0]
    assert np.all(unmasked_feather == 0.0)
    assert float(np.max(res.feathered_mask[res.binary_mask > 0])) > 0.9


def test_auto_masker_rejects_malformed_inputs(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    manifest = load_case_manifest(tmp_path / "test_cases" / "case_01_kova_zip_jacket.png")
    masker = AutoMasker(MaskingConfig())

    bad_gray = np.zeros((128, 128), dtype=np.uint8)
    good_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(LocalizationError):
        masker.localize_and_mask(bad_gray, good_rgb, manifest)


class _StubVLMInspector(Qwen3VLInspector):
    """Test double simulating live Qwen3-VL-8B inspection without network calls."""

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
    """Test double simulating live ComfyUI Qwen-Image 2.1 Edit without network calls."""

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
    ) -> tuple[NDArray[np.uint8], str]:
        self._validate_inputs(cropped_model_roi, cropped_product_ref, cropped_mask_greyscale)
        self.restore_calls += 1
        return cropped_product_ref.copy(), "comfyui_qwen_2_1 (stub)"


def test_comfy_workflow_injection_and_greyscale_mask_enforcement() -> None:
    workflow_path = ROOT_DIR / "workflows" / "qwen_2_1_edit_api.json"
    cfg = ComfyConfig(
        server_url="http://127.0.0.1:59999",
        fallback_ports=(),
        seed=99,
        steps=20,
        cfg=1.0,
        denoise=0.75,
    )
    bridge = ComfyQwenBridge(cfg, workflow_path)
    payload = bridge.build_workflow_payload(
        uploaded_model_name="roi_1.png",
        uploaded_ref_name="ref_1.png",
        uploaded_mask_name="mask_1.png",
        prompt="Test prompt",
        output_prefix="prefix_01",
    )
    assert payload["1"]["inputs"]["image"] == "roi_1.png"
    assert payload["2"]["inputs"]["image"] == "ref_1.png"
    assert payload["6"]["inputs"]["prompt"] == "Test prompt"
    assert payload["7"]["inputs"]["seed"] == 99
    assert payload["7"]["inputs"]["steps"] == 20
    assert payload["10"]["inputs"]["image"] == "mask_1.png"
    assert payload["11"]["class_type"] == "VAEEncode"
    assert payload["11"]["inputs"]["pixels"] == ["14", 0]
    assert payload["12"]["class_type"] == "ImageToMask"
    assert payload["13"]["class_type"] == "SetLatentNoiseMask"
    assert payload["14"]["class_type"] == "ImageScaleToTotalPixels"

    # Negative test: passing a 3-channel or 4-channel RGBA mask instead of 2D greyscale must fail
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    bad_rgba_mask = np.zeros((128, 128, 4), dtype=np.uint8)
    with pytest.raises(ComfyExecutionError, match="strict 2D single-channel greyscale"):
        bridge.restore_detail_roi(rgb, rgb, bad_rgba_mask, "prompt", "case_x")


def test_fail_fast_when_comfyui_server_is_offline(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    offline_comfy = ComfyConfig(
        server_url="http://127.0.0.1:59999",
        fallback_ports=(),
        timeout_seconds=2.0,
    )
    offline_cfg = PipelineConfig(
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=offline_comfy,
        verification=base_cfg.verification,
    )

    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.ones((128, 128), dtype=np.uint8) * 255

    bridge = ComfyQwenBridge(offline_comfy, offline_cfg.workflow_path)
    with pytest.raises(ComfyExecutionError, match="ComfyUI server is unreachable"):
        bridge.restore_detail_roi(rgb, rgb, mask, "Test prompt", "case_offline")

    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    inspector = Qwen3VLInspector(offline_comfy, inspector_wf)
    with pytest.raises(ComfyExecutionError, match="ComfyUI server is offline"):
        inspector.inspect_garment_pair(rgb, rgb, "case_offline", server_url=None)

    pipeline = GarmentCorrectionPipeline(offline_cfg)
    first_case = pipeline.discover_cases()[0]
    with pytest.raises(ComfyExecutionError, match="ComfyUI server is offline"):
        pipeline.run_case(first_case)


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

    # Negative test: alter a SINGLE channel of a SINGLE pixel outside the mask by +1
    tampered = final_img.copy()
    tampered[10, 10, 0] = np.uint8((int(tampered[10, 10, 0]) + 1) % 256)
    failed_metrics = verifier.verify("tampered_case", orig, tampered, binary_mask)
    assert failed_metrics.passed is False
    assert failed_metrics.unmasked_max_abs_diff == 1
    assert failed_metrics.unmasked_changed_pixels_count == 1

    # Negative test: empty mask must raise VerificationError
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(VerificationError):
        verifier.verify("empty_mask_case", orig, final_img, empty_mask)


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
        assert (
            artifact.corrected_image_path
            == test_cfg.outputs_dir / f"{artifact.manifest.case_id}.png"
        )
        assert artifact.corrected_image_path.is_file()
        assert artifact.report_image_path.is_file()
        assert artifact.metrics_json_path.is_file()

        # Verify EU AI Act Article 50 & Pixel-Lock provenance chunks were embedded
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
    from PIL import Image

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

    # Reload pixel data and verify 100% bitwise pixel identity (zero recompression)
    with Image.open(png_path) as reloaded_img:
        reloaded = np.asarray(reloaded_img.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(arr, reloaded)

    tags = read_png_text_chunks(png_path)
    assert tags["Case-ID"] == "case_meta_test"
    assert tags["Pixel-Lock-Verified"] == "true"

    # Negative test: non-PNG / corrupt file returns False safely without raising
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not_a_valid_png_stream")
    assert embed_png_provenance_metadata(corrupt_path) is False


def test_housekeeper_protects_preexisting_and_blocks_traversal(tmp_path: Path) -> None:
    assert is_loopback_host("http://127.0.0.1:8188") is True
    assert is_loopback_host("https://remote-gpu.example.com:8188") is False

    input_dir = tmp_path / "comfy_input"
    input_dir.mkdir()
    pre_existing = input_dir / "keep_me.png"
    pre_existing.write_bytes(b"original")

    hk = ComfyUIHousekeeper(input_dir=input_dir, enabled=True)
    hk.record_pre_upload("keep_me.png")
    hk.record_uploaded_name("keep_me.png")

    session_file = input_dir / "temp_roi.png"
    hk.record_pre_upload("temp_roi.png")
    session_file.write_bytes(b"temporary")
    hk.record_uploaded_name("temp_roi.png")

    # Negative test: path traversal outside input_dir must be ignored
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside")
    hk.record_uploaded_name("../outside.png")

    purged = hk.cleanup_session_inputs()
    assert purged == 1
    assert pre_existing.exists() is True
    assert outside_file.exists() is True
    assert session_file.exists() is False


def test_gradio_custom_upload_and_box_preview_handlers(tmp_path: Path) -> None:
    from app import (
        build_gradio_app,
        detect_custom_ui,
        preview_custom_boxes_ui,
        run_custom_upload_ui,
    )

    rng = np.random.default_rng(99)
    # Non-square portrait model image vs square product ref to validate aspect preservation
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

    # Negative test: Auto-VLM and Edit must both fail fast when ComfyUI is unreachable
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

    # Positive test: Auto-Detect preview + Auto-VLM full custom run with injected stubs
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

    # Negative test: missing image input raises PipelineError
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

    # Verify Gradio Blocks app constructs cleanly without error
    demo = build_gradio_app()
    assert demo is not None

    # Verify full-garment reference alignment shifts vertically at native resolution
    ref_canvas = np.full((320, 240, 3), 210, dtype=np.uint8)
    ref_canvas[60:90, 120:150] = 20
    aligned = AutoMasker._align_full_garment_reference(
        ref_rgb=ref_canvas,
        ref_detail_bbox=PixelBBox(120, 60, 150, 90),
        target_bbox=PixelBBox(120, 180, 150, 210),
        model_size=(240, 320),
    )
    assert aligned.shape == (320, 240, 3)
    assert int(aligned[195, 135, 0]) == 20
    assert int(aligned[10, 10, 0]) == 210


def test_qwen3_vl_inspector_json_parsing_and_negative_rejection() -> None:
    from src.vlm_inspector import Qwen3VLInspector

    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    raw_vlm_output = """```json
    {
      "has_defect": true,
      "defect_summary": "Chest clover instead of geometric hexagon crest",
      "model_bbox_2d": [550, 560, 670, 680],
      "ref_bbox_2d": [540, 430, 600, 510],
      "edit_prompt": "Replace clover with crisp hexagonal crest."
    }
    ```"""
    res = Qwen3VLInspector.parse_vlm_json_response(
        raw_text=raw_vlm_output,
        model_rgb=dummy_rgb,
        ref_rgb=dummy_rgb,
    )
    assert res.has_defect is True
    assert "clover" in res.defect_summary
    assert res.edit_prompt.startswith("Edit <image1>:")
    assert 0.50 <= res.model_target_box.x_min < res.model_target_box.x_max <= 0.75

    # Negative test: non-JSON response must raise ComfyExecutionError
    with pytest.raises(ComfyExecutionError, match="did not contain JSON"):
        Qwen3VLInspector.parse_vlm_json_response(
            raw_text="No JSON here at all",
            model_rgb=dummy_rgb,
            ref_rgb=dummy_rgb,
        )


def test_vlm_multi_defect_parsing_dedup_and_cap() -> None:
    from src.vlm_inspector import MAX_DEFECTS_PER_IMAGE, Qwen3VLInspector

    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    # 1. Two separated defects sorted deterministically top-to-bottom + overlapping duplicate merged
    raw_multi = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Two separated garment embroidery defects",
            "defects": [
                {
                    "defect_summary": "Thigh patch gibberish",
                    "model_bbox_2d": [650, 660, 820, 820],
                    "ref_bbox_2d": [620, 620, 740, 740],
                    "edit_prompt": "Replace thigh patch with small mountain crest from <image2>.",
                },
                {
                    "defect_summary": "Chest clover",
                    "model_bbox_2d": [160, 170, 320, 320],
                    "ref_bbox_2d": [180, 180, 300, 300],
                    "edit_prompt": "Replace chest clover with KOVA crest from <image2>.",
                },
                {
                    "defect_summary": "Duplicate overlapping chest region",
                    "model_bbox_2d": [165, 175, 315, 315],
                    "ref_bbox_2d": [180, 180, 300, 300],
                    "edit_prompt": "Replace chest clover and subtitle.",
                },
            ],
        }
    )
    parsed = Qwen3VLInspector.parse_vlm_json_response(raw_multi, dummy_rgb, dummy_rgb)
    assert parsed.has_defect is True
    assert len(parsed.resolved_defects) == 2
    assert "Chest clover" in parsed.resolved_defects[0].defect_summary
    assert "Thigh patch" in parsed.resolved_defects[1].defect_summary
    assert parsed.to_dict()["defect_count"] == 2

    # 2. Enforce MAX_DEFECTS_PER_IMAGE cap on separated boxes
    six_boxes = [
        {
            "defect_summary": f"Defect {i}",
            "model_bbox_2d": [50 + i * 130, 100, 130 + i * 130, 180],
            "ref_bbox_2d": [50, 100, 130, 180],
            "edit_prompt": f"Replace defect {i} with reference <image2>.",
        }
        for i in range(6)
    ]
    capped = Qwen3VLInspector.parse_vlm_json_response(
        json.dumps({"has_defect": True, "defect_summary": "Many", "defects": six_boxes}),
        dummy_rgb,
        dummy_rgb,
    )
    assert len(capped.resolved_defects) == MAX_DEFECTS_PER_IMAGE

    # 3. Clean no-defect response yields zero defect targets
    clean = Qwen3VLInspector.parse_vlm_json_response(
        json.dumps({"has_defect": False, "defect_summary": "All intact", "defects": []}),
        dummy_rgb,
        dummy_rgb,
    )
    assert clean.has_defect is False
    assert clean.resolved_defects == ()


def test_multi_defect_sequential_loop_pipeline_and_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app as ui_app
    from src.models import DefectTarget, NormalizedBox
    from src.pipeline import execute_multi_defect_restoration
    from src.vlm_inspector import VLMInspectionResult

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
        ) -> tuple[NDArray[np.uint8], str]:
            _ = cropped_product_ref
            _ = cropped_mask_greyscale
            recorded_calls.append((case_id, prompt))
            edited = cropped_model_roi.copy()
            # Paint distinctive colors over whole image to prove per-pass Pixel-Lock containment
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

    # Verify BOTH defect regions were modified relative to the original input
    h, w = artifact.original_model_rgb.shape[:2]
    chest_center = (int(0.25 * h), int(0.25 * w))
    thigh_center = (int(0.75 * h), int(0.75 * w))
    assert not np.array_equal(
        artifact.final_corrected_rgb[chest_center], artifact.original_model_rgb[chest_center]
    )
    assert not np.array_equal(
        artifact.final_corrected_rgb[thigh_center], artifact.original_model_rgb[thigh_center]
    )

    vlm_json = json.loads(
        (artifact.output_case_dir / f"{artifact.manifest.case_id}_vlm_inspection.json").read_text(
            "utf-8"
        )
    )
    assert vlm_json["defect_count"] == 2

    # Negative boundary test: zero defect targets must fail fast with ConfigurationError
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

    # Verify Gradio Tab 3 multi-defect detection overlay and UI run handler
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


def test_flat_reference_discovery_mixed_extensions_and_negatives(tmp_path: Path) -> None:
    from PIL import Image
    from src.auto_masker import discover_case_paths, has_manifest_roi, resolve_case_path

    tc = tmp_path / "test_cases"
    tc.mkdir()

    # 1. Mixed extensions (.jpg On-Model + .png _reference) work with zero JSON
    model_jpg = tc / "lookbook_a.jpg"
    ref_png = tc / "lookbook_a_reference.png"
    img = Image.new("RGB", (128, 128), color=(30, 40, 50))
    img.save(model_jpg)
    img.save(ref_png)

    discovered = discover_case_paths(tc)
    assert discovered == [model_jpg]
    manifest = load_case_manifest(model_jpg)
    assert manifest.case_id == "lookbook_a"
    assert manifest.model_image_filename == "lookbook_a.jpg"
    assert manifest.reference_image_filename == "lookbook_a_reference.png"
    assert has_manifest_roi(model_jpg) is False
    assert resolve_case_path(tc, "lookbook_a") == model_jpg

    # Negative: manifest_roi on JSON-less flat pair must fail fast
    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    roi_cfg = replace(
        base_cfg,
        test_cases_dir=tc,
        outputs_dir=tmp_path / "outputs",
        masking=MaskingConfig(strategy=MaskingStrategy.MANIFEST_ROI),
    )

    class _OnlineBridge(ComfyQwenBridge):
        def _probe_live_server(self) -> str:
            return "http://127.0.0.1:8000"

    roi_pipeline = GarmentCorrectionPipeline(
        roi_cfg, bridge=_OnlineBridge(roi_cfg.comfyui, roi_cfg.workflow_path)
    )
    with pytest.raises(ConfigurationError, match="manifest_roi"):
        roi_pipeline.run_case(model_jpg)

    # Negative: path traversal rejected in resolve_case_path
    with pytest.raises(ConfigurationError, match="Path traversal rejected"):
        resolve_case_path(tc, "../escape")

    # Negative: orphan On-Model image missing _reference
    orphan_model = tc / "stray_shirt.png"
    img.save(orphan_model)
    with pytest.raises(ConfigurationError, match="Orphan On-Model image"):
        discover_case_paths(tc)
    orphan_model.unlink()

    # Negative: orphan _reference image missing base On-Model
    orphan_ref = tc / "typo_reference.png"
    img.save(orphan_ref)
    with pytest.raises(ConfigurationError, match="Orphan reference image"):
        discover_case_paths(tc)
    orphan_ref.unlink()

    # Negative: duplicate model stems with different extensions
    dup_model = tc / "lookbook_a.png"
    img.save(dup_model)
    with pytest.raises(ConfigurationError, match="Duplicate On-Model images"):
        discover_case_paths(tc)


def test_ui_custom_upload_saves_matching_filename_to_outputs(tmp_path: Path) -> None:
    from app import run_custom_upload_ui
    from PIL import Image

    tc = tmp_path / "inputs"
    tc.mkdir()
    model_file = tc / "hoodie_campaign_shot.jpg"
    ref_file = tc / "hoodie_campaign_shot_reference.png"
    Image.new("RGB", (192, 256), color=(25, 30, 38)).save(model_file)
    ref_img = Image.new("RGB", (192, 192), color=(25, 30, 38))
    ref_arr = np.asarray(ref_img, dtype=np.uint8).copy()
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
    assert expected_out.is_file()
    assert Path(payload["saved_output_path"]) == expected_out
    assert (
        ui_cfg.outputs_dir / "_reports" / "hoodie_campaign_shot_verification_report.png"
    ).is_file()
    assert final_img.shape == (256, 192, 3)


def test_comfy_launcher_detection_command_and_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.comfy_launcher import (
        ComfyInstance,
        build_headless_command,
        detect_comfy_desktop_instance,
        ensure_comfyui_server,
    )

    # 1. Parse synthetic Comfy Desktop installations.json + YAML path cleanly
    appdata = tmp_path / "appdata"
    desktop_cfg = appdata / "Comfy Desktop"
    (desktop_cfg / "instance-model-paths").mkdir(parents=True)
    py_exe = tmp_path / "python.exe"
    py_exe.write_bytes(b"")
    install_dir = tmp_path / "ComfyUI"
    (install_dir / "ComfyUI").mkdir(parents=True)
    (install_dir / "ComfyUI" / "main.py").write_text("# mock", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    yaml_file = desktop_cfg / "instance-model-paths" / "inst-1.yaml"
    yaml_file.write_text("comfyui:\n", encoding="utf-8")

    (desktop_cfg / "installations.json").write_text(
        json.dumps(
            [
                {
                    "id": "inst-1",
                    "name": "Local Comfy Desktop",
                    "status": "installed",
                    "adoptedPythonPath": str(py_exe),
                    "installPath": str(install_dir),
                    "adoptedBaseDir": str(base_dir),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.comfy_launcher.get_desktop_appdata_dir", lambda: desktop_cfg)
    inst = detect_comfy_desktop_instance()
    assert inst is not None
    assert inst.extra_model_paths == yaml_file

    cmd = build_headless_command(inst, "127.0.0.1", "8000")
    assert cmd[0] == str(py_exe)
    assert "--extra-model-paths-config" in cmd

    # 2. Negative: auto_launch=False or missing installation fails fast cleanly
    cfg_disabled = ComfyConfig(
        server_url="http://127.0.0.1:59999", fallback_ports=(), auto_launch=False
    )
    with pytest.raises(ComfyExecutionError, match="auto_launch is disabled"):
        ensure_comfyui_server(cfg_disabled, startup_timeout=0.5)

    monkeypatch.setattr("src.comfy_launcher.detect_comfyui", lambda: None)
    cfg_no_inst = ComfyConfig(
        server_url="http://127.0.0.1:59999", fallback_ports=(), auto_launch=True
    )
    with pytest.raises(ComfyExecutionError, match="no local ComfyUI Desktop"):
        ensure_comfyui_server(cfg_no_inst, startup_timeout=0.5)

    # Suppress unused ComfyInstance warning in typing
    assert isinstance(inst, ComfyInstance)


def test_ui_inbox_single_view_and_browser_lifecycle(tmp_path: Path) -> None:
    import threading
    import time

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

    # Browser lifecycle: F5 refresh cancels grace timer; real disconnect triggers shutdown
    shutdown_called = threading.Event()
    lifecycle = BrowserSessionLifecycle(on_shutdown=shutdown_called.set, grace_seconds=0.15)
    lifecycle.on_connect()
    lifecycle.on_disconnect()
    lifecycle.on_connect()  # Simulated browser reload before 0.15s expires
    time.sleep(0.22)
    assert shutdown_called.is_set() is False
    assert lifecycle.active_clients == 1

    lifecycle.on_disconnect()
    assert shutdown_called.wait(timeout=1.0) is True


def test_ui_add_pair_to_inbox_tile_and_collision_handling(tmp_path: Path) -> None:
    from app import (
        add_pair_to_inbox,
        build_add_look_tile,
        build_artist_heatmap,
        gallery_with_add_tile,
        get_inbox_overview_data,
        is_case_repaired,
    )
    from PIL import Image

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

    # Duplicate upload increments suffix cleanly without overwriting existing case
    second_id = add_pair_to_inbox(model_file, ref_file, config_override=cfg)
    assert second_id == "street_look_2"

    _, thumbs, ids = get_inbox_overview_data(cfg)
    assert ids == ["street_look", "street_look_2"]
    gallery = gallery_with_add_tile(thumbs)
    assert len(gallery) == 3
    assert gallery[-1][1] == "Add Look"
    assert build_add_look_tile().shape == (320, 320, 3)

    # Verify repaired visibility & balanced heatmap gating
    assert (
        is_case_repaired(model_rgb, model_rgb.copy(), {"masked_changed_pixels_count": 0}) is False
    )
    assert is_case_repaired(model_rgb, model_rgb.copy(), None) is False
    assert build_artist_heatmap(model_rgb, model_rgb.copy(), config_override=cfg) is None

    changed_rgb = model_rgb.copy()
    changed_rgb[40:70, 40:70] = 200
    assert is_case_repaired(model_rgb, changed_rgb, {"masked_changed_pixels_count": 900}) is True
    assert build_artist_heatmap(model_rgb, changed_rgb, config_override=cfg) is not None

    # Negative: missing either image raises PipelineError
    with pytest.raises(PipelineError):
        add_pair_to_inbox(None, ref_file, config_override=cfg)
    with pytest.raises(PipelineError):
        add_pair_to_inbox(model_file, None, config_override=cfg)


def test_vlm_misplaced_logo_erase_prompt_and_linux_launcher_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from src.comfy_launcher import detect_standalone_instance

    # 1. Misplaced logo two-pass JSON (erase + insert) keeps erase prompt intact
    #    and skips contrast logo re-lookup on the plain-fabric erase defect.
    model_rgb = np.full((256, 256, 3), 100, dtype=np.uint8)
    model_rgb[155:195, 95:130] = 210  # misplaced lower logo
    ref_rgb = np.full((256, 256, 3), 100, dtype=np.uint8)
    ref_rgb[70:95, 80:105] = 220  # true higher pocket badge

    raw_json = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Misplaced patch",
            "defects": [
                {
                    "defect_summary": "Remove misplaced lower patch",
                    "model_bbox_2d": [360, 590, 520, 770],
                    "ref_bbox_2d": [300, 600, 450, 750],
                    "edit_prompt": (
                        "Edit <image1>: Remove and erase the misplaced patch, "
                        "leaving plain matching fabric from <image2>."
                    ),
                },
                {
                    "defect_summary": "Add authentic upper pocket badge",
                    "model_bbox_2d": [350, 460, 460, 560],
                    "ref_bbox_2d": [310, 270, 410, 370],
                    "edit_prompt": (
                        "Edit <image1>: Add ONLY the authentic circular badge from <image2>."
                    ),
                },
            ],
        }
    )
    parsed = Qwen3VLInspector.parse_vlm_json_response(
        raw_text=raw_json, model_rgb=model_rgb, ref_rgb=ref_rgb
    )
    assert len(parsed.resolved_defects) == 2
    d_erase, d_insert = parsed.resolved_defects
    assert Qwen3VLInspector._is_erase_defect(d_erase.edit_prompt) is True
    assert Qwen3VLInspector._is_erase_defect(d_insert.edit_prompt) is False
    assert "Remove and erase" in d_erase.edit_prompt

    # Verify inspect_garment_pair only invokes solo ref lookup for non-erase defects
    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    inspector = Qwen3VLInspector(ComfyConfig(), inspector_wf)
    solo_calls: list[str] = []
    plain_ref = np.full((256, 256, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(inspector, "_run_comfy_vlm", lambda *args, **kwargs: raw_json)
    monkeypatch.setattr(
        inspector,
        "_locate_ref_detail_single_image",
        lambda **kwargs: solo_calls.append(kwargs["case_id"]) or None,
    )
    inspector.inspect_garment_pair(model_rgb, plain_ref, "sage_sweatpant", "http://127.0.0.1:8000")
    assert solo_calls == ["sage_sweatpant_d2"]

    # 2. Linux standalone ComfyUI detection resolves ~/ComfyUI/venv/bin/python
    fake_home = tmp_path / "home"
    comfy_dir = fake_home / "ComfyUI"
    venv_bin = comfy_dir / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    venv_bin.mkdir(parents=True)
    py_file = venv_bin / ("python.exe" if sys.platform == "win32" else "python")
    py_file.write_bytes(b"")
    (comfy_dir / "main.py").write_text("# mock", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("COMFYUI_DIR", raising=False)
    inst = detect_standalone_instance()
    assert inst is not None
    assert inst.python_exe == py_file
    assert inst.main_py == comfy_dir / "main.py"
