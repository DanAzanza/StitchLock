"""Unit, Boundary, and Negative Tests for Core Models, Config, Discovery, and AutoMasker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms
from setup_test_cases import setup_all_test_cases
from src.auto_masker import (
    AutoMasker,
    discover_case_paths,
    has_manifest_roi,
    load_case_manifest,
    resolve_case_path,
)
from src.comfy_client import ComfyQwenBridge
from src.models import (
    ComfyConfig,
    ConfigurationError,
    LocalizationError,
    MaskingConfig,
    MaskingStrategy,
    NormalizedBox,
    PipelineConfig,
    PixelBBox,
    ReferenceType,
    load_pipeline_config,
    resolve_safe_path,
)
from src.pipeline import GarmentCorrectionPipeline, convert_reference_image_to_rgb
from src.vlm_inspector import Qwen3VLInspector, VLMInspectionResult

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_normalized_box_validation_and_conversion() -> None:
    box = NormalizedBox(0.25, 0.50, 0.75, 0.80)
    px = box.to_pixel_bbox(width=800, height=1000)
    assert px == PixelBBox(x1=200, y1=500, x2=600, y2=800)
    assert px.width == 400
    assert px.height == 300

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


def test_load_pipeline_config_reads_new_fields_and_rejects_invalid(tmp_path: Path) -> None:
    cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    assert (
        cfg.resolved_inspector_workflow_path
        == (ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json").resolve()
    )
    assert cfg.verification.min_masked_similarity == pytest.approx(0.15)
    assert cfg.heuristics.min_graphic_contrast_std == pytest.approx(14.0)

    bad_cfg = tmp_path / "bad_config.yaml"
    bad_cfg.write_text("verification:\n  min_masked_similarity: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="min_masked_similarity"):
        load_pipeline_config(bad_cfg, base_dir=tmp_path)

    bad_dilation = tmp_path / "bad_dilation.yaml"
    bad_dilation.write_text("masking:\n  mask_dilation_ratio: 0.9\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mask_dilation_ratio"):
        load_pipeline_config(bad_dilation, base_dir=tmp_path)


def test_flat_reference_discovery_mixed_extensions_and_negatives(tmp_path: Path) -> None:
    tc = tmp_path / "test_cases"
    tc.mkdir()

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

    with pytest.raises(ConfigurationError, match="Path traversal rejected"):
        resolve_case_path(tc, "../escape")

    orphan_model = tc / "stray_shirt.png"
    img.save(orphan_model)
    with pytest.raises(ConfigurationError, match="Orphan On-Model image"):
        discover_case_paths(tc)
    orphan_model.unlink()

    orphan_ref = tc / "typo_reference.png"
    img.save(orphan_ref)
    with pytest.raises(ConfigurationError, match="Orphan reference image"):
        discover_case_paths(tc)
    orphan_ref.unlink()

    dup_model = tc / "lookbook_a.png"
    img.save(dup_model)
    with pytest.raises(ConfigurationError, match="Duplicate On-Model images"):
        discover_case_paths(tc)


def test_auto_masker_enforces_strict_zero_outside_binary_mask(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    case_dir = tmp_path / "test_cases" / "case_01_kova_zip_jacket.png"
    manifest = load_case_manifest(case_dir)

    rng = np.random.default_rng(123)
    model_rgb = rng.integers(0, 256, size=(600, 400, 3), dtype=np.uint8)
    ref_rgb = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    masker = AutoMasker(MaskingConfig(strategy=MaskingStrategy.AUTO_VISION))
    res = masker.localize_and_mask(model_rgb, ref_rgb, manifest)

    assert res.binary_mask.shape == (600, 400)
    assert res.feathered_mask.shape == (600, 400)
    assert res.cropped_model_roi.shape == (600, 400, 3)
    assert res.cropped_product_ref.shape == (512, 512, 3)
    assert res.cropped_mask_greyscale.shape == (600, 400)
    assert res.cropped_mask_greyscale.ndim == 2

    large_model = np.zeros((1600, 1200, 3), dtype=np.uint8)
    res_large = masker.localize_and_mask(large_model, ref_rgb, manifest)
    assert np.count_nonzero(res_large.binary_mask) > np.count_nonzero(res.binary_mask) * 3

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

    too_small = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(LocalizationError, match="minimum supported dimension"):
        masker.localize_and_mask(too_small, good_rgb, manifest)


def test_auto_masker_micro_roi_supersampling_and_boundaries(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    manifest = load_case_manifest(tmp_path / "test_cases" / "case_01_kova_zip_jacket.png")
    micro_manifest = replace(
        manifest,
        model_target_box=NormalizedBox(0.57, 0.33, 0.62, 0.37),
        reference_detail_box=NormalizedBox(0.58, 0.29, 0.64, 0.34),
    )
    large_model = np.full((1024, 1024, 3), 40, dtype=np.uint8)
    large_ref = np.full((1024, 1024, 3), 45, dtype=np.uint8)
    masker = AutoMasker(MaskingConfig(strategy=MaskingStrategy.AUTO_VISION))

    micro_loc = masker.localize_and_mask(large_model, large_ref, micro_manifest)
    assert micro_loc.context_bbox.width == 256
    assert micro_loc.context_bbox.height == 256
    assert micro_loc.cropped_model_roi.shape == (256, 256, 3)
    assert micro_loc.cropped_product_ref.shape == (256, 256, 3)
    assert micro_loc.cropped_mask_greyscale.shape == (256, 256)
    assert np.all(micro_loc.feathered_mask[micro_loc.binary_mask == 0] == 0.0)
    assert np.count_nonzero(micro_loc.cropped_mask_greyscale) == np.count_nonzero(
        micro_loc.binary_mask
    )

    macro_manifest = replace(
        manifest,
        model_target_box=NormalizedBox(0.50, 0.30, 0.59, 0.39),
        reference_detail_box=NormalizedBox(0.50, 0.30, 0.59, 0.39),
    )
    assert masker.localize_and_mask(
        large_model, large_ref, macro_manifest
    ).context_bbox == PixelBBox(0, 0, 1024, 1024)

    erase_micro_manifest = replace(
        micro_manifest,
        prompt="Edit <image1>: Remove and erase misplaced badge from <image2>.",
    )
    assert masker.localize_and_mask(
        large_model, large_ref, erase_micro_manifest
    ).context_bbox == PixelBBox(0, 0, 1024, 1024)

    insert_micro_manifest = replace(
        micro_manifest,
        prompt=(
            "Edit <image1>: Add ONLY the authentic small detail from Image 2 onto this spot, "
            "matching <image2> in size and design."
        ),
    )
    assert masker.localize_and_mask(
        large_model, large_ref, insert_micro_manifest
    ).context_bbox == PixelBBox(0, 0, 1024, 1024)

    sub512_model = np.full((511, 511, 3), 40, dtype=np.uint8)
    assert masker.localize_and_mask(sub512_model, large_ref, micro_manifest).context_bbox == (
        PixelBBox(0, 0, 511, 511)
    )

    elongated_model = np.full((512, 4096, 3), 40, dtype=np.uint8)
    elongated_manifest = replace(
        manifest,
        model_target_box=NormalizedBox(0.50, 0.40, 0.555, 0.45),
        reference_detail_box=NormalizedBox(0.50, 0.40, 0.555, 0.45),
    )
    elongated_loc = masker.localize_and_mask(elongated_model, large_ref, elongated_manifest)
    assert elongated_loc.context_bbox == PixelBBox(0, 0, 4096, 512)

    corner_manifest = replace(
        manifest,
        model_target_box=NormalizedBox(0.005, 0.005, 0.045, 0.045),
        reference_detail_box=NormalizedBox(0.01, 0.01, 0.06, 0.06),
    )
    small_ref = np.full((200, 200, 3), 45, dtype=np.uint8)
    corner_loc = masker.localize_and_mask(large_model, small_ref, corner_manifest)
    assert corner_loc.context_bbox.width == 256
    assert np.count_nonzero(corner_loc.cropped_mask_greyscale) == np.count_nonzero(
        corner_loc.binary_mask
    )


def test_align_full_garment_reference_shifts_vertically() -> None:
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


def test_rgba_alpha_and_icc_profile_preservation_roundtrip(tmp_path: Path) -> None:
    tc = tmp_path / "test_cases"
    tc.mkdir()

    icc_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rng = np.random.default_rng(77)
    rgba = rng.integers(30, 80, size=(256, 256, 4), dtype=np.uint8)
    rgba[:, :, 3] = 200
    rgba[:20, :20, 3] = 0  # Transparent cutout outside mask
    ref_rgb = rng.integers(80, 140, size=(256, 256, 3), dtype=np.uint8)

    model_path = tc / "alpha_case.png"
    ref_path = tc / "alpha_case_reference.png"
    Image.fromarray(rgba, mode="RGBA").save(model_path, format="PNG", icc_profile=icc_bytes)
    Image.fromarray(ref_rgb, mode="RGB").save(ref_path, format="PNG")

    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    cfg = PipelineConfig(
        test_cases_dir=tc,
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=base_cfg.verification,
        inspector_workflow_path=base_cfg.resolved_inspector_workflow_path,
    )

    class _LocalInspector(Qwen3VLInspector):
        def inspect_garment_pair(
            self,
            model_rgb: np.ndarray,
            ref_rgb: np.ndarray,
            case_id: str,
            server_url: str | None = None,
        ) -> VLMInspectionResult:
            _ = (model_rgb, ref_rgb, case_id, server_url)
            return VLMInspectionResult(
                has_defect=True,
                defect_summary="Alpha test defect",
                model_target_box=NormalizedBox(0.35, 0.35, 0.65, 0.65),
                reference_detail_box=NormalizedBox(0.35, 0.35, 0.65, 0.65),
                edit_prompt="Edit <image1>: Replace center logo matching <image2>.",
                inspector_source="stub",
            )

    class _LocalBridge(ComfyQwenBridge):
        def _probe_live_server(self) -> str:
            return "http://127.0.0.1:8000"

        def restore_detail_roi(
            self,
            cropped_model_roi: np.ndarray,
            cropped_product_ref: np.ndarray,
            cropped_mask_greyscale: np.ndarray,
            prompt: str,
            case_id: str,
        ) -> tuple[np.ndarray, str]:
            _ = (cropped_model_roi, cropped_mask_greyscale, prompt, case_id)
            return cropped_product_ref.copy(), "stub_bridge"

    pipeline = GarmentCorrectionPipeline(
        cfg,
        inspector=_LocalInspector(cfg.comfyui, cfg.resolved_inspector_workflow_path),
        bridge=_LocalBridge(cfg.comfyui, cfg.workflow_path),
    )
    artifact = pipeline.run_case(model_path)
    assert artifact.metrics.passed is True

    with Image.open(artifact.corrected_image_path) as out_im:
        assert out_im.mode == "RGBA"
        assert out_im.info.get("icc_profile") == icc_bytes
        out_arr = np.asarray(out_im, dtype=np.uint8)
        assert np.array_equal(out_arr[:, :, 3], rgba[:, :, 3])

    # Negative/boundary: non-RGB ICC profile (e.g., GRAY) must not be embedded on RGB PNG output
    gray_icc = bytearray(icc_bytes)
    gray_icc[16:20] = b"GRAY"
    gray_tc = tmp_path / "gray_cases"
    gray_tc.mkdir()
    gray_model = gray_tc / "gray_icc_case.png"
    gray_ref = gray_tc / "gray_icc_case_reference.png"
    gray_arr = rng.integers(40, 120, size=(256, 256), dtype=np.uint8)
    Image.fromarray(gray_arr, mode="L").save(gray_model, format="PNG", icc_profile=bytes(gray_icc))
    Image.fromarray(ref_rgb, mode="RGB").save(gray_ref, format="PNG")
    gray_cfg = replace(cfg, test_cases_dir=gray_tc, outputs_dir=tmp_path / "gray_outputs")
    gray_pipeline = GarmentCorrectionPipeline(
        gray_cfg,
        inspector=_LocalInspector(gray_cfg.comfyui, gray_cfg.resolved_inspector_workflow_path),
        bridge=_LocalBridge(gray_cfg.comfyui, gray_cfg.workflow_path),
    )
    gray_artifact = gray_pipeline.run_case(gray_model)
    with Image.open(gray_artifact.corrected_image_path) as gray_out:
        assert gray_out.mode == "RGB"
        assert gray_out.info.get("icc_profile") is None


def test_standalone_logo_masker_and_transparent_png_handling(tmp_path: Path) -> None:
    setup_all_test_cases(tmp_path)
    base_manifest = load_case_manifest(tmp_path / "test_cases" / "case_01_kova_zip_jacket.png")
    masker = AutoMasker(MaskingConfig(strategy=MaskingStrategy.AUTO_VISION))

    # Dark hoodie on light studio backdrop with graphic close to left garment edge
    model_rgb = np.full((600, 600, 3), 240, dtype=np.uint8)
    model_rgb[120:540, 165:400] = (15, 16, 20)
    model_rgb[250:300, 172:285] = (230, 50, 130)

    # Wide standalone PAC-MAN style logo (spans almost full width on pure black)
    logo_ref = np.zeros((400, 400, 3), dtype=np.uint8)
    logo_ref[150:250, 20:380] = (245, 205, 60)

    logo_manifest = replace(
        base_manifest,
        model_target_box=NormalizedBox(0.28, 0.41, 0.48, 0.50),
        reference_detail_box=NormalizedBox(0.04, 0.36, 0.96, 0.64),
        reference_type=ReferenceType.LOGO,
        prompt="Edit <image1>: Replace back wordmark with <image2>.",
    )
    loc = masker.localize_and_mask(model_rgb, logo_ref, logo_manifest)
    # Studio backdrop left of x=165 must remain unmasked (binary_mask == 0 and feathered_mask == 0)
    assert np.all(loc.binary_mask[:, :164] == 0)
    assert np.all(loc.feathered_mask[loc.binary_mask == 0] == 0.0)
    assert np.count_nonzero(loc.binary_mask[250:300, 172:285]) > 100
    # Prepared logo reference stays centered and unclipped on both left & right sides
    prep = loc.cropped_product_ref
    assert prep.shape[0] == prep.shape[1]
    assert np.all(prep[:, :15] == 0)
    assert np.all(prep[:, -15:] == 0)
    assert int(np.max(prep[:, prep.shape[1] // 2])) > 180

    # Micro target with LOGO reference also keeps full centered logo (not a middle slice)
    micro_logo_manifest = replace(
        logo_manifest,
        model_target_box=NormalizedBox(0.45, 0.35, 0.50, 0.40),
        reference_detail_box=NormalizedBox(0.04, 0.36, 0.96, 0.64),
    )
    micro_loc = masker.localize_and_mask(model_rgb, logo_ref, micro_logo_manifest)
    assert micro_loc.context_bbox.width == 192
    assert np.all(micro_loc.cropped_product_ref[:, :8] == 0)
    assert np.all(micro_loc.cropped_product_ref[:, -8:] == 0)

    # Negative/boundary: uniform, white-on-light, and near-backdrop garments must NOT clip interiors
    uniform = np.full((256, 256, 3), 240, dtype=np.uint8)
    u_mask, _ = masker._build_masks(
        uniform, PixelBBox(60, 60, 140, 140), dilation_px=8, feather_sigma=3.0
    )
    assert int(np.count_nonzero(u_mask)) == 96 * 96

    white_on_light = np.full((320, 320, 3), 235, dtype=np.uint8)
    white_on_light[40:310, 60:260] = (245, 245, 245)  # White tee on light studio backdrop
    white_on_light[120:155, 110:210] = (30, 30, 30)  # Dark wordmark inside white chest
    wol_mask, wol_feather = masker._build_masks(
        white_on_light, PixelBBox(105, 115, 215, 160), dilation_px=6, feather_sigma=2.0
    )
    assert int(np.count_nonzero(wol_mask)) == 122 * 57
    assert float(np.min(wol_feather[120:155, 110:210])) > 0.5

    cream_on_light = np.full((320, 320, 3), (233, 233, 228), dtype=np.uint8)
    cream_on_light[40:310, 50:270] = (231, 229, 222)  # Cream cotton touching corner tolerance
    cream_on_light[120:155, 110:210] = (35, 55, 45)
    cream_mask, _ = masker._build_masks(
        cream_on_light, PixelBBox(105, 115, 215, 160), dilation_px=6, feather_sigma=2.0
    )
    assert int(np.count_nonzero(cream_mask)) == 122 * 57

    real_cream = ROOT_DIR / "test_cases" / "cream_crewneck.png"
    if real_cream.is_file():
        with Image.open(real_cream) as cream_im:
            cream_arr = np.asarray(cream_im.convert("RGB"), dtype=np.uint8)
        rc_mask, _ = masker._build_masks(
            cream_arr, PixelBBox(540, 388, 642, 472), dilation_px=0, feather_sigma=2.0
        )
        assert int(np.count_nonzero(rc_mask[388:472, 540:642])) == (642 - 540) * (472 - 388)

    # Transparent PNG logos composite onto contrasting neutral; opaque images pass unchanged
    dark_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    dark_rgba[30:70, 20:80, :3] = (15, 20, 25)
    dark_rgba[30:70, 20:80, 3] = 255
    dark_comp = convert_reference_image_to_rgb(Image.fromarray(dark_rgba, mode="RGBA"))
    assert int(dark_comp[0, 0, 0]) == 245

    light_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    light_rgba[30:70, 20:80, :3] = (245, 220, 80)
    light_rgba[30:70, 20:80, 3] = 255
    light_comp = convert_reference_image_to_rgb(Image.fromarray(light_rgba, mode="RGBA"))
    assert int(light_comp[0, 0, 0]) == 18

    opaque_rgb = np.full((64, 64, 3), (33, 66, 99), dtype=np.uint8)
    opaque_out = convert_reference_image_to_rgb(Image.fromarray(opaque_rgb, mode="RGB"))
    assert np.array_equal(opaque_out, opaque_rgb)

    # 3/4 pose perspective skew estimation & 2D homography warp direction
    pose_model = np.full((400, 400, 3), 240, dtype=np.uint8)
    pose_model[80:360, 100:280] = (15, 16, 20)  # Garment silhouette spanning x=100..280
    left_back_skew = AutoMasker._estimate_garment_perspective_skew(
        pose_model, PixelBBox(110, 160, 180, 210)
    )
    assert left_back_skew > 0.08
    center_skew = AutoMasker._estimate_garment_perspective_skew(
        pose_model, PixelBBox(155, 160, 225, 210)
    )
    assert center_skew == 0.0

    warped_logo = AutoMasker._prepare_standalone_logo_reference(
        ref_rgb=logo_ref,
        ref_detail_bbox=PixelBBox(20, 150, 380, 250),
        perspective_skew=0.18,
    )
    wy, wx = np.where(warped_logo[:, :, 0] > 180)
    mid_x = warped_logo.shape[1] // 2
    left_span = int(wy[wx < mid_x].max() - wy[wx < mid_x].min())
    right_span = int(wy[wx >= mid_x].max() - wy[wx >= mid_x].min())
    assert left_span > right_span
