"""8-Case Holdout Generalization Suite Across Diverse Garments, Ratios, and Defect Types."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from src.comfy_client import ComfyQwenBridge
from src.models import (
    ComfyConfig,
    PipelineConfig,
    load_pipeline_config,
)
from src.pipeline import GarmentCorrectionPipeline
from src.vlm_inspector import Qwen3VLInspector

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _HoldoutCaseSpec:
    case_id: str
    model_rgb: NDArray[np.uint8]
    ref_rgb: NDArray[np.uint8]
    vlm_dual_json: str
    vlm_solo_json: str | None
    expected_defect_count: int
    expected_bridge_calls: int
    expect_micro_roi: bool = False


class _ScriptedHoldoutInspector(Qwen3VLInspector):
    """Runs Qwen3VLInspector parsing, heuristics, and solo lookup with scripted VLM JSON."""

    def __init__(
        self,
        comfy_config: ComfyConfig,
        workflow_path: Path,
        specs_by_id: dict[str, _HoldoutCaseSpec],
    ) -> None:
        super().__init__(comfy_config, workflow_path)
        self._specs = specs_by_id

    def _run_comfy_vlm(
        self,
        server_url: str,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        case_id: str,
    ) -> str:
        _ = (server_url, model_rgb, ref_rgb)
        return self._specs[case_id].vlm_dual_json

    def _execute_vlm_workflow(self, server_url: str, workflow: dict[str, Any], case_id: str) -> str:
        _ = (server_url, workflow)
        base_id = case_id.split("_d")[0]
        solo_json = self._specs[base_id].vlm_solo_json
        assert solo_json is not None, f"Unexpected solo lookup for {case_id}"
        return solo_json


class _HoldoutBridge(ComfyQwenBridge):
    """Records ComfyQwenBridge edit invocations and returns reference-aligned ROI edits."""

    def __init__(self, config: ComfyConfig, workflow_path: Path) -> None:
        super().__init__(config, workflow_path)
        self.calls: list[tuple[str, tuple[int, int], str]] = []

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
        self._validate_inputs(cropped_model_roi, cropped_product_ref, cropped_mask_greyscale)
        self.calls.append(
            (case_id, (cropped_model_roi.shape[0], cropped_model_roi.shape[1]), prompt)
        )
        return cropped_product_ref.copy(), "comfyui_qwen_2_1 (holdout_stub)"


def _build_holdout_specs() -> list[_HoldoutCaseSpec]:
    specs: list[_HoldoutCaseSpec] = []

    # 1. Cream hoodie (4:5 portrait 400x500 model vs 512x512 cross-res reference)
    m1 = np.full((500, 400, 3), (42, 48, 56), dtype=np.uint8)
    m1[70:440, 80:320] = (232, 224, 210)
    m1[140:190, 145:255] = (180, 50, 50)
    r1 = np.full((512, 512, 3), (42, 48, 56), dtype=np.uint8)
    r1[72:450, 102:410] = (232, 224, 210)
    r1[143:194, 184:328] = (32, 36, 44)
    specs.append(
        _HoldoutCaseSpec(
            case_id="01_cream_hoodie",
            model_rgb=m1,
            ref_rgb=r1,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Garbled chest wordmark",
                    "defects": [
                        {
                            "defect_summary": "Misspelled chest print",
                            "model_bbox_2d": [360, 280, 640, 380],
                            "ref_bbox_2d": [360, 280, 640, 380],
                            "edit_prompt": (
                                "Edit <image1>: Replace red chest print with dark wordmark "
                                "matching <image2>."
                            ),
                        }
                    ],
                }
            ),
            vlm_solo_json=None,
            expected_defect_count=1,
            expected_bridge_calls=1,
        )
    )

    # 2. Charcoal parka (1:1, 1024x1024): micro crest missed by VLM, caught by CV + micro-ROI
    m2 = np.full((1024, 1024, 3), 244, dtype=np.uint8)
    m2[120:960, 200:820] = (38, 42, 48)
    m2[356:384, 594:634] = (220, 224, 228)  # filled-in blob
    r2 = np.full((1024, 1024, 3), 244, dtype=np.uint8)
    r2[120:960, 200:820] = (38, 42, 48)
    r2[356:384, 608:620] = (220, 224, 228)
    r2[364:384, 594:604] = (220, 224, 228)
    r2[364:384, 624:634] = (220, 224, 228)
    specs.append(
        _HoldoutCaseSpec(
            case_id="02_charcoal_parka",
            model_rgb=m2,
            ref_rgb=r2,
            vlm_dual_json=json.dumps(
                {"has_defect": False, "defect_summary": "Looks fine", "defects": []}
            ),
            vlm_solo_json=None,
            expected_defect_count=1,
            expected_bridge_calls=1,
            expect_micro_roi=True,
        )
    )

    # 3. Cobalt jersey (3:4 portrait, 450x600): vertically shifted patch -> split ERASE + INSERT
    m3 = np.full((600, 450, 3), (238, 240, 244), dtype=np.uint8)
    m3[220:540, 140:310] = (35, 82, 175)
    m3[396:456, 170:220] = (225, 115, 45)
    r3 = np.full((600, 450, 3), (238, 240, 244), dtype=np.uint8)
    r3[40:560, 120:330] = (35, 82, 175)
    r3[140:176, 150:186] = (225, 115, 45)
    specs.append(
        _HoldoutCaseSpec(
            case_id="03_cobalt_jersey",
            model_rgb=m3,
            ref_rgb=r3,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Patch sits too low on garment",
                    "model_bbox_2d": [365, 650, 500, 770],
                    "ref_bbox_2d": [320, 220, 430, 310],
                    "edit_prompt": (
                        "Edit <image1>: Replace low patch with upper patch matching <image2>."
                    ),
                }
            ),
            vlm_solo_json=None,
            expected_defect_count=2,
            expected_bridge_calls=2,
        )
    )

    # 4. Crimson polo (1:1, 512x512): horizontally misplaced emblem -> split ERASE + INSERT
    m4 = np.full((512, 512, 3), (245, 242, 238), dtype=np.uint8)
    m4[40:472, 128:384] = (168, 36, 48)
    m4[144:194, 300:350] = (240, 235, 220)
    r4 = np.full((512, 512, 3), (245, 242, 238), dtype=np.uint8)
    r4[40:472, 128:384] = (168, 36, 48)
    r4[144:184, 158:198] = (240, 235, 220)
    specs.append(
        _HoldoutCaseSpec(
            case_id="04_crimson_polo",
            model_rgb=m4,
            ref_rgb=r4,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Emblem mirrored onto wrong side",
                    "model_bbox_2d": [575, 275, 695, 385],
                    "ref_bbox_2d": [300, 275, 395, 370],
                    "edit_prompt": "Edit <image1>: Replace wrong-side emblem matching <image2>.",
                }
            ),
            vlm_solo_json=None,
            expected_defect_count=2,
            expected_bridge_calls=2,
        )
    )

    # 5. Heather grey crewneck (16:10 640x400 model vs 512x512 cross-res reference): dual defects
    m5 = np.full((400, 640, 3), (20, 26, 36), dtype=np.uint8)
    m5[40:360, 120:520] = (145, 148, 154)
    m5[90:150, 260:340] = (220, 70, 50)
    m5[220:290, 420:490] = (220, 70, 50)
    r5 = np.full((512, 512, 3), (20, 26, 36), dtype=np.uint8)
    r5[51:460, 96:416] = (145, 148, 154)
    r5[118:188, 212:268] = (30, 110, 90)
    r5[138:168, 226:254] = (242, 242, 242)
    r5[284:368, 338:388] = (30, 110, 90)
    r5[308:346, 352:376] = (242, 242, 242)
    specs.append(
        _HoldoutCaseSpec(
            case_id="05_heather_crewneck",
            model_rgb=m5,
            ref_rgb=r5,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Two separated graphic defects",
                    "defects": [
                        {
                            "defect_summary": "Chest crest",
                            "model_bbox_2d": [400, 220, 535, 380],
                            "ref_bbox_2d": [400, 220, 535, 380],
                            "edit_prompt": "Replace chest crest with <image2>.",
                        },
                        {
                            "defect_summary": "Sleeve stripe",
                            "model_bbox_2d": [650, 545, 770, 730],
                            "ref_bbox_2d": [650, 545, 770, 730],
                            "edit_prompt": "Replace sleeve stripe with <image2>.",
                        },
                    ],
                }
            ),
            vlm_solo_json=None,
            expected_defect_count=2,
            expected_bridge_calls=2,
        )
    )

    # 6. Navy trousers (3:4 portrait, 512x680): intact micro emblem control (0 bridge calls)
    m6 = np.full((680, 512, 3), (236, 238, 242), dtype=np.uint8)
    m6[60:640, 110:400] = (26, 38, 66)
    m6[210:240, 290:320] = (228, 200, 120)
    r6 = m6.copy()
    specs.append(
        _HoldoutCaseSpec(
            case_id="06_navy_trousers_clean",
            model_rgb=m6,
            ref_rgb=r6,
            vlm_dual_json=json.dumps(
                {"has_defect": False, "defect_summary": "All details intact", "defects": []}
            ),
            vlm_solo_json=None,
            expected_defect_count=0,
            expected_bridge_calls=0,
        )
    )

    # 7. Sand trench coat (4:5 480x600 model vs 512x512 cross-res reference): connected signature
    m7 = np.full((600, 480, 3), (35, 38, 44), dtype=np.uint8)
    m7[60:540, 90:390] = (202, 184, 154)
    m7[190:235, 255:315] = (32, 34, 38)
    m7[244:262, 250:320] = (32, 34, 38)
    r7 = np.full((512, 512, 3), (35, 38, 44), dtype=np.uint8)
    r7[51:460, 96:416] = (202, 184, 154)
    r7[160:225, 264:343] = (40, 80, 120)
    specs.append(
        _HoldoutCaseSpec(
            case_id="07_sand_trench",
            model_rgb=m7,
            ref_rgb=r7,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Emblem and detached wordmark distorted",
                    "defects": [
                        {
                            "defect_summary": "Truncated emblem box",
                            "model_bbox_2d": [520, 310, 665, 395],
                            "ref_bbox_2d": [515, 310, 670, 440],
                            "edit_prompt": "Replace chest mark and wordmark from <image2>.",
                        }
                    ],
                }
            ),
            vlm_solo_json=None,
            expected_defect_count=1,
            expected_bridge_calls=1,
        )
    )

    # 8. Forest fleece vest (1:1, 512x512): plain-cloth dual ref_bbox triggers solo contrast lookup
    m8 = np.full((512, 512, 3), (240, 242, 240), dtype=np.uint8)
    m8[50:460, 110:400] = (34, 86, 56)
    m8[140:200, 260:330] = (210, 80, 60)
    r8 = np.full((512, 512, 3), (240, 242, 240), dtype=np.uint8)
    r8[50:460, 110:400] = (34, 86, 56)
    r8[100:155, 255:325] = (242, 236, 224)
    specs.append(
        _HoldoutCaseSpec(
            case_id="08_forest_vest",
            model_rgb=m8,
            ref_rgb=r8,
            vlm_dual_json=json.dumps(
                {
                    "has_defect": True,
                    "defect_summary": "Echoed wrong reference coordinates",
                    "defects": [
                        {
                            "defect_summary": "Chest patch mismatch",
                            "model_bbox_2d": [500, 270, 650, 395],
                            "ref_bbox_2d": [500, 420, 640, 540],
                            "edit_prompt": (
                                "Replace red chest patch with cream crest from <image2>."
                            ),
                        }
                    ],
                }
            ),
            vlm_solo_json=json.dumps({"bbox_2d": [485, 185, 645, 315]}),
            expected_defect_count=1,
            expected_bridge_calls=1,
        )
    )

    return specs


def test_eight_holdout_garments_generalize_without_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.vlm_inspector.upload_png_bytes", lambda _url, name, _png: name)
    specs = _build_holdout_specs()
    specs_by_id = {s.case_id: s for s in specs}

    tc_dir = tmp_path / "test_cases"
    tc_dir.mkdir()
    for spec in specs:
        Image.fromarray(spec.model_rgb, mode="RGB").save(tc_dir / f"{spec.case_id}.png")
        Image.fromarray(spec.ref_rgb, mode="RGB").save(tc_dir / f"{spec.case_id}_reference.png")

    base_cfg = load_pipeline_config(ROOT_DIR / "config.yaml", base_dir=ROOT_DIR)
    cfg = PipelineConfig(
        test_cases_dir=tc_dir,
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=ComfyConfig(server_url="http://127.0.0.1:59999", fallback_ports=()),
        verification=base_cfg.verification,
        inspector_workflow_path=base_cfg.resolved_inspector_workflow_path,
        heuristics=base_cfg.heuristics,
    )

    inspector = _ScriptedHoldoutInspector(
        cfg.comfyui, cfg.resolved_inspector_workflow_path, specs_by_id
    )
    bridge = _HoldoutBridge(cfg.comfyui, cfg.workflow_path)
    pipeline = GarmentCorrectionPipeline(cfg, inspector=inspector, bridge=bridge)

    for case_path in pipeline.discover_cases():
        case_id = case_path.stem
        spec = specs_by_id[case_id]
        calls_before = len(bridge.calls)
        artifact = pipeline.run_case(case_path)
        calls_for_case = bridge.calls[calls_before:]

        assert artifact.metrics.passed is True, f"Holdout failed: {case_id}"
        assert artifact.metrics.unmasked_max_abs_diff == 0
        assert artifact.metrics.unmasked_changed_pixels_count == 0
        assert artifact.metrics.unmasked_ssim == 1.0
        assert len(calls_for_case) == spec.expected_bridge_calls

        if spec.expected_defect_count == 0:
            assert np.array_equal(artifact.original_model_rgb, artifact.final_corrected_rgb)
        if spec.expect_micro_roi:
            assert artifact.localization.context_bbox.width == 256
            assert artifact.localization.context_bbox.height == 256
        if case_id == "07_sand_trench":
            assert artifact.localization.target_bbox.y2 >= int(0.43 * 600)
        if case_id == "08_forest_vest":
            assert artifact.localization.ref_detail_bbox.y1 <= int(0.22 * 512)
