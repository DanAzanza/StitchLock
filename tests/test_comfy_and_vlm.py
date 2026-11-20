"""Unit, Boundary, and Negative Tests for ComfyUI Bridge, VLM Inspector, Heuristics & Launcher."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from setup_test_cases import setup_all_test_cases
from src.comfy_client import ComfyQwenBridge
from src.comfy_launcher import (
    ComfyInstance,
    ComfyServerManager,
    build_headless_command,
    detect_comfy_desktop_instance,
    detect_standalone_instance,
    ensure_comfyui_server,
)
from src.housekeeper import ComfyUIHousekeeper, is_loopback_host
from src.models import (
    ComfyConfig,
    ComfyExecutionError,
    NormalizedBox,
    PipelineConfig,
    ReferenceType,
    load_pipeline_config,
)
from src.pipeline import GarmentCorrectionPipeline
from src.vlm_heuristics import tighten_standalone_logo_box
from src.vlm_inspector import (
    MAX_DEFECTS_PER_IMAGE,
    REFERENCE_CLASSIFICATION_PROMPT,
    Qwen3VLInspector,
    _resize_for_vlm,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


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
    assert bridge.probe_live_server() is None

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
        auto_launch=False,
    )
    offline_cfg = PipelineConfig(
        test_cases_dir=tmp_path / "test_cases",
        outputs_dir=tmp_path / "outputs",
        workflow_path=base_cfg.workflow_path,
        masking=base_cfg.masking,
        comfyui=offline_comfy,
        verification=base_cfg.verification,
        inspector_workflow_path=base_cfg.resolved_inspector_workflow_path,
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


def test_qwen3_vl_inspector_json_parsing_and_negative_rejection() -> None:
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

    with pytest.raises(ComfyExecutionError, match="did not contain JSON"):
        Qwen3VLInspector.parse_vlm_json_response(
            raw_text="No JSON here at all",
            model_rgb=dummy_rgb,
            ref_rgb=dummy_rgb,
        )


def test_vlm_multi_defect_parsing_dedup_and_cap() -> None:
    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

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

    clean = Qwen3VLInspector.parse_vlm_json_response(
        json.dumps({"has_defect": False, "defect_summary": "All intact", "defects": []}),
        dummy_rgb,
        dummy_rgb,
    )
    assert clean.has_defect is False
    assert clean.resolved_defects == ()


def test_vlm_misplaced_logo_erase_prompt_and_solo_lookup_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_rgb = np.full((256, 256, 3), 100, dtype=np.uint8)
    model_rgb[155:195, 95:130] = 210
    ref_rgb = np.full((256, 256, 3), 100, dtype=np.uint8)
    ref_rgb[70:95, 80:105] = 220

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

    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    inspector = Qwen3VLInspector(ComfyConfig(), inspector_wf)
    solo_calls: list[str] = []
    plain_ref = np.full((256, 256, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(
        inspector, "_classify_reference_type", lambda *_a, **_kw: ReferenceType.GARMENT
    )
    monkeypatch.setattr(inspector, "_run_comfy_vlm", lambda *args, **kwargs: raw_json)
    monkeypatch.setattr(
        inspector,
        "_locate_ref_detail_single_image",
        lambda **kwargs: solo_calls.append(kwargs["case_id"]) or None,
    )
    inspector.inspect_garment_pair(model_rgb, plain_ref, "sage_sweatpant", "http://127.0.0.1:8000")
    assert solo_calls == ["sage_sweatpant_d2"]


def test_vlm_solo_lookup_logs_warning_on_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    inspector = Qwen3VLInspector(ComfyConfig(), inspector_wf)
    ref_rgb = np.full((128, 128, 3), 100, dtype=np.uint8)

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise requests.ConnectionError("Simulated ComfyUI upload timeout")

    monkeypatch.setattr("src.vlm_inspector.upload_png_bytes", _boom)
    with caplog.at_level(logging.WARNING, logger="garment_fix.vlm_inspector"):
        res = inspector._locate_ref_detail_single_image(
            server_url="http://127.0.0.1:8000",
            ref_rgb=ref_rgb,
            target_description="chest crest",
            case_id="warn_case",
        )
    assert res is None
    assert "Single-image reference lookup fallback failed" in caplog.text


def test_connected_graphic_box_refinement_and_boundaries() -> None:
    dark_jacket = np.full((256, 256, 3), 24, dtype=np.uint8)
    dark_jacket[146:162, 148:172] = 235
    dark_jacket[166:174, 146:174] = 235
    truncated_box = NormalizedBox(0.55, 0.46, 0.70, 0.625)
    refined = Qwen3VLInspector._refine_connected_graphic_box(dark_jacket, truncated_box, None)
    assert refined.y_max >= 0.68

    prior = NormalizedBox(0.50, 0.45, 0.72, 0.68)
    assert (
        Qwen3VLInspector._refine_connected_graphic_box(
            dark_jacket, NormalizedBox(0.52, 0.47, 0.70, 0.66), prior
        )
        == prior
    )

    edge_stripe = np.full((256, 256, 3), 24, dtype=np.uint8)
    edge_stripe[146:162, 148:172] = 235
    edge_stripe[166:174, 134:186] = 235
    edge_refined = Qwen3VLInspector._refine_connected_graphic_box(edge_stripe, truncated_box, None)
    assert edge_refined.y_max < 0.66

    disconnected = np.full((256, 256, 3), 24, dtype=np.uint8)
    disconnected[140:152, 148:172] = 235
    disconnected[174:184, 148:172] = 235
    disc_refined = Qwen3VLInspector._refine_connected_graphic_box(disconnected, truncated_box, None)
    assert disc_refined.y_max < 0.66

    tiny_img = np.full((64, 64, 3), 24, dtype=np.uint8)
    tiny_refined = Qwen3VLInspector._refine_connected_graphic_box(
        tiny_img, NormalizedBox(0.10, 0.10, 0.20, 0.20), None
    )
    assert tiny_refined.y_max <= 0.225


def test_shifted_defect_split_vertical_horizontal_and_boundaries() -> None:
    model_pants = np.full((256, 256, 3), 242, dtype=np.uint8)
    model_pants[106:235, 92:166] = np.array([130, 140, 115], dtype=np.uint8)
    model_pants[144:176, 98:124] = np.array([195, 95, 55], dtype=np.uint8)
    ref_pants = np.full((256, 256, 3), 242, dtype=np.uint8)
    ref_pants[14:242, 74:182] = np.array([130, 140, 115], dtype=np.uint8)
    ref_pants[72:88, 82:96] = np.array([195, 95, 55], dtype=np.uint8)

    shifted_json = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Oversized lower badge instead of small hip patch",
            "model_bbox_2d": [373, 550, 500, 700],
            "ref_bbox_2d": [280, 220, 410, 390],
            "edit_prompt": "Edit <image1>: Replace oversized patch with small badge from <image2>.",
        }
    )
    split_res = Qwen3VLInspector.parse_vlm_json_response(shifted_json, model_pants, ref_pants)
    assert len(split_res.resolved_defects) == 2
    d_erase, d_insert = split_res.resolved_defects
    assert Qwen3VLInspector._is_erase_defect(d_erase.edit_prompt) is True
    assert Qwen3VLInspector._is_erase_defect(d_insert.edit_prompt) is False
    assert d_erase.model_target_box.y_min >= 0.54
    assert d_insert.model_target_box.y_max < d_erase.model_target_box.y_min + 0.02

    aligned_pants = ref_pants.copy()
    aligned_json = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Aligned defect stays single pass",
            "model_bbox_2d": [310, 270, 390, 360],
            "ref_bbox_2d": [310, 270, 390, 360],
            "edit_prompt": "Edit <image1>: Replace aligned patch with <image2>.",
        }
    )
    aligned_res = Qwen3VLInspector.parse_vlm_json_response(aligned_json, aligned_pants, ref_pants)
    assert len(aligned_res.resolved_defects) == 1

    small_arr = np.full((16, 16, 3), 120, dtype=np.uint8)
    assert (
        len(
            Qwen3VLInspector._split_misplaced_defect_if_shifted(
                small_arr, small_arr, list(aligned_res.resolved_defects)
            )
        )
        == 1
    )

    horiz_ref = np.full((256, 256, 3), 242, dtype=np.uint8)
    horiz_ref[14:242, 74:182] = np.array([130, 140, 115], dtype=np.uint8)
    horiz_ref[72:88, 82:96] = np.array([195, 95, 55], dtype=np.uint8)
    horiz_model = np.full((256, 256, 3), 242, dtype=np.uint8)
    horiz_model[14:242, 74:182] = np.array([130, 140, 115], dtype=np.uint8)
    horiz_model[72:96, 152:174] = np.array([195, 95, 55], dtype=np.uint8)
    horiz_json = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Logo on wrong side",
            "model_bbox_2d": [580, 275, 690, 385],
            "ref_bbox_2d": [300, 260, 390, 360],
            "edit_prompt": "Edit <image1>: Replace wrong-side patch with <image2>.",
        }
    )
    horiz_res = Qwen3VLInspector.parse_vlm_json_response(horiz_json, horiz_model, horiz_ref)
    assert len(horiz_res.resolved_defects) == 2
    assert horiz_res.resolved_defects[0].model_target_box.x_min >= 0.55
    assert horiz_res.resolved_defects[1].model_target_box.x_max <= 0.45


def test_micro_emblem_detection_and_negative_boundaries() -> None:
    clean_jkt = np.full((512, 512, 3), 248, dtype=np.uint8)
    clean_jkt[60:480, 100:410] = np.array([36, 46, 52], dtype=np.uint8)
    crest_rgb = np.array([210, 212, 218], dtype=np.uint8)
    clean_jkt[178:192, 304:310] = crest_rgb
    clean_jkt[182:192, 297:302] = crest_rgb
    clean_jkt[182:192, 312:317] = crest_rgb
    ref_jkt = clean_jkt.copy()
    no_defect_json = json.dumps({"has_defect": False, "defect_summary": "Looks ok", "defects": []})

    intact_res = Qwen3VLInspector.parse_vlm_json_response(no_defect_json, clean_jkt, ref_jkt)
    assert intact_res.has_defect is False

    broken_jkt = clean_jkt.copy()
    broken_jkt[178:192, 297:317] = crest_rgb
    broken_res = Qwen3VLInspector.parse_vlm_json_response(no_defect_json, broken_jkt, ref_jkt)
    assert broken_res.has_defect is True
    assert "chest emblem" in broken_res.defect_summary
    assert "leaves" not in broken_res.edit_prompt

    zipper_jkt = np.full((512, 512, 3), 248, dtype=np.uint8)
    zipper_jkt[60:480, 100:410] = np.array([36, 46, 52], dtype=np.uint8)
    zipper_jkt[155:210, 304:308] = crest_rgb
    assert (
        Qwen3VLInspector._find_salient_emblem_box(zipper_jkt, NormalizedBox(0.50, 0.26, 0.68, 0.46))
        is None
    )
    one_sided = Qwen3VLInspector.parse_vlm_json_response(no_defect_json, zipper_jkt, ref_jkt)
    assert one_sided.has_defect is False
    assert (
        Qwen3VLInspector._find_salient_emblem_box(
            np.zeros((24, 24, 3), dtype=np.uint8), NormalizedBox(0.50, 0.26, 0.68, 0.46)
        )
        is None
    )

    # Faint lower plaid/fold highlights must not override a real chest emblem or clean plaid
    plaid_clean = clean_jkt.copy()
    plaid_clean[295:325, 195:225] = np.array([75, 84, 88], dtype=np.uint8)
    plaid_ref = ref_jkt.copy()
    plaid_ref[290:315, 210:238] = np.array([72, 82, 86], dtype=np.uint8)
    assert (
        Qwen3VLInspector.parse_vlm_json_response(no_defect_json, plaid_clean, plaid_ref).has_defect
        is False
    )

    plaid_broken = broken_jkt.copy()
    plaid_broken[295:325, 195:225] = np.array([75, 84, 88], dtype=np.uint8)
    plaid_res = Qwen3VLInspector.parse_vlm_json_response(no_defect_json, plaid_broken, plaid_ref)
    assert plaid_res.has_defect is True
    assert plaid_res.model_target_box.y_max < 0.45


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

    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside")
    hk.record_uploaded_name("../outside.png")

    purged = hk.cleanup_session_inputs()
    assert purged == 1
    assert pre_existing.exists() is True
    assert outside_file.exists() is True
    assert session_file.exists() is False


def test_comfy_launcher_detection_and_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert isinstance(inst, ComfyInstance)

    fake_home = tmp_path / "home"
    comfy_dir = fake_home / "ComfyUI"
    venv_bin = comfy_dir / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    venv_bin.mkdir(parents=True)
    py_file = venv_bin / ("python.exe" if sys.platform == "win32" else "python")
    py_file.write_bytes(b"")
    (comfy_dir / "main.py").write_text("# mock", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("COMFYUI_DIR", raising=False)
    standalone = detect_standalone_instance()
    assert standalone is not None
    assert standalone.python_exe == py_file
    assert standalone.main_py == comfy_dir / "main.py"


def test_comfy_server_manager_isolated_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeProcess:
        def __init__(self, pid: int, returncode: int | None = None) -> None:
            self.pid = pid
            self.returncode = returncode

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0 if self.returncode is None else self.returncode

    cfg = ComfyConfig(server_url="http://127.0.0.1:8123", fallback_ports=(), auto_launch=True)
    fake_inst = ComfyInstance(
        name="Isolated Headless",
        mode="headless",
        python_exe=tmp_path / "python.exe",
        main_py=tmp_path / "main.py",
        working_dir=tmp_path,
    )

    mgr_a = ComfyServerManager(register_atexit=False)
    mgr_b = ComfyServerManager(register_atexit=False)

    online_calls = {"n": 0}
    spawned: list[_FakeProcess] = []
    freed_urls: list[str] = []
    terminated_pids: list[int] = []

    def _fake_online(url: str, timeout: float = 1.5) -> bool:
        _ = (url, timeout)
        online_calls["n"] += 1
        return online_calls["n"] >= 2

    def _fake_spawn(inst: ComfyInstance, host: str, port: str) -> _FakeProcess:
        _ = (inst, host, port)
        proc = _FakeProcess(pid=4242 + len(spawned), returncode=None)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("src.comfy_launcher.probe_candidate_servers", lambda _c: None)
    monkeypatch.setattr("src.comfy_launcher.detect_comfyui", lambda: fake_inst)
    monkeypatch.setattr("src.comfy_launcher._spawn_instance", _fake_spawn)
    monkeypatch.setattr("src.comfy_launcher.is_comfy_server_online", _fake_online)
    monkeypatch.setattr("src.comfy_launcher.time.sleep", lambda _s: None)
    monkeypatch.setattr("src.comfy_launcher._free_server_vram", freed_urls.append)
    monkeypatch.setattr(
        "src.comfy_launcher._terminate_process_tree", lambda p: terminated_pids.append(p.pid)
    )

    url = mgr_a.ensure_server(cfg, startup_timeout=5.0)
    assert url == "http://127.0.0.1:8123"
    assert len(spawned) == 1
    assert mgr_a.process is spawned[0]
    assert mgr_a.server_url == "http://127.0.0.1:8123"
    assert mgr_b.process is None
    assert mgr_b.server_url is None

    # URL switch stops prior spawned process cleanly before spawning replacement
    online_calls["n"] = 0
    cfg_alt = ComfyConfig(server_url="http://127.0.0.1:8199", fallback_ports=(), auto_launch=True)
    alt_url = mgr_a.ensure_server(cfg_alt, startup_timeout=5.0)
    assert alt_url == "http://127.0.0.1:8199"
    assert len(spawned) == 2
    assert freed_urls == ["http://127.0.0.1:8123"]
    assert terminated_pids == [4242]
    assert mgr_a.process is spawned[1]
    assert mgr_a.server_url == "http://127.0.0.1:8199"

    mgr_a.stop()
    assert freed_urls == ["http://127.0.0.1:8123", "http://127.0.0.1:8199"]
    assert terminated_pids == [4242, 4243]
    assert mgr_a.process is None
    assert mgr_a.server_url is None

    # Idempotent second stop
    mgr_a.stop()
    assert freed_urls == ["http://127.0.0.1:8123", "http://127.0.0.1:8199"]
    assert terminated_pids == [4242, 4243]

    # Premature exit clears manager state and raises ComfyExecutionError
    crashed = _FakeProcess(pid=5001, returncode=1)
    monkeypatch.setattr("src.comfy_launcher._spawn_instance", lambda *_a: crashed)
    monkeypatch.setattr("src.comfy_launcher.is_comfy_server_online", lambda *_a, **_kw: False)
    with pytest.raises(ComfyExecutionError, match="exited prematurely"):
        mgr_b.ensure_server(cfg, startup_timeout=2.0)
    assert mgr_b.process is None
    assert mgr_b.server_url is None

    # Startup timeout terminates live hung process and clears manager state
    hung = _FakeProcess(pid=5002, returncode=None)
    ticks = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr("src.comfy_launcher._spawn_instance", lambda *_a: hung)
    monkeypatch.setattr("src.comfy_launcher.time.monotonic", lambda: next(ticks, 10.0))
    with pytest.raises(ComfyExecutionError, match="did not respond"):
        mgr_b.ensure_server(cfg, startup_timeout=2.0)
    assert terminated_pids == [4242, 4243, 5002]
    assert mgr_b.process is None
    assert mgr_b.server_url is None


def test_reference_type_classification_and_logo_inspection_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Qwen3VLInspector.parse_reference_type_json('{"reference_type": "logo"}') == (
        ReferenceType.LOGO
    )
    assert Qwen3VLInspector.parse_reference_type_json('{"reference_type": "garment"}') == (
        ReferenceType.GARMENT
    )
    # Malformed or unknown output falls back safely to garment
    assert Qwen3VLInspector.parse_reference_type_json("not json") == ReferenceType.GARMENT
    assert Qwen3VLInspector.parse_reference_type_json('{"reference_type": "unknown"}') == (
        ReferenceType.GARMENT
    )

    # Standalone multicolor logo on pure black (e.g. PAC-MAN style) must keep full extent
    logo_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    logo_rgb[96:160, 24:110] = (245, 205, 65)
    logo_rgb[96:160, 120:232] = (235, 80, 60)
    tight = tighten_standalone_logo_box(logo_rgb)
    assert tight.x_min <= 0.09
    assert tight.x_max >= 0.91
    assert tight.y_min <= 0.37
    assert tight.y_max >= 0.63

    # Uniform blank image falls back cleanly instead of collapsing
    blank = np.full((128, 128, 3), 240, dtype=np.uint8)
    fallback_box = NormalizedBox(0.10, 0.20, 0.90, 0.80)
    assert tighten_standalone_logo_box(blank, fallback_box=fallback_box) == fallback_box

    # Mock on-model hoodie with pink wordmark on lower-left back
    model_rgb = np.full((256, 256, 3), 240, dtype=np.uint8)
    model_rgb[40:240, 60:170] = (15, 15, 18)
    model_rgb[105:130, 72:130] = (230, 40, 125)

    logo_json = json.dumps(
        {
            "has_defect": True,
            "defect_summary": "Replace FORTNITE back wordmark with PAC-MAN logo",
            "defects": [
                {
                    "defect_summary": "Replace FORTNITE wordmark",
                    "model_bbox_2d": [275, 410, 510, 505],
                    "ref_bbox_2d": [45, 370, 955, 630],
                    "edit_prompt": (
                        "Edit <image1>: Replace ONLY the pink FORTNITE wordmark with "
                        "the PAC-MAN logo from <image2>."
                    ),
                }
            ],
        }
    )

    # Shifted model vs logo coords must NOT trigger garment-anatomical erase+insert split
    parsed_logo = Qwen3VLInspector.parse_vlm_json_response(
        raw_text=logo_json,
        model_rgb=model_rgb,
        ref_rgb=logo_rgb,
        reference_type=ReferenceType.LOGO,
    )
    assert parsed_logo.reference_type == ReferenceType.LOGO
    assert parsed_logo.to_dict()["reference_type"] == "logo"
    assert len(parsed_logo.resolved_defects) == 1
    assert Qwen3VLInspector._is_erase_defect(parsed_logo.resolved_defects[0].edit_prompt) is False

    # has_defect=False with LOGO reference must NOT run garment 3x3 micro-emblem scan
    clean_logo_json = json.dumps(
        {"has_defect": False, "defect_summary": "Already matching", "defects": []}
    )
    clean_logo = Qwen3VLInspector.parse_vlm_json_response(
        raw_text=clean_logo_json,
        model_rgb=model_rgb,
        ref_rgb=logo_rgb,
        reference_type=ReferenceType.LOGO,
    )
    assert clean_logo.has_defect is False
    assert clean_logo.reference_type == ReferenceType.LOGO

    # inspect_garment_pair routes LOGO reference to _run_logo_vlm and skips garment solo lookup
    inspector_wf = ROOT_DIR / "workflows" / "qwen3_vl_inspector_api.json"
    inspector = Qwen3VLInspector(ComfyConfig(), inspector_wf)
    routed: list[str] = []
    monkeypatch.setattr(
        inspector, "_classify_reference_type", lambda *_a, **_kw: ReferenceType.LOGO
    )
    monkeypatch.setattr(
        inspector,
        "_run_logo_vlm",
        lambda *_a, **_kw: routed.append("logo") or logo_json,
    )
    monkeypatch.setattr(
        inspector,
        "_run_comfy_vlm",
        lambda *_a, **_kw: routed.append("garment") or logo_json,
    )
    monkeypatch.setattr(
        inspector,
        "_locate_ref_detail_single_image",
        lambda **_kw: routed.append("solo") or None,
    )
    inspected = inspector.inspect_garment_pair(
        model_rgb, logo_rgb, "hoodie_pacman", server_url="http://127.0.0.1:8000"
    )
    assert routed == ["logo"]
    assert inspected.reference_type == ReferenceType.LOGO
    assert inspected.reference_detail_box.x_min <= 0.09
    assert inspected.reference_detail_box.x_max >= 0.91

    # _resize_for_vlm preserves small inputs, downscales oversized images, and keeps min >= 32px
    small = np.zeros((256, 128, 3), dtype=np.uint8)
    assert _resize_for_vlm(small, max_side=512) is small
    wide = np.zeros((1200, 2400, 3), dtype=np.uint8)
    resized = _resize_for_vlm(wide, max_side=1024)
    assert resized.shape == (512, 1024, 3)
    thin = np.zeros((20, 2000, 3), dtype=np.uint8)
    assert _resize_for_vlm(thin, max_side=512).shape == (32, 512, 3)

    # _classify_reference_type strips node "2" / images.image2, downscales to <=512, sets max_len=48
    captured_workflows: list[dict[str, object]] = []
    uploaded_shapes: list[tuple[int, int]] = []

    def _fake_upload(server_url: str, filename: str, png_bytes: bytes) -> str:
        from io import BytesIO

        from PIL import Image

        _ = server_url
        with Image.open(BytesIO(png_bytes)) as im:
            uploaded_shapes.append(im.size)
        return filename

    def _fake_exec(server_url: str, workflow: dict[str, object], case_id: str) -> str:
        _ = (server_url, case_id)
        captured_workflows.append(workflow)
        return '{"reference_type": "logo"}'

    monkeypatch.setattr("src.vlm_inspector.upload_png_bytes", _fake_upload)
    real_cls_inspector = Qwen3VLInspector(ComfyConfig(), inspector_wf)
    monkeypatch.setattr(real_cls_inspector, "_execute_vlm_workflow", _fake_exec)

    big_logo = np.zeros((2400, 2400, 3), dtype=np.uint8)
    cls_res = real_cls_inspector._classify_reference_type("http://127.0.0.1:8000", big_logo, "pac")
    assert cls_res == ReferenceType.LOGO
    assert uploaded_shapes == [(512, 512)]
    wf_sent: Any = captured_workflows[0]
    assert "2" not in wf_sent
    assert "images.image2" not in wf_sent["4"]["inputs"]
    assert wf_sent["4"]["inputs"]["max_length"] == 48
    assert wf_sent["4"]["inputs"]["prompt"] == REFERENCE_CLASSIFICATION_PROMPT

    # Offline/error inside _execute_vlm_workflow must fail fast (not swallow ComfyExecutionError)
    monkeypatch.setattr(
        real_cls_inspector,
        "_execute_vlm_workflow",
        lambda *_a, **_kw: (_ for _ in ()).throw(ComfyExecutionError("timeout")),
    )
    with pytest.raises(ComfyExecutionError, match="timeout"):
        real_cls_inspector._classify_reference_type("http://127.0.0.1:8000", big_logo, "pac")
