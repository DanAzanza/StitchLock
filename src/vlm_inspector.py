"""Stage 1A: Qwen3-VL-8B Automated Garment Defect Inspector & Visual Grounding Engine.

Uses ComfyUI's built-in `TextGenerateV2` node running on the shared `qwen3vl_8b_int8_convrot`
checkpoint (0 MB extra VRAM) to automatically:
  1. Verify whether a garment logo, embroidery, or text on the On-Model image is defective.
  2. Localize the 2D bounding box of the defective logo (`model_bbox_2d`) and the authentic
     reference detail (`ref_bbox_2d`) without manual user coordinates.
  3. Formulate the targeted edit instruction for `Qwen-Image 2.1`.
"""

from __future__ import annotations

import copy
import io
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import websocket
from numpy.typing import NDArray
from PIL import Image

from src.models import ComfyConfig, ComfyExecutionError, DefectTarget, NormalizedBox

MAX_DEFECTS_PER_IMAGE = 4

DEFAULT_INSPECTION_PROMPT = (
    "Compare Image 1 (generated On-Model photo) and Image 2 "
    "(authentic full-garment Product Reference).\n"
    "1. Identify ALL distinct garment defects where a logo, emblem, embroidery, patch, or text "
    "in Image 1 is hallucinated, misspelled, wrong scale, or in the WRONG POSITION compared to "
    "Image 2 (0 if accurate, 1 for an in-place defect, up to 4 separated regions).\n"
    "   - GROUPING RULE: Keep one emblem together with any text/signature directly beneath it "
    "as ONE single defect box. ONLY split into multiple entries in 'defects' when errors sit "
    "on separated parts of the garment.\n"
    "   - MISPLACED LOGO RULE: If a logo/patch on Image 1 sits on the WRONG garment zone compared "
    "to Image 2 (for example, low mid-thigh on Image 1 instead of high near the hip pocket in "
    "Image 2, so the true position in Image 1 is blank fabric), emit TWO SEPARATED defect "
    "entries:\n"
    "     (a) ERASE entry: model_bbox_2d = tight around the misplaced logo in Image 1, "
    "ref_bbox_2d = empty plain fabric at the matching zone in Image 2, edit_prompt = "
    "'Edit <image1>: Remove and erase the misplaced logo completely, leaving clean seamless "
    "plain garment fabric matching <image2>.'\n"
    "     (b) INSERT entry: model_bbox_2d = tight box on the EMPTY garment location in Image 1 "
    "corresponding anatomically to where the detail sits in Image 2 (e.g., upper thigh below "
    "pocket), ref_bbox_2d = tight box around the authentic detail in Image 2, edit_prompt = "
    "'Edit <image1>: Add ONLY the authentic <graphic, colors, count> from Image 2 onto this "
    "spot, matching <image2> in position and size.'\n"
    "2. For each defect entry, provide two INDEPENDENT 2D bounding boxes [xmin, ymin, xmax, ymax] "
    "(scale 0 to 1000):\n"
    "   - model_bbox_2d: TIGHT box enclosing ONLY that defective/target spot in Image 1 "
    "(never touching hands/skin or spanning unrelated intact logos).\n"
    "   - ref_bbox_2d: TIGHT box in Image 2 measured strictly in Image 2's own pixel space.\n"
    "3. For in-place defects, write ONE direct imperative edit_prompt starting with 'Replace':\n"
    "'Edit <image1>: Replace <the defective graphic and any text beneath it> with ONLY "
    "<the exact authentic graphic, emblem count, colors, and text from Image 2>, erasing the old "
    "shape completely and matching <image2> in position and size.'\n"
    "IMPORTANT rules for edit_prompt:\n"
    "- NEVER say 'left chest', 'right chest', 'left leg', or 'right thigh' (viewer vs. wearer "
    "perspective swaps sides); refer directly to the visible graphic or garment pocket/seam.\n"
    "- State exact emblem count and clear details matching <image2> directly.\n"
    "Respond ONLY with valid JSON:\n"
    '{"has_defect": true, "defect_summary": "...", "defects": ['
    '{"defect_summary": "...", "model_bbox_2d": [xmin, ymin, xmax, ymax], '
    '"ref_bbox_2d": [xmin, ymin, xmax, ymax], '
    '"edit_prompt": "Edit <image1>: Replace ... with ..."}]}'
)


@dataclass(frozen=True)
class VLMInspectionResult:
    """Structured multi-defect output from Qwen3-VL-8B garment inspection."""

    has_defect: bool
    defect_summary: str
    model_target_box: NormalizedBox
    reference_detail_box: NormalizedBox
    edit_prompt: str
    inspector_source: str
    defects: tuple[DefectTarget, ...] = ()

    @property
    def resolved_defects(self) -> tuple[DefectTarget, ...]:
        """Return ordered defect targets (falls back cleanly to primary fields)."""
        if not self.has_defect:
            return ()
        if self.defects:
            return self.defects
        return (
            DefectTarget(
                defect_summary=self.defect_summary,
                model_target_box=self.model_target_box,
                reference_detail_box=self.reference_detail_box,
                edit_prompt=self.edit_prompt,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        targets = self.resolved_defects
        return {
            "has_defect": self.has_defect,
            "defect_count": len(targets),
            "defect_summary": self.defect_summary,
            "model_target_box": [
                round(self.model_target_box.x_min, 4),
                round(self.model_target_box.y_min, 4),
                round(self.model_target_box.x_max, 4),
                round(self.model_target_box.y_max, 4),
            ],
            "reference_detail_box": [
                round(self.reference_detail_box.x_min, 4),
                round(self.reference_detail_box.y_min, 4),
                round(self.reference_detail_box.x_max, 4),
                round(self.reference_detail_box.y_max, 4),
            ],
            "edit_prompt": self.edit_prompt,
            "defects": [item.to_dict() for item in targets],
            "inspector_source": self.inspector_source,
        }


class Qwen3VLInspector:
    """Executes zero-shot garment defect detection and localization via Qwen3-VL-8B in ComfyUI."""

    def __init__(self, comfy_config: ComfyConfig, inspector_workflow_path: Path) -> None:
        self._config = comfy_config
        self._workflow_path = inspector_workflow_path
        with inspector_workflow_path.open("r", encoding="utf-8") as handle:
            self._workflow_template: dict[str, Any] = json.load(handle)

    def inspect_garment_pair(
        self,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        case_id: str,
        server_url: str | None = None,
    ) -> VLMInspectionResult:
        """Run live Qwen3-VL-8B inspection on ComfyUI; fail fast if ComfyUI is offline."""
        if server_url is None:
            raise ComfyExecutionError(
                f"Cannot run Qwen3-VL inspection for case '{case_id}': ComfyUI server is offline. "
                f"Root cause: No active ComfyUI server responded at {self._config.server_url} "
                f"or fallback ports {list(self._config.fallback_ports)}. "
                "Recovery: Start the ComfyUI server and retry."
            )

        raw_text = self._run_comfy_vlm(server_url, model_rgb, ref_rgb, case_id)
        parsed = self.parse_vlm_json_response(
            raw_text=raw_text,
            model_rgb=model_rgb,
            ref_rgb=ref_rgb,
            source_label=f"qwen3_vl_8b ({server_url})",
        )
        if not parsed.has_defect:
            return parsed

        refined_defects: list[DefectTarget] = []
        for idx, defect in enumerate(parsed.resolved_defects, start=1):
            ref_box = defect.reference_detail_box
            if not self._is_erase_defect(defect.edit_prompt) and not self._has_graphic_contrast(
                ref_rgb, ref_box
            ):
                located = self._locate_ref_detail_single_image(
                    server_url=server_url,
                    ref_rgb=ref_rgb,
                    target_description=defect.edit_prompt,
                    case_id=f"{case_id}_d{idx}",
                )
                if located is not None:
                    ref_box = located
            refined_defects.append(
                DefectTarget(
                    defect_summary=defect.defect_summary,
                    model_target_box=defect.model_target_box,
                    reference_detail_box=ref_box,
                    edit_prompt=defect.edit_prompt,
                )
            )

        primary = refined_defects[0]
        return VLMInspectionResult(
            has_defect=True,
            defect_summary=parsed.defect_summary,
            model_target_box=primary.model_target_box,
            reference_detail_box=primary.reference_detail_box,
            edit_prompt=primary.edit_prompt,
            inspector_source=parsed.inspector_source,
            defects=tuple(refined_defects),
        )

    def _run_comfy_vlm(
        self,
        server_url: str,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        case_id: str,
    ) -> str:
        """Upload full-res pair and execute TextGenerateV2 on Qwen3-VL-8B."""
        run_token = uuid.uuid4().hex[:8]
        try:
            m_name = self._upload_rgb(server_url, f"vlm_model_{case_id}_{run_token}.png", model_rgb)
            r_name = self._upload_rgb(server_url, f"vlm_ref_{case_id}_{run_token}.png", ref_rgb)
        except Exception as exc:
            raise ComfyExecutionError(
                f"Failed to upload images for Qwen3-VL inspection on '{case_id}': {exc}"
            ) from exc

        workflow = copy.deepcopy(self._workflow_template)
        workflow["1"]["inputs"]["image"] = m_name
        workflow["2"]["inputs"]["image"] = r_name
        workflow["4"]["inputs"]["prompt"] = DEFAULT_INSPECTION_PROMPT
        return self._execute_vlm_workflow(server_url, workflow, case_id)

    def _locate_ref_detail_single_image(
        self,
        server_url: str,
        ref_rgb: NDArray[np.uint8],
        target_description: str,
        case_id: str,
    ) -> NormalizedBox | None:
        """Locate graphic inside Image 2 solo when dual-view ref_box points at plain fabric."""
        run_token = uuid.uuid4().hex[:8]
        try:
            r_name = self._upload_rgb(
                server_url, f"vlm_ref_solo_{case_id}_{run_token}.png", ref_rgb
            )
        except Exception:
            return None

        workflow = copy.deepcopy(self._workflow_template)
        workflow.pop("2", None)
        workflow["1"]["inputs"]["image"] = r_name
        inputs_4 = workflow["4"]["inputs"]
        inputs_4.pop("images.image2", None)
        inputs_4["prompt"] = (
            f"Look at this garment photo and locate ONLY the target graphic described here: "
            f"{target_description}\n"
            "Return the tight 2D bounding box [xmin, ymin, xmax, ymax] on a 0 to 1000 scale "
            "enclosing that graphic on this image.\n"
            'Respond ONLY with JSON: {"bbox_2d": [xmin, ymin, xmax, ymax]}'
        )
        try:
            raw = self._execute_vlm_workflow(server_url, workflow, f"{case_id}_ref_solo")
            match = re.search(r"\{.*\}", raw.strip(), flags=re.DOTALL)
            if not match:
                return None
            data: dict[str, Any] = json.loads(match.group(0))
            coords = data.get("bbox_2d") or data.get("ref_bbox_2d") or data.get("model_bbox_2d")
            candidate = self._parse_box_1000(coords, None)
            if self._has_graphic_contrast(ref_rgb, candidate):
                return self._tighten_ref_box_to_salient(ref_rgb, candidate)
        except Exception:
            return None
        return None

    def _execute_vlm_workflow(self, server_url: str, workflow: dict[str, Any], case_id: str) -> str:
        """Execute a prepared TextGenerateV2 workflow graph against ComfyUI WebSocket/HTTP."""
        client_id = str(uuid.uuid4())
        parsed = urlparse(server_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{parsed.netloc}/ws?clientId={client_id}"

        ws = websocket.WebSocket()
        ws.settimeout(5.0)
        try:
            ws.connect(ws_url)
        except Exception as exc:
            raise ComfyExecutionError(f"Failed to connect VLM WebSocket: {exc}") from exc

        try:
            resp = requests.post(
                f"{server_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=15.0,
            )
            resp.raise_for_status()
            prompt_id = str(resp.json()["prompt_id"])
            self._wait_ws(ws, prompt_id, self._config.timeout_seconds)
            hist = requests.get(f"{server_url}/history/{prompt_id}", timeout=15.0).json()
        except ComfyExecutionError:
            raise
        except Exception as exc:
            raise ComfyExecutionError(
                f"Qwen3-VL ComfyUI execution failed for case '{case_id}': {exc}"
            ) from exc
        finally:
            ws.close()

        outputs = hist.get(prompt_id, {}).get("outputs", {})
        for node_out in outputs.values():
            if "text" in node_out and node_out["text"]:
                return str(node_out["text"][0])

        raise ComfyExecutionError(f"No text output from Qwen3-VL for prompt_id={prompt_id}")

    @staticmethod
    def _has_graphic_contrast(image_rgb: NDArray[np.uint8], box: NormalizedBox) -> bool:
        """Return True if the localized crop contains visible graphic contrast vs. plain cloth."""
        h, w = image_rgb.shape[:2]
        px = box.to_pixel_bbox(w, h)
        patch = image_rgb[px.y1 : px.y2, px.x1 : px.x2].astype(np.float32)
        return float(np.std(patch)) >= 14.0

    @staticmethod
    def _tighten_ref_box_to_salient(
        ref_rgb: NDArray[np.uint8], ref_box: NormalizedBox
    ) -> NormalizedBox:
        """Trim empty cloth padding around ref_box color-agnostically via border median."""
        h, w = ref_rgb.shape[:2]
        px = ref_box.to_pixel_bbox(w, h)
        sx1 = max(0, px.x1 - int(0.03 * w))
        sx2 = min(w, px.x2 + int(0.03 * w))
        sy1 = max(0, px.y1 - int(0.10 * h))
        sy2 = min(h, px.y2 + int(0.04 * h))
        patch = ref_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        if patch.shape[0] < 6 or patch.shape[1] < 6:
            return ref_box
        border_px = np.concatenate(
            [patch[0:2, :, :].reshape(-1, 3), patch[-2:, :, :].reshape(-1, 3)], axis=0
        )
        cloth_med = np.median(border_px, axis=0)
        diff = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
        ys, xs = np.where(diff > 32.0)
        if xs.size < 40 or ys.size < 40:
            return ref_box
        nx1 = max(0.0, (sx1 + int(xs.min()) - 4) / float(w))
        ny1 = max(0.0, (sy1 + int(ys.min()) - 4) / float(h))
        nx2 = min(1.0, (sx1 + int(xs.max()) + 4) / float(w))
        ny2 = min(1.0, (sy1 + int(ys.max()) + 4) / float(h))
        if nx2 - nx1 < 0.02 or ny2 - ny1 < 0.02:
            return ref_box
        return NormalizedBox(nx1, ny1, nx2, ny2)

    @staticmethod
    def _upload_rgb(server_url: str, filename: str, rgb: NDArray[np.uint8]) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        resp = requests.post(
            f"{server_url}/upload/image",
            files={"image": (filename, buf.getvalue(), "image/png")},
            data={"overwrite": "true"},
            timeout=15.0,
        )
        resp.raise_for_status()
        return str(resp.json()["name"])

    @staticmethod
    def _wait_ws(ws: websocket.WebSocket, prompt_id: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not isinstance(msg, str):
                continue
            ev = json.loads(msg)
            d = ev.get("data", {})
            if d.get("prompt_id") != prompt_id:
                continue
            if ev.get("type") == "execution_error":
                raise ComfyExecutionError(str(d.get("exception_message", "VLM error")))
            if ev.get("type") == "executing" and d.get("node") is None:
                return
        raise ComfyExecutionError("Qwen3-VL execution timed out.")

    @classmethod
    def parse_vlm_json_response(
        cls,
        raw_text: str,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        fallback_model_box: NormalizedBox | None = None,
        fallback_ref_box: NormalizedBox | None = None,
        fallback_prompt: str | None = None,
        source_label: str = "qwen3_vl_8b",
    ) -> VLMInspectionResult:
        """Parse multi-defect or single-defect Qwen3-VL JSON and refine target boxes."""
        cleaned = raw_text.strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ComfyExecutionError(f"Qwen3-VL output did not contain JSON: {raw_text!r}")

        data: dict[str, Any] = json.loads(match.group(0))
        has_defect = bool(data.get("has_defect", True))
        defect_summary = str(data.get("defect_summary", "Garment logo/detail mismatch detected."))

        raw_entries = cls._extract_raw_defect_entries(data)
        if not has_defect or not raw_entries:
            empty_box = fallback_model_box or NormalizedBox(0.30, 0.30, 0.70, 0.70)
            empty_ref = fallback_ref_box or NormalizedBox(0.30, 0.30, 0.70, 0.70)
            return VLMInspectionResult(
                has_defect=False,
                defect_summary=defect_summary,
                model_target_box=empty_box,
                reference_detail_box=empty_ref,
                edit_prompt=fallback_prompt or "",
                inspector_source=source_label,
                defects=(),
            )

        parsed_defects = [
            cls._parse_single_defect(
                entry=entry,
                default_summary=defect_summary,
                model_rgb=model_rgb,
                ref_rgb=ref_rgb,
                fallback_model_box=fallback_model_box if idx == 0 else None,
                fallback_ref_box=fallback_ref_box if idx == 0 else None,
                fallback_prompt=fallback_prompt if idx == 0 else None,
            )
            for idx, entry in enumerate(raw_entries[: MAX_DEFECTS_PER_IMAGE * 2])
        ]
        deduped = cls._merge_overlapping_defects(parsed_defects)
        ordered = sorted(
            deduped[:MAX_DEFECTS_PER_IMAGE],
            key=lambda d: (
                0 if cls._is_erase_defect(d.edit_prompt) else 1,
                d.model_target_box.y_min,
                d.model_target_box.x_min,
            ),
        )
        primary = ordered[0]
        combined_summary = (
            defect_summary if len(ordered) == 1 else "; ".join(d.defect_summary for d in ordered)
        )
        return VLMInspectionResult(
            has_defect=True,
            defect_summary=combined_summary,
            model_target_box=primary.model_target_box,
            reference_detail_box=primary.reference_detail_box,
            edit_prompt=primary.edit_prompt,
            inspector_source=source_label,
            defects=tuple(ordered),
        )

    @staticmethod
    def _extract_raw_defect_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize either `defects` list schema or legacy top-level defect fields."""
        raw_list = data.get("defects")
        if isinstance(raw_list, list):
            return [item for item in raw_list if isinstance(item, dict)]
        if data.get("has_defect", True):
            return [data]
        return []

    @classmethod
    def _parse_single_defect(
        cls,
        *,
        entry: dict[str, Any],
        default_summary: str,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        fallback_model_box: NormalizedBox | None,
        fallback_ref_box: NormalizedBox | None,
        fallback_prompt: str | None,
    ) -> DefectTarget:
        raw_m_box = entry.get("model_bbox_2d") or entry.get("model_bbox_1000")
        raw_r_box = entry.get("ref_bbox_2d") or entry.get("ref_bbox_1000")

        m_box = cls._parse_box_1000(raw_m_box, fallback_model_box)
        r_box = cls._parse_box_1000(raw_r_box, fallback_ref_box)

        m_box = cls._refine_connected_graphic_box(model_rgb, m_box, fallback_model_box)
        if fallback_ref_box is not None:
            r_box = fallback_ref_box
        elif cls._has_graphic_contrast(ref_rgb, r_box):
            r_box = cls._tighten_ref_box_to_salient(ref_rgb, r_box)

        summary = str(entry.get("defect_summary") or default_summary).strip()
        prompt = cls._normalize_edit_prompt(str(entry.get("edit_prompt", "")), fallback_prompt)
        return DefectTarget(
            defect_summary=summary,
            model_target_box=m_box,
            reference_detail_box=r_box,
            edit_prompt=prompt,
        )

    @staticmethod
    def _is_erase_defect(edit_prompt: str) -> bool:
        """Return True when the defect only removes/erases a misplaced graphic."""
        lower = edit_prompt.lower()
        has_erase = any(tok in lower for tok in ("remove", "erase", "delete"))
        has_replace_or_add = any(tok in lower for tok in ("replace", "add ", "insert"))
        return has_erase and not has_replace_or_add

    @staticmethod
    def _normalize_edit_prompt(raw_prompt: str, fallback_prompt: str | None) -> str:
        """Enforce unambiguous `Edit <image1>: ... <image2>` instruction format."""
        cleaned = raw_prompt.strip()
        if not cleaned:
            return fallback_prompt or (
                "Edit <image1>: Replace the hallucinated graphic with the exact "
                "authentic logo from <image2>, matching <image2> in size and position."
            )
        body = (
            cleaned[len("Edit <image1>:") :].strip()
            if cleaned.startswith("Edit <image1>:")
            else cleaned
        )
        action_verbs = (
            "replace",
            "change",
            "fix",
            "restore",
            "correct",
            "swap",
            "add",
            "remove",
            "erase",
            "delete",
            "insert",
        )
        if not any(v in body.lower() for v in action_verbs):
            body = f"Replace the defective graphic in <image1> with {body}"
        body = re.sub(
            r"\bon the (?:viewer's |wearer's )?(?:left|right) (?:chest|thigh|leg|side)\b",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = re.sub(r"\s{2,}", " ", body).strip()
        formatted = f"Edit <image1>: {body}"
        if "<image2>" not in formatted:
            formatted = f"{formatted.rstrip('. ')}, matching <image2> exactly."
        return formatted

    @classmethod
    def _merge_overlapping_defects(cls, defects: list[DefectTarget]) -> list[DefectTarget]:
        """Merge strongly overlapping model defect boxes to prevent duplicate edits."""
        merged: list[DefectTarget] = []
        for candidate in defects:
            overlap_idx = next(
                (
                    i
                    for i, existing in enumerate(merged)
                    if cls._boxes_overlap_strongly(
                        existing.model_target_box, candidate.model_target_box
                    )
                ),
                None,
            )
            if overlap_idx is None:
                merged.append(candidate)
                continue
            merged[overlap_idx] = cls._combine_defect_pair(merged[overlap_idx], candidate)
        return merged

    @staticmethod
    def _boxes_overlap_strongly(a: NormalizedBox, b: NormalizedBox) -> bool:
        ix1 = max(a.x_min, b.x_min)
        iy1 = max(a.y_min, b.y_min)
        ix2 = min(a.x_max, b.x_max)
        iy2 = min(a.y_max, b.y_max)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1e-9, (a.x_max - a.x_min) * (a.y_max - a.y_min))
        area_b = max(1e-9, (b.x_max - b.x_min) * (b.y_max - b.y_min))
        iou = inter / (area_a + area_b - inter)
        containment = inter / min(area_a, area_b)
        return iou >= 0.35 or containment >= 0.65

    @staticmethod
    def _combine_defect_pair(first: DefectTarget, second: DefectTarget) -> DefectTarget:
        m_union = NormalizedBox(
            min(first.model_target_box.x_min, second.model_target_box.x_min),
            min(first.model_target_box.y_min, second.model_target_box.y_min),
            max(first.model_target_box.x_max, second.model_target_box.x_max),
            max(first.model_target_box.y_max, second.model_target_box.y_max),
        )
        r_union = NormalizedBox(
            min(first.reference_detail_box.x_min, second.reference_detail_box.x_min),
            min(first.reference_detail_box.y_min, second.reference_detail_box.y_min),
            max(first.reference_detail_box.x_max, second.reference_detail_box.x_max),
            max(first.reference_detail_box.y_max, second.reference_detail_box.y_max),
        )
        primary_prompt = (
            first.edit_prompt
            if len(first.edit_prompt) >= len(second.edit_prompt)
            else second.edit_prompt
        )
        combined_summary = (
            first.defect_summary
            if second.defect_summary in first.defect_summary
            else f"{first.defect_summary}; {second.defect_summary}"
        )
        return DefectTarget(
            defect_summary=combined_summary,
            model_target_box=m_union,
            reference_detail_box=r_union,
            edit_prompt=primary_prompt,
        )

    @staticmethod
    def _parse_box_1000(coords: Any, fallback: NormalizedBox | None) -> NormalizedBox:
        if isinstance(coords, list) and len(coords) == 4:
            vals = [float(v) for v in coords]
            scale = 1000.0 if max(vals) > 1.0 else 1.0
            x1, y1, x2, y2 = vals[0] / scale, vals[1] / scale, vals[2] / scale, vals[3] / scale
            xmin, xmax = max(0.0, min(x1, x2)), min(1.0, max(x1, x2))
            ymin, ymax = max(0.0, min(y1, y2)), min(1.0, max(y1, y2))
            if xmax - xmin >= 0.02 and ymax - ymin >= 0.02:
                return NormalizedBox(xmin, ymin, xmax, ymax)
        if fallback is not None:
            return fallback
        return NormalizedBox(0.30, 0.30, 0.70, 0.70)

    @classmethod
    def _refine_connected_graphic_box(
        cls,
        image_rgb: NDArray[np.uint8],
        vlm_box: NormalizedBox,
        prior_box: NormalizedBox | None,
    ) -> NormalizedBox:
        """Apply a clean color-agnostic safety margin around Qwen3-VL's target bounding box."""
        _ = image_rgb
        if prior_box is not None:
            cx = 0.5 * (vlm_box.x_min + vlm_box.x_max)
            cy = 0.5 * (vlm_box.y_min + vlm_box.y_max)
            if (
                prior_box.x_min - 0.12 <= cx <= prior_box.x_max + 0.12
                and prior_box.y_min - 0.12 <= cy <= prior_box.y_max + 0.12
            ):
                return prior_box

        return NormalizedBox(
            max(0.0, vlm_box.x_min - 0.025),
            max(0.0, vlm_box.y_min - 0.025),
            min(1.0, vlm_box.x_max + 0.025),
            min(1.0, vlm_box.y_max + 0.025),
        )
