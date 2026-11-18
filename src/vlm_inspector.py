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

import cv2
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
    def _sample_studio_bg_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Estimate studio backdrop RGB from upper-left and upper-right image corners."""
        h, w = image_rgb.shape[:2]
        c_w = max(4, w // 20)
        c_h = max(4, h // 20)
        corners = np.concatenate(
            [
                image_rgb[0:c_h, 0:c_w, :].reshape(-1, 3),
                image_rgb[0:c_h, w - c_w : w, :].reshape(-1, 3),
            ],
            axis=0,
        )
        return np.median(corners.astype(np.float32), axis=0)

    @staticmethod
    def _has_graphic_contrast(image_rgb: NDArray[np.uint8], box: NormalizedBox) -> bool:
        """Return True if the localized crop contains visible graphic contrast vs. plain cloth."""
        h, w = image_rgb.shape[:2]
        px = box.to_pixel_bbox(w, h)
        patch = image_rgb[px.y1 : px.y2, px.x1 : px.x2].astype(np.float32)
        return float(np.std(patch)) >= 14.0

    @classmethod
    def _tighten_ref_box_to_salient(
        cls, ref_rgb: NDArray[np.uint8], ref_box: NormalizedBox
    ) -> NormalizedBox:
        """Trim empty cloth and studio backdrop padding around ref_box color-agnostically."""
        h, w = ref_rgb.shape[:2]
        px = ref_box.to_pixel_bbox(w, h)
        sx1 = max(0, px.x1 - int(0.03 * w))
        sx2 = min(w, px.x2 + int(0.03 * w))
        sy1 = max(0, px.y1 - int(0.06 * h))
        sy2 = min(h, px.y2 + int(0.04 * h))
        patch = ref_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        if patch.shape[0] < 6 or patch.shape[1] < 6:
            return ref_box

        bg_med = cls._sample_studio_bg_rgb(ref_rgb)
        diff_bg = np.linalg.norm(patch - bg_med.reshape(1, 1, 3), axis=2)
        fg_pixels = patch[diff_bg > 28.0]
        if fg_pixels.shape[0] >= 40:
            cloth_med = np.median(fg_pixels, axis=0)
            c_rg = float(cloth_med[0] - cloth_med[1])
            c_gb = float(cloth_med[1] - cloth_med[2])
            p_chroma = np.hypot(
                (patch[:, :, 0] - patch[:, :, 1]) - c_rg,
                (patch[:, :, 1] - patch[:, :, 2]) - c_gb,
            )
            diff_cloth = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
            chroma_mask = (p_chroma > 15.0) & (diff_bg > 28.0)
            salient_mask = (
                chroma_mask
                if int(np.count_nonzero(chroma_mask)) >= 40
                else ((diff_cloth > 44.0) & (diff_bg > 28.0))
            )
        else:
            border_px = np.concatenate(
                [patch[0:2, :, :].reshape(-1, 3), patch[-2:, :, :].reshape(-1, 3)], axis=0
            )
            cloth_med = np.median(border_px, axis=0)
            diff = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
            salient_mask = diff > 32.0

        ys, xs = np.where(salient_mask)
        if xs.size < 25 or ys.size < 25:
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
            micro = cls._detect_micro_emblem_defect(model_rgb, ref_rgb, source_label)
            if micro is not None:
                return micro
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
        if len(deduped) == 1 and not cls._is_erase_defect(deduped[0].edit_prompt):
            deduped = cls._split_misplaced_defect_if_shifted(model_rgb, ref_rgb, deduped)
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
        """Merge strongly overlapping model defect boxes unless erase vs. non-erase differ."""
        merged: list[DefectTarget] = []
        for candidate in defects:
            cand_erase = cls._is_erase_defect(candidate.edit_prompt)
            overlap_idx = next(
                (
                    i
                    for i, existing in enumerate(merged)
                    if cls._is_erase_defect(existing.edit_prompt) == cand_erase
                    and cls._boxes_overlap_strongly(
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
        """Expand truncated VLM boxes to enclose connected graphic/signature components."""
        if prior_box is not None:
            cx = 0.5 * (vlm_box.x_min + vlm_box.x_max)
            cy = 0.5 * (vlm_box.y_min + vlm_box.y_max)
            if (
                prior_box.x_min - 0.12 <= cx <= prior_box.x_max + 0.12
                and prior_box.y_min - 0.12 <= cy <= prior_box.y_max + 0.12
            ):
                return prior_box

        padded = NormalizedBox(
            max(0.0, vlm_box.x_min - 0.025),
            max(0.0, vlm_box.y_min - 0.025),
            min(1.0, vlm_box.x_max + 0.025),
            min(1.0, vlm_box.y_max + 0.025),
        )
        h, w = image_rgb.shape[:2]
        px = padded.to_pixel_bbox(w, h)
        inner = image_rgb[px.y1 : px.y2, px.x1 : px.x2].astype(np.float32)
        if inner.shape[0] < 12 or inner.shape[1] < 12:
            return padded

        cloth_med = np.median(inner.reshape(-1, 3), axis=0)
        inner_salient = np.linalg.norm(inner - cloth_med.reshape(1, 1, 3), axis=2) > 45.0
        if int(np.count_nonzero(inner_salient)) < 15:
            return padded
        graphic_med = np.median(inner[inner_salient], axis=0)
        if float(np.linalg.norm(graphic_med - cloth_med)) < 45.0:
            return padded

        sy1, sy2 = px.y1, min(h, px.y2 + int(0.10 * h))
        search = image_rgb[sy1:sy2, px.x1 : px.x2].astype(np.float32)
        d_cloth = np.linalg.norm(search - cloth_med.reshape(1, 1, 3), axis=2)
        d_graphic = np.linalg.norm(search - graphic_med.reshape(1, 1, 3), axis=2)
        salient = ((d_cloth > 45.0) & (d_graphic < d_cloth * 0.85) & (d_graphic < 85.0)).astype(
            np.uint8
        )

        k = max(5, int(round(min(h, w) * 0.022)))
        bridged = cv2.dilate(salient, np.ones((k, k), np.uint8))
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bridged)
        in_h = px.y2 - px.y1
        keep = np.zeros_like(salient)
        for lbl in range(1, num_labels):
            comp = labels == lbl
            if not np.any(comp[:in_h, :]):
                continue
            pts = (salient > 0) & comp
            _, cxs = np.where(pts)
            if cxs.size == 0 or int(cxs.min()) <= 2 or int(cxs.max()) >= search.shape[1] - 3:
                continue
            keep[pts] = 1

        ys, xs = np.where(keep > 0)
        if xs.size < 20:
            return padded
        gy1, gy2 = (sy1 + int(ys.min())) / float(h), (sy1 + int(ys.max())) / float(h)
        if gy2 <= padded.y_max - 0.005:
            return padded
        gx1, gx2 = (px.x1 + int(xs.min())) / float(w), (px.x1 + int(xs.max())) / float(w)
        nx1 = max(0.0, min(padded.x_min, gx1 - 0.025))
        ny1 = min(padded.y_max - 0.02, max(0.0, max(padded.y_min, gy1 - 0.035)))
        nx2 = min(1.0, max(padded.x_max, gx2 + 0.025))
        ny2 = min(1.0, max(padded.y_max, gy2 + 0.025))
        if nx2 - nx1 < 0.02 or ny2 - ny1 < 0.02:
            return padded
        return NormalizedBox(round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4))

    @classmethod
    def _segment_garment_bounds(
        cls, model_rgb: NDArray[np.uint8], ref_rgb: NDArray[np.uint8]
    ) -> (
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.uint8],
        ]
        | None
    ):
        """Segment matching garment silhouette bounds between Product Reference and On-Model."""
        h, w = model_rgb.shape[:2]
        rh, rw = ref_rgb.shape[:2]
        if h < 32 or w < 32 or rh < 32 or rw < 32:
            return None

        rbg = cls._sample_studio_bg_rgb(ref_rgb)
        rm = np.linalg.norm(ref_rgb.astype(np.float32) - rbg.reshape(1, 1, 3), axis=2) > 24.0
        if int(np.count_nonzero(rm)) < 100:
            return None
        ry_idx, rx_idx = np.where(rm.sum(axis=1) >= 20)[0], np.where(rm.sum(axis=0) >= 20)[0]
        if ry_idx.size < 10 or rx_idx.size < 10:
            return None
        ref_bounds = (
            int(rx_idx[0]) / float(rw),
            int(ry_idx[0]) / float(rh),
            int(rx_idx[-1]) / float(rw),
            int(ry_idx[-1]) / float(rh),
        )
        ref_cloth = np.median(ref_rgb[rm].astype(np.float32), axis=0)

        mbg = cls._sample_studio_bg_rgb(model_rgb)
        m_f = model_rgb.astype(np.float32)
        c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
        m_chroma = np.hypot(
            (m_f[:, :, 0] - m_f[:, :, 1]) - c_rg, (m_f[:, :, 1] - m_f[:, :, 2]) - c_gb
        )
        diff_c = np.linalg.norm(m_f - ref_cloth.reshape(1, 1, 3), axis=2)
        diff_mbg = np.linalg.norm(m_f - mbg.reshape(1, 1, 3), axis=2)
        mm_raw = ((diff_mbg > 18.0) & (diff_c < 52.0) & (m_chroma < 14.0)).astype(np.uint8)
        mm = cv2.morphologyEx(mm_raw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        my_idx, mx_idx = np.where(mm.sum(axis=1) >= 30)[0], np.where(mm.sum(axis=0) >= 30)[0]
        if my_idx.size < 10 or mx_idx.size < 10:
            return None
        model_bounds = (
            int(mx_idx[0]) / float(w),
            int(my_idx[0]) / float(h),
            int(mx_idx[-1]) / float(w),
            int(my_idx[-1]) / float(h),
        )
        return ref_bounds, model_bounds, rbg, ref_cloth, np.asarray(mm, dtype=np.uint8)

    @staticmethod
    def _find_salient_ref_bounds(
        ref_rgb: NDArray[np.uint8],
        ref_box: NormalizedBox,
        rbg: NDArray[np.float32],
        ref_cloth: NDArray[np.float32],
    ) -> tuple[float, float, float, float]:
        """Locate tight chromatic/high-contrast graphic bounds inside reference detail crop."""
        rh, rw = ref_rgb.shape[:2]
        rpx = ref_box.to_pixel_bbox(rw, rh)
        rp = ref_rgb[rpx.y1 : rpx.y2, rpx.x1 : rpx.x2].astype(np.float32)
        c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
        r_chroma = np.hypot((rp[:, :, 0] - rp[:, :, 1]) - c_rg, (rp[:, :, 1] - rp[:, :, 2]) - c_gb)
        r_cloth_dist = np.linalg.norm(rp - ref_cloth.reshape(1, 1, 3), axis=2)
        r_bg_dist = np.linalg.norm(rp - rbg.reshape(1, 1, 3), axis=2)
        chroma_sal = (r_chroma > 16.0) & (r_bg_dist > 28.0)
        sal = (
            chroma_sal
            if int(np.count_nonzero(chroma_sal)) >= 40
            else ((r_cloth_dist > 44.0) & (r_bg_dist > 28.0))
        )
        rys, rxs = np.where(sal)
        if rxs.size < 20 or rys.size < 20:
            return ref_box.x_min, ref_box.y_min, ref_box.x_max, ref_box.y_max
        return (
            (rpx.x1 + int(rxs.min())) / float(rw),
            (rpx.y1 + int(rys.min())) / float(rh),
            (rpx.x1 + int(rxs.max())) / float(rw),
            (rpx.y1 + int(rys.max())) / float(rh),
        )

    @classmethod
    def _split_misplaced_defect_if_shifted(
        cls,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        defects: list[DefectTarget],
    ) -> list[DefectTarget]:
        """Split a misplaced lower/upper logo defect into separate ERASE + INSERT targets."""
        defect = defects[0]
        seg = cls._segment_garment_bounds(model_rgb, ref_rgb)
        if seg is None:
            return defects
        ref_bounds, model_bounds, rbg, ref_cloth, mm = seg
        rx1, ry1, rx2, ry2 = ref_bounds
        mx1, my1, mx2, my2 = model_bounds

        m_box, r_box = defect.model_target_box, defect.reference_detail_box
        true_rx1, true_ry1, true_rx2, true_ry2 = cls._find_salient_ref_bounds(
            ref_rgb, r_box, rbg, ref_cloth
        )
        m_cx_raw = 0.5 * (m_box.x_min + m_box.x_max)
        r_cx = 0.5 * (true_rx1 + true_rx2)
        rel_rx = (r_cx - rx1) / max(0.1, rx2 - rx1)
        expected_mcx = mx1 + rel_rx * (mx2 - mx1)

        m_cy = 0.5 * (m_box.y_min + m_box.y_max)
        r_cy = 0.5 * (true_ry1 + true_ry2)
        rel_ry = (r_cy - ry1) / max(0.1, ry2 - ry1)
        expected_mcy = my1 + rel_ry * (my2 - my1)

        y_shifted = abs(m_cy - r_cy) > 0.25 and abs(m_cy - expected_mcy) > 0.04
        x_shifted = abs(m_cx_raw - r_cx) > 0.22 and abs(m_cx_raw - expected_mcx) > 0.12
        if not y_shifted and not x_shifted:
            return defects

        h, w = model_rgb.shape[:2]
        mpx = m_box.to_pixel_bbox(w, h)
        mp = model_rgb[mpx.y1 : mpx.y2, mpx.x1 : mpx.x2].astype(np.float32)
        c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
        m_chroma = np.hypot((mp[:, :, 0] - mp[:, :, 1]) - c_rg, (mp[:, :, 1] - mp[:, :, 2]) - c_gb)
        m_cloth_dist = np.linalg.norm(mp - ref_cloth.reshape(1, 1, 3), axis=2)
        mys, mxs = np.where((m_chroma > 18.0) | (m_cloth_dist > 48.0))
        erase_box = m_box
        if mxs.size >= 40:
            ex1 = max(m_box.x_min + 0.012, (mpx.x1 + int(np.percentile(mxs, 2))) / w - 0.015)
            ey1 = max(m_box.y_min, (mpx.y1 + int(np.percentile(mys, 5))) / h - 0.015)
            ex2 = min(m_box.x_max, (mpx.x1 + int(np.percentile(mxs, 98))) / w + 0.015)
            ey2 = min(m_box.y_max, (mpx.y1 + int(np.percentile(mys, 98))) / h + 0.015)
            if ex2 - ex1 >= 0.03 and ey2 - ey1 >= 0.03:
                erase_box = NormalizedBox(
                    round(ex1, 4), round(ey1, 4), round(ex2, 4), round(ey2, 4)
                )

        ref_erase_box = NormalizedBox(
            round(max(0.0, min(0.88, true_rx1)), 4),
            round(max(0.0, min(0.80, m_box.y_min)), 4),
            round(min(1.0, max(true_rx1 + 0.05, min(0.95, true_rx1 + 0.10))), 4),
            round(min(1.0, max(min(0.80, m_box.y_min) + 0.05, min(0.92, m_box.y_max))), 4),
        )
        erase_target = DefectTarget(
            defect_summary="Erase misplaced graphic from wrong garment zone",
            model_target_box=erase_box,
            reference_detail_box=ref_erase_box,
            edit_prompt=(
                "Edit <image1>: Remove and erase the misplaced logo patch completely, "
                "leaving clean seamless plain garment fabric matching <image2>."
            ),
        )

        ins_w = min(
            0.074, max(0.060, (true_rx2 - true_rx1) * (mx2 - mx1) / max(0.1, rx2 - rx1) + 0.026)
        )
        ins_h = min(
            0.076, max(0.068, (true_ry2 - true_ry1) * (my2 - my1) / max(0.1, ry2 - ry1) + 0.030)
        )
        if not y_shifted:
            target_cy = max(my1 + 0.10 * (my2 - my1), min(my2 - 0.10 * (my2 - my1), expected_mcy))
        elif m_cy > expected_mcy:
            min_allowed_cy = my1 + 0.15 * (my2 - my1)
            clear_above_cy = max(min_allowed_cy, erase_box.y_min - 0.5 * ins_h - 0.004)
            target_cy = min(expected_mcy, clear_above_cy)
        else:
            max_allowed_cy = my2 - 0.15 * (my2 - my1)
            clear_below_cy = min(max_allowed_cy, erase_box.y_max + 0.5 * ins_h + 0.004)
            target_cy = max(expected_mcy, clear_below_cy)

        iy1 = max(0.0, min(1.0 - ins_h, target_cy - 0.5 * ins_h))
        iy2 = min(1.0, iy1 + ins_h)
        if x_shifted:
            ix1 = max(mx1 + 0.02, min(mx2 - ins_w - 0.02, expected_mcx - 0.5 * ins_w))
        else:
            m_cx = 0.5 * (erase_box.x_min + erase_box.x_max)
            ix1 = max(m_box.x_min + 0.025, m_cx - 0.5 * ins_w)
            y_band = mm[int(iy1 * h) : max(int(iy1 * h) + 1, int(iy2 * h)), :]
            cloth_cols = np.where(y_band.mean(axis=0) > 0.5)[0]
            half_mid = 0.5 * (mx1 + mx2) * w
            side_cols = (
                cloth_cols[cloth_cols < half_mid]
                if m_cx * w < half_mid
                else cloth_cols[cloth_cols >= half_mid]
            )
            if side_cols.size >= 10:
                edge_x = int(side_cols[0]) / float(w)
                ix1 = max(ix1, min(int(side_cols[-1]) / float(w) - ins_w, edge_x - 0.025))
        ix1 = max(0.0, min(1.0 - ins_w, ix1))
        ix2 = min(1.0, ix1 + ins_w)

        tight_ref_box = NormalizedBox(
            round(max(0.0, true_rx1 - 0.015), 4),
            round(max(0.0, true_ry1 - 0.015), 4),
            round(min(1.0, true_rx2 + 0.015), 4),
            round(min(1.0, true_ry2 + 0.015), 4),
        )
        insert_target = DefectTarget(
            defect_summary="Insert authentic logo detail at aligned garment zone",
            model_target_box=NormalizedBox(
                round(ix1, 4), round(iy1, 4), round(ix2, 4), round(iy2, 4)
            ),
            reference_detail_box=tight_ref_box,
            edit_prompt=(
                "Edit <image1>: Add ONLY the authentic small detail from Image 2 onto this spot, "
                "matching <image2> in size and design."
            ),
        )
        return [erase_target, insert_target]

    @classmethod
    def _find_salient_emblem_box(
        cls, image_rgb: NDArray[np.uint8], hint_box: NormalizedBox
    ) -> NormalizedBox | None:
        """Locate a compact high-contrast chest emblem inside the garment zone."""
        h, w = image_rgb.shape[:2]
        if h < 32 or w < 32:
            return None
        bg_med = cls._sample_studio_bg_rgb(image_rgb)
        px = hint_box.to_pixel_bbox(w, h)
        sx1, sx2 = max(0, px.x1 - int(0.06 * w)), min(w, px.x2 + int(0.06 * w))
        sy1, sy2 = max(0, px.y1 - int(0.06 * h)), min(h, px.y2 + int(0.06 * h))
        patch = image_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            return None

        fg_mask = np.linalg.norm(patch - bg_med.reshape(1, 1, 3), axis=2) > 28.0
        if int(np.count_nonzero(fg_mask)) < 100:
            return None
        cloth_med = np.median(patch[fg_mask], axis=0)
        diff = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
        thr = max(75.0, float(np.percentile(diff[fg_mask], 98.5)))
        salient = ((diff >= thr) & fg_mask).astype(np.uint8)

        k = max(3, int(round(min(h, w) * 0.005)))
        bridged = cv2.dilate(salient, np.ones((k, k), np.uint8))
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bridged)

        best_box: NormalizedBox | None = None
        best_score = -1.0
        ph, pw = patch.shape[:2]
        max_pts = int(0.12 * ph * pw)
        for lbl in range(1, num_labels):
            pts = (salient > 0) & (labels == lbl)
            count = int(np.count_nonzero(pts))
            if count < 15 or count > max_pts:
                continue
            ys, xs = np.where(pts)
            if (
                int(xs.min()) <= 2
                or int(ys.min()) <= 2
                or int(xs.max()) >= pw - 3
                or int(ys.max()) >= ph - 3
            ):
                continue
            bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if max(bw / max(1.0, float(bh)), bh / max(1.0, float(bw))) > 3.0:
                continue
            score = float(count) * float(diff[pts].mean())
            if score > best_score:
                best_score = score
                nx1, ny1 = (sx1 + int(xs.min())) / float(w), (sy1 + int(ys.min())) / float(h)
                nx2, ny2 = (sx1 + int(xs.max())) / float(w), (sy1 + int(ys.max())) / float(h)
                if nx2 - nx1 >= 0.02 or ny2 - ny1 >= 0.02:
                    best_box = NormalizedBox(
                        round(max(0.0, nx1), 4),
                        round(max(0.0, ny1), 4),
                        round(min(1.0, max(nx1 + 0.02, nx2)), 4),
                        round(min(1.0, max(ny1 + 0.02, ny2)), 4),
                    )
        return best_box

    @staticmethod
    def _emblem_pattern_correlation(
        model_rgb: NDArray[np.uint8],
        m_box: NormalizedBox,
        ref_rgb: NDArray[np.uint8],
        r_box: NormalizedBox,
    ) -> float:
        """Return normalized 64x64 contrast-map correlation between two small emblems."""

        def _norm_patch(img: NDArray[np.uint8], box: NormalizedBox) -> NDArray[np.float32]:
            h, w = img.shape[:2]
            px = box.to_pixel_bbox(w, h)
            pad = 3
            crop = img[
                max(0, px.y1 - pad) : min(h, px.y2 + pad),
                max(0, px.x1 - pad) : min(w, px.x2 + pad),
            ].astype(np.float32)
            cloth = np.median(crop.reshape(-1, 3), axis=0)
            dist = np.linalg.norm(crop - cloth.reshape(1, 1, 3), axis=2)
            norm = (dist - float(dist.min())) / max(1.0, float(dist.max() - dist.min()))
            return np.asarray(
                cv2.resize(norm.astype(np.float32), (64, 64), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )

        pm = _norm_patch(model_rgb, m_box)
        pr = _norm_patch(ref_rgb, r_box)
        if float(np.std(pm)) < 1e-4 or float(np.std(pr)) < 1e-4:
            return 1.0
        return float(np.corrcoef(pm.ravel(), pr.ravel())[0, 1])

    @classmethod
    def _detect_micro_emblem_defect(
        cls,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        source_label: str,
    ) -> VLMInspectionResult | None:
        """Fallback localizer catching small emblem mismatches across general garment zones."""
        candidate_zones = (
            NormalizedBox(0.50, 0.26, 0.68, 0.46),  # right chest (viewer perspective)
            NormalizedBox(0.32, 0.26, 0.50, 0.46),  # left chest (viewer perspective)
            NormalizedBox(0.41, 0.22, 0.59, 0.44),  # center chest / upper back
            NormalizedBox(0.28, 0.22, 0.48, 0.44),  # upper left hip / thigh
            NormalizedBox(0.52, 0.22, 0.72, 0.44),  # upper right hip / thigh
        )
        best_match: tuple[float, NormalizedBox, NormalizedBox] | None = None
        for zone in candidate_zones:
            m_emblem = cls._find_salient_emblem_box(model_rgb, zone)
            r_emblem = cls._find_salient_emblem_box(ref_rgb, zone)
            if m_emblem is None or r_emblem is None:
                continue
            corr = cls._emblem_pattern_correlation(model_rgb, m_emblem, ref_rgb, r_emblem)
            if corr < 0.75:
                best_match = (corr, m_emblem, r_emblem)
                break

        if best_match is None:
            return None
        corr, m_emblem, r_emblem = best_match

        m_w = m_emblem.x_max - m_emblem.x_min
        m_h = m_emblem.y_max - m_emblem.y_min
        r_w = r_emblem.x_max - r_emblem.x_min
        r_h = r_emblem.y_max - r_emblem.y_min
        box_w = max(m_w, r_w * 0.78) + 0.020
        box_h = max(m_h, r_h * 0.78) + 0.020
        m_cx = 0.5 * (m_emblem.x_min + m_emblem.x_max)
        m_cy = 0.5 * (m_emblem.y_min + m_emblem.y_max)

        model_box = NormalizedBox(
            round(max(0.0, m_cx - 0.5 * box_w), 4),
            round(max(0.0, m_cy - 0.5 * box_h), 4),
            round(min(1.0, m_cx + 0.5 * box_w), 4),
            round(min(1.0, m_cy + 0.5 * box_h), 4),
        )
        ref_box = NormalizedBox(
            round(max(0.0, r_emblem.x_min - 0.012), 4),
            round(max(0.0, r_emblem.y_min - 0.012), 4),
            round(min(1.0, r_emblem.x_max + 0.012), 4),
            round(min(1.0, r_emblem.y_max + 0.012), 4),
        )
        prompt = (
            "Edit <image1>: Replace the defective chest logo with ONLY the exact authentic logo "
            "from <image2>, restoring all leaves, internal cutout stripes, and text to match "
            "<image2> in position and size."
        )
        defect = DefectTarget(
            defect_summary=f"Small chest emblem structural mismatch detected (corr={corr:.2f})",
            model_target_box=model_box,
            reference_detail_box=ref_box,
            edit_prompt=prompt,
        )
        return VLMInspectionResult(
            has_defect=True,
            defect_summary=defect.defect_summary,
            model_target_box=model_box,
            reference_detail_box=ref_box,
            edit_prompt=prompt,
            inspector_source=source_label,
            defects=(defect,),
        )
