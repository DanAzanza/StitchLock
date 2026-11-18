"""Stage 2: ComfyUI Qwen-Image 2.1 Headless API Bridge & Shared Transport Helpers."""

from __future__ import annotations

import copy
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import websocket
from numpy.typing import NDArray
from PIL import Image

from src.comfy_launcher import ensure_comfyui_server, probe_candidate_servers
from src.models import ComfyConfig, ComfyExecutionError

_TRANSPORT_ERRORS = (
    requests.RequestException,
    websocket.WebSocketException,
    json.JSONDecodeError,
    KeyError,
    ValueError,
    OSError,
)


def encode_png_rgb(image_rgb: NDArray[np.uint8]) -> bytes:
    """Encode a 3-channel uint8 RGB image array as lossless PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def encode_png_greyscale(mask_grey: NDArray[np.uint8]) -> bytes:
    """Encode a single-channel 2D uint8 mask array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(mask_grey, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def build_ws_url(server_url: str, client_id: str) -> str:
    """Build the ComfyUI WebSocket endpoint URL for a given client ID."""
    parsed = urlparse(server_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/ws?clientId={client_id}"


def upload_png_bytes(server_url: str, filename: str, png_bytes: bytes) -> str:
    """Upload PNG bytes to ComfyUI /upload/image and return the server-assigned filename."""
    try:
        resp = requests.post(
            f"{server_url}/upload/image",
            files={"image": (filename, png_bytes, "image/png")},
            data={"overwrite": "true"},
            timeout=15.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        return str(payload["name"])
    except _TRANSPORT_ERRORS as exc:
        raise ComfyExecutionError(f"Failed to upload '{filename}' to ComfyUI: {exc}") from exc


def queue_prompt(server_url: str, workflow: dict[str, Any], client_id: str) -> str:
    """Submit a prepared API workflow graph to ComfyUI /prompt and return prompt_id."""
    try:
        resp = requests.post(
            f"{server_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=15.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        return str(payload["prompt_id"])
    except _TRANSPORT_ERRORS as exc:
        raise ComfyExecutionError(f"Failed to queue prompt on ComfyUI: {exc}") from exc


def wait_for_prompt_completion(ws: websocket.WebSocket, prompt_id: str, timeout_s: float) -> None:
    """Wait on an open ComfyUI WebSocket until prompt_id finishes or raises an error."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw_msg = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except _TRANSPORT_ERRORS as exc:
            raise ComfyExecutionError(f"WebSocket error during execution: {exc}") from exc

        if not isinstance(raw_msg, str):
            continue

        try:
            event = json.loads(raw_msg)
        except json.JSONDecodeError:
            continue
        msg_type = event.get("type")
        data = event.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue

        if msg_type == "execution_error":
            err_msg = data.get("exception_message", "Unknown ComfyUI execution error")
            raise ComfyExecutionError(f"ComfyUI node execution failed: {err_msg}")

        if msg_type == "executing" and data.get("node") is None:
            return

    raise ComfyExecutionError(
        f"ComfyUI execution timed out after {timeout_s:.1f}s for prompt_id={prompt_id}."
    )


def fetch_prompt_history(server_url: str, prompt_id: str) -> dict[str, Any]:
    """Fetch ComfyUI /history/<prompt_id> outputs mapping."""
    try:
        hist_resp = requests.get(f"{server_url}/history/{prompt_id}", timeout=15.0)
        hist_resp.raise_for_status()
        history: dict[str, Any] = hist_resp.json().get(prompt_id, {})
        outputs: dict[str, Any] = history.get("outputs", {})
        return outputs
    except _TRANSPORT_ERRORS as exc:
        raise ComfyExecutionError(
            f"Failed to retrieve execution history for prompt_id={prompt_id}: {exc}"
        ) from exc


def execute_comfy_prompt(
    server_url: str,
    workflow: dict[str, Any],
    timeout_s: float,
    case_id: str,
) -> dict[str, Any]:
    """Subscribe to ComfyUI WebSocket, submit workflow, wait for completion, and return outputs."""
    client_id = str(uuid.uuid4())
    ws_url = build_ws_url(server_url, client_id)

    ws = websocket.WebSocket()
    ws.settimeout(5.0)
    try:
        ws.connect(ws_url)
    except _TRANSPORT_ERRORS as exc:
        raise ComfyExecutionError(f"Failed to connect WebSocket to {ws_url}: {exc}") from exc

    try:
        prompt_id = queue_prompt(server_url, workflow, client_id)
        wait_for_prompt_completion(ws, prompt_id, timeout_s)
    except ComfyExecutionError:
        raise
    except _TRANSPORT_ERRORS as exc:
        raise ComfyExecutionError(f"ComfyUI execution failed for case '{case_id}': {exc}") from exc
    finally:
        ws.close()

    return fetch_prompt_history(server_url, prompt_id)


class ComfyQwenBridge:
    """Headless ComfyUI API client for Qwen-Image 2.1 Edit."""

    def __init__(self, config: ComfyConfig, workflow_path: Path) -> None:
        self._config = config
        self._workflow_path = workflow_path
        self._workflow_template = self._load_workflow_template(workflow_path)

    @staticmethod
    def _load_workflow_template(workflow_path: Path) -> dict[str, Any]:
        """Load and validate the ComfyUI API JSON graph."""
        if not workflow_path.is_file():
            raise ComfyExecutionError(f"ComfyUI workflow file not found: {workflow_path}")
        with workflow_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not data:
            raise ComfyExecutionError(
                f"Invalid ComfyUI API workflow in {workflow_path}: expected non-empty node dict."
            )
        return data

    def probe_live_server(self) -> str | None:
        """Public wrapper probing configured ComfyUI ports (auto-launching if enabled)."""
        return self._probe_live_server()

    def _probe_live_server(self) -> str | None:
        """Probe configured ComfyUI ports and auto-launch backend when configured."""
        active = probe_candidate_servers(self._config)
        if active is not None:
            return active
        if self._config.auto_launch:
            return ensure_comfyui_server(self._config)
        return None

    def restore_detail_roi(
        self,
        cropped_model_roi: NDArray[np.uint8],
        cropped_product_ref: NDArray[np.uint8],
        cropped_mask_greyscale: NDArray[np.uint8],
        prompt: str,
        case_id: str,
    ) -> tuple[NDArray[np.uint8], str]:
        """Restore the garment detail ROI via live ComfyUI Qwen-Image 2.1 Edit.

        Fails fast with ComfyExecutionError if the ComfyUI server is offline or fails.
        """
        self._validate_inputs(cropped_model_roi, cropped_product_ref, cropped_mask_greyscale)

        active_server = self._probe_live_server()
        if active_server is None:
            raise ComfyExecutionError(
                f"ComfyUI server is unreachable for case '{case_id}'. "
                f"Root cause: No active ComfyUI instance responded at {self._config.server_url} "
                f"or fallback ports {list(self._config.fallback_ports)}. "
                "Recovery: Start your local ComfyUI server (e.g. on port 8000 or 8188) with "
                "Qwen3-VL and Qwen-Image 2.1 loaded and retry."
            )

        restored = self._execute_comfy_workflow(
            server_url=active_server,
            cropped_model_roi=cropped_model_roi,
            cropped_product_ref=cropped_product_ref,
            cropped_mask_greyscale=cropped_mask_greyscale,
            prompt=prompt,
            case_id=case_id,
        )
        return restored, f"comfyui_qwen_2_1 ({active_server})"

    @staticmethod
    def _validate_inputs(
        model_roi: NDArray[np.uint8],
        product_ref: NDArray[np.uint8],
        mask_grey: NDArray[np.uint8],
    ) -> None:
        if model_roi.ndim != 3 or model_roi.shape[2] != 3:
            raise ComfyExecutionError(f"Expected RGB model_roi, got shape {model_roi.shape}.")
        if product_ref.ndim != 3 or product_ref.shape[2] != 3:
            raise ComfyExecutionError(f"Expected RGB product_ref, got shape {product_ref.shape}.")
        if mask_grey.ndim != 2:
            raise ComfyExecutionError(
                "Expected strict 2D single-channel greyscale mask (no alpha), "
                f"got {mask_grey.shape}."
            )
        if model_roi.shape[:2] != mask_grey.shape[:2]:
            raise ComfyExecutionError(
                f"Model ROI {model_roi.shape[:2]} and greyscale mask {mask_grey.shape[:2]} "
                "dimensions must match."
            )

    def build_workflow_payload(
        self,
        uploaded_model_name: str,
        uploaded_ref_name: str,
        uploaded_mask_name: str,
        prompt: str,
        output_prefix: str,
    ) -> dict[str, Any]:
        """Inject uploaded asset filenames, prompt, and sampler parameters into workflow JSON."""
        workflow = copy.deepcopy(self._workflow_template)
        replacements = {
            "{{CROPPED_MODEL_ROI}}": uploaded_model_name,
            "{{CROPPED_PRODUCT_REF}}": uploaded_ref_name,
            "{{CROPPED_MASK_GREYSCALE}}": uploaded_mask_name,
            "{{PROMPT}}": prompt,
            "{{OUTPUT_PREFIX}}": output_prefix,
        }

        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for key, val in list(inputs.items()):
                if isinstance(val, str) and val in replacements:
                    inputs[key] = replacements[val]

            if node.get("class_type") == "KSampler":
                inputs["seed"] = self._config.seed
                inputs["steps"] = self._config.steps
                inputs["cfg"] = self._config.cfg
                inputs["denoise"] = self._config.denoise

        return workflow

    def _execute_comfy_workflow(
        self,
        server_url: str,
        cropped_model_roi: NDArray[np.uint8],
        cropped_product_ref: NDArray[np.uint8],
        cropped_mask_greyscale: NDArray[np.uint8],
        prompt: str,
        case_id: str,
    ) -> NDArray[np.uint8]:
        """Execute pure Qwen-Image 2.1 Edit on full On-Model, full Product Ref, and mask."""
        run_token = uuid.uuid4().hex[:8]
        model_server_name = upload_png_bytes(
            server_url, f"roi_{case_id}_{run_token}.png", encode_png_rgb(cropped_model_roi)
        )
        ref_server_name = upload_png_bytes(
            server_url, f"ref_{case_id}_{run_token}.png", encode_png_rgb(cropped_product_ref)
        )
        mask_server_name = upload_png_bytes(
            server_url,
            f"mask_{case_id}_{run_token}.png",
            encode_png_greyscale(cropped_mask_greyscale),
        )

        workflow = self.build_workflow_payload(
            uploaded_model_name=model_server_name,
            uploaded_ref_name=ref_server_name,
            uploaded_mask_name=mask_server_name,
            prompt=prompt,
            output_prefix=f"garment_fix_{case_id}",
        )
        outputs = execute_comfy_prompt(
            server_url=server_url,
            workflow=workflow,
            timeout_s=self._config.timeout_seconds,
            case_id=case_id,
        )
        return self._download_output_image(
            server_url=server_url,
            outputs=outputs,
            target_size=(cropped_model_roi.shape[1], cropped_model_roi.shape[0]),
            case_id=case_id,
        )

    @staticmethod
    def _download_output_image(
        server_url: str,
        outputs: dict[str, Any],
        target_size: tuple[int, int],
        case_id: str,
    ) -> NDArray[np.uint8]:
        try:
            for node_output in outputs.values():
                images = node_output.get("images", [])
                if not images:
                    continue
                img_meta = images[0]
                view_resp = requests.get(
                    f"{server_url}/view",
                    params={
                        "filename": img_meta["filename"],
                        "subfolder": img_meta.get("subfolder", ""),
                        "type": img_meta.get("type", "output"),
                    },
                    timeout=15.0,
                )
                view_resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(view_resp.content)).convert("RGB")
                if pil_img.size != target_size:
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                return np.asarray(pil_img, dtype=np.uint8)
        except _TRANSPORT_ERRORS as exc:
            raise ComfyExecutionError(
                f"Failed to retrieve output image for case '{case_id}': {exc}"
            ) from exc

        raise ComfyExecutionError(f"No output images found in history for case '{case_id}'.")
