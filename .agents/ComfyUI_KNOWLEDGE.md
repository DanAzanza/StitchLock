# ComfyUI_KNOWLEDGE.md — ComfyUI & Qwen-Image 2.1 Platform Memory

## 1. Local Hardware & Server Environment
- **GPU Hardware**: CUDA 0 — NVIDIA GeForce RTX 4060 Ti (16 GB VRAM).
- **Server Endpoints**:
  - Standalone ComfyUI default: `http://127.0.0.1:8188`
  - Comfy Desktop headless default: `http://127.0.0.1:8000`
  - `ComfyQwenBridge` automatically probes `server_url` and `fallback_ports: [8188, 8000]`.
- **Shared Model Storage (`C:\Users\danie\ComfyUI-Shared\models`)**:
  - Diffusion Transformer (UNET): `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
  - Text/Vision Encoder (CLIP): `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (`type: "qwen_image"`)
  - VAE: `vae/qwen_image_2.1_vae_bf16.safetensors`

---

## 2. ComfyUI Headless API Invariants & Traps
- **Prompt API Graph Schema vs. UI Canvas JSON**:
  - `POST /prompt` requires a dictionary mapping node ID strings (`"1"`, `"2"`, ...) to `{"class_type": ..., "inputs": ...}`. Passing the UI editor format (`nodes` / `links` arrays) fails immediately.
- **Asset Upload & Server Filename Resolution (`POST /upload/image`)**:
  - `LoadImage` only resolves files inside ComfyUI's `input/` directory.
  - Always parse `response.json()["name"]` from the upload response and inject that exact string into the workflow graph rather than assuming the local filename was preserved.
- **WebSocket Subscribe-Before-Dispatch Protocol**:
  - ComfyUI does **not** buffer WebSocket messages for disconnected `clientId`s. If a workflow's upstream nodes are cached, execution completes in $< 10\text{ ms}$.
  - Always open `ws://<host>:<port>/ws?clientId=<uuid>` **before** calling `POST /prompt`.
  - Filter out binary latent preview frames (`isinstance(raw_msg, str)`) before calling `json.loads`.
  - Track wall-clock deadlines exclusively with `time.monotonic()`.
  - Completion is signaled by `{"type": "executing", "data": {"node": None, "prompt_id": "<id>"}}`.
- **VRAM Cache Hygiene**:
  - Keep ComfyUI's execution cache warm across sequential batch items so the 8B text encoder and INT8 UNET stay resident in VRAM.
  - Only invoke `POST /free` (`{"unload_models": True, "free_memory": True}`) upon OOM recovery or during final session teardown.
- **Windows NTFS Handle Locking on Intermediate Assets**:
  - Purging temporary files from local ComfyUI `input/` or `output/` directories can raise transient `PermissionError` (`WinError 32`) while ComfyUI or Windows Search holds an open handle. Use bounded micro-retries (3 attempts, 50 ms backoff) and verify canonical path containment (`target.resolve().is_relative_to(input_dir.resolve())`).

---

## 3. Qwen-Image 2.1 DiT & 4-Channel VAE Quirks
- **4-Channel VAE Alpha Channel Quirk (Strict Greyscale Mask Requirement)**:
  - Passing an RGBA image with an embedded alpha mask into `LoadImage` $\rightarrow$ `TextEncodeQwenImage21` causes Qwen-Image 2.1's 4-channel VAE to interpret transparent border pixels as dark/black color shifts around the inpaint boundary.
  - **Mandatory Workaround**: Upload `cropped_model_roi.png` and `cropped_product_ref.png` strictly as 3-channel `RGB` PNGs, and upload `cropped_mask.png` strictly as a single-channel 8-bit greyscale (`mode="L"`) PNG routed through `ImageToMask (channel="red")` $\rightarrow$ `SetLatentNoiseMask`.
- **Rectified Flow Matching Sampler & Native `CFG = 1.0`**:
  - Qwen-Image 2.1 uses straight-line Rectified Flow trajectories (`sampler_name: "euler"`, `scheduler: "simple"`, `steps: 20`, `denoise: 1.0`).
  - Keep `cfg: 1.0` and `negative_prompt: ""`. Raising `CFG > 1.5` triggers multimodal cross-attention crosstalk where studio packshot backgrounds from `<image2>` contaminate the surrounding garment in `<image1>`.
- **Shared Qwen3-VL-8B Text/Vision Encoder (`TextGenerateV2`)**:
  - `workflows/qwen3_vl_inspector_api.json` reuses `qwen3vl_8b_int8_convrot.safetensors` inside ComfyUI (`sampling_mode: "off"`, captured via `PreviewAny`) with zero extra VRAM overhead.
  - Both VLM inspection and `Qwen-Image 2.1 Edit` strictly fail fast with `ComfyExecutionError` when ComfyUI is offline (no offline/OpenCV fallbacks).
- **`TextEncodeQwenImage21` + `VAEEncode` + `SetLatentNoiseMask` Architecture**:
  - `TextEncodeQwenImage21` (`comfy_extras/nodes_qwen.py`) returns an all-zero latent tensor (`torch.zeros`) while injecting `image_1` and `image_2` into `reference_latents`. Connecting `SetLatentNoiseMask` onto that zero latent leaves unmasked regions as solid grey/black.
  - Therefore, `workflows/qwen_2_1_edit_api.json` encodes `<image1>` via `VAEEncode(pixels=["1", 0], vae=["5", 0])` and wires `ImageToMask(image=["10", 0], channel="red")` -> `SetLatentNoiseMask` -> `KSampler (denoise=1.0)` on full `1024x1024` garment frames so unmasked pose/identity/logos stay anchored while `Qwen-Image 2.1` sees full garment proportions from `<image2>`.
- **Bypassing Full-Frame VAE Reconstruction Drift**:
  - Even with a latent noise mask (`SetLatentNoiseMask`), decoding the latent tensor through `VAEDecode` slightly alters pixel intensities (`16x` spatial block boundaries).
  - Therefore, the ComfyUI output must **never** be used directly as the final image; it must always pass through Stage 3 (`PixelLockCompositor`) where perimeter VAE offset compensation, interior distance-transform feathering, and `np.where(binary_mask[..., None] > 0, blended_image, original_image)` restore bit-identical unmasked pixels.
- **VLM Erase-Before-Insert Ordering**:
  - When a misplaced graphic requires two sequential passes (erase misplaced spot + insert authentic detail at correct garment coordinates), `Qwen3VLInspector` sorts `_is_erase_defect` passes first and skips `_locate_ref_detail_single_image` on pure erase passes (since their `<image2>` target is plain cloth, not a contrasting emblem).

---

## 4. Packaging, Static Checking & Gradio 6.x Runtime Quirks
- **OpenCV 5.1 Stubs & Pyright Interpreter Binding**:
  - `opencv-python-headless>=5.1` annotates functions such as `cv2.cvtColor` with `cv2.typing.MatLike` (`integer | floating`). Explicitly convert arrays returned from strict `NDArray[np.uint8]` helpers using `np.asarray(..., dtype=np.uint8)`.
  - `src/ci.py` must pass `--pythonpath <python_executable>` to `python -m pyright` so CI runners and virtualenvs type-check against the active environment instead of probing a host Python outside `.venv`.
- **Gradio 6.32 UI & Modal Overlay Traps**:
  - Pass `theme`, `css_paths`, `footer_links`, and `run_history` to `demo.launch(...)` rather than `gr.Blocks(...)`, and import `from gradio.themes import Soft` directly to avoid Pyright private-import warnings.
  - Under `from __future__ import annotations`, attach `handler.__annotations__ = {"evt": gr.SelectData}` explicitly (without `@wraps` hiding it or a string `"return"` annotation) so Gradio injects `SelectData` without warnings.
  - Gradio mirrors `elem_classes` onto nested `.gr-group` wrappers. Target full-screen modal backdrops via a unique `elem_id` (scoped with `:not(.hide):not(.hidden):not([hidden])`) and set `overflow: hidden` / transparent inner `.styler` and `.form` containers so rounded modal corners render cleanly without square bleed-through.
