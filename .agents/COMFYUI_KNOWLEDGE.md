# COMFYUI_KNOWLEDGE.md — ComfyUI & Qwen-Image 2.1 Platform Memory

## 1. Local Hardware & Server Environment
- **GPU Hardware**: CUDA 0 — NVIDIA GeForce RTX 4060 Ti (16 GB VRAM).
- **Server Endpoints**:
  - Standalone ComfyUI default: `http://127.0.0.1:8188`
  - Comfy Desktop headless default: `http://127.0.0.1:8000`
  - Headless clients probe `server_url` and fallback ports `[8000, 8188]`.
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

## 3. Qwen-Image 2.1 DiT, Qwen3-VL-8B & 4-Channel VAE Quirks
- **4-Channel VAE Alpha Channel Quirk (Strict Greyscale Mask Requirement)**:
  - Passing an RGBA image with an embedded alpha mask into `LoadImage` $\rightarrow$ `TextEncodeQwenImage21` causes Qwen-Image 2.1's 4-channel VAE to interpret transparent border pixels as dark/black color shifts around the inpaint boundary.
  - **Mandatory Workaround**: Upload input and reference images strictly as 3-channel `RGB` PNGs, and upload inpaint masks strictly as single-channel 8-bit greyscale (`mode="L"`) PNGs routed through `ImageToMask (channel="red")` $\rightarrow$ `SetLatentNoiseMask`.
- **Rectified Flow Matching Sampler & Native `CFG = 1.0`**:
  - Qwen-Image 2.1 is trained under the Rectified Flow Matching objective ($\frac{dX_t}{dt} = v_\theta(X_t, t, c)$, $X_t = (1 - t)X_0 + tX_1$) and uses straight-line Euler integration (`sampler_name: "euler"`, `scheduler: "simple"`, `steps: 20`, `denoise: 1.0`).
  - Keep `cfg: 1.0` and `negative_prompt: ""`. Raising `CFG > 1.5` in dual-image conditioning triggers multimodal cross-attention crosstalk where studio packshot backgrounds from `<image2>` bleed into `<image1>`.
- **Shared Qwen3-VL-8B Text/Vision Encoder (`TextGenerateV2`)**:
  - ComfyUI's `TextGenerateV2` node reuses `qwen3vl_8b_int8_convrot.safetensors` (`sampling_mode: "off"`, captured via `PreviewAny`) with zero extra VRAM overhead when paired with Qwen-Image 2.1.
- **`TextEncodeQwenImage21` + `VAEEncode` + `SetLatentNoiseMask` Graph Topology**:
  - `TextEncodeQwenImage21` (`comfy_extras/nodes_qwen.py`) outputs an all-zero latent tensor (`torch.zeros`) while injecting `image_1` and `image_2` into `reference_latents`. Wiring `SetLatentNoiseMask` onto that zero latent leaves unmasked regions as solid grey/black.
  - Encode `<image1>` via `VAEEncode(pixels, vae)` first and wire `ImageToMask(channel="red")` $\rightarrow$ `SetLatentNoiseMask` $\rightarrow$ `KSampler (denoise=1.0)` so unmasked latent tokens remain anchored to `<image1>`.
- **ViT Token Grid (`28x28 px`) & `8x` VAE Latent Bottleneck on Small Details**:
  - At `1024x1024` full-frame scale (`ImageScaleToTotalPixels` at `1.0 MP`), `Qwen3-VL-8B`'s merged ViT patch grid (`28x28 px` per visual token) compresses a `~25–35 px` chest emblem into a single visual token, and `Qwen-Image 2.1`'s `8x` VAE downsampling reduces it to `~3x4` latent cells (with `VAEDecode` still introducing slight reconstruction drift of `2–18` RGB levels across `16x` block boundaries).
  - Resolving fine internal cutouts or `6 px`-tall typography requires cropping a local sub-region (`256x256`) around the target before `ImageScaleToTotalPixels (1.0 MP)` (`4x` effective latent supersampling) and compositing the decoded result back into the original frame outside ComfyUI.
