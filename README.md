# StitchLocker (`garment-fix-pipeline`)

**Automated E-Commerce Garment Detail Correction Pipeline with Strict Unmasked RGB Pixel-Lock Verification**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Release v1.0.1](https://img.shields.io/badge/release-v1.0.1-emerald.svg)](https://github.com/DanAzanza/StitchLock/releases)
[![CI Quality Gate](https://github.com/DanAzanza/StitchLock/actions/workflows/ci.yml/badge.svg)](https://github.com/DanAzanza/StitchLock/actions/workflows/ci.yml)
[![ComfyUI API](https://img.shields.io/badge/ComfyUI-REST%20%2F%20WebSocket-orange.svg)](workflows/qwen_2_1_edit_api.json)
[![Architecture: DiT](https://img.shields.io/badge/Model-Qwen--Image%202.1%20Edit-purple.svg)](.agents/PROJECT.md)
[![Pixel-Lock](https://img.shields.io/badge/Pixel--Lock-Unmasked%20RGB%20Verified-success.svg)](src/verifier.py)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Executive Summary

When generating On-Model lookbook imagery from e-commerce product packshots, generative models frequently capture the pose, lighting, and garment drape well while hallucinating fine-grained brand details—producing misspelled typography (`"KROVX"` instead of `"KOVA"`), oversized or mutated emblems, or distorted embroidered patches.

Passing the entire image through another diffusion pass degrades skin texture, alters facial identity, and shifts background pixels due to lossy VAE encode/decode cycles.

**StitchLocker (`garment-fix-pipeline`)** solves this with **zero manual intervention**:
1. **Stage 0 — Qwen3-VL-8B Multi-Defect Inspector & Visual Grounding (`src/vlm_inspector.py` & `src/vlm_heuristics.py`)**: Uses ComfyUI's built-in `TextGenerateV2` node on the shared `qwen3vl_8b_int8_convrot.safetensors` checkpoint (0 MB extra VRAM) to verify whether garment logos/text are defective (`has_defect`), localize one or multiple separated defects (`model_bbox_2d` & `ref_bbox_2d`, with overlap deduplication, border-connected backdrop segmentation, dynamic garment-relative 3×3 micro-emblem scanning, and solo-image contrast re-grounding), and generate direct imperative `edit_prompt`s.
2. **Stage 1 — Full-Frame Masking & Reference Garment Alignment (`src/auto_masker.py`)**: Builds a resolution-relative, interior-feathered full-image mask around each localized defect while preserving the native On-Model resolution/aspect ratio and aligning the full Product Reference garment frame per defect (so `Qwen-Image 2.1` sees all logos and exact garment-to-logo proportions).
3. **Stage 2 — Sequential Multi-Defect Qwen-Image 2.1 Edit (`src/comfy_client.py`)**: Executes a 14-node headless **ComfyUI Qwen-Image 2.1 (`ImageScaleToTotalPixels` + `TextEncodeQwenImage21` + `VAEEncode` + `SetLatentNoiseMask`)** workflow sequentially per detected defect on full native-aspect frames and greyscale masks, scaling to ~1.0 MP (`32px`-aligned) inside ComfyUI to replace oversized/hallucinated logos in true 1:1 garment proportion while anchoring unmasked pose and fabric.
4. **Stage 3 & 4 — Strict Pixel-Lock Compositor & Automated Verification Gate (`src/compositor.py` & `src/verifier.py`)**: Composites each repaired region using perimeter VAE color-shift compensation and interior distance-transform feathering, then enforces a hard NumPy pixel lock outside the union mask in the decoded 8-bit RGB buffer (`unmasked_max_abs_diff == 0`, 2D Gaussian-windowed `unmasked_ssim == 1.0000`), verifies masked-region non-degeneracy and calibrated `128×128` salient-cropped reference similarity (`masked_reference_similarity`), supports opt-in fail-closed VLM post-checks (`enable_vlm_post_check`), preserves source ICC color profiles and RGBA alpha channels on lossless PNG export, and renders a 4-panel visual report.

---

## Interactive Artist Studio UI (`app.py`)

StitchLocker ships with a clean, full-height 2-column **Artist Studio** (`python app.py` or `run_pipeline.bat`) that manages `test_cases/` pairs (`<name>.png` + `<name>_reference.png`), auto-starts ComfyUI in the background, and shuts it down cleanly when the browser closes:

### Before/After Compare Slider & 3-Up Look Inspection
![StitchLocker Studio — Compare Slider](docs/assets/ui_studio_compare.png)

### 1-Click Change Heatmap View (`4×` Balanced Pixel-Diff)
![StitchLocker Studio — Change Heatmap](docs/assets/ui_studio_heatmap.png)

### `+ Add Look` Upload Pop-Up (Drag & Drop directly into Inbox)
![StitchLocker Studio — Add Look Pop-Up](docs/assets/ui_add_look_popup.png)

---

## Visual Verification Showcase (Zero-Touch Batch Output on ComfyUI Qwen-Image 2.1)

All showcase cases use **100% synthetic, trademark-free studio assets** (front product packshots paired with photorealistic On-Model lookbook photos containing intentional logo/patch hallucinations), restored and verified with **zero manual brush work**:

### Case 1: Minimalist Full-Zip Track Jacket — Left-Chest Hexagon Compass Crest & `'KOVA'` Embroidery
![Case 1 Verification Report](docs/assets/kova_zip_jacket_verification_report.png)

### Case 2: Sage-Olive Fleece Sweatpants — Right-Thigh Terracotta Twin-Mountain Oval Patch
![Case 2 Verification Report](docs/assets/sage_sweatpant_verification_report.png)

| Case ID | Garment Detail | Localization Strategy | Backend | Unmasked Pixels Verified | `unmasked_max_abs_diff` | `unmasked_changed_pixels` | `unmasked_ssim` | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `kova_zip_jacket` | White hexagonal compass crest and `KOVA` chest embroidery | `qwen3_vl_8b + vlm_grounded_box` | `comfyui_qwen_2_1` | `1,006,576 / 1,048,576` | **`0`** | **`0 (0.0000%)`** | **`1.0000`** | **PASS** |
| `sage_sweatpant` | Small rust-terracotta and cream oval twin-mountain crest | `qwen3_vl_8b + vlm_grounded_box` | `comfyui_qwen_2_1` | `1,028,136 / 1,048,576` | **`0`** | **`0 (0.0000%)`** | **`1.0000`** | **PASS** |

---

## 1. What I Built (Architecture & 5-Stage Pipeline)

> [!NOTE]
> For mathematical derivations of the Interior Distance-Transform Feathering, `128×128` Masked Reference Similarity, Zero-Recompression PNG Provenance chunks, and Rectified Flow Matching ODE dynamics, see [.agents/PROJECT.md](.agents/PROJECT.md) and [.agents/COMFYUI_KNOWLEDGE.md](.agents/COMFYUI_KNOWLEDGE.md).

```mermaid
sequenceDiagram
    autonumber
    participant CLI as run_batch.py / app.py
    participant VLM as Stage 0: Qwen3VLInspector
    participant Masker as Stage 1: AutoMasker
    participant Bridge as Stage 2: ComfyQwenBridge
    participant Comfy as ComfyUI Server (:8000/:8188)
    participant Comp as Stage 3: PixelLockCompositor
    participant Verifier as Stage 4: PixelLockVerifier

    CLI->>Bridge: Probe ComfyUI /system_stats
    alt ComfyUI Offline / Unreachable
        Bridge--xCLI: Raise ComfyExecutionError (Fail-Fast)
    else ComfyUI Online
        CLI->>VLM: inspect_garment_pair(model_rgb, ref_rgb)
        VLM->>Comfy: TextGenerateV2 (qwen3vl_8b_int8_convrot)
        Comfy-->>VLM: has_defect, defects[] (boxes + edit_prompts)
        alt No Defect Detected (has_defect == false)
            CLI-->>Verifier: Zero-Compute Short-Circuit (Original Untouched)
        else One or More Defects Detected
            loop Per Separated Defect in defects[]
                CLI->>Masker: localize_and_mask(current_rgb, ref_rgb, defect)
                Masker-->>CLI: binary_mask, feathered_mask, native frames & aligned ref
                CLI->>Bridge: restore_detail_roi(full_model, aligned_ref, greyscale_mask)
                Bridge->>Comfy: POST /upload/image + WebSocket (/ws) + POST /prompt
                Comfy-->>Bridge: Masked Full-Frame Diffusion Output (1.0 MP scaled inside ComfyUI)
                CLI->>Comp: composite(current_rgb, restored_frame, binary_mask, feathered_mask)
                Comp-->>CLI: Updated current_rgb (Previous fixes & background preserved)
            end
            CLI->>Verifier: verify(union_mask, reference_crop) + render_four_panel_report()
            Verifier-->>CLI: PASS/FAIL Metrics JSON + 4-Panel PNG (with Non-Cryptographic PNG tEXt Provenance)
        end
    end
```

### Repository Structure
```text
StitchLocker/
├── .github/workflows/
│   ├── ci.yml                          # Multi-OS (Ubuntu & Windows) & Python 3.11/3.12 CI Quality Gate
│   └── release.yml                     # Tag-triggered GitHub Release builder & checksum uploader
├── config.yaml                         # Centralized pipeline, masking, ComfyUI, verification & heuristic config
├── pyproject.toml                      # Project metadata, dependencies, Ruff, Pyright & Pytest config
├── requirements.txt                    # Runtime, Studio UI & CI dependencies
├── install.bat                         # Windows 1-Click venv installer & CI self-test runner
├── install.sh                          # Linux/macOS 1-Click venv installer & CI self-test runner
├── run_pipeline.bat                    # Windows 1-Click Artist Studio UI launcher (auto ComfyUI lifecycle)
├── run_pipeline.sh                     # Linux/macOS 1-Click Artist Studio UI launcher (auto ComfyUI lifecycle)
├── setup_test_cases.py                 # Ensures flat `<name>.png` + `<name>_reference.png` fixtures exist
├── run_batch.py                        # Zero-touch CLI batch runner (fails fast if ComfyUI offline)
├── app.py                              # Gradio 6.0+ 2-Column Artist Studio UI with `+ Add Look` pop-up
├── docs/
│   └── assets/                         # Artist Studio UI & verification showcase screenshots
├── workflows/
│   ├── qwen3_vl_inspector_api.json     # ComfyUI API graph for Qwen3-VL-8B defect detection
│   └── qwen_2_1_edit_api.json          # 14-node ComfyUI graph (ImageScaleToTotalPixels + TextEncodeQwenImage21 + SetLatentNoiseMask)
├── scripts/
│   ├── run_ci_locally.py               # Local CI Quality Gate runner (Ruff + Formatter + Pyright + Pytest)
│   └── package_release.py              # Standalone release zip & SHA256SUMS.txt archive builder
├── src/
│   ├── models.py                       # Immutable domain dataclasses, config loader & path safety guards
│   ├── vlm_heuristics.py               # Configurable CV box refinement, 3x3 garment scan & shift-splitting heuristics
│   ├── vlm_inspector.py                # Stage 0: Qwen3-VL-8B defect inspection, grounding & opt-in post-check
│   ├── auto_masker.py                  # Stage 1: Full-frame mask & reference alignment engine
│   ├── comfy_client.py                 # Stage 2: Shared headless ComfyUI WebSocket/HTTP transport & Edit bridge
│   ├── comfy_launcher.py               # Thread-safe ComfyServerManager for ComfyUI Desktop/standalone startup & cleanup
│   ├── compositor.py                   # Stage 3: Lanczos stitching & strict NumPy pixel-lock compositor
│   ├── fixtures.py                     # Deterministic procedural test case fixture generator
│   ├── verifier.py                     # Stage 4: 2D SSIM, Sobel edge-NCC similarity, Pixel-Lock gate & 4-panel renderer
│   ├── metadata.py                     # Zero-recompression PNG tEXt provenance chunk processor (IDAT-preserving)
│   ├── housekeeper.py                  # Safe local ComfyUI intermediate asset cleanup with NTFS lock retries
│   ├── ui_handlers.py                  # Inbox, batch & BrowserSessionLifecycle handlers for Gradio Studio
│   ├── ui_custom_handlers.py           # Custom upload, interactive VLM box overlay & ROI preview handlers
│   ├── ui_style.css                    # Full-height 2-column & modal overlay Artist Studio styling
│   ├── ci.py                           # Multi-stage local CI verification orchestrator
│   ├── packaging.py                    # Deterministic zip & SHA-256 release packaging engine
│   └── pipeline.py                     # End-to-end orchestrator with shared VLM post-check & atomic .part writes
├── test_cases/                         # 6 synthetic lookbook/packshot pairs (12 PNGs: kova, sage, ember, cobalt, cream, burgundy)
├── outputs/
│   ├── <name>.png                      # Repaired final images matching input case filenames (gitignored locally)
│   └── _reports/                       # 4-panel verification PNGs, diff heatmaps (`_diff_heatmap_x4.png`) & JSONs
└── tests/
    ├── test_core_and_masker.py         # Models, config, case discovery, AutoMasker & ICC/RGBA round-trip tests
    ├── test_comfy_and_vlm.py           # ComfyUI bridge, VLM parsing, heuristics, housekeeper & ComfyServerManager tests
    ├── test_pipeline_and_ui.py         # Compositor, 2D SSIM verifier, metadata, pipeline & Studio UI tests
    ├── test_verifier_and_post_check.py # Typography SSIM/Sobel discrimination & fail-closed VLM post-check tests
    ├── test_packaging.py               # Release archive, SHA-256 checksum & version alignment tests
    └── test_generalization.py          # 10-case holdout generalization + real studio asset verification suite
```

---

## 2. How Unmasked RGB Pixel-Lock & Verification Work (Bypassing VAE Degradation)

Passing a full-resolution image through a latent diffusion VAE (`VAEEncode` $\rightarrow$ `VAEDecode`) introduces lossy reconstruction noise across **every pixel** in the frame (typically $\Delta \in [2, 18]$ intensity levels per RGB channel), subtly blurring skin pores, eyes, hair strands, and studio grain.

Furthermore, applying a standard outward Gaussian blur to a binary mask bleeds non-zero alpha values *outside* the mask boundary (`binary_mask == 0`), contaminating surrounding pixels during alpha compositing.

We enforce unmasked preservation and masked-region quality through three complementary mechanisms and provenance tagging:

1. **Strictly Interior Distance-Transform Feathering (`src/auto_masker.py`)**:
   Instead of blurring outward across the mask boundary, we compute the Euclidean distance transform `cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)` **strictly inside** `binary_mask > 0` and apply a cubic Hermite smoothstep ramp:
   $$\alpha(y, x) = \begin{cases} 3t^2 - 2t^3, \quad t = \min\!\left(1, \frac{\text{dist}(y, x)}{w_{\text{ramp}}}\right) & \text{if } M_{\text{bin}}(y, x) = 1 \\ 0 & \text{if } M_{\text{bin}}(y, x) = 0 \end{cases}$$
   This creates a seamless transition inside the dilated mask border while reaching exact $0.0$ at the boundary.

2. **Hard NumPy Pixel Lock & Source ICC/RGBA Preservation (`src/compositor.py`, `src/pipeline.py`)**:
   After blending inside the ROI, the compositor enforces a bitwise selection guard in the decoded 8-bit RGB buffer:
   ```python
   final_image = np.where(binary_mask[..., None] > 0, blended_image, original_image)
   ```
   Every pixel where `binary_mask == 0` is copied directly from the untouched `original_image` RGB buffer in memory (`unmasked_max_abs_diff = 0`, `0.0000%` alteration relative to the decoded 8-bit RGB input). On lossless PNG export, `src/pipeline.py` also re-attaches the original image's ICC color profile (`icc_profile`) and untouched RGBA alpha channel when present (inputs using 16-bit depth or lossy JPEG source compression are normalized to 8-bit RGB/RGBA PNGs).

3. **2D Windowed SSIM, Calibrated Masked Similarity & Opt-In VLM Post-Check (`src/verifier.py`, `src/vlm_inspector.py`)**:
   `PixelLockVerifier` validates unmasked integrity using 2D Gaussian-windowed SSIM ($11 \times 11, \sigma = 1.5$, isolated on an unmasked canvas to prevent boundary bleed) and checks the *masked* region for non-degeneracy (rejecting blackouts or flat collapses), plain-cloth `ERASE` uniformity, and `128×128` salient-cropped 2D SSIM + Sobel edge-gradient NCC similarity against the aligned reference crop (`masked_reference_similarity >= min_masked_similarity`), optionally backed by a fail-closed `Qwen3-VL` semantic/OCR post-check (`enable_vlm_post_check`).

4. **Zero-Recompression PNG Provenance Metadata (`src/metadata.py`)**:
   To support EU AI Act Article 50 transparency workflows without re-encoding `IDAT` pixel streams, `embed_png_provenance_metadata` injects `tEXt` audit chunks (`Software`, `AI-Generated`, `AI-Compliance`, `Pixel-Lock-Verified`, etc.) directly before `IEND`. PNG `tEXt` chunks are non-cryptographic metadata (distinct from cryptographic C2PA manifests or imperceptible watermarks) and can be stripped by downstream re-encoders.

---

## 3. What Didn't Work & Key Technical Findings

1. **Tight Logo Cropping Loses Relative Garment Scale**:
   - Cropping out just the defective region and conditioning on a tight crop of the reference emblem removes global garment proportions (leg width, chest span, opposite emblem), causing **Qwen-Image 2.1** to over-scale replacement logos.
   - **Solution**: Conditioning on the full On-Model image (`<image1>`) and full Product Reference garment (`<image2>`, frame-aligned to match torso/leg coordinates when packshot headroom differs) while constraining denoising to the `Qwen3-VL` mask (`VAEEncode` + `SetLatentNoiseMask`) preserves authentic 1:1 logo proportions and keeps the model's pose intact.
2. **High CFG Cross-Attention Bleeding ("Pink Elephant" Effect)**:
   - Raising `CFG > 1.5` or adding verbose negative prompts in Qwen-Image 2.1 causes global layout contamination from `<image2>` into `<image1>`. Keeping `CFG = 1.0` (`euler` / `simple`, `steps = 20`, `denoise = 1.0`) with `SetLatentNoiseMask` yields natural fold integration while `PixelLockCompositor` restores exact outer pixels.
3. **Dual-Image VLM Coordinate Echoing on Differently Framed Packshots**:
   - When comparing `<image1>` and `<image2>` in one pass, VLMs occasionally echo Image 1 coordinates onto plain cloth in Image 2 when vertical headroom differs, and viewer/wearer `"left/right chest"` phrasing misleads directional edits.
   - **Solution**: `Qwen3VLInspector` detects low-contrast reference boxes automatically and runs a solo-image lookup on `<image2>`, applies universal padding without color-specific hacks, deduplicates overlapping detections, and enforces direct `"Replace..."` prompts without ambiguous left/right labels.

---

## 4. Quickstart & Reproducible Verification

### Option A: 1-Click Artist Studio Launcher (Windows & Linux/macOS)
- **Windows**: Double-click **`install.bat`** once to set up the environment, then launch **`run_pipeline.bat`** to start ComfyUI automatically, open the interactive Web UI (`http://127.0.0.1:7860`) in your browser, manage the `test_cases/` inbox (single-case processing or full batch), and cleanly shut down ComfyUI when the browser tab closes.
- **Linux / macOS**: Run `./install.sh` once, then `./run_pipeline.sh` (set `COMFYUI_DIR=/path/to/ComfyUI` if your local ComfyUI install is in a custom path).

### Option B: CLI Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic sample test cases (or drop flat `<name>.png` + `<name>_reference.png` pairs into test_cases/)
python setup_test_cases.py

# 3. Execute zero-touch batch pipeline (requires live ComfyUI on :8000/:8188; fails fast if offline)
python run_batch.py

# 4. Launch interactive Artist Studio UI (auto-opens http://127.0.0.1:7860 in browser)
python app.py

# 5. Run full local CI Quality Gate (Ruff linter + formatter + Pyright + Pytest on Windows/Linux parity)
python scripts/run_ci_locally.py
```
