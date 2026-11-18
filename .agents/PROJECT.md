# PROJECT.md — StitchLocker (`garment-fix-pipeline`) Project Memory

## 1. Project Overview & Architectural Boundaries
**StitchLocker (`garment-fix-pipeline`)** is an automated E-Commerce Garment Detail Correction pipeline that repairs hallucinated garment details (e.g., distorted logos, garbled typography, faulty stripes) on generated On-Model lookbook imagery using an authentic product reference packshot, while enforcing **0.0000% pixel alteration** (`unmasked_max_abs_diff == 0`, 2D Gaussian-windowed `unmasked_ssim == 1.0000`) outside the binary mask in the decoded 8-bit RGB buffer and verifying masked-region non-degeneracy and reference similarity (`masked_reference_similarity`).

### Layer Separation
- **Contracts & Schemas ([`src/models.py`](../src/models.py))**: Frozen dataclasses (`NormalizedBox`, `PixelBBox`, `CaseManifest`, `LocalizationResult`, `VerificationMetrics`, `HeuristicConfig`, `PipelineConfig` with `inspector_workflow_path`), configuration loader (`load_pipeline_config`), path traversal guard (`resolve_safe_path`), and domain exceptions (`PipelineError`, `ConfigurationError`, `LocalizationError`, `ComfyExecutionError`, `VerificationError`).
- **Stage 0 — VLM Defect Inspection & Heuristics ([`src/vlm_inspector.py`](../src/vlm_inspector.py), [`src/vlm_heuristics.py`](../src/vlm_heuristics.py))**: Uses ComfyUI's built-in `TextGenerateV2` on `qwen3vl_8b_int8_convrot.safetensors` to verify whether a defect exists (`has_defect`), localize `model_bbox_2d` / `ref_bbox_2d` (with single-image solo contrast re-grounding when `<image2>` framing differs), apply color-agnostic chroma/contrast salience tightening, shift splitting, and micro-emblem checks governed by `HeuristicConfig`, and write the direct `"Replace..."` edit instruction; fails fast with `ComfyExecutionError` when ComfyUI is offline.
- **Stage 1 — Auto-Localization & Masking ([`src/auto_masker.py`](../src/auto_masker.py))**: Zero-brush localization (`auto_vision` or deterministic `manifest_roi`). Produces full-resolution `binary_mask` ($\{0, 1\}$), strictly interior Euclidean distance-transform smoothstep `feathered_mask` ($[0.0, 1.0]$), `cropped_model_roi` (full On-Model frame or `256x256` micro-ROI), `cropped_product_ref` (full Product Reference garment aligned on-frame), and `cropped_mask_greyscale`.
- **Stage 2 — Generative Restoration Bridge ([`src/comfy_client.py`](../src/comfy_client.py))**: Shared headless ComfyUI REST/WebSocket transport helpers and `ComfyQwenBridge` executing [`workflows/qwen_2_1_edit_api.json`](../workflows/qwen_2_1_edit_api.json) (`TextEncodeQwenImage21` + `VAEEncode` + `ImageToMask` + `SetLatentNoiseMask` + `KSampler`). Contains zero OpenCV/offline fallbacks and fails fast with `ComfyExecutionError` if ComfyUI is unreachable.
- **Stage 3 — Strict Pixel-Lock Compositor ([`src/compositor.py`](../src/compositor.py))**: Resizes restored ROI back to `context_bbox`, compensates boundary VAE color shift, blends inside the mask via `feathered_mask`, and enforces the hard bitwise pixel lock (`np.where(binary_mask[..., None] > 0, blended_image, original_image)`).
- **Stage 4 — Automated Verification Gate ([`src/verifier.py`](../src/verifier.py))**: Computes boundary-isolated 2D Gaussian-windowed `unmasked_ssim`, enforces `strict_zero_outside_mask` (strict 0-diff or bounded tolerance when disabled), verifies masked non-degeneracy and `masked_reference_similarity >= min_masked_similarity`, and renders the amplified diff heatmap (`diff_heatmap_filename`) and 4-panel verification report image.
- **Compliance & Housekeeping Utilities ([`src/metadata.py`](../src/metadata.py), [`src/housekeeper.py`](../src/housekeeper.py))**: Zero-recompression PNG `tEXt` chunk processor injecting non-cryptographic provenance/audit tags without touching `IDAT` pixels or `iCCP` profiles; sandboxed cleanup of temporary local ComfyUI assets.
- **Orchestration & Entrypoints ([`src/pipeline.py`](../src/pipeline.py), [`src/ui_handlers.py`](../src/ui_handlers.py), [`src/fixtures.py`](../src/fixtures.py), [`src/comfy_launcher.py`](../src/comfy_launcher.py), [`run_batch.py`](../run_batch.py), [`app.py`](../app.py), [`setup_test_cases.py`](../setup_test_cases.py))**: Flat `<name>.*` + `<name>_reference.*` case discovery, cross-platform automated ComfyUI Desktop/standalone detection & headless launch (`auto_launch: true` in `config.yaml`, `COMFYUI_DIR` override on Linux), sequential multi-defect loop (`execute_multi_defect_restoration`), flat `outputs/<name>.png` (preserving source RGB ICC profiles & RGBA alpha channels) + `outputs/_reports/` (`_diff_heatmap_x4.png` configurable via `diff_amplification_factor`), atomic `.part` staging writes, Gradio Studio UI (`app.py` + `src/ui_handlers.py`), and modular test suites (`tests/test_core_and_masker.py`, `tests/test_comfy_and_vlm.py`, `tests/test_pipeline_and_ui.py`, `tests/test_generalization.py`).

---

## 2. Runtime & Filesystem Constraints
- **Platform**: Windows 11 / PowerShell 7+ and Linux (`bash`) hosts; multi-OS GitHub Actions CI matrix (`ubuntu-latest`, `windows-latest`, Python `3.11` & `3.12`).
- **Python & Dependency Runtime**: Project targets Python `>= 3.11`; uses `opencv-python-headless` for X11/libGL-free execution on Linux CI runners; `.gitattributes` enforces `eol=lf` on `*.sh` launchers (`install.sh`, `run_pipeline.sh`) and `eol=crlf` on `*.bat`.
- **Windows Console & Google Drive Mount (`G:\`)**:
  - Always reconfigure standard streams (`sys.stdout.reconfigure(encoding="utf-8")`) in CLI entrypoints.
  - Write all image and JSON outputs via temporary `.part` staging files (`*.<pid>_<ns>.part`) followed by atomic `Path.replace()` to prevent partial reads by background cloud sync or Windows Search indexers.
  - Validate all manifest and config paths with `resolve_safe_path(root, candidate)`.

---

## 3. Central Verification & Execution Commands
- **Full Local CI Parity Gate (Linter + Formatter + Pytest Suite)**:
  ```powershell
  python scripts/run_ci_locally.py
  ```
- **Generate / Reset Sample Test Cases (`test_cases/`)**:
  ```powershell
  python setup_test_cases.py --force
  ```
- **Run Zero-Touch Batch Pipeline & Verification Gate** *(requires live ComfyUI server)*:
  ```powershell
  python run_batch.py
  ```
- **Launch Interactive Gradio Web UI**:
  ```powershell
  python app.py
  ```
- **Individual Quality Gate Commands**:
  ```powershell
  python -m ruff check .
  python -m ruff format --check .
  python -m pyright
  python -m pytest tests -v
  ```

---

## 4. Commit Scopes
- `[core]` — Domain models, configuration, and path guards in `src/models.py`
- `[masker]` — Auto-localization and distance-transform mask generation in `src/auto_masker.py`
- `[comfy]` — Headless ComfyUI VLM/Edit clients and workflow graphs in `src/vlm_inspector.py`, `src/comfy_client.py`, and `workflows/`
- `[compositor]` — Strict pixel-lock stitching in `src/compositor.py`
- `[verifier]` — Automated verification gate and 4-panel report rendering in `src/verifier.py`
- `[metadata]` — Zero-recompression PNG chunk provenance processing in `src/metadata.py`
- `[housekeeper]` — Local ComfyUI intermediate asset cleanup in `src/housekeeper.py`
- `[pipeline]` — Orchestration, CLI runner, Gradio UI, and fixtures (`src/pipeline.py`, `run_batch.py`, `app.py`, `setup_test_cases.py`)
- `[ci]` — CI runner, GitHub workflows, and repository configs (`src/ci.py`, `scripts/run_ci_locally.py`, `.github/`, `pyproject.toml`)
- `[test]` — Automated unit, boundary, and negative tests in `tests/`
- `[docs]` — Architecture documentation and `README.md`
