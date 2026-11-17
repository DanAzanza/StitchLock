# StitchLocker (`garment-fix-pipeline`): Systems Architecture & Mathematical Reference

This document details the engineering and mathematical foundations of **StitchLocker (`garment-fix-pipeline`)**: how the pipeline guarantees **0.0000% pixel alteration** outside the corrected garment region while leveraging **Qwen-Image 2.1 DiT** and headless **ComfyUI** execution.

---

## 1. Mathematical Proof of Zero-Loss Unmasked Pixel Preservation

Standard latent diffusion pipelines pass an entire image $I_{\text{orig}} \in \{0, \dots, 255\}^{H \times W \times 3}$ through a variational autoencoder ($\text{VAE}_{\text{dec}}(\text{VAE}_{\text{enc}}(I_{\text{orig}}))$). Because the latent bottleneck compresses spatial dimensions by $8\times$, every unmasked pixel $(y, x)$ suffers reconstruction drift:

$$\| I_{\text{VAE}}(y, x) - I_{\text{orig}}(y, x) \|_\infty \in [2, 18] > 0$$

Furthermore, applying a standard isotropic Gaussian blur $G_\sigma * M_{\text{bin}}$ to a binary mask $M_{\text{bin}} \in \{0, 1\}^{H \times W}$ leaks non-zero weights outside the mask support ($\text{supp}(G_\sigma * M_{\text{bin}}) \supset \text{supp}(M_{\text{bin}})$), mutating surrounding pixels during alpha blending.

### 1.1 Strictly Interior Euclidean Distance-Transform Feathering

In `src/auto_masker.py`, we restrict all blending transitions to the **interior** of the dilated binary mask $M_{\text{bin}}$ using the Euclidean Distance Transform $D(y, x)$:

$$D(y, x) = \min_{(y', x') : M_{\text{bin}}(y', x') = 0} \sqrt{(y - y')^2 + (x - x')^2}$$

By construction, $D(y, x) = 0$ for every pixel where $M_{\text{bin}}(y, x) = 0$, and $D(y, x) > 0$ strictly inside the mask. We normalize $D(y, x)$ over a ramp width $w_{\text{ramp}} = 1.8 \sigma$ and apply a $C^1$-continuous cubic Hermite smoothstep function:

$$t(y, x) = \text{clip}\!\left(\frac{D(y, x)}{w_{\text{ramp}}}, 0, 1\right), \quad \alpha(y, x) = \begin{cases} 3 t(y, x)^2 - 2 t(y, x)^3 & \text{if } M_{\text{bin}}(y, x) = 1 \\ 0 & \text{if } M_{\text{bin}}(y, x) = 0 \end{cases}$$

### 1.2 Hard Bitwise Pixel-Lock Compositing (`src/compositor.py`)

After blending the restored $1024 \times 1024$ ROI into the context window, `PixelLockCompositor.composite` applies a hard piecewise selection guard:

$$I_{\text{final}}(y, x, c) = \begin{cases} \text{round}\big(\alpha(y, x) I_{\text{restored}}(y, x, c) + (1 - \alpha(y, x)) I_{\text{orig}}(y, x, c)\big) & \text{if } M_{\text{bin}}(y, x) = 1 \\ I_{\text{orig}}(y, x, c) & \text{if } M_{\text{bin}}(y, x) = 0 \end{cases}$$

Consequently, over the entire unmasked domain $\Omega_{\text{unmasked}} = \{(y, x) \mid M_{\text{bin}}(y, x) = 0\}$:

$$\max_{(y, x) \in \Omega_{\text{unmasked}}, \, c \in \{R, G, B\}} \left| I_{\text{final}}(y, x, c) - I_{\text{orig}}(y, x, c) \right| = 0, \quad \text{SSIM}(\Omega_{\text{unmasked}}) = 1.0000$$

---

## 2. Qwen-Image 2.1 DiT Dynamics & 4-Channel VAE Topology

### 2.1 Rectified Flow Matching (RFM) & Native `CFG = 1.0`

**Qwen-Image 2.1** is a multimodal Diffusion Transformer trained under the Rectified Flow Matching objective, transporting latent samples along straight deterministic trajectories:

$$\frac{d X_t}{dt} = v_\theta(X_t, t, c), \quad X_t = (1 - t) X_0 + t X_1$$

In `workflows/qwen_2_1_edit_api.json`, we configure:
- `sampler_name: "euler"` and `scheduler: "simple"` (20 linear steps at `denoise: 1.0` along the straight ODE path).
- `cfg: 1.0` and `negative_prompt: ""`: Raising $s > 1.0$ in multimodal dual-image conditioning ($c = \{\text{Prompt}, \langle\text{image1}\rangle, \langle\text{image2}\rangle\}$) triggers **Cross-Attention Crosstalk**, where global background textures from the reference packshot $\langle\text{image2}\rangle$ bleed into the On-Model crop $\langle\text{image1}\rangle$.
- Strictly fail-fast execution: there is no OpenCV or cached fallback; if ComfyUI is offline, `ComfyQwenBridge` and `Qwen3VLInspector` immediately raise `ComfyExecutionError`.

### 2.2 Avoiding 4-Channel VAE Alpha Channel Artifacts

When an RGBA image is passed to ComfyUI's `LoadImage` node, the RGB channels at transparent pixels ($\alpha = 0$) are often zeroed or premultiplied, causing Qwen 2.1's 4-channel VAE to encode dark boundary halos around the inpaint mask.

To prevent this:
1. `cropped_model_roi.png` and `cropped_product_ref.png` are encoded strictly as 3-channel `RGB` PNGs.
2. `cropped_mask.png` is encoded strictly as a single-channel 8-bit greyscale (`mode="L"`) PNG and routed via `ImageToMask (channel="red")` $\rightarrow$ `SetLatentNoiseMask`.

---

## 3. Zero-Recompression PNG Provenance Metadata (`src/metadata.py`)

To comply with **EU AI Act Article 50** (machine-readable marking of AI-manipulated media) without risking pixel re-encoding drift:
- `embed_png_provenance_metadata` parses the binary PNG stream (`IHDR` $\dots$ `IEND`), validates every chunk's IEEE 802.3 CRC32 checksum, strips proprietary ComfyUI `workflow`/`prompt` graphs, and injects `tEXt` provenance chunks (`Software`, `AI-Compliance`, `Pixel-Lock-Verified`, `Unmasked-Max-Diff`, `Unmasked-SSIM`) immediately before `IEND` **without decompressing or touching `IDAT` pixel data**.
