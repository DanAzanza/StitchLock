# OPENCV_KNOWLEDGE.md — OpenCV & Image Runtime Memory

## 1. Headless Packaging for Multi-OS CI
- **Headless Wheel on Linux (`opencv-python-headless>=4.9.0`)**:
  - Standard `opencv-python` links against system X11/OpenGL shared libraries (`libGL.so.1`) that are absent on default `ubuntu-latest` GitHub Actions runners and minimal Docker images.
  - Keep `opencv-python-headless>=4.9.0` in both `requirements.txt` and `pyproject.toml`; the pipeline never invokes `cv2.imshow` or OpenCV GUI windows.

---

## 2. Type Stubs & Static Checking (`OpenCV 5.1+` + Pyright)
- **`cv2.typing.MatLike` Return Annotations**:
  - `opencv-python-headless>=5.1` annotates image transforms (such as `cv2.cvtColor`, `cv2.morphologyEx`, `cv2.distanceTransform`) as returning `cv2.typing.MatLike` (`integer | floating`) rather than `NDArray[np.uint8]`.
  - Wrap returned buffers explicitly with `np.asarray(..., dtype=np.uint8)` in functions annotated with `NDArray[np.uint8]` so strict `pyright` passes cleanly across `Python 3.11+` and Linux/Windows environments.
- **CI Interpreter Alignment (`src/ci.py`)**:
  - Always invoke Pyright with `--pythonpath <active_python_executable>` so local virtualenvs and CI runners validate against the installed `opencv-python-headless` stubs rather than falling back to a host Python outside `.venv`.

---

## 3. Masking & Textile Image-Processing Invariants
- **Strictly Interior Euclidean Distance-Transform Feathering (`cv2.distanceTransform` vs. `cv2.GaussianBlur`)**:
  - Applying an isotropic Gaussian blur (`cv2.GaussianBlur`) to a binary mask $M_{\text{bin}} \in \{0, 1\}^{H \times W}$ leaks non-zero alpha weights outside the mask boundary ($\text{supp}(G_\sigma * M_{\text{bin}}) \supset \text{supp}(M_{\text{bin}})$), mutating unmasked background pixels during compositing.
  - Instead, compute Euclidean distances strictly inside $M_{\text{bin}} == 1$ via `dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)`, normalize by $w_{\text{ramp}} = 1.8\sigma$, and apply a $C^1$-continuous cubic Hermite smoothstep ($3t^2 - 2t^3$) clamped to `0.0` wherever `binary_mask == 0`.
- **Chroma vs. Luminance Separation on Garment Fabrics**:
  - Garment seam shadows and folds exhibit large Euclidean RGB distance (`> 35`) from the median cloth color while sharing almost identical chromaticity (`R - G`, `G - B`).
  - Combining chromaticity distance (`np.hypot((R - G) - c_rg, (G - B) - c_gb)`) with upper-corner studio backdrop rejection (`diff_bg > 24..28`) isolates colored patches, high-contrast emblems, and skin regions without false-triggering on fold shadows or studio margins.
