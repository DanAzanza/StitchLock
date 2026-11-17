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
