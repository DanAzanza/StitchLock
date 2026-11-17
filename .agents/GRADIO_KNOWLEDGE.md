# GRADIO_KNOWLEDGE.md — Gradio 6.x Runtime & UI Quirks

## 1. Gradio 6 Launch & Typing Contracts
- **`demo.launch(...)` vs. `gr.Blocks(...)`**:
  - In Gradio 6.x (`6.32.0`), `theme`, `css_paths`, `footer_links` (e.g. `footer_links=[]`), and `run_history` (`run_history=False`) belong on `demo.launch(...)`, not `gr.Blocks(...)`.
  - Import themes via `from gradio.themes import Soft` instead of `gr.themes.Soft` to prevent Pyright `reportPrivateImportUsage` errors.
- **Event Handler Annotations under `from __future__ import annotations`**:
  - Gradio inspects runtime `__annotations__` to inject `gr.SelectData` payloads for `.select(...)` callbacks (e.g. `gr.Gallery.select`).
  - Assign `handler.__annotations__ = {"evt": gr.SelectData}` directly on the inner handler (without wrapping it in `@wraps` or adding stringified return annotations) to avoid `UserWarning` or skipped injection.

---

## 2. Layout, Gallery & Modal Overlay Quirks
- **Gallery Thumbnails & Scrollbars**:
  - Set `allow_preview=False` on selection galleries (`inbox_gallery`) so clicking a thumbnail immediately loads the item instead of opening an unwanted lightbox modal.
  - Omit hardcoded pixel `height` (`height=None`) and combine `overflow-y: auto` with bounded thumbnail heights so scrollbars appear only when items exceed column height.
- **Modal Pop-Ups via `gr.Group`**:
  - Gradio mirrors `elem_classes` onto nested `.gr-group` wrappers; applying `position: fixed; inset: 0` via class stacks two backdrops. Target the overlay via a unique `elem_id` scoped with `:not(.hide):not(.hidden):not([hidden])` for `display: flex`, set inner `.styler`/`.form`/`.gr-group` wrappers to `background: transparent; border: none`, and keep `overflow: hidden` on the inner card for clean rounded corners.
