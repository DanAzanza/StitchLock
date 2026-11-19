# GRADIO_KNOWLEDGE.md — Gradio 6.x Runtime & UI Quirks

## 1. Gradio 6 Launch & Typing Contracts
- **`demo.launch(...)` vs. `gr.Blocks(...)`**:
  - In Gradio 6.x (`6.32.0`), `theme`, `css_paths`, `footer_links` (e.g. `footer_links=[]`), and `run_history` (`run_history=False`) belong on `demo.launch(...)`, not `gr.Blocks(...)`.
  - Import themes via `from gradio.themes import Soft` instead of `gr.themes.Soft` to prevent Pyright `reportPrivateImportUsage` errors.
- **Event Handler Annotations under `from __future__ import annotations`**:
  - Gradio inspects runtime `__annotations__` to inject `gr.SelectData` payloads for `.select(...)` callbacks (e.g. `gr.Gallery.select`).
  - Assign `handler.__annotations__ = {"evt": gr.SelectData}` directly on the inner handler (without wrapping it in `@wraps` or adding stringified return annotations) to avoid `UserWarning` or skipped injection.
- **Deferred Theme/UI Import & Unbuffered Console Feedback (`python -u`)**:
  - Importing `gradio` (`from gradio.themes import Soft` / `import gradio as gr`) loads hundreds of submodules and can take `~9s` on local SSDs (or `~80s` on virtual/cloud mounts).
  - Defer importing Gradio until *after* CLI argument parsing, config loading, and fast backend healthchecks/auto-start, and launch entrypoints with `python -u` (`flush=True`) so users see immediate startup progress.

---

## 2. Layout, Gallery & Modal Overlay Quirks
- **Gallery Thumbnails & Scrollbars**:
  - Set `allow_preview=False` on selection galleries (`inbox_gallery`) so clicking a thumbnail immediately loads the item instead of opening an unwanted lightbox modal.
  - Omit hardcoded pixel `height` (`height=None`) and combine `overflow-y: auto` with bounded thumbnail heights so scrollbars appear only when items exceed column height.
- **Modal Pop-Ups via `gr.Group`**:
  - Gradio mirrors `elem_classes` onto nested `.gr-group` wrappers; applying `position: fixed; inset: 0` via class stacks two backdrops. Target the overlay via a unique `elem_id` scoped with `:not(.hide):not(.hidden):not([hidden])` for `display: flex`, set inner `.styler`/`.form`/`.gr-group` wrappers to `background: transparent; border: none`, and keep `overflow: hidden` on the inner card for clean rounded corners.
- **`gr.Row(equal_height=True)` & Compact Button Rows**:
  - `gr.Row(equal_height=True)` attaches `.stretch` (`.stretch > .column > * { flex-grow: 1; flex-shrink: 0 }`), which stretches all direct children vertically and adds wide gaps inside sidebar/workspace columns. Keep `equal_height=False` on the outer layout row, equalize column heights via CSS `align-items: stretch` on the row, and enforce `flex-grow: 0; flex-shrink: 0` on column children.
  - `gr.Row` children default to `min-width: min(160px, 100%)` with `flex-wrap: wrap`; pass explicit small `min_width` values to compact sidebar `gr.Button`s and apply `flex-wrap: nowrap; min-width: 0` in CSS so action buttons remain side-by-side.
- **Scoping Fixed Card Heights Away from `.block.fullscreen`**:
  - Gradio 6 implements `buttons=["fullscreen"]` on `gr.Image` / `gr.ImageSlider` by toggling `.fullscreen` (`position: fixed; inset: 0; width: 100%; height: 100%`) on the outer `.block` wrapper while keeping `.image-container` inside it.
  - Any custom CSS constraining card heights (e.g. `.stitch-triad .image-container { height: 225px !important; }`) must scope via `.block:not(.fullscreen)` and explicitly allow `.block.fullscreen .image-container` (`height: 100%`) plus `.block.fullscreen img` (`max-width: 92vw; max-height: 92vh; object-fit: contain`), otherwise the fullscreen overlay opens with the image trapped at thumbnail height.
