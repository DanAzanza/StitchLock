"""Interactive Gradio Studio UI for StitchLocker Garment Detail Correction.

Clean Artist-First 2-Column Studio Web App:
  - Left Inbox displays a `+ Add Look` tile plus `test_cases/` thumbnails with live Refresh
    and Run All (Batch); clicking `+` opens a clean Model + Reference upload pop-up whose
    uploaded pair is persisted directly into `test_cases/` and added to the thumbnails.
  - Right Workspace displays Selected Look, 3-up (Original | Product Reference | Repaired),
    and an interactive Before/After Compare Slider with a 1-click toggle to the Change Heatmap.
  - Lifecycle: Auto-starts ComfyUI when offline and shuts down the spawned ComfyUI backend
    cleanly when the browser session closes (with a reload-safe grace window).
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray
from src.comfy_launcher import ensure_comfyui_server, stop_auto_started_server
from src.models import ComfyExecutionError, MaskingStrategy, PipelineError, load_pipeline_config
from src.ui_custom_handlers import (
    CONFIG_PATH,
    ROOT_DIR,
    UI_CSS_PATH,
    VIEW_COMPARE,
    VIEW_HEATMAP,
    ImageInput,
    detect_custom_ui,
    ensure_environment_ready,
    preview_custom_boxes_ui,
    render_box_overlay,
    render_multi_box_overlay,
    run_custom_upload_ui,
)
from src.ui_handlers import (
    BrowserSessionLifecycle,
    add_pair_to_inbox,
    build_add_look_tile,
    build_artist_heatmap,
    execute_batch_ui,
    format_look_bar,
    gallery_with_add_tile,
    get_inbox_overview_data,
    inspect_single_case_ui,
    is_case_repaired,
    load_single_case_view_ui,
    slider_pair,
)

__all__ = [
    "BrowserSessionLifecycle",
    "CONFIG_PATH",
    "ImageInput",
    "ROOT_DIR",
    "UI_CSS_PATH",
    "VIEW_COMPARE",
    "VIEW_HEATMAP",
    "add_pair_to_inbox",
    "build_add_look_tile",
    "build_artist_heatmap",
    "build_gradio_app",
    "detect_custom_ui",
    "execute_batch_ui",
    "gallery_with_add_tile",
    "get_inbox_overview_data",
    "inspect_single_case_ui",
    "is_case_repaired",
    "load_single_case_view_ui",
    "main",
    "preview_custom_boxes_ui",
    "render_box_overlay",
    "render_multi_box_overlay",
    "run_custom_upload_ui",
]

P = ParamSpec("P")
R = TypeVar("R")


def _workspace_payload(
    gr: Any,
    case_id: str | None,
    orig: NDArray[np.uint8] | None,
    ref: NDArray[np.uint8] | None,
    final_img: NDArray[np.uint8] | None,
    metrics_data: dict[str, Any] | None,
) -> tuple[Any, ...]:
    repaired = is_case_repaired(orig, final_img, metrics_data)
    shown_final = final_img if repaired else None
    heatmap = build_artist_heatmap(orig, shown_final) if repaired else None
    slider_val = slider_pair(orig, shown_final) if repaired else None
    return (
        case_id,
        format_look_bar(case_id),
        orig,
        ref,
        gr.Image(value=shown_final, visible=repaired),
        gr.Column(visible=repaired),
        slider_val,
        heatmap,
    )


def _build_studio_tab(gr: Any, ui_guard: Callable[..., Any]) -> None:
    """Build the clean 2-column Artist Studio with bottom `+ Add Look` pop-up."""
    _, initial_thumbs, case_ids = get_inbox_overview_data()
    first_case = case_ids[0] if case_ids else None
    init_orig, init_ref, init_final, _, _, init_met = load_single_case_view_ui(first_case)
    init_repaired = is_case_repaired(init_orig, init_final, init_met)
    init_shown_final = init_final if init_repaired else None
    init_heat = build_artist_heatmap(init_orig, init_shown_final) if init_repaired else None

    selected_case_state = gr.State(value=first_case)

    with gr.Row(elem_classes=["stitch-main-row"], equal_height=False):
        with gr.Column(scale=4, min_width=240, elem_classes=["stitch-sidebar"]):
            gr.Markdown("### Inbox")
            with gr.Row(elem_classes=["stitch-inbox-actions"], equal_height=False):
                refresh_inbox_btn = gr.Button("Refresh", size="sm", min_width=80)
                run_batch_btn = gr.Button(
                    "Run All (Batch)", variant="primary", size="sm", min_width=110
                )
            inbox_gallery = gr.Gallery(
                value=gallery_with_add_tile(initial_thumbs),
                show_label=False,
                columns=2,
                height=None,
                allow_preview=False,
                object_fit="contain",
                buttons=[],
            )

        with gr.Column(scale=8, min_width=460, elem_classes=["stitch-workspace"]):
            selected_look_bar = gr.HTML(value=format_look_bar(first_case))

            with gr.Row(elem_classes=["stitch-triad"], equal_height=False):
                with gr.Column(scale=1, min_width=120):
                    out_orig = gr.Image(
                        value=init_orig,
                        label="Original",
                        type="numpy",
                        height=225,
                        buttons=["fullscreen"],
                    )
                with gr.Column(scale=1, min_width=120):
                    out_ref = gr.Image(
                        value=init_ref,
                        label="Product Reference",
                        type="numpy",
                        height=225,
                        buttons=["fullscreen"],
                    )
                with gr.Column(scale=1, min_width=120, elem_classes=["stitch-repaired-col"]):
                    out_final = gr.Image(
                        value=init_shown_final,
                        label="Repaired",
                        type="numpy",
                        height=225,
                        visible=init_repaired,
                        buttons=["download", "fullscreen"],
                    )
                    inspect_btn = gr.Button("Fix Selected Look", variant="primary")

            with gr.Column(visible=init_repaired) as compare_section:
                with gr.Row():
                    view_mode = gr.Radio(
                        choices=[VIEW_COMPARE, VIEW_HEATMAP],
                        value=VIEW_COMPARE,
                        show_label=False,
                        interactive=True,
                    )

                with gr.Column(elem_classes=["stitch-compare-stage"]):
                    compare_slider = gr.ImageSlider(
                        value=slider_pair(init_orig, init_shown_final),
                        label="Compare (Drag slider: Original ↔ Repaired)",
                        type="numpy",
                        height=None,
                        max_height=450,
                        visible=True,
                        buttons=["fullscreen"],
                    )
                    heatmap_view = gr.Image(
                        value=init_heat,
                        label="Change Heatmap",
                        type="numpy",
                        height=450,
                        visible=False,
                        buttons=["fullscreen"],
                    )

    with (
        gr.Group(
            visible=False,
            elem_id="stitch-add-modal",
            elem_classes=["stitch-modal-backdrop"],
        ) as add_modal,
        gr.Column(elem_classes=["stitch-modal-card"]),
    ):
        with gr.Row(elem_classes=["stitch-modal-header"]):
            gr.Markdown("### Add Look")
            close_modal_btn = gr.Button("✕", size="sm", scale=0, min_width=40)
        with gr.Row():
            modal_model_in = gr.Image(label="Model Photo", type="filepath", height=280)
            modal_ref_in = gr.Image(label="Product Reference", type="filepath", height=280)

    def _toggle_compare_view(mode: str) -> tuple[Any, Any]:
        show_heatmap = mode == VIEW_HEATMAP
        return gr.ImageSlider(visible=not show_heatmap), gr.Image(visible=show_heatmap)

    def _refresh_inbox(current_case_id: str | None) -> tuple[Any, ...]:
        _, thumbs, updated_ids = get_inbox_overview_data()
        target_id = (
            current_case_id
            if current_case_id in updated_ids
            else (updated_ids[0] if updated_ids else None)
        )
        orig, ref, final_img, _, _, met = load_single_case_view_ui(target_id)
        return (
            gallery_with_add_tile(thumbs),
            *_workspace_payload(gr, target_id, orig, ref, final_img, met),
        )

    def _open_selected_thumbnail(evt: Any) -> Any:
        try:
            _, _, updated_ids = get_inbox_overview_data()
            raw_idx = getattr(evt, "index", 0)
            idx = int(raw_idx[0] if isinstance(raw_idx, (list, tuple)) else raw_idx)
            if idx == len(updated_ids):
                return (
                    gr.Group(visible=True),
                    None,
                    None,
                    *(gr.skip() for _ in range(8)),
                )
            case_id = updated_ids[idx] if 0 <= idx < len(updated_ids) else None
            orig, ref, final_img, _, _, met = load_single_case_view_ui(case_id)
            return (
                gr.Group(visible=False),
                gr.skip(),
                gr.skip(),
                *_workspace_payload(gr, case_id, orig, ref, final_img, met),
            )
        except PipelineError as exc:
            raise gr.Error(str(exc)) from exc

    _open_selected_thumbnail.__annotations__ = {"evt": gr.SelectData}

    def _handle_popup_upload(model_path: ImageInput, ref_path: ImageInput) -> tuple[Any, ...]:
        if not model_path or not ref_path:
            return tuple(gr.skip() for _ in range(12))
        new_case_id = add_pair_to_inbox(model_path, ref_path)
        _, thumbs, _ = get_inbox_overview_data()
        orig, ref, final_img, _, _, met = load_single_case_view_ui(new_case_id)
        return (
            gr.Group(visible=False),
            None,
            None,
            gallery_with_add_tile(thumbs),
            *_workspace_payload(gr, new_case_id, orig, ref, final_img, met),
        )

    def _close_popup() -> tuple[Any, None, None]:
        return gr.Group(visible=False), None, None

    def _fix_single_case(case_id: str) -> tuple[Any, ...]:
        inspect_single_case_ui(case_id, MaskingStrategy.AUTO_VISION.value)
        _, thumbs, _ = get_inbox_overview_data()
        orig, ref, final_img, _, _, met = load_single_case_view_ui(case_id)
        return (
            gallery_with_add_tile(thumbs),
            *_workspace_payload(gr, case_id, orig, ref, final_img, met),
        )

    def _run_batch_and_refresh(current_case_id: str | None) -> tuple[Any, ...]:
        _, _, _, thumbs = execute_batch_ui(MaskingStrategy.AUTO_VISION.value)
        _, _, updated_ids = get_inbox_overview_data()
        target_id = (
            current_case_id
            if current_case_id in updated_ids
            else (updated_ids[0] if updated_ids else None)
        )
        orig, ref, final_img, _, _, met = load_single_case_view_ui(target_id)
        return (
            gallery_with_add_tile(thumbs),
            *_workspace_payload(gr, target_id, orig, ref, final_img, met),
        )

    workspace_outputs = [
        selected_case_state,
        selected_look_bar,
        out_orig,
        out_ref,
        out_final,
        compare_section,
        compare_slider,
        heatmap_view,
    ]

    view_mode.change(
        fn=_toggle_compare_view,
        inputs=[view_mode],
        outputs=[compare_slider, heatmap_view],
        queue=False,
    )
    refresh_inbox_btn.click(
        fn=ui_guard(_refresh_inbox),
        inputs=[selected_case_state],
        outputs=[inbox_gallery, *workspace_outputs],
    )
    inbox_gallery.select(
        fn=_open_selected_thumbnail,
        outputs=[
            add_modal,
            modal_model_in,
            modal_ref_in,
            *workspace_outputs,
        ],
    )
    close_modal_btn.click(
        fn=_close_popup,
        outputs=[add_modal, modal_model_in, modal_ref_in],
        queue=False,
    )
    for upload_comp in (modal_model_in, modal_ref_in):
        upload_comp.change(
            fn=ui_guard(_handle_popup_upload),
            inputs=[modal_model_in, modal_ref_in],
            outputs=[
                add_modal,
                modal_model_in,
                modal_ref_in,
                inbox_gallery,
                *workspace_outputs,
            ],
        )
    inspect_btn.click(
        fn=ui_guard(_fix_single_case),
        inputs=[selected_case_state],
        outputs=[inbox_gallery, *workspace_outputs],
    )
    run_batch_btn.click(
        fn=ui_guard(_run_batch_and_refresh),
        inputs=[selected_case_state],
        outputs=[inbox_gallery, *workspace_outputs],
    )


def build_gradio_app(lifecycle: BrowserSessionLifecycle | None = None) -> Any:
    """Construct and return the clean Artist-First Gradio Studio application."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required to launch the web UI. Install via: pip install gradio>=6.0.0"
        ) from exc

    def _ui_guard(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return fn(*args, **kwargs)
            except PipelineError as exc:
                raise gr.Error(str(exc)) from exc

        return wrapper

    ensure_environment_ready()

    with gr.Blocks(title="StitchLocker Studio", fill_width=True) as demo:
        with gr.Column(elem_classes=["stitch-header"]):
            gr.Markdown("# StitchLocker Studio\nGarment Logo & Detail Correction")

        _build_studio_tab(gr, _ui_guard)

        if lifecycle is not None:
            demo.load(fn=lifecycle.on_connect, queue=False)
            demo.unload(fn=lifecycle.on_disconnect)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio Web UI for Garment Correction.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the Web UI in the default browser.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(CONFIG_PATH, base_dir=ROOT_DIR)
    shutdown_event = threading.Event()
    lifecycle = BrowserSessionLifecycle(on_shutdown=shutdown_event.set, grace_seconds=2.5)
    demo: Any = None
    demo_started = False
    try:
        if cfg.comfyui.auto_launch:
            print("[1/3] Checking ComfyUI backend...", flush=True)
            try:
                ensure_comfyui_server(cfg.comfyui)
            except ComfyExecutionError as exc:
                print(f"[WARNING] {exc}", flush=True)

        print("[2/3] Loading StitchLocker Artist Studio UI...", flush=True)
        from gradio.themes import Soft

        demo = build_gradio_app(lifecycle=lifecycle)
        theme = Soft(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
        )
        css_paths = [str(UI_CSS_PATH)] if UI_CSS_PATH.is_file() else None
        print(
            f"[3/3] Opening StitchLocker Studio at http://{args.host}:{args.port} ...",
            flush=True,
        )
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
            prevent_thread_lock=True,
            theme=theme,
            css_paths=css_paths,
            footer_links=[],
            run_history=False,
        )
        demo_started = True
        while not shutdown_event.wait(timeout=0.5):
            pass
        print("[SHUTDOWN] Browser closed — stopping ComfyUI and Web UI cleanly...")
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Interrupted — stopping ComfyUI and Web UI cleanly...")
    finally:
        lifecycle.cancel()
        if demo_started and demo is not None:
            demo.close()
        stop_auto_started_server()


if __name__ == "__main__":
    main()
