"""CV Bounding Box Refinement and Defect Heuristics for VLM Inspection."""

from __future__ import annotations

import itertools
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import DefectTarget, HeuristicConfig, NormalizedBox

_DEFAULT_HEURISTICS = HeuristicConfig()
_BOX_PAD_NORM: Final[float] = 0.025
_MIN_BOX_SPAN_NORM: Final[float] = 0.02
_INSERT_BOX_MIN_NORM: Final[float] = 0.064
_INSERT_BOX_MAX_NORM: Final[float] = 0.22
_EMBLEM_MAX_ASPECT: Final[float] = 3.0
_GARMENT_GRID_U: Final[tuple[tuple[float, float], ...]] = (
    (0.08, 0.44),
    (0.32, 0.68),
    (0.56, 0.92),
)
_GARMENT_GRID_V: Final[tuple[tuple[float, float], ...]] = (
    (0.14, 0.46),
    (0.36, 0.66),
    (0.58, 0.90),
)


def sample_studio_bg_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Estimate studio backdrop RGB from upper-left and upper-right image corners."""
    h, w = image_rgb.shape[:2]
    c_w = max(4, w // 20)
    c_h = max(4, h // 20)
    corners = np.concatenate(
        [
            image_rgb[0:c_h, 0:c_w, :].reshape(-1, 3),
            image_rgb[0:c_h, w - c_w : w, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.asarray(np.median(corners.astype(np.float32), axis=0), dtype=np.float32)


def sample_four_corner_bg_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Estimate uniform backdrop RGB from all four outer corners of an image."""
    h, w = image_rgb.shape[:2]
    c_w = max(4, w // 20)
    c_h = max(4, h // 20)
    corners = np.concatenate(
        [
            image_rgb[0:c_h, 0:c_w, :].reshape(-1, 3),
            image_rgb[0:c_h, w - c_w : w, :].reshape(-1, 3),
            image_rgb[h - c_h : h, 0:c_w, :].reshape(-1, 3),
            image_rgb[h - c_h : h, w - c_w : w, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.asarray(np.median(corners.astype(np.float32), axis=0), dtype=np.float32)


def tighten_standalone_logo_box(
    ref_rgb: NDArray[np.uint8],
    fallback_box: NormalizedBox | None = None,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> NormalizedBox:
    """Locate tight bounds of a standalone logo/graphic against its 4-corner backdrop."""
    h, w = ref_rgb.shape[:2]
    default_box = fallback_box or NormalizedBox(0.02, 0.02, 0.98, 0.98)
    if h < 16 or w < 16:
        return default_box

    bg_med = sample_four_corner_bg_rgb(ref_rgb)
    diff_bg = np.linalg.norm(ref_rgb.astype(np.float32) - bg_med.reshape(1, 1, 3), axis=2)
    thr = max(18.0, heuristics.bg_color_distance_threshold - 4.0)
    ys, xs = np.where(diff_bg > thr)
    if xs.size < 25 or ys.size < 25:
        return default_box

    pad_x = max(2, int(round(w * 0.015)))
    pad_y = max(2, int(round(h * 0.015)))
    nx1 = max(0.0, (int(xs.min()) - pad_x) / float(w))
    ny1 = max(0.0, (int(ys.min()) - pad_y) / float(h))
    nx2 = min(1.0, (int(xs.max()) + 1 + pad_x) / float(w))
    ny2 = min(1.0, (int(ys.max()) + 1 + pad_y) / float(h))

    if nx2 - nx1 < _MIN_BOX_SPAN_NORM:
        nx2 = min(1.0, nx1 + _MIN_BOX_SPAN_NORM)
        if nx2 - nx1 < _MIN_BOX_SPAN_NORM:
            nx1 = max(0.0, nx2 - _MIN_BOX_SPAN_NORM)
    if ny2 - ny1 < _MIN_BOX_SPAN_NORM:
        ny2 = min(1.0, ny1 + _MIN_BOX_SPAN_NORM)
        if ny2 - ny1 < _MIN_BOX_SPAN_NORM:
            ny1 = max(0.0, ny2 - _MIN_BOX_SPAN_NORM)
    return NormalizedBox(round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4))


def has_graphic_contrast(
    image_rgb: NDArray[np.uint8],
    box: NormalizedBox,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> bool:
    """Return True if the localized crop contains spatial graphic contrast vs. plain cloth."""
    h, w = image_rgb.shape[:2]
    px = box.to_pixel_bbox(w, h)
    patch = image_rgb[px.y1 : px.y2, px.x1 : px.x2].astype(np.float32)
    if patch.size == 0:
        return False
    spatial_std = np.std(patch, axis=(0, 1))
    return float(np.max(spatial_std)) >= heuristics.min_graphic_contrast_std


def tighten_ref_box_to_salient(
    ref_rgb: NDArray[np.uint8],
    ref_box: NormalizedBox,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> NormalizedBox:
    """Trim empty cloth and studio backdrop padding around ref_box color-agnostically."""
    h, w = ref_rgb.shape[:2]
    px = ref_box.to_pixel_bbox(w, h)
    sx1 = max(0, px.x1 - int(0.03 * w))
    sx2 = min(w, px.x2 + int(0.03 * w))
    sy1 = max(0, px.y1 - int(0.06 * h))
    sy2 = min(h, px.y2 + int(0.04 * h))
    patch = ref_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
    if patch.shape[0] < 6 or patch.shape[1] < 6:
        return ref_box

    bg_med = sample_studio_bg_rgb(ref_rgb)
    bg_thr = heuristics.bg_color_distance_threshold
    diff_bg = np.linalg.norm(patch - bg_med.reshape(1, 1, 3), axis=2)
    fg_pixels = patch[diff_bg > bg_thr]
    if fg_pixels.shape[0] >= 40:
        cloth_med = np.median(fg_pixels, axis=0)
        c_rg = float(cloth_med[0] - cloth_med[1])
        c_gb = float(cloth_med[1] - cloth_med[2])
        p_chroma = np.hypot(
            (patch[:, :, 0] - patch[:, :, 1]) - c_rg,
            (patch[:, :, 1] - patch[:, :, 2]) - c_gb,
        )
        diff_cloth = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
        chroma_mask = (p_chroma > heuristics.chroma_distance_threshold) & (diff_bg > bg_thr)
        salient_mask = (
            chroma_mask
            if int(np.count_nonzero(chroma_mask)) >= 40
            else ((diff_cloth > heuristics.cloth_color_distance_threshold) & (diff_bg > bg_thr))
        )
    else:
        border_px = np.concatenate(
            [patch[0:2, :, :].reshape(-1, 3), patch[-2:, :, :].reshape(-1, 3)], axis=0
        )
        cloth_med = np.median(border_px, axis=0)
        diff = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
        salient_mask = diff > (bg_thr + 4.0)

    ys, xs = np.where(salient_mask)
    if xs.size < 25 or ys.size < 25:
        return ref_box
    nx1 = max(0.0, (sx1 + int(xs.min()) - 4) / float(w))
    ny1 = max(0.0, (sy1 + int(ys.min()) - 4) / float(h))
    nx2 = min(1.0, (sx1 + int(xs.max()) + 4) / float(w))
    ny2 = min(1.0, (sy1 + int(ys.max()) + 4) / float(h))
    if nx2 - nx1 < 0.02 or ny2 - ny1 < 0.02:
        return ref_box
    return NormalizedBox(nx1, ny1, nx2, ny2)


def refine_connected_graphic_box(
    image_rgb: NDArray[np.uint8],
    vlm_box: NormalizedBox,
    prior_box: NormalizedBox | None,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> NormalizedBox:
    """Expand truncated VLM boxes to enclose connected graphic/signature components."""
    if prior_box is not None:
        cx = 0.5 * (vlm_box.x_min + vlm_box.x_max)
        cy = 0.5 * (vlm_box.y_min + vlm_box.y_max)
        if (
            prior_box.x_min - 0.12 <= cx <= prior_box.x_max + 0.12
            and prior_box.y_min - 0.12 <= cy <= prior_box.y_max + 0.12
        ):
            return prior_box

    padded = NormalizedBox(
        max(0.0, vlm_box.x_min - _BOX_PAD_NORM),
        max(0.0, vlm_box.y_min - _BOX_PAD_NORM),
        min(1.0, vlm_box.x_max + _BOX_PAD_NORM),
        min(1.0, vlm_box.y_max + _BOX_PAD_NORM),
    )
    h, w = image_rgb.shape[:2]
    px = padded.to_pixel_bbox(w, h)
    inner = image_rgb[px.y1 : px.y2, px.x1 : px.x2].astype(np.float32)
    if inner.shape[0] < 12 or inner.shape[1] < 12:
        return padded

    sal_thr = heuristics.connected_component_salience_threshold
    cloth_med = np.median(inner.reshape(-1, 3), axis=0)
    inner_salient = np.linalg.norm(inner - cloth_med.reshape(1, 1, 3), axis=2) > sal_thr
    if int(np.count_nonzero(inner_salient)) < 15:
        return padded
    graphic_med = np.median(inner[inner_salient], axis=0)
    if float(np.linalg.norm(graphic_med - cloth_med)) < sal_thr:
        return padded

    sy1, sy2 = px.y1, min(h, px.y2 + int(0.10 * h))
    search = image_rgb[sy1:sy2, px.x1 : px.x2].astype(np.float32)
    d_cloth = np.linalg.norm(search - cloth_med.reshape(1, 1, 3), axis=2)
    d_graphic = np.linalg.norm(search - graphic_med.reshape(1, 1, 3), axis=2)
    salient = ((d_cloth > sal_thr) & (d_graphic < d_cloth * 0.85) & (d_graphic < 85.0)).astype(
        np.uint8
    )

    k = max(5, int(round(min(h, w) * 0.022)))
    bridged = cv2.dilate(salient, np.ones((k, k), np.uint8))
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bridged)
    in_h = px.y2 - px.y1
    keep = np.zeros_like(salient)
    for lbl in range(1, num_labels):
        comp = labels == lbl
        if not np.any(comp[:in_h, :]):
            continue
        pts = (salient > 0) & comp
        _, cxs = np.where(pts)
        if cxs.size == 0 or int(cxs.min()) <= 2 or int(cxs.max()) >= search.shape[1] - 3:
            continue
        keep[pts] = 1

    ys, xs = np.where(keep > 0)
    if xs.size < 20:
        return padded
    gy1, gy2 = (sy1 + int(ys.min())) / float(h), (sy1 + int(ys.max())) / float(h)
    if gy2 <= padded.y_max - 0.005:
        return padded
    gx1, gx2 = (px.x1 + int(xs.min())) / float(w), (px.x1 + int(xs.max())) / float(w)
    nx1 = max(0.0, min(padded.x_min, gx1 - _BOX_PAD_NORM))
    ny1 = min(padded.y_max - _MIN_BOX_SPAN_NORM, max(0.0, max(padded.y_min, gy1 - 0.035)))
    nx2 = min(1.0, max(padded.x_max, gx2 + _BOX_PAD_NORM))
    ny2 = min(1.0, max(padded.y_max, gy2 + _BOX_PAD_NORM))
    if nx2 - nx1 < _MIN_BOX_SPAN_NORM or ny2 - ny1 < _MIN_BOX_SPAN_NORM:
        return padded
    return NormalizedBox(round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4))


def segment_garment_bounds(
    model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> (
    tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.uint8],
    ]
    | None
):
    """Segment matching garment silhouette bounds between Product Reference and On-Model."""
    h, w = model_rgb.shape[:2]
    rh, rw = ref_rgb.shape[:2]
    if h < 32 or w < 32 or rh < 32 or rw < 32:
        return None

    rbg = sample_studio_bg_rgb(ref_rgb)
    ref_fg_thr = max(12.0, heuristics.bg_color_distance_threshold - 4.0)
    rm = np.linalg.norm(ref_rgb.astype(np.float32) - rbg.reshape(1, 1, 3), axis=2) > ref_fg_thr
    if int(np.count_nonzero(rm)) < 100:
        return None
    ry_idx, rx_idx = np.where(rm.sum(axis=1) >= 20)[0], np.where(rm.sum(axis=0) >= 20)[0]
    if ry_idx.size < 10 or rx_idx.size < 10:
        return None
    ref_bounds = (
        int(rx_idx[0]) / float(rw),
        int(ry_idx[0]) / float(rh),
        int(rx_idx[-1]) / float(rw),
        int(ry_idx[-1]) / float(rh),
    )
    ref_cloth = np.asarray(np.median(ref_rgb[rm].astype(np.float32), axis=0), dtype=np.float32)

    mbg = sample_studio_bg_rgb(model_rgb)
    m_f = model_rgb.astype(np.float32)
    c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
    m_chroma = np.hypot((m_f[:, :, 0] - m_f[:, :, 1]) - c_rg, (m_f[:, :, 1] - m_f[:, :, 2]) - c_gb)
    diff_c = np.linalg.norm(m_f - ref_cloth.reshape(1, 1, 3), axis=2)
    diff_mbg = np.linalg.norm(m_f - mbg.reshape(1, 1, 3), axis=2)
    mm_raw = (
        (diff_mbg > 18.0)
        & (diff_c < heuristics.cloth_color_distance_threshold + 8.0)
        & (m_chroma < heuristics.chroma_distance_threshold)
    ).astype(np.uint8)
    mm = cv2.morphologyEx(mm_raw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    my_idx, mx_idx = np.where(mm.sum(axis=1) >= 30)[0], np.where(mm.sum(axis=0) >= 30)[0]
    if my_idx.size < 10 or mx_idx.size < 10:
        return None
    model_bounds = (
        int(mx_idx[0]) / float(w),
        int(my_idx[0]) / float(h),
        int(mx_idx[-1]) / float(w),
        int(my_idx[-1]) / float(h),
    )
    return ref_bounds, model_bounds, rbg, ref_cloth, np.asarray(mm, dtype=np.uint8)


def find_salient_ref_bounds(
    ref_rgb: NDArray[np.uint8],
    ref_box: NormalizedBox,
    rbg: NDArray[np.float32],
    ref_cloth: NDArray[np.float32],
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> tuple[float, float, float, float]:
    """Locate tight chromatic/high-contrast graphic bounds inside reference detail crop."""
    rh, rw = ref_rgb.shape[:2]
    rpx = ref_box.to_pixel_bbox(rw, rh)
    rp = ref_rgb[rpx.y1 : rpx.y2, rpx.x1 : rpx.x2].astype(np.float32)
    c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
    r_chroma = np.hypot((rp[:, :, 0] - rp[:, :, 1]) - c_rg, (rp[:, :, 1] - rp[:, :, 2]) - c_gb)
    r_cloth_dist = np.linalg.norm(rp - ref_cloth.reshape(1, 1, 3), axis=2)
    r_bg_dist = np.linalg.norm(rp - rbg.reshape(1, 1, 3), axis=2)
    bg_thr = heuristics.bg_color_distance_threshold
    chroma_sal = (r_chroma > heuristics.chroma_distance_threshold + 1.0) & (r_bg_dist > bg_thr)
    sal = (
        chroma_sal
        if int(np.count_nonzero(chroma_sal)) >= 40
        else ((r_cloth_dist > heuristics.cloth_color_distance_threshold) & (r_bg_dist > bg_thr))
    )
    rys, rxs = np.where(sal)
    if rxs.size < 20 or rys.size < 20:
        return ref_box.x_min, ref_box.y_min, ref_box.x_max, ref_box.y_max
    return (
        (rpx.x1 + int(rxs.min())) / float(rw),
        (rpx.y1 + int(rys.min())) / float(rh),
        (rpx.x1 + int(rxs.max())) / float(rw),
        (rpx.y1 + int(rys.max())) / float(rh),
    )


def _compute_erase_box(
    model_rgb: NDArray[np.uint8],
    m_box: NormalizedBox,
    ref_cloth: NDArray[np.float32],
    heuristics: HeuristicConfig,
) -> NormalizedBox:
    h, w = model_rgb.shape[:2]
    mpx = m_box.to_pixel_bbox(w, h)
    mp = model_rgb[mpx.y1 : mpx.y2, mpx.x1 : mpx.x2].astype(np.float32)
    c_rg, c_gb = float(ref_cloth[0] - ref_cloth[1]), float(ref_cloth[1] - ref_cloth[2])
    m_chroma = np.hypot((mp[:, :, 0] - mp[:, :, 1]) - c_rg, (mp[:, :, 1] - mp[:, :, 2]) - c_gb)
    m_cloth_dist = np.linalg.norm(mp - ref_cloth.reshape(1, 1, 3), axis=2)
    mys, mxs = np.where(
        (m_chroma > heuristics.chroma_distance_threshold + 3.0)
        | (m_cloth_dist > heuristics.cloth_color_distance_threshold + 4.0)
    )
    if mxs.size < 40:
        return m_box
    ex1 = max(m_box.x_min + 0.012, (mpx.x1 + int(np.percentile(mxs, 2))) / w - 0.015)
    ey1 = max(m_box.y_min, (mpx.y1 + int(np.percentile(mys, 5))) / h - 0.015)
    ex2 = min(m_box.x_max, (mpx.x1 + int(np.percentile(mxs, 98))) / w + 0.015)
    ey2 = min(m_box.y_max, (mpx.y1 + int(np.percentile(mys, 98))) / h + 0.015)
    if ex2 - ex1 >= 0.03 and ey2 - ey1 >= 0.03:
        return NormalizedBox(round(ex1, 4), round(ey1, 4), round(ex2, 4), round(ey2, 4))
    return m_box


def split_misplaced_defect_if_shifted(
    model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    defects: list[DefectTarget],
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> list[DefectTarget]:
    """Split a misplaced lower/upper logo defect into separate ERASE + INSERT targets."""
    defect = defects[0]
    seg = segment_garment_bounds(model_rgb, ref_rgb, heuristics=heuristics)
    if seg is None:
        return defects
    ref_bounds, model_bounds, rbg, ref_cloth, mm = seg
    rx1, ry1, rx2, ry2 = ref_bounds
    mx1, my1, mx2, my2 = model_bounds

    m_box, r_box = defect.model_target_box, defect.reference_detail_box
    true_rx1, true_ry1, true_rx2, true_ry2 = find_salient_ref_bounds(
        ref_rgb, r_box, rbg, ref_cloth, heuristics=heuristics
    )
    m_cx_raw = 0.5 * (m_box.x_min + m_box.x_max)
    r_cx = 0.5 * (true_rx1 + true_rx2)
    rel_rx = (r_cx - rx1) / max(0.1, rx2 - rx1)
    expected_mcx = mx1 + rel_rx * (mx2 - mx1)

    m_cy = 0.5 * (m_box.y_min + m_box.y_max)
    r_cy = 0.5 * (true_ry1 + true_ry2)
    rel_ry = (r_cy - ry1) / max(0.1, ry2 - ry1)
    expected_mcy = my1 + rel_ry * (my2 - my1)

    y_shifted = abs(m_cy - r_cy) > heuristics.shift_y_threshold and abs(m_cy - expected_mcy) > 0.04
    x_shifted = (
        abs(m_cx_raw - r_cx) > heuristics.shift_x_threshold and abs(m_cx_raw - expected_mcx) > 0.12
    )
    if not y_shifted and not x_shifted:
        return defects

    erase_box = _compute_erase_box(model_rgb, m_box, ref_cloth, heuristics)
    ref_erase_box = NormalizedBox(
        round(max(0.0, min(0.88, true_rx1)), 4),
        round(max(0.0, min(0.80, m_box.y_min)), 4),
        round(min(1.0, max(true_rx1 + 0.05, min(0.95, true_rx1 + 0.10))), 4),
        round(min(1.0, max(min(0.80, m_box.y_min) + 0.05, min(0.92, m_box.y_max))), 4),
    )
    erase_target = DefectTarget(
        defect_summary="Erase misplaced graphic from wrong garment zone",
        model_target_box=erase_box,
        reference_detail_box=ref_erase_box,
        edit_prompt=(
            "Edit <image1>: Remove and erase the misplaced logo patch completely, "
            "leaving clean seamless plain garment fabric matching <image2>."
        ),
    )

    raw_w = (true_rx2 - true_rx1) * (mx2 - mx1) / max(0.1, rx2 - rx1) + 0.026
    raw_h = (true_ry2 - true_ry1) * (my2 - my1) / max(0.1, ry2 - ry1) + 0.030
    ins_w = min(_INSERT_BOX_MAX_NORM, max(_INSERT_BOX_MIN_NORM, raw_w))
    ins_h = min(_INSERT_BOX_MAX_NORM, max(_INSERT_BOX_MIN_NORM, raw_h))
    if not y_shifted:
        target_cy = max(my1 + 0.10 * (my2 - my1), min(my2 - 0.10 * (my2 - my1), expected_mcy))
    elif m_cy > expected_mcy:
        min_allowed_cy = my1 + 0.15 * (my2 - my1)
        clear_above_cy = max(min_allowed_cy, erase_box.y_min - 0.5 * ins_h + 0.012)
        target_cy = min(expected_mcy, clear_above_cy)
    else:
        max_allowed_cy = my2 - 0.15 * (my2 - my1)
        clear_below_cy = min(max_allowed_cy, erase_box.y_max + 0.5 * ins_h - 0.012)
        target_cy = max(expected_mcy, clear_below_cy)

    h, w = model_rgb.shape[:2]
    iy1 = max(0.0, min(1.0 - ins_h, target_cy - 0.5 * ins_h))
    iy2 = min(1.0, iy1 + ins_h)
    if x_shifted:
        ix1 = max(mx1 + 0.02, min(mx2 - ins_w - 0.02, expected_mcx - 0.5 * ins_w))
    else:
        m_cx = 0.5 * (erase_box.x_min + erase_box.x_max)
        ix1 = max(m_box.x_min + 0.025, m_cx - 0.5 * ins_w)
        y_band = mm[int(iy1 * h) : max(int(iy1 * h) + 1, int(iy2 * h)), :]
        cloth_cols = np.where(y_band.mean(axis=0) > 0.5)[0]
        half_mid = 0.5 * (mx1 + mx2) * w
        side_cols = (
            cloth_cols[cloth_cols < half_mid]
            if m_cx * w < half_mid
            else cloth_cols[cloth_cols >= half_mid]
        )
        if side_cols.size >= 10:
            edge_x = int(side_cols[0]) / float(w)
            ix1 = max(ix1, min(int(side_cols[-1]) / float(w) - ins_w, edge_x - 0.025))
    ix1 = max(0.0, min(1.0 - ins_w, ix1))
    ix2 = min(1.0, ix1 + ins_w)

    tight_ref_box = NormalizedBox(
        round(max(0.0, true_rx1 - 0.015), 4),
        round(max(0.0, true_ry1 - 0.015), 4),
        round(min(1.0, true_rx2 + 0.015), 4),
        round(min(1.0, true_ry2 + 0.015), 4),
    )
    insert_target = DefectTarget(
        defect_summary="Insert authentic logo detail at aligned garment zone",
        model_target_box=NormalizedBox(round(ix1, 4), round(iy1, 4), round(ix2, 4), round(iy2, 4)),
        reference_detail_box=tight_ref_box,
        edit_prompt=(
            "Edit <image1>: Add ONLY the authentic small detail from Image 2 onto this spot, "
            "matching <image2> in size and design."
        ),
    )
    return [erase_target, insert_target]


def compute_foreground_mask(
    patch: NDArray[np.float32], bg_med: NDArray[np.float32], bg_thr: float
) -> NDArray[np.bool_]:
    """Exclude only border-connected backdrop components so interior white logos stay foreground."""
    bg_raw = (np.linalg.norm(patch - bg_med.reshape(1, 1, 3), axis=2) <= bg_thr).astype(np.uint8)
    _, labels = cv2.connectedComponents(bg_raw)
    border_vals = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_labels = [int(lbl) for lbl in np.unique(border_vals) if int(lbl) != 0]
    if not border_labels:
        return np.ones(patch.shape[:2], dtype=np.bool_)
    return ~np.isin(labels, border_labels)


_compute_foreground_mask = compute_foreground_mask


def find_salient_emblem_box(
    image_rgb: NDArray[np.uint8],
    hint_box: NormalizedBox,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
    garment_bounds: tuple[float, float, float, float] | None = None,
) -> NormalizedBox | None:
    """Locate a compact high-contrast garment emblem inside the candidate zone."""
    h, w = image_rgb.shape[:2]
    if h < 32 or w < 32:
        return None
    bg_med = sample_studio_bg_rgb(image_rgb)
    px = hint_box.to_pixel_bbox(w, h)
    pad_ratio = 0.03 if garment_bounds is not None else 0.06
    gx1, gy1, gx2, gy2 = (
        (
            max(0, int(round(garment_bounds[0] * w))),
            max(0, int(round(garment_bounds[1] * h))),
            min(w, int(round(garment_bounds[2] * w))),
            min(h, int(round(garment_bounds[3] * h))),
        )
        if garment_bounds is not None
        else (0, 0, w, h)
    )
    sx1, sx2 = max(gx1, px.x1 - int(pad_ratio * w)), min(gx2, px.x2 + int(pad_ratio * w))
    sy1, sy2 = max(gy1, px.y1 - int(pad_ratio * h)), min(gy2, px.y2 + int(pad_ratio * h))
    patch = image_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        return None

    fg_mask = _compute_foreground_mask(patch, bg_med, heuristics.bg_color_distance_threshold)
    if int(np.count_nonzero(fg_mask)) < 100:
        return None
    cloth_med = np.median(patch[fg_mask], axis=0)
    diff = np.linalg.norm(patch - cloth_med.reshape(1, 1, 3), axis=2)
    thr = max(75.0, float(np.percentile(diff[fg_mask], 98.5)))
    salient = ((diff >= thr) & fg_mask).astype(np.uint8)

    k = max(3, int(round(min(h, w) * 0.005)))
    bridged = cv2.dilate(salient, np.ones((k, k), np.uint8))
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bridged)

    best_box: NormalizedBox | None = None
    best_score = -1.0
    ph, pw = patch.shape[:2]
    max_pts = int(0.12 * ph * pw)
    for lbl in range(1, num_labels):
        pts = (salient > 0) & (labels == lbl)
        count = int(np.count_nonzero(pts))
        if count < 15 or count > max_pts:
            continue
        ys, xs = np.where(pts)
        if (
            int(xs.min()) <= 2
            or int(ys.min()) <= 2
            or int(xs.max()) >= pw - 3
            or int(ys.max()) >= ph - 3
        ):
            continue
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if max(bw / max(1.0, float(bh)), bh / max(1.0, float(bw))) > _EMBLEM_MAX_ASPECT:
            continue
        score = float(count) * float(diff[pts].mean())
        if score > best_score:
            best_score = score
            nx1, ny1 = (sx1 + int(xs.min())) / float(w), (sy1 + int(ys.min())) / float(h)
            nx2, ny2 = (sx1 + int(xs.max())) / float(w), (sy1 + int(ys.max())) / float(h)
            if nx2 - nx1 >= _MIN_BOX_SPAN_NORM or ny2 - ny1 >= _MIN_BOX_SPAN_NORM:
                best_box = NormalizedBox(
                    round(max(0.0, nx1), 4),
                    round(max(0.0, ny1), 4),
                    round(min(1.0, max(nx1 + _MIN_BOX_SPAN_NORM, nx2)), 4),
                    round(min(1.0, max(ny1 + _MIN_BOX_SPAN_NORM, ny2)), 4),
                )
    return best_box


def emblem_pattern_correlation(
    model_rgb: NDArray[np.uint8],
    m_box: NormalizedBox,
    ref_rgb: NDArray[np.uint8],
    r_box: NormalizedBox,
) -> float:
    """Return normalized 64x64 contrast-map correlation between two small emblems."""

    def _norm_patch(img: NDArray[np.uint8], box: NormalizedBox) -> NDArray[np.float32]:
        h, w = img.shape[:2]
        px = box.to_pixel_bbox(w, h)
        pad = 3
        crop = img[
            max(0, px.y1 - pad) : min(h, px.y2 + pad),
            max(0, px.x1 - pad) : min(w, px.x2 + pad),
        ].astype(np.float32)
        cloth = np.median(crop.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(crop - cloth.reshape(1, 1, 3), axis=2)
        norm = (dist - float(dist.min())) / max(1.0, float(dist.max() - dist.min()))
        return np.asarray(
            cv2.resize(norm.astype(np.float32), (64, 64), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )

    pm = _norm_patch(model_rgb, m_box)
    pr = _norm_patch(ref_rgb, r_box)
    if float(np.std(pm)) < 1e-4 or float(np.std(pr)) < 1e-4:
        return 1.0
    return float(np.corrcoef(pm.ravel(), pr.ravel())[0, 1])


def _map_relative_zone(
    bounds: tuple[float, float, float, float], u1: float, v1: float, u2: float, v2: float
) -> NormalizedBox | None:
    bx1, by1, bx2, by2 = bounds
    span_x = bx2 - bx1
    span_y = by2 - by1
    x1 = max(0.0, min(1.0, bx1 + u1 * span_x))
    y1 = max(0.0, min(1.0, by1 + v1 * span_y))
    x2 = max(0.0, min(1.0, bx1 + u2 * span_x))
    y2 = max(0.0, min(1.0, by1 + v2 * span_y))
    if x2 - x1 < _MIN_BOX_SPAN_NORM or y2 - y1 < _MIN_BOX_SPAN_NORM:
        return None
    return NormalizedBox(x1, y1, x2, y2)


def _build_garment_scan_zones(
    model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    heuristics: HeuristicConfig,
) -> tuple[
    list[tuple[NormalizedBox, NormalizedBox]],
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    """Build garment-silhouette-relative 3x3 zone pairs (or interior full-frame fallback)."""
    seg = segment_garment_bounds(model_rgb, ref_rgb, heuristics=heuristics)
    if seg is None:
        m_bounds: tuple[float, float, float, float] | None = None
        r_bounds: tuple[float, float, float, float] | None = None
        eval_m_bounds = (0.15, 0.15, 0.85, 0.85)
        eval_r_bounds = (0.15, 0.15, 0.85, 0.85)
    else:
        r_bounds, m_bounds, _, _, _ = seg
        eval_m_bounds = m_bounds
        eval_r_bounds = r_bounds

    pairs: list[tuple[NormalizedBox, NormalizedBox]] = []
    for (v1, v2), (u1, u2) in itertools.product(_GARMENT_GRID_V, _GARMENT_GRID_U):
        mz = _map_relative_zone(eval_m_bounds, u1, v1, u2, v2)
        rz = _map_relative_zone(eval_r_bounds, u1, v1, u2, v2)
        if mz is not None and rz is not None:
            pairs.append((mz, rz))
    return pairs, m_bounds, r_bounds


def _emblem_patch_salience(image_rgb: NDArray[np.uint8], box: NormalizedBox) -> tuple[float, float]:
    """Return (95th-percentile RGB diff from local cloth median, patch RGB std)."""
    h, w = image_rgb.shape[:2]
    px = box.to_pixel_bbox(w, h)
    pad = 4
    crop = image_rgb[
        max(0, px.y1 - pad) : min(h, px.y2 + pad),
        max(0, px.x1 - pad) : min(w, px.x2 + pad),
    ].astype(np.float32)
    if crop.size == 0:
        return 0.0, 0.0
    cloth = np.median(crop.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(crop - cloth.reshape(1, 1, 3), axis=2)
    return float(np.percentile(dist, 95)), float(np.std(crop))


def detect_micro_emblem_defect(
    model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    source_label: str,
    heuristics: HeuristicConfig = _DEFAULT_HEURISTICS,
) -> tuple[str, NormalizedBox, NormalizedBox, str] | None:
    """Detect small emblem mismatches dynamically across garment-interior zones.

    Returns (defect_summary, model_box, ref_box, edit_prompt) or None if intact.
    """
    _ = source_label
    zone_pairs, m_bounds, r_bounds = _build_garment_scan_zones(model_rgb, ref_rgb, heuristics)
    best_match: tuple[float, float, NormalizedBox, NormalizedBox] | None = None
    for m_zone, r_zone in zone_pairs:
        m_emblem = find_salient_emblem_box(
            model_rgb, m_zone, heuristics=heuristics, garment_bounds=m_bounds
        )
        r_emblem = find_salient_emblem_box(
            ref_rgb, r_zone, heuristics=heuristics, garment_bounds=r_bounds
        )
        if m_emblem is None or r_emblem is None:
            continue
        m_p95, m_std = _emblem_patch_salience(model_rgb, m_emblem)
        r_p95, r_std = _emblem_patch_salience(ref_rgb, r_emblem)
        min_p95 = min(m_p95, r_p95)
        min_std = min(m_std, r_std)
        # Reject faint plaid texture, fabric folds, and sleeve-seam highlights
        if min_p95 < 120.0 or min_std < 34.0:
            continue
        corr = emblem_pattern_correlation(model_rgb, m_emblem, ref_rgb, r_emblem)
        if corr < heuristics.emblem_correlation_threshold:
            salience_score = min_p95 * min_std * (1.0 - max(-0.5, corr))
            if best_match is None or salience_score > best_match[0]:
                best_match = (salience_score, corr, m_emblem, r_emblem)

    if best_match is None:
        return None
    _, corr, m_emblem, r_emblem = best_match

    m_w = m_emblem.x_max - m_emblem.x_min
    m_h = m_emblem.y_max - m_emblem.y_min
    r_w = r_emblem.x_max - r_emblem.x_min
    r_h = r_emblem.y_max - r_emblem.y_min
    box_w = max(m_w, r_w * 0.78) + 0.020
    box_h = max(m_h, r_h * 0.78) + 0.020
    m_cx = 0.5 * (m_emblem.x_min + m_emblem.x_max)
    m_cy = 0.5 * (m_emblem.y_min + m_emblem.y_max)

    model_box = NormalizedBox(
        round(max(0.0, m_cx - 0.5 * box_w), 4),
        round(max(0.0, m_cy - 0.5 * box_h), 4),
        round(min(1.0, m_cx + 0.5 * box_w), 4),
        round(min(1.0, m_cy + 0.5 * box_h), 4),
    )
    ref_box = NormalizedBox(
        round(max(0.0, r_emblem.x_min - 0.012), 4),
        round(max(0.0, r_emblem.y_min - 0.012), 4),
        round(min(1.0, r_emblem.x_max + 0.012), 4),
        round(min(1.0, r_emblem.y_max + 0.012), 4),
    )
    prompt = (
        "Edit <image1>: Replace the defective emblem with ONLY the exact authentic emblem "
        "from <image2>, restoring all fine geometric details, internal cutouts, and "
        "typography to match <image2> in position and size."
    )
    zone_label = "chest emblem" if m_cy < 0.50 else "garment emblem"
    summary = f"Small {zone_label} structural mismatch detected (corr={corr:.2f})"
    return summary, model_box, ref_box, prompt
