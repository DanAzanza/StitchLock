"""2.5D Garment Surface Fitting, Cylindrical Wrap, Fold Displacement & Relighting."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import LocalizationError, PixelBBox
from src.vlm_heuristics import (
    compute_foreground_mask,
    sample_four_corner_bg_rgb,
    sample_studio_bg_rgb,
)


def _clamp_bbox_to_image(bbox: PixelBBox, width: int, height: int) -> PixelBBox | None:
    """Clamp bbox to image bounds; return None when intersection is empty."""
    x1 = max(0, min(width, bbox.x1))
    y1 = max(0, min(height, bbox.y1))
    x2 = max(0, min(width, bbox.x2))
    y2 = max(0, min(height, bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return PixelBBox(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True)
class GarmentSurfaceGeometry:
    """Estimated 2.5D cloth surface parameters across a target bounding box."""

    skew: float
    baseline_slope: float
    cyl_center_x: float
    cyl_radius: float
    arc_strength: float


def estimate_garment_perspective_skew(
    model_rgb: NDArray[np.uint8],
    target_bbox: PixelBBox,
) -> float:
    """Estimate horizontal 3/4 garment surface foreshortening (+ right recedes, - left)."""
    h, w = model_rgb.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    bg_med = sample_studio_bg_rgb(model_rgb)
    band = model_rgb[target_bbox.y1 : target_bbox.y2, :, :].astype(np.float32)
    if band.size == 0:
        return 0.0
    fg = compute_foreground_mask(band, bg_med, 24.0)
    fg_cols = np.where(fg.mean(axis=0) > 0.25)[0]
    if fg_cols.size < max(16, int(round(w * 0.12))):
        return 0.0
    gx1, gx2 = int(fg_cols[0]), int(fg_cols[-1])
    span = float(gx2 - gx1)
    if span < w * 0.12:
        return 0.0
    t_cx = 0.5 * (target_bbox.x1 + target_bbox.x2)
    rel_cx = (t_cx - gx1) / span
    if 0.42 <= rel_cx <= 0.58:
        return 0.0
    return float(np.clip((rel_cx - 0.5) * 0.90, -0.26, 0.26))


def _estimate_baseline_slope(
    original_model_rgb: NDArray[np.uint8] | None,
    clean_model_rgb: NDArray[np.uint8],
    clamped_bbox: PixelBBox,
    fg_patch: NDArray[np.bool_],
) -> float:
    """Robustly fit dy/dx surface slope from erased print pixels inside clamped_bbox."""
    if original_model_rgb is None or original_model_rgb.shape != clean_model_rgb.shape:
        return 0.0

    orig_patch = original_model_rgb[
        clamped_bbox.y1 : clamped_bbox.y2, clamped_bbox.x1 : clamped_bbox.x2
    ].astype(np.float32)
    clean_patch = clean_model_rgb[
        clamped_bbox.y1 : clamped_bbox.y2, clamped_bbox.x1 : clamped_bbox.x2
    ].astype(np.float32)
    delta = np.linalg.norm(orig_patch - clean_patch, axis=2)
    if float(delta.max(initial=0.0)) < 22.0 or delta.shape != fg_patch.shape:
        return 0.0

    fg_delta = delta[fg_patch] if np.any(fg_patch) else delta.ravel()
    thr = max(18.0, float(np.percentile(fg_delta, 92)) * 0.45)
    active = (delta >= thr) & fg_patch

    bw = active.shape[1]
    x_start = int(round(bw * 0.08))
    x_end = int(round(bw * 0.90))
    cols: list[tuple[float, float, float]] = []
    for x in range(x_start, x_end):
        ys = np.where(active[:, x])[0]
        if ys.size >= 4:
            y_lo = float(np.percentile(ys, 10))
            y_hi = float(np.percentile(ys, 90))
            cols.append((float(x), 0.5 * (y_lo + y_hi), max(1.0, y_hi - y_lo)))

    if len(cols) < 16:
        return 0.0

    heights = np.asarray([c[2] for c in cols], dtype=np.float32)
    med_h = max(8.0, float(np.median(heights)))
    filtered = [c for c in cols if 0.55 * med_h <= c[2] <= 1.38 * med_h]
    use_cols = filtered if len(filtered) >= 12 else cols

    xs_arr = np.asarray([c[0] for c in use_cols], dtype=np.float32)
    ys_arr = np.asarray([c[1] for c in use_cols], dtype=np.float32)
    if xs_arr.size > 48:
        idx = np.linspace(0, xs_arr.size - 1, 48, dtype=np.int32)
        xs_arr = xs_arr[idx]
        ys_arr = ys_arr[idx]

    # Theil-Sen pairwise median slope over separated columns for occlusion robustness
    min_dx = max(8.0, bw * 0.18)
    dx_mat = xs_arr[None, :] - xs_arr[:, None]
    dy_mat = ys_arr[None, :] - ys_arr[:, None]
    valid = dx_mat >= min_dx
    if not np.any(valid):
        return 0.0

    raw_slope = float(np.median(dy_mat[valid] / dx_mat[valid]))
    return float(np.clip(raw_slope, -0.24, 0.24))


def fit_garment_surface_geometry(
    *,
    clean_model_rgb: NDArray[np.uint8],
    target_bbox: PixelBBox,
    perspective_skew: float,
    original_model_rgb: NDArray[np.uint8] | None = None,
) -> GarmentSurfaceGeometry:
    """Fit torso cylinder curvature and surface slope around target_bbox."""
    h, w = clean_model_rgb.shape[:2]
    clamped = _clamp_bbox_to_image(target_bbox, w, h)
    arc_strength = (
        float(np.clip(abs(perspective_skew) / 0.18, 0.35, 0.88))
        if abs(perspective_skew) >= 0.04
        else 0.0
    )
    if clamped is None:
        return GarmentSurfaceGeometry(
            skew=float(perspective_skew),
            baseline_slope=0.0,
            cyl_center_x=0.5 * float(w),
            cyl_radius=max(24.0, 0.5 * float(w)),
            arc_strength=arc_strength,
        )

    bg_med = sample_studio_bg_rgb(clean_model_rgb)
    band = clean_model_rgb[clamped.y1 : clamped.y2, :, :].astype(np.float32)
    fg_band = compute_foreground_mask(band, bg_med, 24.0)
    fg_cols = np.where(fg_band.mean(axis=0) > 0.25)[0]

    if fg_cols.size >= max(16, int(round(w * 0.12))):
        gx1, gx2 = int(fg_cols[0]), int(fg_cols[-1])
        span = float(gx2 - gx1)
        # Anchor torso limb cylinder on the receding outer edge so an extended opposite sleeve
        # does not over-flatten local curvature.
        eff_radius = max(float(clamped.width) * 0.72, min(0.5 * span, 0.42 * span))
        if perspective_skew < 0:
            cyl_center_x = float(gx1) + eff_radius
        elif perspective_skew > 0:
            cyl_center_x = float(gx2) - eff_radius
        else:
            cyl_center_x = float(gx1) + 0.5 * span
            eff_radius = 0.5 * span
    else:
        cyl_center_x = 0.5 * float(w)
        eff_radius = max(32.0, 0.5 * float(w))

    fg_patch = fg_band[:, clamped.x1 : clamped.x2]
    baseline_slope = _estimate_baseline_slope(
        original_model_rgb=original_model_rgb,
        clean_model_rgb=clean_model_rgb,
        clamped_bbox=clamped,
        fg_patch=fg_patch,
    )
    return GarmentSurfaceGeometry(
        skew=float(perspective_skew),
        baseline_slope=baseline_slope,
        cyl_center_x=cyl_center_x,
        cyl_radius=max(24.0, eff_radius),
        arc_strength=arc_strength,
    )


def _normalized_cloth_blur(
    lum: NDArray[np.float32],
    fg_float: NDArray[np.float32],
    sigma: float,
) -> NDArray[np.float32]:
    """Blur luminance strictly over foreground garment pixels to prevent backdrop bleed."""
    num = np.asarray(cv2.GaussianBlur(lum * fg_float, (0, 0), sigmaX=sigma, sigmaY=sigma))
    den = np.asarray(cv2.GaussianBlur(fg_float, (0, 0), sigmaX=sigma, sigmaY=sigma))
    has_fg = bool(np.any(fg_float > 0.5))
    fallback = float(np.median(lum[fg_float > 0.5])) if has_fg else float(lum.mean())
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback).astype(np.float32)


def _compute_fold_displacement_and_shading(
    *,
    clean_model_rgb: NDArray[np.uint8],
    ox1: int,
    oy1: int,
    placed_w: int,
    placed_h: int,
    geometry: GarmentSurfaceGeometry,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Derive smooth cloth-fold displacement (dx, dy), shading, and fine weave texture."""
    img_h, img_w = clean_model_rgb.shape[:2]
    margin = max(12, min(48, int(round(max(placed_w, placed_h) * 0.18))))
    rx1 = max(0, ox1 - margin)
    ry1 = max(0, oy1 - margin)
    rx2 = min(img_w, ox1 + placed_w + margin)
    ry2 = min(img_h, oy1 + placed_h + margin)
    if rx2 <= rx1 or ry2 <= ry1:
        zeros = np.zeros((placed_h, placed_w), dtype=np.float32)
        ones = np.ones((placed_h, placed_w), dtype=np.float32)
        return zeros, zeros, ones, zeros

    patch = clean_model_rgb[ry1:ry2, rx1:rx2].astype(np.float32)
    lum = np.asarray(
        np.float32(0.2126) * patch[..., 0]
        + np.float32(0.7152) * patch[..., 1]
        + np.float32(0.0722) * patch[..., 2],
        dtype=np.float32,
    )
    bg_med = sample_studio_bg_rgb(clean_model_rgb)
    band_full = clean_model_rgb[ry1:ry2, :, :].astype(np.float32)
    fg_mask = compute_foreground_mask(band_full, bg_med, 22.0)[:, rx1:rx2]
    if int(np.count_nonzero(fg_mask)) < 32:
        fg_mask = np.ones(lum.shape, dtype=bool)
    fg_float = fg_mask.astype(np.float32)

    lum_fg = lum[fg_mask]
    dyn_range = float(np.percentile(lum_fg, 90) - np.percentile(lum_fg, 10))
    if dyn_range < 2.0:
        zeros = np.zeros((placed_h, placed_w), dtype=np.float32)
        ones = np.ones((placed_h, placed_w), dtype=np.float32)
        return zeros, zeros, ones, zeros

    sigma_irrad = max(2.2, placed_w * 0.014)
    sigma_fold = max(3.5, placed_w * 0.024)
    sigma_macro = max(14.0, placed_w * 0.14)

    l_smooth = _normalized_cloth_blur(lum, fg_float, sigma_irrad)
    l_fold_base = _normalized_cloth_blur(lum, fg_float, sigma_fold)
    l_macro = _normalized_cloth_blur(lum, fg_float, sigma_macro)
    fold_band = (l_fold_base - l_macro) * fg_float
    fine_weave = (lum - l_smooth) * fg_float

    gx_fold = np.asarray(cv2.Sobel(fold_band, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32) * 0.25
    gy_fold = np.asarray(cv2.Sobel(fold_band, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32) * 0.25

    fold_abs_p90 = float(np.percentile(np.abs(fold_band[fg_mask]), 90))
    norm_scale = max(14.0, fold_abs_p90 * 2.2)
    max_disp = min(14.0, max(2.5, placed_h * 0.075))
    skew_sign = -1.0 if geometry.skew < 0 else (1.0 if geometry.skew > 0 else 0.0)

    dx_full = max_disp * np.tanh((0.55 * skew_sign * fold_band + 2.4 * gx_fold) / norm_scale)
    dy_full = (0.75 * max_disp) * np.tanh((fold_band + 1.4 * gy_fold) / norm_scale)
    dx_full *= fg_float
    dy_full *= fg_float

    l_lit = float(np.percentile(l_smooth[fg_mask], 88))
    rel_light = l_smooth / max(12.0, l_lit)
    shade_full = np.clip(np.power(np.clip(rel_light, 0.35, 1.15), 0.75), 0.48, 1.06)

    # Extract region matching [ox1:ox1+placed_w, oy1:oy1+placed_h] cleanly (clamp-padded if OOB)
    py_idx = np.clip(np.arange(oy1, oy1 + placed_h) - ry1, 0, ry2 - ry1 - 1)
    px_idx = np.clip(np.arange(ox1, ox1 + placed_w) - rx1, 0, rx2 - rx1 - 1)
    dx_patch = dx_full[np.ix_(py_idx, px_idx)].astype(np.float32)
    dy_patch = dy_full[np.ix_(py_idx, px_idx)].astype(np.float32)
    shade_patch = shade_full[np.ix_(py_idx, px_idx)].astype(np.float32)
    weave_patch = fine_weave[np.ix_(py_idx, px_idx)].astype(np.float32)
    return dx_patch, dy_patch, shade_patch, weave_patch


def _build_surface_uv_grid(
    *,
    ox1: int,
    oy1: int,
    placed_w: int,
    placed_h: int,
    crop_w: int,
    crop_h: int,
    geometry: GarmentSurfaceGeometry,
    dx_fold: NDArray[np.float32],
    dy_fold: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Build inverse surface map (map_x, map_y) and cylinder limb shading factor."""
    gx_local, gy_local = np.meshgrid(
        np.arange(placed_w, dtype=np.float32),
        np.arange(placed_h, dtype=np.float32),
    )
    px_surf = float(ox1) + gx_local - dx_fold
    py_surf = float(oy1) + gy_local - dy_fold

    u_lin = (px_surf - float(ox1)) / max(1.0, float(placed_w - 1))

    theta_left = float(
        np.arcsin(np.clip((float(ox1) - geometry.cyl_center_x) / geometry.cyl_radius, -0.94, 0.94))
    )
    theta_right = float(
        np.arcsin(
            np.clip(
                (float(ox1 + placed_w - 1) - geometry.cyl_center_x) / geometry.cyl_radius,
                -0.94,
                0.94,
            )
        )
    )
    theta_span = theta_right - theta_left
    theta = np.arcsin(np.clip((px_surf - geometry.cyl_center_x) / geometry.cyl_radius, -0.94, 0.94))

    if geometry.arc_strength > 0.0 and theta_span > 0.08:
        u_cyl = (theta - theta_left) / theta_span
        u_surf = (1.0 - geometry.arc_strength) * u_lin + geometry.arc_strength * u_cyl
    else:
        u_surf = u_lin

    s = abs(geometry.skew)
    ox_mid = float(ox1) + 0.5 * float(placed_w - 1)
    oy_mid = float(oy1) + 0.5 * float(placed_h - 1)

    # Local vertical center tracks garment baseline slope + gentle pitch curve on curved back
    y_slope_shift = geometry.baseline_slope * (px_surf - ox_mid)
    y_center_local = oy_mid + y_slope_shift

    # Perspective height foreshortening: narrower on receding far side, taller on near side
    if geometry.skew < 0:
        h_scale = (1.0 - 0.85 * s) + (1.00 * s) * u_surf
    elif geometry.skew > 0:
        h_scale = (1.0 + 0.15 * s) - (1.00 * s) * u_surf
    else:
        h_scale = np.ones_like(u_surf)
    local_half_h = 0.5 * float(placed_h) * np.clip(h_scale, 0.55, 1.35)

    # Subtle vertical shear aligned with tilted surface slope
    shear_u = np.clip(-0.18 * geometry.baseline_slope, -0.05, 0.05) * (
        (py_surf - y_center_local) / max(1.0, float(placed_w))
    )
    u_src = u_surf + shear_u
    v_src = 0.5 + (py_surf - y_center_local) / (2.0 * local_half_h)

    map_x = (u_src * float(crop_w)).astype(np.float32)
    map_y = (v_src * float(crop_h)).astype(np.float32)

    limb_factor = (
        (1.0 - 0.18 * geometry.arc_strength) + 0.18 * geometry.arc_strength * np.cos(theta)
    ).astype(np.float32)
    return map_x, map_y, limb_factor


def project_logo_onto_garment_surface(
    *,
    clean_model_rgb: NDArray[np.uint8],
    ref_rgb: NDArray[np.uint8],
    target_bbox: PixelBBox,
    ref_detail_bbox: PixelBBox,
    perspective_skew: float = 0.0,
    binary_mask: NDArray[np.uint8] | None = None,
    original_model_rgb: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Project a standalone logo onto clean fabric with 2.5D curvature, folds, and relighting."""
    img_h, img_w = clean_model_rgb.shape[:2]
    if binary_mask is not None and binary_mask.shape[:2] != (img_h, img_w):
        raise LocalizationError(
            f"binary_mask shape {binary_mask.shape[:2]} does not match "
            f"clean_model_rgb shape {(img_h, img_w)}."
        )

    clamped_target = _clamp_bbox_to_image(target_bbox, img_w, img_h)
    if clamped_target is None:
        return clean_model_rgb.copy()

    logo_crop = ref_rgb[
        ref_detail_bbox.y1 : ref_detail_bbox.y2,
        ref_detail_bbox.x1 : ref_detail_bbox.x2,
    ]
    if logo_crop.size == 0:
        return clean_model_rgb.copy()

    bg_med = sample_four_corner_bg_rgb(ref_rgb)
    dist = np.linalg.norm(logo_crop.astype(np.float32) - bg_med.reshape(1, 1, 3), axis=2)
    alpha = np.clip((dist - 18.0) / 26.0, 0.0, 1.0).astype(np.float32)
    alpha = np.asarray(cv2.GaussianBlur(alpha, (3, 3), 0.6), dtype=np.float32)

    crop_h, crop_w = logo_crop.shape[:2]
    pad_x = min(clamped_target.width // 4, max(4, int(round(clamped_target.width * 0.08))))
    pad_y = min(clamped_target.height // 4, max(4, int(round(clamped_target.height * 0.12))))
    dst_x1 = clamped_target.x1 + pad_x
    dst_x2 = max(dst_x1 + 1, clamped_target.x2 - pad_x)
    dst_y1 = clamped_target.y1 + pad_y
    dst_y2 = max(dst_y1 + 1, clamped_target.y2 - pad_y)
    avail_w = float(dst_x2 - dst_x1)
    avail_h = float(dst_y2 - dst_y1)
    scale = min(avail_w / max(1.0, float(crop_w)), avail_h / max(1.0, float(crop_h)))
    placed_w = max(1, int(round(crop_w * scale)))
    placed_h = max(1, int(round(crop_h * scale)))
    ox1 = int(round(0.5 * (dst_x1 + dst_x2) - 0.5 * placed_w))
    oy1 = int(round(0.5 * (dst_y1 + dst_y2) - 0.5 * placed_h))
    ox2 = ox1 + placed_w
    oy2 = oy1 + placed_h

    cx1 = max(clamped_target.x1, ox1)
    cy1 = max(clamped_target.y1, oy1)
    cx2 = min(clamped_target.x2, ox2)
    cy2 = min(clamped_target.y2, oy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return clean_model_rgb.copy()

    geometry = fit_garment_surface_geometry(
        clean_model_rgb=clean_model_rgb,
        target_bbox=clamped_target,
        perspective_skew=perspective_skew,
        original_model_rgb=original_model_rgb,
    )
    dx_fold, dy_fold, shade_patch, weave_patch = _compute_fold_displacement_and_shading(
        clean_model_rgb=clean_model_rgb,
        ox1=ox1,
        oy1=oy1,
        placed_w=placed_w,
        placed_h=placed_h,
        geometry=geometry,
    )
    map_x, map_y, limb_factor = _build_surface_uv_grid(
        ox1=ox1,
        oy1=oy1,
        placed_w=placed_w,
        placed_h=placed_h,
        crop_w=crop_w,
        crop_h=crop_h,
        geometry=geometry,
        dx_fold=dx_fold,
        dy_fold=dy_fold,
    )

    warped_rgb = cv2.remap(
        logo_crop,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.remap(
        alpha,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # Apply cloth irradiance shading + cylinder falloff + subtle matte fabric weave grain
    total_shade = np.clip(shade_patch * limb_factor, 0.45, 1.06)[..., None]
    relit_rgb = warped_rgb.astype(np.float32) * total_shade + 0.35 * weave_patch[..., None]

    out = clean_model_rgb.copy()
    sx1, sy1 = cx1 - ox1, cy1 - oy1
    sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
    a_patch = warped_alpha[sy1:sy2, sx1:sx2, None]
    if binary_mask is not None:
        mask_patch = (binary_mask[cy1:cy2, cx1:cx2] > 0).astype(np.float32)[..., None]
        a_patch = a_patch * mask_patch
    rgb_patch = relit_rgb[sy1:sy2, sx1:sx2]
    cloth_patch = out[cy1:cy2, cx1:cx2].astype(np.float32)
    blended = rgb_patch * a_patch + cloth_patch * (1.0 - a_patch)
    out[cy1:cy2, cx1:cx2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return out
