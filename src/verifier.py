"""Stage 4: Automated Pixel-Lock Verification Gate and 4-Panel Visual Report Generator."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import (
    VerificationConfig,
    VerificationError,
    VerificationMetrics,
)


class PixelLockVerifier:
    """Verifies zero pixel alteration outside the binary mask and generates visual proof."""

    def __init__(self, config: VerificationConfig) -> None:
        self._config = config

    def verify(
        self,
        case_id: str,
        original_image: NDArray[np.uint8],
        final_image: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
        backend_used: str = "unknown",
        detection_method: str = "unknown",
    ) -> VerificationMetrics:
        """Compute exact pixel-wise verification metrics outside and inside the mask."""
        self._validate_inputs(original_image, final_image, binary_mask)

        height, width = original_image.shape[:2]
        total_pixels = int(height * width)
        unmasked_bool = binary_mask == 0
        masked_bool = binary_mask > 0

        unmasked_pixels = int(np.count_nonzero(unmasked_bool))
        masked_pixels = int(np.count_nonzero(masked_bool))
        if unmasked_pixels == 0 or masked_pixels == 0:
            raise VerificationError(
                f"Invalid binary_mask for case '{case_id}': "
                f"unmasked_pixels={unmasked_pixels}, masked_pixels={masked_pixels}. "
                "Both masked and unmasked regions must be non-empty."
            )

        abs_diff = np.abs(original_image.astype(np.int16) - final_image.astype(np.int16))
        pixel_max_channel_diff = np.max(abs_diff, axis=2)

        # Unmasked region metrics (must be strictly 0 changed pixels)
        unmasked_diffs = pixel_max_channel_diff[unmasked_bool]
        unmasked_max_abs_diff = int(np.max(unmasked_diffs))
        unmasked_changed_count = int(np.count_nonzero(unmasked_diffs > 0))
        unmasked_changed_pct = (unmasked_changed_count / float(unmasked_pixels)) * 100.0

        unmasked_mse = float(np.mean(abs_diff[unmasked_bool].astype(np.float64) ** 2))
        unmasked_psnr = (
            100.0 if unmasked_mse == 0.0 else float(10.0 * np.log10((255.0**2) / unmasked_mse))
        )
        unmasked_ssim = self._compute_unmasked_ssim(original_image, final_image, unmasked_bool)

        # Masked region metrics (ensure the detail was actually updated)
        masked_diffs = pixel_max_channel_diff[masked_bool]
        masked_mean_abs_diff = float(np.mean(abs_diff[masked_bool]))
        masked_changed_count = int(np.count_nonzero(masked_diffs > 0))

        passed = (
            unmasked_max_abs_diff == 0
            and unmasked_changed_count == 0
            and unmasked_ssim == 1.0
            and masked_changed_count > 0
        )

        return VerificationMetrics(
            case_id=case_id,
            passed=passed,
            unmasked_max_abs_diff=unmasked_max_abs_diff,
            unmasked_changed_pixels_count=unmasked_changed_count,
            unmasked_changed_pixels_pct=unmasked_changed_pct,
            unmasked_psnr=unmasked_psnr,
            unmasked_ssim=unmasked_ssim,
            masked_mean_abs_diff=masked_mean_abs_diff,
            masked_changed_pixels_count=masked_changed_count,
            total_pixels=total_pixels,
            unmasked_pixels=unmasked_pixels,
            masked_pixels=masked_pixels,
            backend_used=backend_used,
            detection_method=detection_method,
        )

    def build_diff_heatmap(
        self,
        original_image: NDArray[np.uint8],
        final_image: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Generate an amplified (x50) pixel-difference heatmap with mask boundary contour."""
        abs_diff = np.abs(original_image.astype(np.float32) - final_image.astype(np.float32))
        max_channel_diff = np.max(abs_diff, axis=2)
        amplified = np.clip(
            max_channel_diff * self._config.diff_amplification_factor, 0.0, 255.0
        ).astype(np.uint8)

        # Apply inferno/turbo style colormap in RGB, keeping 0 diff strictly pitch-black (0, 0, 0)
        heatmap_bgr = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        zero_mask = max_channel_diff == 0
        heatmap_rgb[zero_mask] = np.array([6, 10, 18], dtype=np.uint8)

        # Draw crisp cyan contour marking the exact binary mask lock boundary
        mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heatmap_rgb, contours, -1, (0, 255, 220), 2)
        return heatmap_rgb

    def render_four_panel_report(
        self,
        original_image: NDArray[np.uint8],
        final_image: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
        cropped_product_ref: NDArray[np.uint8],
        metrics: VerificationMetrics,
    ) -> NDArray[np.uint8]:
        """Render the 4-panel verification report image:
        [1. Original On-Model (Hallucinated) | 2. Auto-Mask & Product Ref Crop |
         3. Corrected Output | 4. Amplified Pixel-Diff Heatmap (x50)]
        """
        height, width = original_image.shape[:2]

        # Panel 1: Original On-Model with highlighted red box around hallucinated region
        panel_1 = original_image.copy()
        mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel_1, contours, -1, (255, 70, 70), 2)

        # Panel 2: Auto-Mask overlay + picture-in-picture Product Reference detail crop
        panel_2 = (original_image.astype(np.float32) * 0.42).astype(np.uint8)
        green_tint = np.zeros_like(original_image)
        green_tint[:, :, 1] = 230
        green_tint[:, :, 2] = 180
        panel_2 = np.where(
            binary_mask[..., None] > 0,
            np.clip(original_image.astype(np.float32) * 0.55 + green_tint * 0.45, 0, 255).astype(
                np.uint8
            ),
            panel_2,
        )
        cv2.drawContours(panel_2, contours, -1, (40, 255, 160), 2)

        # Inset authentic product reference crop in bottom-right of Panel 2
        inset_size = max(96, int(min(height, width) * 0.34))
        ref_thumb = cv2.resize(
            cropped_product_ref, (inset_size, inset_size), interpolation=cv2.INTER_LANCZOS4
        )
        pad = 14
        y_start = height - inset_size - pad
        x_start = width - inset_size - pad
        cv2.rectangle(
            panel_2,
            (x_start - 3, y_start - 24),
            (x_start + inset_size + 3, y_start + inset_size + 3),
            (20, 24, 34),
            -1,
        )
        cv2.rectangle(
            panel_2,
            (x_start - 3, y_start - 24),
            (x_start + inset_size + 3, y_start + inset_size + 3),
            (0, 230, 180),
            2,
        )
        cv2.putText(
            panel_2,
            "AUTHENTIC REF CROP",
            (x_start + 4, y_start - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 248, 255),
            1,
            cv2.LINE_AA,
        )
        panel_2[y_start : y_start + inset_size, x_start : x_start + inset_size] = ref_thumb

        # Panel 3: Corrected Output
        panel_3 = final_image.copy()

        # Panel 4: Amplified Pixel-Diff Heatmap (x50)
        panel_4 = self.build_diff_heatmap(original_image, final_image, binary_mask)

        # Add headers to each panel
        titles = [
            "1. Original On-Model (Hallucinated)",
            "2. Auto-Mask & Product Ref Crop",
            "3. Corrected Output (Pixel-Locked)",
            f"4. Amplified Diff Heatmap (x{int(self._config.diff_amplification_factor)})",
        ]
        panels = [
            self._add_panel_header(p, t)
            for p, t in zip([panel_1, panel_2, panel_3, panel_4], titles, strict=True)
        ]

        # Concatenate horizontally with 6px dividers
        divider = np.full((panels[0].shape[0], 6, 3), 28, dtype=np.uint8)
        grid = np.concatenate(
            [panels[0], divider, panels[1], divider, panels[2], divider, panels[3]], axis=1
        )

        # Add bottom verification banner summarizing mathematical guarantee
        banner_h = 54
        banner = np.full((banner_h, grid.shape[1], 3), (15, 20, 28), dtype=np.uint8)
        status_label = (
            "PASS - ZERO DETAIL LOSS GUARANTEED"
            if metrics.passed
            else "FAIL - PIXEL DRIFT DETECTED"
        )
        status_color = (40, 235, 130) if metrics.passed else (255, 75, 75)

        summary_text = (
            f"[{metrics.case_id}] {status_label}   |   "
            f"Unmasked Changed Pixels: {metrics.unmasked_changed_pixels_count} "
            f"({metrics.unmasked_changed_pixels_pct:.4f}%)   |   "
            f"Unmasked Max Diff: {metrics.unmasked_max_abs_diff}   |   "
            f"Unmasked SSIM: {metrics.unmasked_ssim:.4f}   |   "
            f"Backend: {metrics.backend_used}"
        )
        cv2.putText(
            banner,
            summary_text,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return np.concatenate([grid, banner], axis=0)

    @staticmethod
    def _add_panel_header(panel: NDArray[np.uint8], title: str) -> NDArray[np.uint8]:
        header_h = 40
        header = np.full((header_h, panel.shape[1], 3), (22, 27, 36), dtype=np.uint8)
        cv2.putText(
            header,
            title,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 240, 248),
            1,
            cv2.LINE_AA,
        )
        return np.concatenate([header, panel], axis=0)

    @staticmethod
    def _compute_unmasked_ssim(
        original_image: NDArray[np.uint8],
        final_image: NDArray[np.uint8],
        unmasked_bool: NDArray[np.bool_],
    ) -> float:
        """Compute exact structural similarity over the unmasked pixel set."""
        orig_vals = original_image[unmasked_bool].astype(np.float64)
        final_vals = final_image[unmasked_bool].astype(np.float64)
        if np.array_equal(orig_vals, final_vals):
            return 1.0

        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_x = float(np.mean(orig_vals))
        mu_y = float(np.mean(final_vals))
        var_x = float(np.var(orig_vals))
        var_y = float(np.var(final_vals))
        cov_xy = float(np.mean((orig_vals - mu_x) * (final_vals - mu_y)))

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        return float(numerator / denominator)

    @staticmethod
    def _validate_inputs(
        original_image: NDArray[np.uint8],
        final_image: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
    ) -> None:
        if original_image.shape != final_image.shape:
            raise VerificationError(
                f"Dimension mismatch: original_image={original_image.shape} "
                f"vs final_image={final_image.shape}."
            )
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise VerificationError(f"Expected 3-channel RGB images, got {original_image.shape}.")
        if binary_mask.shape != original_image.shape[:2]:
            raise VerificationError(
                f"binary_mask shape {binary_mask.shape} "
                f"does not match image {original_image.shape[:2]}."
            )
