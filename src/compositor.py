"""Stage 3: Strict Pixel-Lock Compositor (Bypassing Full-Frame VAE Degradation)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import PipelineError, PixelBBox


class PixelLockCompositor:
    """Stitches a restored ROI back into the original image with a mathematical pixel lock."""

    def composite(
        self,
        original_image: NDArray[np.uint8],
        restored_roi: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
        feathered_mask: NDArray[np.float32],
        context_bbox: PixelBBox,
    ) -> NDArray[np.uint8]:
        """Composite restored ROI into original_image while locking all unmasked pixels.

        Mathematical guarantee:
        For every pixel (y, x) where `binary_mask[y, x] == 0`,
        `final_image[y, x, c] == original_image[y, x, c]` is bit-identical.
        """
        self._validate_inputs(
            original_image=original_image,
            restored_roi=restored_roi,
            binary_mask=binary_mask,
            feathered_mask=feathered_mask,
            context_bbox=context_bbox,
        )

        # Resize restored 1024x1024 ROI back to the exact full-res context_bbox dimensions
        resized_roi = cv2.resize(
            restored_roi,
            (context_bbox.width, context_bbox.height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Place resized ROI onto a full-resolution canvas initialized from original_image
        stitched_canvas = original_image.copy()
        stitched_canvas[
            context_bbox.y1 : context_bbox.y2,
            context_bbox.x1 : context_bbox.x2,
        ] = resized_roi

        # Clamp feathered alpha strictly to [0.0, 1.0] and zero out outside binary_mask
        alpha = np.where(
            binary_mask > 0,
            np.clip(feathered_mask, 0.0, 1.0),
            np.float32(0.0),
        )[..., None]

        # Perform interior feathered blend in float32 with VAE perimeter color-shift compensation
        orig_f32 = original_image.astype(np.float32)
        stitched_f32 = stitched_canvas.astype(np.float32)

        border_band = (feathered_mask > 0.02) & (feathered_mask < 0.55)
        if int(np.count_nonzero(border_band)) >= 32:
            diff_band = orig_f32[border_band] - stitched_f32[border_band]
            # Compensate small constant VAE luminance/tint offset (clamped to [-12, +12] RGB levels)
            vae_offset = np.clip(np.median(diff_band, axis=0), -12.0, 12.0)
            stitched_f32 = np.clip(stitched_f32 + vae_offset.reshape(1, 1, 3), 0.0, 255.0)

        blended_f32 = alpha * stitched_f32 + (1.0 - alpha) * orig_f32
        blended_image = np.clip(np.round(blended_f32), 0, 255).astype(np.uint8)

        # Hard NumPy pixel lock outside the binary mask (binary_mask == 0)
        final_image = np.where(binary_mask[..., None] > 0, blended_image, original_image)
        return np.asarray(final_image, dtype=np.uint8)

    @staticmethod
    def _validate_inputs(
        original_image: NDArray[np.uint8],
        restored_roi: NDArray[np.uint8],
        binary_mask: NDArray[np.uint8],
        feathered_mask: NDArray[np.float32],
        context_bbox: PixelBBox,
    ) -> None:
        if (
            original_image.ndim != 3
            or original_image.shape[2] != 3
            or original_image.dtype != np.uint8
        ):
            raise PipelineError(
                "original_image must be (H, W, 3) uint8, "
                f"got {original_image.shape} {original_image.dtype}."
            )
        if restored_roi.ndim != 3 or restored_roi.shape[2] != 3 or restored_roi.dtype != np.uint8:
            raise PipelineError(
                "restored_roi must be (H_roi, W_roi, 3) uint8, "
                f"got {restored_roi.shape} {restored_roi.dtype}."
            )
        height, width = original_image.shape[:2]
        if binary_mask.shape != (height, width):
            raise PipelineError(
                f"binary_mask shape {binary_mask.shape} does not match image {(height, width)}."
            )
        if feathered_mask.shape != (height, width):
            raise PipelineError(
                f"feathered_mask shape {feathered_mask.shape} "
                f"does not match image {(height, width)}."
            )
        if context_bbox.x2 > width or context_bbox.y2 > height:
            raise PipelineError(
                f"context_bbox {context_bbox} exceeds original_image bounds {width}x{height}."
            )
