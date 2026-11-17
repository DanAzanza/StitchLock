"""Stage 1: Automated Garment Detail Localization and Feathered Mask Generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.models import (
    CaseManifest,
    ConfigurationError,
    LocalizationError,
    LocalizationResult,
    MaskingConfig,
    MaskingStrategy,
    NormalizedBox,
    PixelBBox,
    resolve_safe_path,
)

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
REFERENCE_STEM_SUFFIX = "_reference"


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _find_flat_reference_file(model_path: Path) -> Path:
    """Find the matching `<stem>_reference.<ext>` file next to a flat On-Model image."""
    target_ref_stem = f"{model_path.stem}{REFERENCE_STEM_SUFFIX}"
    matches = [
        p
        for p in sorted(model_path.parent.iterdir())
        if _is_supported_image(p) and p.stem == target_ref_stem
    ]
    if not matches:
        raise ConfigurationError(
            f"Missing reference image for '{model_path.name}': "
            f"expected '{target_ref_stem}.*' in {model_path.parent}."
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ConfigurationError(
            f"Ambiguous multiple reference images for '{model_path.name}': {names}."
        )
    return matches[0]


def discover_case_paths(test_cases_dir: Path, *, allow_empty: bool = False) -> list[Path]:
    """Discover flat `<name>.*` + `<name>_reference.*` pairs and legacy manifest folders."""
    if not test_cases_dir.is_dir():
        if allow_empty:
            return []
        raise ConfigurationError(
            f"Test cases directory does not exist: {test_cases_dir}. "
            "Run 'python setup_test_cases.py' first."
        )

    models_by_stem: dict[str, Path] = {}
    refs_by_stem: dict[str, Path] = {}
    for entry in sorted(test_cases_dir.iterdir()):
        if not _is_supported_image(entry):
            continue
        if entry.stem.endswith(REFERENCE_STEM_SUFFIX):
            base_stem = entry.stem[: -len(REFERENCE_STEM_SUFFIX)]
            if not base_stem:
                raise ConfigurationError(
                    f"Invalid reference filename '{entry.name}' in {test_cases_dir}."
                )
            if base_stem in refs_by_stem:
                raise ConfigurationError(
                    f"Duplicate reference images for '{base_stem}': "
                    f"'{refs_by_stem[base_stem].name}' and '{entry.name}'."
                )
            refs_by_stem[base_stem] = entry
            continue

        if entry.stem in models_by_stem:
            raise ConfigurationError(
                f"Duplicate On-Model images for '{entry.stem}': "
                f"'{models_by_stem[entry.stem].name}' and '{entry.name}'."
            )
        models_by_stem[entry.stem] = entry

    missing_models = sorted(set(refs_by_stem) - set(models_by_stem))
    if missing_models:
        stem = missing_models[0]
        raise ConfigurationError(
            f"Orphan reference image '{refs_by_stem[stem].name}' in {test_cases_dir}: "
            f"missing matching On-Model image '{stem}.*'."
        )

    missing_refs = sorted(set(models_by_stem) - set(refs_by_stem))
    if missing_refs:
        stem = missing_refs[0]
        raise ConfigurationError(
            f"Orphan On-Model image '{models_by_stem[stem].name}' in {test_cases_dir}: "
            f"missing matching reference image '{stem}{REFERENCE_STEM_SUFFIX}.*'."
        )

    discovered_by_id: dict[str, Path] = dict(models_by_stem)
    for manifest_file in sorted(test_cases_dir.glob("*/case_manifest.json")):
        folder = manifest_file.parent
        if folder.name in discovered_by_id:
            raise ConfigurationError(
                f"Duplicate test case ID '{folder.name}': present as both flat image pair "
                f"and legacy folder in {test_cases_dir}."
            )
        discovered_by_id[folder.name] = folder

    if not discovered_by_id and not allow_empty:
        raise ConfigurationError(
            f"No image pairs (`<name>.*` + `<name>_reference.*`) or manifests found in "
            f"{test_cases_dir}. Run 'python setup_test_cases.py' first."
        )
    return [discovered_by_id[cid] for cid in sorted(discovered_by_id)]


def resolve_case_path(test_cases_dir: Path, case_id: str) -> Path:
    """Resolve a case ID safely to its flat On-Model image file or legacy case folder."""
    resolve_safe_path(test_cases_dir, case_id)
    for path in discover_case_paths(test_cases_dir):
        cid = path.stem if path.is_file() else path.name
        if cid == case_id:
            return path
    raise ConfigurationError(f"Test case '{case_id}' not found in {test_cases_dir}.")


def has_manifest_roi(case_path: Path) -> bool:
    """Return True if the case explicitly supplies ROI boxes via JSON."""
    if case_path.is_dir():
        return (case_path / "case_manifest.json").is_file()
    sidecar = case_path.with_suffix(".json")
    if not sidecar.is_file():
        return False
    with sidecar.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    return "model_target_box" in raw and "reference_detail_box" in raw


def load_case_manifest(case_path: Path) -> CaseManifest:
    """Load a CaseManifest from either a flat `<name>.*` + `<name>_reference.*` pair or folder."""
    if case_path.is_dir():
        manifest_path = resolve_safe_path(case_path, "case_manifest.json")
        if not manifest_path.is_file():
            raise ConfigurationError(f"Case manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        required_keys = {
            "case_id",
            "title",
            "garment_category",
            "target_detail",
            "prompt",
            "model_image_filename",
            "reference_image_filename",
            "model_target_box",
            "reference_detail_box",
        }
        missing = required_keys - set(raw.keys())
        if missing:
            raise ConfigurationError(
                f"Manifest {manifest_path} missing required keys: {sorted(missing)}"
            )
        m_box = raw["model_target_box"]
        r_box = raw["reference_detail_box"]
        return CaseManifest(
            case_id=str(raw["case_id"]),
            title=str(raw["title"]),
            garment_category=str(raw["garment_category"]),
            target_detail=str(raw["target_detail"]),
            prompt=str(raw["prompt"]),
            model_image_filename=str(raw["model_image_filename"]),
            reference_image_filename=str(raw["reference_image_filename"]),
            model_target_box=NormalizedBox(
                float(m_box[0]), float(m_box[1]), float(m_box[2]), float(m_box[3])
            ),
            reference_detail_box=NormalizedBox(
                float(r_box[0]), float(r_box[1]), float(r_box[2]), float(r_box[3])
            ),
        )

    if not _is_supported_image(case_path):
        raise ConfigurationError(f"Unsupported or missing case path: {case_path}")
    if case_path.stem.endswith(REFERENCE_STEM_SUFFIX):
        raise ConfigurationError(
            f"Expected On-Model image path, got reference image '{case_path.name}'."
        )

    ref_path = _find_flat_reference_file(case_path)
    sidecar = case_path.with_suffix(".json")
    raw_sidecar: dict[str, Any] = {}
    if sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as handle:
            raw_sidecar = json.load(handle)

    m_box = raw_sidecar.get("model_target_box", [0.30, 0.30, 0.70, 0.70])
    r_box = raw_sidecar.get("reference_detail_box", [0.30, 0.30, 0.70, 0.70])
    stem_title = case_path.stem.replace("_", " ").title()
    return CaseManifest(
        case_id=str(raw_sidecar.get("case_id", case_path.stem)),
        title=str(raw_sidecar.get("title", stem_title)),
        garment_category=str(raw_sidecar.get("garment_category", "garment")),
        target_detail=str(raw_sidecar.get("target_detail", "Auto-detected garment detail")),
        prompt=str(raw_sidecar.get("prompt", "")),
        model_image_filename=case_path.name,
        reference_image_filename=ref_path.name,
        model_target_box=NormalizedBox(
            float(m_box[0]), float(m_box[1]), float(m_box[2]), float(m_box[3])
        ),
        reference_detail_box=NormalizedBox(
            float(r_box[0]), float(r_box[1]), float(r_box[2]), float(r_box[3])
        ),
    )


class AutoMasker:
    """Pluggable garment detail localizer and strict-boundary mask generator."""

    def __init__(self, config: MaskingConfig) -> None:
        self._config = config

    def localize_and_mask(
        self,
        model_rgb: NDArray[np.uint8],
        ref_rgb: NDArray[np.uint8],
        manifest: CaseManifest,
    ) -> LocalizationResult:
        """Build resolution-independent full-frame masks without distorting aspect ratios."""
        self._validate_rgb_image(model_rgb, "model_rgb")
        self._validate_rgb_image(ref_rgb, "ref_rgb")

        height, width = model_rgb.shape[:2]
        ref_h, ref_w = ref_rgb.shape[:2]

        ref_detail_bbox = manifest.reference_detail_box.to_pixel_bbox(ref_w, ref_h)
        target_bbox = manifest.model_target_box.to_pixel_bbox(width, height)
        detection_method = (
            "vlm_grounded_box"
            if self._config.strategy == MaskingStrategy.AUTO_VISION
            else "manifest_roi_deterministic"
        )

        context_bbox = PixelBBox(x1=0, y1=0, x2=width, y2=height)

        ref_dim = float(min(height, width))
        dilation_px = max(2, int(round(ref_dim * self._config.mask_dilation_ratio)))
        feather_sigma = max(1.5, ref_dim * self._config.feather_sigma_ratio)

        binary_mask, feathered_mask = self._build_masks(
            model_rgb=model_rgb,
            target_bbox=target_bbox,
            dilation_px=dilation_px,
            feather_sigma=feather_sigma,
        )

        cropped_model_roi = model_rgb.copy()
        cropped_product_ref = self._align_full_garment_reference(
            ref_rgb=ref_rgb,
            ref_detail_bbox=ref_detail_bbox,
            target_bbox=target_bbox,
            model_size=(width, height),
        )
        cropped_mask_greyscale = np.clip(np.round(feathered_mask * 255.0), 0, 255).astype(np.uint8)

        return LocalizationResult(
            binary_mask=binary_mask,
            feathered_mask=feathered_mask,
            cropped_model_roi=cropped_model_roi,
            cropped_product_ref=cropped_product_ref,
            cropped_mask_greyscale=cropped_mask_greyscale,
            context_bbox=context_bbox,
            target_bbox=target_bbox,
            ref_detail_bbox=ref_detail_bbox,
            detection_method=detection_method,
        )

    @staticmethod
    def _validate_rgb_image(image: NDArray[np.uint8], name: str) -> None:
        """Verify that input array is a valid 3-channel uint8 RGB image."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise LocalizationError(
                f"Expected '{name}' to be (H, W, 3) uint8 RGB array, "
                f"got shape={image.shape}, dtype={image.dtype}."
            )
        if image.shape[0] < 32 or image.shape[1] < 32:
            raise LocalizationError(
                f"Image '{name}' is too small ({image.shape[1]}x{image.shape[0]}); "
                "minimum supported dimension is 32x32."
            )

    @staticmethod
    def _build_masks(
        model_rgb: NDArray[np.uint8],
        target_bbox: PixelBBox,
        dilation_px: int,
        feather_sigma: float,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Generate binary mask {0, 1} and strictly interior-feathered blending mask [0.0, 1.0].

        Crucial invariant:
        - `binary_mask` is 1 inside the dilated target region (clipped to garment silhouette
          if bordering a bright studio background) and 0 everywhere else.
        - `feathered_mask` transitions smoothly from 0.0 at the outer boundary of `binary_mask`
          to 1.0 on the interior, and is strictly 0.0 wherever `binary_mask == 0`.
        """
        image_height, image_width = model_rgb.shape[:2]
        pad = max(0, dilation_px)
        x1 = max(0, target_bbox.x1 - pad)
        y1 = max(0, target_bbox.y1 - pad)
        x2 = min(image_width, target_bbox.x2 + pad)
        y2 = min(image_height, target_bbox.y2 + pad)

        binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        binary_mask[y1:y2, x1:x2] = 1

        # Compute Euclidean distance to the nearest zero pixel (strictly inside binary_mask == 1)
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        ramp_width = max(2.0, float(feather_sigma) * 1.8)
        feathered = np.clip(dist / ramp_width, 0.0, 1.0).astype(np.float32)

        # Apply smoothstep S-curve for seamless visual transition inside the mask
        feathered = feathered * feathered * (3.0 - 2.0 * feathered)

        # Enforce hard zero outside binary_mask
        feathered = np.where(binary_mask > 0, feathered, np.float32(0.0))
        return binary_mask, feathered

    @staticmethod
    def _align_full_garment_reference(
        ref_rgb: NDArray[np.uint8],
        ref_detail_bbox: PixelBBox,
        target_bbox: PixelBBox,
        model_size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        """Align reference garment logo plane at native aspect ratio using edge replication."""
        m_w, m_h = model_size
        r_h, r_w = ref_rgb.shape[:2]

        t_cx = 0.5 * (target_bbox.x1 + target_bbox.x2) / float(m_w)
        t_cy = 0.5 * (target_bbox.y1 + target_bbox.y2) / float(m_h)
        r_cx = 0.5 * (ref_detail_bbox.x1 + ref_detail_bbox.x2) / float(r_w)
        r_cy = 0.5 * (ref_detail_bbox.y1 + ref_detail_bbox.y2) / float(r_h)

        dx_norm = t_cx - r_cx
        dy_norm = t_cy - r_cy
        if abs(dx_norm) <= 0.08 and abs(dy_norm) <= 0.08:
            return np.asarray(ref_rgb.copy(), dtype=np.uint8)

        # Sample outer top corners cleanly so background padding never smears foreground/chin pixels
        corner_w = max(4, r_w // 20)
        corner_h = max(4, r_h // 20)
        corners = np.concatenate(
            [
                ref_rgb[0:corner_h, 0:corner_w, :].reshape(-1, 3),
                ref_rgb[0:corner_h, r_w - corner_w : r_w, :].reshape(-1, 3),
            ],
            axis=0,
        )
        border_rgb = tuple(int(v) for v in np.median(corners, axis=0))

        dx_px = int(round(dx_norm * r_w))
        dy_px = int(round(dy_norm * r_h))
        transform = np.asarray(
            [[1.0, 0.0, float(dx_px)], [0.0, 1.0, float(dy_px)]], dtype=np.float32
        )
        aligned = cv2.warpAffine(
            ref_rgb,
            transform,
            (r_w, r_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_rgb,
        )
        return np.asarray(aligned, dtype=np.uint8)
