"""Domain models, immutable data contracts, and configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray


class PipelineError(Exception):
    """Base domain exception for garment correction pipeline errors."""


class ConfigurationError(PipelineError):
    """Raised when configuration or case manifest data is invalid or missing."""


class LocalizationError(PipelineError):
    """Raised when target garment region cannot be localized."""


class ComfyExecutionError(PipelineError):
    """Raised when ComfyUI communication or workflow execution fails."""


class VerificationError(PipelineError):
    """Raised when pixel-lock verification fails or inputs are malformed."""


class MaskingStrategy(StrEnum):
    """Supported auto-localization strategies."""

    AUTO_VISION = "auto_vision"
    MANIFEST_ROI = "manifest_roi"


@dataclass(frozen=True)
class NormalizedBox:
    """Normalized bounding box [x_min, y_min, x_max, y_max] in [0.0, 1.0]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x_min < self.x_max <= 1.0 and 0.0 <= self.y_min < self.y_max <= 1.0):
            raise ConfigurationError(
                f"Invalid NormalizedBox({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}): "
                "coordinates must satisfy 0.0 <= min < max <= 1.0."
            )

    def to_pixel_bbox(self, width: int, height: int) -> PixelBBox:
        """Convert normalized coordinates to integer pixel bounding box."""
        if width <= 0 or height <= 0:
            raise ConfigurationError(f"Image dimensions must be positive, got {width}x{height}.")
        x1 = max(0, min(width - 1, int(round(self.x_min * width))))
        y1 = max(0, min(height - 1, int(round(self.y_min * height))))
        x2 = max(x1 + 1, min(width, int(round(self.x_max * width))))
        y2 = max(y1 + 1, min(height, int(round(self.y_max * height))))
        return PixelBBox(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True)
class PixelBBox:
    """Integer pixel bounding box [x1, y1, x2, y2] in image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ConfigurationError(
                f"Invalid PixelBBox({self.x1}, {self.y1}, {self.x2}, {self.y2}): "
                "must have non-negative coordinates and positive width/height."
            )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DefectTarget:
    """Single localized garment defect paired with its authentic reference detail."""

    defect_summary: str
    model_target_box: NormalizedBox
    reference_detail_box: NormalizedBox
    edit_prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize defect target using normalized coordinates for audit logs and UI."""
        return {
            "defect_summary": self.defect_summary,
            "model_target_box": [
                round(self.model_target_box.x_min, 4),
                round(self.model_target_box.y_min, 4),
                round(self.model_target_box.x_max, 4),
                round(self.model_target_box.y_max, 4),
            ],
            "reference_detail_box": [
                round(self.reference_detail_box.x_min, 4),
                round(self.reference_detail_box.y_min, 4),
                round(self.reference_detail_box.x_max, 4),
                round(self.reference_detail_box.y_max, 4),
            ],
            "edit_prompt": self.edit_prompt,
        }


@dataclass(frozen=True)
class CaseManifest:
    """Metadata and ROI specification for a garment correction test case."""

    case_id: str
    title: str
    garment_category: str
    target_detail: str
    prompt: str
    model_image_filename: str
    reference_image_filename: str
    model_target_box: NormalizedBox
    reference_detail_box: NormalizedBox


@dataclass(frozen=True)
class MaskingConfig:
    """Configuration parameters for resolution-independent mask generation."""

    strategy: MaskingStrategy = MaskingStrategy.AUTO_VISION
    mask_dilation_ratio: float = 0.018
    feather_sigma_ratio: float = 0.006


@dataclass(frozen=True)
class ComfyConfig:
    """Configuration parameters for ComfyUI Qwen-Image 2.1 backend."""

    server_url: str = "http://127.0.0.1:8000"
    fallback_ports: tuple[int, ...] = (8000, 8188)
    timeout_seconds: float = 180.0
    seed: int = 42
    steps: int = 20
    cfg: float = 1.0
    denoise: float = 1.0
    auto_launch: bool = False


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration thresholds for CV box refinement and defect heuristics."""

    min_graphic_contrast_std: float = 14.0
    bg_color_distance_threshold: float = 28.0
    cloth_color_distance_threshold: float = 44.0
    chroma_distance_threshold: float = 15.0
    connected_component_salience_threshold: float = 45.0
    emblem_correlation_threshold: float = 0.75
    shift_y_threshold: float = 0.25
    shift_x_threshold: float = 0.22


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration parameters for automated pixel-lock verification."""

    diff_amplification_factor: float = 4.0
    strict_zero_outside_mask: bool = True
    min_masked_similarity: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object for the garment correction pipeline."""

    test_cases_dir: Path
    outputs_dir: Path
    workflow_path: Path
    masking: MaskingConfig
    comfyui: ComfyConfig
    verification: VerificationConfig
    inspector_workflow_path: Path | None = None
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)

    @property
    def resolved_inspector_workflow_path(self) -> Path:
        """Return configured inspector workflow path or default next to edit workflow."""
        if self.inspector_workflow_path is not None:
            return self.inspector_workflow_path
        return self.workflow_path.parent / "qwen3_vl_inspector_api.json"


@dataclass(frozen=True)
class LocalizationResult:
    """Output of Stage 1: Auto-Localization & Masking."""

    binary_mask: NDArray[np.uint8]  # Full-res (H, W) in {0, 1}
    feathered_mask: NDArray[
        np.float32
    ]  # Full-res (H, W) in [0.0, 1.0], strictly 0 where binary_mask == 0
    cropped_model_roi: NDArray[np.uint8]  # (crop_res, crop_res, 3) RGB
    cropped_product_ref: NDArray[np.uint8]  # (crop_res, crop_res, 3) RGB
    cropped_mask_greyscale: NDArray[np.uint8]  # (crop_res, crop_res) single-channel uint8 [0..255]
    context_bbox: PixelBBox  # Full-res context crop window stitched back in Stage 3
    target_bbox: PixelBBox  # Full-res tight localized detail box
    ref_detail_bbox: PixelBBox  # Reference image detail crop box
    detection_method: str  # Audit string describing how localization succeeded


@dataclass(frozen=True)
class VerificationMetrics:
    """Output of Stage 4: Automated Pixel-Lock Verification Gate."""

    case_id: str
    passed: bool
    unmasked_max_abs_diff: int
    unmasked_changed_pixels_count: int
    unmasked_changed_pixels_pct: float
    unmasked_psnr: float
    unmasked_ssim: float
    masked_mean_abs_diff: float
    masked_changed_pixels_count: int
    total_pixels: int
    unmasked_pixels: int
    masked_pixels: int
    backend_used: str
    detection_method: str
    masked_reference_similarity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return {
            "case_id": self.case_id,
            "status": "PASS" if self.passed else "FAIL",
            "passed": self.passed,
            "unmasked_max_abs_diff": self.unmasked_max_abs_diff,
            "unmasked_changed_pixels_count": self.unmasked_changed_pixels_count,
            "unmasked_changed_pixels_pct": round(self.unmasked_changed_pixels_pct, 6),
            "unmasked_psnr": self.unmasked_psnr,
            "unmasked_ssim": round(self.unmasked_ssim, 6),
            "masked_mean_abs_diff": round(self.masked_mean_abs_diff, 4),
            "masked_changed_pixels_count": self.masked_changed_pixels_count,
            "total_pixels": self.total_pixels,
            "unmasked_pixels": self.unmasked_pixels,
            "masked_pixels": self.masked_pixels,
            "backend_used": self.backend_used,
            "detection_method": self.detection_method,
            "masked_reference_similarity": round(self.masked_reference_similarity, 4),
        }


def resolve_safe_path(root: Path, candidate: Path | str) -> Path:
    """Resolve a path and ensure it resides strictly inside the allowed root directory."""
    resolved_root = root.resolve()
    resolved_target = (resolved_root / candidate).resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ConfigurationError(
            f"Path traversal rejected: '{candidate}' escapes root '{resolved_root}'."
        )
    return resolved_target


def load_pipeline_config(config_path: Path, base_dir: Path | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML."""
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    root = (base_dir or config_path.parent).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    pipe_raw = raw.get("pipeline", {})
    mask_raw = raw.get("masking", {})
    comfy_raw = raw.get("comfyui", {})
    ver_raw = raw.get("verification", {})
    heur_raw = raw.get("heuristics", {})

    strategy = MaskingStrategy(mask_raw.get("strategy", MaskingStrategy.AUTO_VISION.value))

    dilation_ratio = float(mask_raw.get("mask_dilation_ratio", 0.018))
    feather_ratio = float(mask_raw.get("feather_sigma_ratio", 0.006))
    if not (0.0 <= dilation_ratio <= 0.25):
        raise ConfigurationError(
            f"mask_dilation_ratio must be in [0.0, 0.25], got {dilation_ratio}."
        )
    if not (0.0 < feather_ratio <= 0.1):
        raise ConfigurationError(f"feather_sigma_ratio must be in (0.0, 0.1], got {feather_ratio}.")

    min_masked_sim = float(ver_raw.get("min_masked_similarity", 0.0))
    if not (0.0 <= min_masked_sim <= 1.0):
        raise ConfigurationError(
            f"min_masked_similarity must be in [0.0, 1.0], got {min_masked_sim}."
        )

    inspector_wf_raw = pipe_raw.get(
        "inspector_workflow_path", "workflows/qwen3_vl_inspector_api.json"
    )

    return PipelineConfig(
        test_cases_dir=resolve_safe_path(root, pipe_raw.get("test_cases_dir", "test_cases")),
        outputs_dir=resolve_safe_path(root, pipe_raw.get("outputs_dir", "outputs")),
        workflow_path=resolve_safe_path(
            root, pipe_raw.get("workflow_path", "workflows/qwen_2_1_edit_api.json")
        ),
        inspector_workflow_path=resolve_safe_path(root, inspector_wf_raw),
        masking=MaskingConfig(
            strategy=strategy,
            mask_dilation_ratio=dilation_ratio,
            feather_sigma_ratio=feather_ratio,
        ),
        comfyui=ComfyConfig(
            server_url=str(comfy_raw.get("server_url", "http://127.0.0.1:8000")),
            fallback_ports=tuple(int(p) for p in comfy_raw.get("fallback_ports", [8000, 8188])),
            timeout_seconds=float(comfy_raw.get("timeout_seconds", 180.0)),
            seed=int(comfy_raw.get("seed", 42)),
            steps=int(comfy_raw.get("steps", 20)),
            cfg=float(comfy_raw.get("cfg", 1.0)),
            denoise=float(comfy_raw.get("denoise", 1.0)),
            auto_launch=bool(comfy_raw.get("auto_launch", False)),
        ),
        verification=VerificationConfig(
            diff_amplification_factor=float(ver_raw.get("diff_amplification_factor", 4.0)),
            strict_zero_outside_mask=bool(ver_raw.get("strict_zero_outside_mask", True)),
            min_masked_similarity=min_masked_sim,
        ),
        heuristics=HeuristicConfig(
            min_graphic_contrast_std=float(heur_raw.get("min_graphic_contrast_std", 14.0)),
            bg_color_distance_threshold=float(heur_raw.get("bg_color_distance_threshold", 28.0)),
            cloth_color_distance_threshold=float(
                heur_raw.get("cloth_color_distance_threshold", 44.0)
            ),
            chroma_distance_threshold=float(heur_raw.get("chroma_distance_threshold", 15.0)),
            connected_component_salience_threshold=float(
                heur_raw.get("connected_component_salience_threshold", 45.0)
            ),
            emblem_correlation_threshold=float(heur_raw.get("emblem_correlation_threshold", 0.75)),
            shift_y_threshold=float(heur_raw.get("shift_y_threshold", 0.25)),
            shift_x_threshold=float(heur_raw.get("shift_x_threshold", 0.22)),
        ),
    )
