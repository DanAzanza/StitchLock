"""Zero-Touch Batch CLI Runner for the Garment Detail Correction Pipeline.

Usage:
    python run_batch.py
    python run_batch.py --config config.yaml --strategy auto_vision
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.auto_masker import discover_case_paths
from src.comfy_launcher import stop_auto_started_server
from src.fixtures import setup_all_test_cases
from src.models import (
    MaskingStrategy,
    PipelineError,
    load_pipeline_config,
)
from src.pipeline import CaseExecutionArtifact, GarmentCorrectionPipeline


def _configure_utf8_console() -> None:
    """Ensure Windows console stdout/stderr supports UTF-8 box drawing and symbols."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _print_case_report(artifact: CaseExecutionArtifact, root_dir: Path) -> None:
    m = artifact.metrics
    badge = "PASS" if m.passed else "FAIL"
    print("-" * 86)
    print(f"[{badge}] Case: {artifact.manifest.title} ({m.case_id})")
    print(f"  Target Detail               : {artifact.manifest.target_detail}")
    print(f"  Localization Strategy       : {m.detection_method}")
    print(f"  Restoration Backend         : {m.backend_used}")
    print(f"  Unmasked Pixels Verified    : {m.unmasked_pixels:,} / {m.total_pixels:,} total px")
    print(f"  unmasked_max_abs_diff       : {m.unmasked_max_abs_diff} (Required: 0)")
    print(
        f"  unmasked_changed_pixels     : {m.unmasked_changed_pixels_count} "
        f"({m.unmasked_changed_pixels_pct:.4f}%)"
    )
    print(f"  unmasked_psnr / ssim        : {m.unmasked_psnr:.1f} dB / {m.unmasked_ssim:.4f}")
    print(
        f"  masked_changed_pixels       : {m.masked_changed_pixels_count:,} px "
        f"(mean delta: {m.masked_mean_abs_diff:.2f})"
    )
    print(f"  Corrected Output            : {artifact.corrected_image_path.relative_to(root_dir)}")
    print(f"  4-Panel Verification Report : {artifact.report_image_path.relative_to(root_dir)}")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(
        description="Execute all Garment Detail Correction test cases with zero manual work."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to pipeline YAML configuration (default: config.yaml).",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in MaskingStrategy],
        default=None,
        help="Override masking localization strategy (auto_vision or manifest_roi).",
    )
    args = parser.parse_args(argv)

    root_dir = Path(__file__).resolve().parent
    config_path = (root_dir / args.config).resolve()

    try:
        config = load_pipeline_config(config_path, base_dir=root_dir)
        if args.strategy is not None:
            config = replace(
                config,
                masking=replace(config.masking, strategy=MaskingStrategy(args.strategy)),
            )

        # Ensure sample test cases exist if running on a fresh checkout
        if not discover_case_paths(config.test_cases_dir, allow_empty=True):
            print("[INFO] Initializing sample generic garment test cases...")
            setup_all_test_cases(root_dir)

        pipeline = GarmentCorrectionPipeline(config)
        print("=" * 86)
        print(" GARMENT DETAIL CORRECTION PIPELINE - AUTOMATED PIXEL-LOCK BATCH VERIFICATION")
        print("=" * 86)

        artifacts = pipeline.run_all_cases()
        for artifact in artifacts:
            _print_case_report(artifact, root_dir)

        all_passed = all(a.metrics.passed for a in artifacts)
        print("=" * 86)
        overall_status = (
            "ALL CASES PASSED (0.0000% UNMASKED PIXEL LOSS)" if all_passed else "FAILURE DETECTED"
        )
        print(f" SUMMARY: {len(artifacts)} Cases Processed -> {overall_status}")
        summary_rel = (
            config.outputs_dir / "_reports" / "batch_verification_summary.json"
        ).relative_to(root_dir)
        print(f" Batch JSON Report: {summary_rel}")
        print("=" * 86)
        return 0 if all_passed else 1

    except PipelineError as exc:
        print(f"[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        return 2
    finally:
        stop_auto_started_server()


if __name__ == "__main__":
    sys.exit(main())
