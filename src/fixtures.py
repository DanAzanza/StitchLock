"""Deterministic Procedural Test Case Fixture Generator for StitchLocker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _draw_procedural_fixture_pair(
    test_cases_root: Path,
    *,
    case_id: str,
    seed: int,
) -> Path:
    """Write a lightweight deterministic flat `<case_id>.png` + `<case_id>_reference.png` pair."""
    test_cases_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = 512, 512

    on_model = rng.integers(24, 34, size=(h, w, 3), dtype=np.uint8)
    cv2.putText(
        on_model,
        "HALLUCINATED",
        (170, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 80, 70),
        2,
        cv2.LINE_AA,
    )

    ref_img = rng.integers(22, 30, size=(h, w, 3), dtype=np.uint8)
    cv2.rectangle(ref_img, (180, 200), (330, 300), (45, 95, 165), -1)
    cv2.putText(
        ref_img,
        "KOVA",
        (210, 260),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (245, 242, 235),
        2,
        cv2.LINE_AA,
    )

    model_path = test_cases_root / f"{case_id}.png"
    ref_path = test_cases_root / f"{case_id}_reference.png"
    Image.fromarray(on_model, mode="RGB").save(model_path)
    Image.fromarray(ref_img, mode="RGB").save(ref_path)
    return model_path


def setup_all_test_cases(base_dir: Path, force: bool = False) -> list[Path]:
    """Ensure flat `<case_id>.png` + `<case_id>_reference.png` fixtures exist in `test_cases/`."""
    test_cases_root = base_dir / "test_cases"
    if not force and test_cases_root.is_dir():
        existing_refs = [p for p in test_cases_root.glob("*_reference.*") if p.is_file()]
        if existing_refs:
            return sorted(existing_refs)

    specs = [
        ("case_01_kova_zip_jacket", 101),
        ("case_02_sage_sweatpant", 202),
    ]

    created: list[Path] = []
    for case_id, seed in specs:
        model_path = test_cases_root / f"{case_id}.png"
        ref_path = test_cases_root / f"{case_id}_reference.png"
        has_pair = model_path.is_file() and ref_path.is_file()
        if force or not has_pair:
            _draw_procedural_fixture_pair(
                test_cases_root,
                case_id=case_id,
                seed=seed,
            )
        created.append(model_path)
    return created
