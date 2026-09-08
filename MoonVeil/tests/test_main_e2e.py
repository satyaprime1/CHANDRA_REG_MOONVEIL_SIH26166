"""
MoonVeil MK1 — tests/test_main_e2e.py

End-to-end test of run_moonveil() itself, not just individual modules.
Two cases, mirroring the blueprint's "Case A / Case C" demonstration
requirement:
  - Case A (success): a good synthetic OHRC-like pair -> full pipeline
    runs, registration succeeds, metrics are sane.
  - Case C (failure/ambiguity): a pair with no real correspondence (two
    unrelated textures) -> pipeline runs WITHOUT crashing and correctly
    reports REGISTRATION NOT RELIABLE.
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_moonveil
from _helpers import make_synthetic_pair, make_textured_image


def _write_tiff(arr, path):
    tifffile.imwrite(path, arr)


def test_case_a_success():
    reference, source, _ = make_synthetic_pair()
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.tif"
        src_path = Path(tmp) / "source.tif"
        _write_tiff(reference, ref_path)
        _write_tiff(source, src_path)

        result = run_moonveil(ref_path, src_path, reference_sensor="OHRC")

        assert result.evaluation.registration_success is True
        assert result.evaluation.reliability_status in ("HIGH", "MEDIUM")
        assert result.registration.registered_image is not None
        assert result.registration.registered_image.shape == reference.shape

        print(
            f"Case A (success): {result.evaluation.candidate_matches} candidates, "
            f"{result.evaluation.verified_inliers} inliers, "
            f"RMSE {result.evaluation.rmse:.2f}px, "
            f"overall {result.evaluation.overall_reliability:.1f} "
            f"({result.evaluation.reliability_status}) — OK"
        )


def test_case_c_failure_is_graceful():
    # Two completely unrelated textured images — no real correspondence
    # should exist between them beyond coincidental noise.
    reference = make_textured_image(size=200, seed=1)
    unrelated_source = make_textured_image(size=200, seed=12345)

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.tif"
        src_path = Path(tmp) / "source.tif"
        _write_tiff(reference, ref_path)
        _write_tiff(unrelated_source, src_path)

        result = run_moonveil(ref_path, src_path, reference_sensor="OHRC")

        # The pipeline must complete without raising, and must NOT claim
        # a trustworthy registration on unrelated images.
        assert result.evaluation.registration_success is False
        assert result.status.success is False
        assert "REGISTRATION NOT RELIABLE" in (result.registration.failure_reason or "")

        print(
            f"Case C (failure): registration correctly rejected — "
            f"'{result.registration.failure_reason}' — OK, no crash"
        )


if __name__ == "__main__":
    test_case_a_success()
    test_case_c_failure_is_graceful()
    print("\nAll end-to-end tests passed.")
