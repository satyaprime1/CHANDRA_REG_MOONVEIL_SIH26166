"""
MoonVeil MK1 — tests/test_registration.py

Sanity tests for src/registration.py:
  - known-transform pair: warping source with the RANSAC-estimated
    transformation should align it closely with the reference image over
    the valid (non-border) overlap region
  - transformation_matrix=None -> graceful failure, not an exception
  - wrong-shaped matrix -> graceful failure
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import PreprocessedImage
from src.features import detect_features
from src.matching import match_features
from src.geometry import estimate_geometry
from src.registration import register_image
from config import FeatureConfig, MatchingConfig, GeometryConfig

from _helpers import make_synthetic_pair


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def test_registration_aligns_with_reference():
    reference, source, _ = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)
    matches = match_features(src_features, ref_features, MatchingConfig())
    geometry = estimate_geometry(matches, GeometryConfig())
    assert geometry.success

    source_wrapped = PreprocessedImage(
        image=source, original_shape=source.shape, processed_shape=source.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )

    result = register_image(source_wrapped, geometry.transformation_matrix, reference.shape)

    assert result.success
    assert result.registered_image.shape == reference.shape

    # Compare only the "valid" region (where the warp actually placed
    # source content, i.e. non-zero pixels) — border pixels from warping
    # are expected to be black and shouldn't be judged as misalignment.
    valid_mask = result.registered_image > 0
    assert valid_mask.sum() > 0.5 * reference.size, "warp produced too little valid content"

    diff = np.abs(
        result.registered_image[valid_mask].astype(np.int16)
        - reference[valid_mask].astype(np.int16)
    )
    mean_abs_diff = diff.mean()
    assert mean_abs_diff < 15.0, f"registered image too misaligned: mean abs diff {mean_abs_diff:.2f}"
    print(f"Registration alignment: mean abs diff = {mean_abs_diff:.2f} (0-255 scale) — OK")


def test_none_matrix_fails_gracefully():
    reference, source, _ = make_synthetic_pair()
    source_wrapped = PreprocessedImage(
        image=source, original_shape=source.shape, processed_shape=source.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    result = register_image(source_wrapped, None, reference.shape)
    assert result.success is False
    assert result.failure_reason is not None
    print(f"None matrix -> graceful failure: '{result.failure_reason}' — OK")


def test_bad_shape_matrix_fails_gracefully():
    reference, source, _ = make_synthetic_pair()
    source_wrapped = PreprocessedImage(
        image=source, original_shape=source.shape, processed_shape=source.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    bad_matrix = np.eye(2)  # wrong shape, not (3,3)
    result = register_image(source_wrapped, bad_matrix, reference.shape)
    assert result.success is False
    print(f"Bad-shape matrix -> graceful failure: '{result.failure_reason}' — OK")


if __name__ == "__main__":
    test_registration_aligns_with_reference()
    test_none_matrix_fails_gracefully()
    test_bad_shape_matrix_fails_gracefully()
    print("\nAll registration tests passed.")
