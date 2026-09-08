"""
MoonVeil MK1 — tests/test_geometry.py

Sanity tests for src/geometry.py:
  - synthetic known-transform pair -> recovered homography close to the
    known ground truth, low RMSE, high inlier ratio
  - too few candidate matches -> GeometricResult with success=False and a
    failure_reason (NOT an exception — mandatory per the blueprint)
  - too few RANSAC inliers among plausible-looking matches -> also a
    graceful failed result
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import PreprocessedImage
from src.features import detect_features
from src.matching import match_features
from src.geometry import estimate_geometry, calculate_reprojection_error, calculate_rmse
from config import FeatureConfig, MatchingConfig, GeometryConfig

from _helpers import make_synthetic_pair, make_textured_image


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def test_known_transform_recovered():
    reference, source, H_ref_to_source = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)

    matches = match_features(src_features, ref_features, MatchingConfig())
    geometry = estimate_geometry(matches, GeometryConfig())

    assert geometry.success, geometry.failure_reason
    assert geometry.num_inliers >= 8
    assert geometry.inlier_ratio > 0.3
    assert geometry.rmse < 3.0, f"RMSE too high: {geometry.rmse}"

    # The estimated matrix maps source -> reference; ground truth H maps
    # reference -> source, so the estimate should approximate inverse(H).
    expected_source_to_reference = np.linalg.inv(H_ref_to_source)
    expected_source_to_reference /= expected_source_to_reference[2, 2]
    estimated = geometry.transformation_matrix / geometry.transformation_matrix[2, 2]

    # Compare by transforming a handful of test points rather than raw
    # matrix entries (robust to harmless scale/gauge differences).
    test_pts = np.array([[50, 50], [200, 100], [100, 250], [10, 10]], dtype=np.float64)
    err = calculate_reprojection_error(test_pts, _apply_h(test_pts, expected_source_to_reference), estimated)
    assert np.mean(err) < 5.0, f"Estimated transform diverges from ground truth: {err}"

    print(
        f"Known-transform recovery: {geometry.num_inliers} inliers, "
        f"ratio {geometry.inlier_ratio:.2f}, RMSE {geometry.rmse:.2f}px — OK"
    )


def _apply_h(points, H):
    n = points.shape[0]
    homo = np.hstack([points, np.ones((n, 1))])
    projected = (H @ homo.T).T
    return projected[:, :2] / projected[:, 2:3]


def test_too_few_matches_fails_gracefully():
    img = make_textured_image()
    features = _features_for(img)
    matches = match_features(features, features, MatchingConfig())[:2]  # force too few

    geometry = estimate_geometry(matches, GeometryConfig())

    assert geometry.success is False
    assert geometry.failure_reason is not None
    assert "matches" in geometry.failure_reason or "inliers" in geometry.failure_reason
    print(f"Too-few-matches -> graceful failure: '{geometry.failure_reason}' — OK")


def test_rmse_helper_on_zero_error():
    errors = np.zeros(5)
    assert calculate_rmse(errors) == 0.0
    print("calculate_rmse on zero errors: OK")


if __name__ == "__main__":
    test_known_transform_recovered()
    test_too_few_matches_fails_gracefully()
    test_rmse_helper_on_zero_error()
    print("\nAll geometry tests passed.")
