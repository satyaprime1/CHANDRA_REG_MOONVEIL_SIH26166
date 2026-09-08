"""
MoonVeil MK1 — tests/test_reliability.py

Sanity tests for src/reliability.py:
  - end-to-end on a synthetic known-transform pair: RANSAC inliers should
    mostly score HIGH/MEDIUM, RANSAC outliers must ALWAYS be REJECTED
    regardless of how good their descriptor evidence looked
  - invalid (non-summing-to-1) weights raise ValueError
  - empty match list returns an empty list without error
  - a failed GeometricResult (no successful RANSAC) results in every match
    being REJECTED — the "insufficient confidence" behaviour
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import PreprocessedImage
from src.features import detect_features
from src.matching import match_features
from src.geometry import estimate_geometry
from src.spatial import analyze_spatial_distribution
from src.reliability import calculate_match_reliability
from config import FeatureConfig, MatchingConfig, GeometryConfig, SpatialConfig, ReliabilityConfig

from _helpers import make_synthetic_pair, make_textured_image


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def _full_pipeline():
    reference, source, _ = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)
    matches = match_features(src_features, ref_features, MatchingConfig())
    geometry = estimate_geometry(matches, GeometryConfig())
    inlier_ref_pts = np.array([m.reference_point for m in geometry.inlier_matches])
    spatial = analyze_spatial_distribution(inlier_ref_pts, reference.shape, SpatialConfig())
    return matches, geometry, spatial


def test_inliers_score_well_outliers_rejected():
    matches, geometry, spatial = _full_pipeline()
    assert geometry.success

    reliabilities = calculate_match_reliability(matches, geometry, spatial, ReliabilityConfig())
    assert len(reliabilities) == len(matches)

    by_id = {r.match_id: r for r in reliabilities}
    inlier_ids = {m.match_id for m in geometry.inlier_matches}
    outlier_ids = {m.match_id for m in geometry.outlier_matches}

    for mid in outlier_ids:
        assert by_id[mid].status == "REJECTED", "every outlier must be REJECTED"

    inlier_statuses = [by_id[mid].status for mid in inlier_ids]
    high_or_medium = sum(1 for s in inlier_statuses if s in ("HIGH", "MEDIUM"))
    assert high_or_medium / len(inlier_statuses) > 0.7, (
        f"expected most inliers HIGH/MEDIUM, got {inlier_statuses}"
    )

    high_score = by_id[next(iter(inlier_ids))]
    print(
        f"{len(inlier_ids)} inliers, {len(outlier_ids)} outliers. "
        f"{high_or_medium}/{len(inlier_statuses)} inliers HIGH/MEDIUM. "
        f"All outliers REJECTED. Sample score={high_score.reliability_score:.1f} — OK"
    )


def test_bad_weights_raise():
    bad_config = ReliabilityConfig(weight_descriptor=0.5)  # sum will be > 1
    matches, geometry, spatial = _full_pipeline()
    try:
        calculate_match_reliability(matches, geometry, spatial, bad_config)
        raise AssertionError("Expected ValueError for bad weights")
    except ValueError:
        print("Weights not summing to 1 -> ValueError: OK")


def test_empty_matches_returns_empty():
    matches, geometry, spatial = _full_pipeline()
    result = calculate_match_reliability([], geometry, spatial, ReliabilityConfig())
    assert result == []
    print("Empty match list -> empty result: OK")


def test_failed_geometry_rejects_everything():
    img = make_textured_image()
    features = _features_for(img)
    matches = match_features(features, features, MatchingConfig())[:2]  # force failure
    geometry = estimate_geometry(matches, GeometryConfig())
    assert not geometry.success

    spatial = analyze_spatial_distribution(np.zeros((0, 2)), img.shape, SpatialConfig())
    reliabilities = calculate_match_reliability(matches, geometry, spatial, ReliabilityConfig())

    assert all(r.status == "REJECTED" for r in reliabilities)
    print("Failed geometry -> all matches REJECTED: OK")


if __name__ == "__main__":
    test_inliers_score_well_outliers_rejected()
    test_bad_weights_raise()
    test_empty_matches_returns_empty()
    test_failed_geometry_rejects_everything()
    print("\nAll reliability tests passed.")
