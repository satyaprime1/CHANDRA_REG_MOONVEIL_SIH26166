"""
MoonVeil MK1 — tests/test_evaluation.py

Sanity tests for src/evaluation.py:
  - end-to-end synthetic pair: every aggregated number matches what can be
    hand-verified from the upstream results it was built from
  - a failed pipeline (no successful geometry, no registration) still
    produces a valid, non-crashing EvaluationResult with LOW status
  - bad weights raise ValueError
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
from src.registration import register_image
from src.evaluation import evaluate
from config import (
    FeatureConfig, MatchingConfig, GeometryConfig, SpatialConfig,
    ReliabilityConfig, EvaluationConfig,
)

from _helpers import make_synthetic_pair, make_textured_image


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def test_full_pipeline_metrics_are_consistent():
    reference, source, _ = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)
    matches = match_features(src_features, ref_features, MatchingConfig())
    geometry = estimate_geometry(matches, GeometryConfig())
    assert geometry.success

    inlier_ref_pts = np.array([m.reference_point for m in geometry.inlier_matches])
    spatial = analyze_spatial_distribution(inlier_ref_pts, reference.shape, SpatialConfig())

    reliabilities = calculate_match_reliability(matches, geometry, spatial, ReliabilityConfig())

    source_wrapped = PreprocessedImage(
        image=source, original_shape=source.shape, processed_shape=source.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    registration = register_image(source_wrapped, geometry.transformation_matrix, reference.shape)
    assert registration.success

    result = evaluate(matches, geometry, reliabilities, spatial, registration, EvaluationConfig())

    # Hand-verifiable consistency checks against the upstream objects.
    assert result.candidate_matches == len(matches)
    assert result.verified_inliers == geometry.num_inliers
    assert result.rejected_matches == len(matches) - geometry.num_inliers
    assert abs(result.inlier_ratio - geometry.inlier_ratio) < 1e-9
    assert abs(result.rmse - geometry.rmse) < 1e-9
    assert abs(result.mean_reprojection_error - float(np.mean(geometry.reprojection_errors))) < 1e-6
    assert abs(result.median_reprojection_error - float(np.median(geometry.reprojection_errors))) < 1e-6
    assert abs(result.max_reprojection_error - float(np.max(geometry.reprojection_errors))) < 1e-6
    assert abs(result.spatial_coverage - spatial.coverage_ratio) < 1e-9
    assert result.registration_success is True
    assert 0.0 <= result.overall_reliability <= 100.0
    assert result.reliability_status in ("HIGH", "MEDIUM", "LOW")

    print(
        f"Full pipeline eval: {result.candidate_matches} candidates, "
        f"{result.verified_inliers} inliers ({result.inlier_ratio:.1%}), "
        f"RMSE {result.rmse:.2f}px, spatial coverage {result.spatial_coverage:.1%}, "
        f"avg reliability {result.average_reliability:.1f}, "
        f"overall {result.overall_reliability:.1f} ({result.reliability_status}) — OK"
    )


def test_failed_pipeline_produces_low_status_not_a_crash():
    img = make_textured_image()
    features = _features_for(img)
    matches = match_features(features, features, MatchingConfig())[:2]  # force geometry failure
    geometry = estimate_geometry(matches, GeometryConfig())
    assert not geometry.success

    spatial = analyze_spatial_distribution(np.zeros((0, 2)), img.shape, SpatialConfig())
    reliabilities = calculate_match_reliability(matches, geometry, spatial, ReliabilityConfig())

    wrapped = PreprocessedImage(
        image=img, original_shape=img.shape, processed_shape=img.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    registration = register_image(wrapped, geometry.transformation_matrix, img.shape)
    assert not registration.success  # no matrix was ever produced

    result = evaluate(matches, geometry, reliabilities, spatial, registration, EvaluationConfig())

    assert result.registration_success is False
    assert result.reliability_status == "LOW"
    assert result.overall_reliability < 50.0
    print(
        f"Failed pipeline -> overall {result.overall_reliability:.1f} "
        f"({result.reliability_status}), registration_success=False — OK, no crash"
    )


def test_bad_weights_raise():
    bad_config = EvaluationConfig(weight_average_reliability=0.9)  # sum != 1
    reference, source, _ = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)
    matches = match_features(src_features, ref_features, MatchingConfig())
    geometry = estimate_geometry(matches, GeometryConfig())
    spatial = analyze_spatial_distribution(
        np.array([m.reference_point for m in geometry.inlier_matches]), reference.shape, SpatialConfig()
    )
    reliabilities = calculate_match_reliability(matches, geometry, spatial, ReliabilityConfig())
    source_wrapped = PreprocessedImage(
        image=source, original_shape=source.shape, processed_shape=source.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    registration = register_image(source_wrapped, geometry.transformation_matrix, reference.shape)

    try:
        evaluate(matches, geometry, reliabilities, spatial, registration, bad_config)
        raise AssertionError("Expected ValueError for bad weights")
    except ValueError:
        print("Bad evaluation weights -> ValueError: OK")


if __name__ == "__main__":
    test_full_pipeline_metrics_are_consistent()
    test_failed_pipeline_produces_low_status_not_a_crash()
    test_bad_weights_raise()
    print("\nAll evaluation tests passed.")
