"""
MoonVeil MK1 — tests/test_visualization.py

Smoke tests for src/visualization.py. These functions are draw-only, so
correctness here means: runs without error, returns a BGR uint8 image of
sane shape. Pixel-perfect appearance is a human/visual judgment call, not
something worth over-specifying in an automated test.
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
from src.visualization import (
    draw_matches, draw_reliability_matches, draw_registration_overlay,
    draw_spatial_distribution,
)
from config import FeatureConfig, MatchingConfig, GeometryConfig, SpatialConfig, ReliabilityConfig

from _helpers import make_synthetic_pair


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def test_all_visualizations_run():
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

    img1 = draw_matches(reference, source, matches, status="all")
    assert img1.ndim == 3 and img1.shape[2] == 3 and img1.dtype == np.uint8
    assert img1.shape[0] == reference.shape[0]
    assert img1.shape[1] == reference.shape[1] + source.shape[1]

    img2 = draw_matches(reference, source, geometry.inlier_matches, status="inliers")
    img3 = draw_matches(reference, source, geometry.outlier_matches, status="outliers")
    for img in (img2, img3):
        assert img.dtype == np.uint8 and img.ndim == 3

    img4 = draw_reliability_matches(reference, source, matches, reliabilities)
    assert img4.dtype == np.uint8 and img4.ndim == 3

    img5 = draw_registration_overlay(reference, registration.registered_image, alpha=0.7)
    assert img5.shape[:2] == reference.shape
    assert img5.dtype == np.uint8

    img6 = draw_spatial_distribution(reference, spatial)
    assert img6.shape[:2] == reference.shape
    assert img6.dtype == np.uint8

    print("All visualization functions ran and returned well-formed BGR images: OK")


if __name__ == "__main__":
    test_all_visualizations_run()
    print("\nAll visualization tests passed.")
