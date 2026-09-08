"""
MoonVeil MK1 — tests/test_matching.py

Sanity tests for src/matching.py:
  - same image vs itself -> high match count, ratios all pass and are low
  - a real transformed pair -> a reasonable number of candidate matches
  - empty descriptors -> ValueError
  - unsupported matcher -> ValueError
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import PreprocessedImage, FeatureSet
from src.features import detect_features
from src.matching import match_features
from config import FeatureConfig, MatchingConfig

from _helpers import make_textured_image, make_synthetic_pair


def _features_for(arr):
    wrapped = PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )
    return detect_features(wrapped, FeatureConfig())


def test_self_match_is_strong():
    img = make_textured_image()
    features = _features_for(img)

    matches = match_features(features, features, MatchingConfig())

    assert len(matches) > 20, f"expected many self-matches, got {len(matches)}"
    assert all(m.is_ratio_pass for m in matches)
    # A point matched against itself should have a very tight ratio.
    avg_ratio = np.mean([m.ratio for m in matches])
    assert avg_ratio < 0.5, f"expected low avg ratio for self-match, got {avg_ratio:.3f}"
    print(f"Self-match: {len(matches)} matches, avg ratio {avg_ratio:.3f} — OK")


def test_transformed_pair_has_candidates():
    reference, source, _ = make_synthetic_pair()
    ref_features = _features_for(reference)
    src_features = _features_for(source)

    matches = match_features(src_features, ref_features, MatchingConfig())

    assert len(matches) >= 10, f"expected reasonable matches, got {len(matches)}"
    for m in matches:
        assert m.is_ratio_pass
        assert 0.0 <= m.ratio < 1.0
    print(f"Transformed pair: {len(matches)} candidate matches — OK")


def test_empty_descriptors_raise():
    img = make_textured_image()
    features = _features_for(img)
    empty = FeatureSet(
        keypoints=[], descriptors=np.zeros((0, 128), dtype=np.float32),
        image_shape=img.shape, detector_name="SIFT",
    )
    try:
        match_features(features, empty, MatchingConfig())
        raise AssertionError("Expected ValueError for empty descriptors")
    except ValueError:
        print("Empty descriptors -> ValueError: OK")


def test_bad_matcher_raises():
    img = make_textured_image()
    features = _features_for(img)
    try:
        match_features(features, features, MatchingConfig(matcher="FLANN_UNSUPPORTED"))
        raise AssertionError("Expected ValueError for unsupported matcher")
    except ValueError:
        print("Unsupported matcher -> ValueError: OK")


if __name__ == "__main__":
    test_self_match_is_strong()
    test_transformed_pair_has_candidates()
    test_empty_descriptors_raise()
    test_bad_matcher_raises()
    print("\nAll matching tests passed.")
