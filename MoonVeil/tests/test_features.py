"""
MoonVeil MK1 — tests/test_features.py

Sanity tests for src/features.py:
  - a textured synthetic image yields a non-empty FeatureSet
  - a perfectly flat/uniform image raises ValueError (no usable features)
  - a non-2D input raises ValueError
  - an unsupported detector name raises ValueError
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import PreprocessedImage
from src.features import detect_features
from config import FeatureConfig


def _make_textured_image(size=200):
    """Draw shapes/text to give SIFT plenty of real corners/blobs to find —
    more representative of lunar crater/rock texture than pure noise."""
    img = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(40):
        center = tuple(rng.integers(10, size - 10, size=2).tolist())
        radius = int(rng.integers(3, 15))
        color = int(rng.integers(80, 255))
        cv2.circle(img, center, radius, color, -1)
    for _ in range(15):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple(rng.integers(0, size, size=2).tolist())
        cv2.line(img, pt1, pt2, int(rng.integers(50, 255)), 2)
    return img


def _wrap(arr):
    return PreprocessedImage(
        image=arr, original_shape=arr.shape, processed_shape=arr.shape,
        sensor="OHRC", normalization_method="none", scale_factor=1.0,
    )


def test_textured_image_has_features():
    img = _wrap(_make_textured_image())
    result = detect_features(img, FeatureConfig())
    assert len(result.keypoints) > 0
    assert result.descriptors.shape[0] == len(result.keypoints)
    assert result.descriptors.shape[1] == 128
    assert result.detector_name == "SIFT"
    print(f"Textured image: {len(result.keypoints)} keypoints found — OK")


def test_flat_image_raises():
    flat = _wrap(np.full((100, 100), 128, dtype=np.uint8))
    try:
        detect_features(flat, FeatureConfig())
        raise AssertionError("Expected ValueError for flat image")
    except ValueError:
        print("Flat image -> ValueError: OK")


def test_non_2d_raises():
    bad = PreprocessedImage(
        image=np.zeros((3, 50, 50), dtype=np.uint8),
        original_shape=(3, 50, 50), processed_shape=(3, 50, 50),
        sensor="IIRS", normalization_method="none", scale_factor=1.0,
    )
    try:
        detect_features(bad, FeatureConfig())
        raise AssertionError("Expected ValueError for non-2D input")
    except ValueError:
        print("Non-2D input -> ValueError: OK")


def test_bad_detector_raises():
    img = _wrap(_make_textured_image())
    try:
        detect_features(img, FeatureConfig(detector="ORB_UNSUPPORTED"))
        raise AssertionError("Expected ValueError for unsupported detector")
    except ValueError:
        print("Unsupported detector -> ValueError: OK")


if __name__ == "__main__":
    test_textured_image_has_features()
    test_flat_image_raises()
    test_non_2d_raises()
    test_bad_detector_raises()
    print("\nAll features tests passed.")
