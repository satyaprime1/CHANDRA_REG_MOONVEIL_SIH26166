"""
MoonVeil MK1 — tests/test_preprocessing.py

Sanity tests for src/preprocessing.py using synthetic data:
  - basic normalization preserves shape and produces uint8 in [0,255]
  - original ImageData.image is never mutated
  - multi-band input collapses to a 2D working representation
  - NaN/invalid values get filled rather than propagating
  - resize_factor changes processed_shape as expected
  - CLAHE and illumination normalization run without error
  - bad contrast_method raises ValueError
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import ImageData
from src.preprocessing import preprocess_image
from config import PreprocessingConfig


def _make_image_data(arr, sensor="OHRC"):
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    else:
        c, h, w = arr.shape
    return ImageData(
        image=arr, sensor=sensor, path="dummy.tif",
        width=w, height=h, channels=c, dtype=str(arr.dtype),
    )


def test_basic_normalization_shape_and_range():
    arr = (np.random.rand(64, 80) * 1000).astype(np.float32)
    data = _make_image_data(arr)
    original_copy = arr.copy()

    result = preprocess_image(data, PreprocessingConfig())

    assert result.processed_shape == (64, 80)
    assert result.image.dtype == np.uint8
    assert result.image.min() >= 0 and result.image.max() <= 255
    assert np.array_equal(data.image, original_copy)
    print("Basic normalization + no-mutation: OK")


def test_multiband_reduction():
    arr = (np.random.rand(6, 30, 40) * 255).astype(np.uint8)
    data = _make_image_data(arr, sensor="IIRS")

    result = preprocess_image(data, PreprocessingConfig())

    assert result.image.ndim == 2
    assert result.processed_shape == (30, 40)
    assert result.metadata["band_derivation_method"] == "mean_across_bands"
    print("Multi-band reduction: OK")


def test_nodata_handling():
    arr = np.random.rand(20, 20).astype(np.float64) * 255
    arr[5, 5] = np.nan
    arr[10, 10] = np.inf
    data = _make_image_data(arr)

    result = preprocess_image(data, PreprocessingConfig())

    assert np.isfinite(result.image).all()
    print("NaN/Inf handling: OK")


def test_resize_factor():
    arr = (np.random.rand(100, 100) * 255).astype(np.uint8)
    data = _make_image_data(arr)

    result = preprocess_image(data, PreprocessingConfig(resize_factor=0.5))

    assert result.processed_shape == (50, 50)
    assert result.scale_factor == 0.5
    print("Resize factor: OK")


def test_clahe_and_illumination_run():
    arr = (np.random.rand(60, 60) * 255).astype(np.uint8)
    data = _make_image_data(arr)

    result = preprocess_image(
        data,
        PreprocessingConfig(contrast_method="CLAHE", illumination_normalization=True),
    )
    assert result.image.shape == (60, 60)
    assert "CLAHE" in result.metadata["preprocessing_steps"]
    assert any("illumination" in s for s in result.metadata["preprocessing_steps"])
    print("CLAHE + illumination normalization: OK")


def test_bad_contrast_method_raises():
    arr = (np.random.rand(10, 10) * 255).astype(np.uint8)
    data = _make_image_data(arr)
    try:
        preprocess_image(data, PreprocessingConfig(contrast_method="NOT_A_METHOD"))
        raise AssertionError("Expected ValueError")
    except ValueError:
        print("Bad contrast_method -> ValueError: OK")


if __name__ == "__main__":
    test_basic_normalization_shape_and_range()
    test_multiband_reduction()
    test_nodata_handling()
    test_resize_factor()
    test_clahe_and_illumination_run()
    test_bad_contrast_method_raises()
    print("\nAll preprocessing tests passed.")
