"""
MoonVeil MK1 — tests/test_loader.py

Sanity tests for src/data_loader.py using synthetic (non-ISRO) images,
since real Chandrayaan-2 data is not yet available. Covers:
  - standard format (PNG) load
  - single-band TIFF load
  - multi-band TIFF load + band selection
  - missing file -> FileNotFoundError
  - bad sensor string -> ValueError
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_image


def test_load_png():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.png"
        arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
        PILImage.fromarray(arr).save(path)

        data = load_image(path, sensor="OHRC")
        assert data.height == 64 and data.width == 64
        assert data.channels == 1
        assert data.sensor == "OHRC"
        print("PNG load: OK")


def test_load_single_band_tiff():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.tif"
        arr = (np.random.rand(50, 40) * 255).astype(np.uint8)
        tifffile.imwrite(path, arr)

        data = load_image(path, sensor="TMC")
        assert data.height == 50 and data.width == 40
        assert data.channels == 1
        print("Single-band TIFF load: OK")


def test_load_multiband_tiff_and_band_select():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cube.tif"
        # Simulate an IIRS-like hyperspectral cube: (bands, H, W)
        arr = (np.random.rand(10, 30, 20) * 255).astype(np.uint8)
        tifffile.imwrite(path, arr)

        full = load_image(path, sensor="IIRS")
        assert full.channels == 10
        assert full.height == 30 and full.width == 20

        single = load_image(path, sensor="IIRS", band=3)
        assert single.channels == 1
        assert single.metadata["band_selected"] == 3
        print("Multi-band TIFF load + band selection: OK")


def test_missing_file_raises():
    try:
        load_image("/tmp/does_not_exist_moonveil.tif", sensor="OHRC")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        print("Missing file -> FileNotFoundError: OK")


def test_bad_sensor_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.png"
        arr = (np.random.rand(10, 10) * 255).astype(np.uint8)
        PILImage.fromarray(arr).save(path)
        try:
            load_image(path, sensor="NOT_A_SENSOR")
            raise AssertionError("Expected ValueError")
        except ValueError:
            print("Bad sensor -> ValueError: OK")


if __name__ == "__main__":
    test_load_png()
    test_load_single_band_tiff()
    test_load_multiband_tiff_and_band_select()
    test_missing_file_raises()
    test_bad_sensor_raises()
    print("\nAll data_loader tests passed.")
