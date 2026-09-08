"""
MoonVeil MK1 — src/data_loader.py

Responsibility: load ISRO (or any raster/standard-format) image products
into NumPy arrays while preserving useful metadata, WITHOUT any
preprocessing, feature detection, or CV logic. That belongs downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .types import ImageData, SensorType

_ALLOWED_SENSORS = {s.value for s in SensorType}

# Formats readable via plain Pillow/OpenCV (no georeferencing needed)
_STANDARD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
# Formats that may carry georeferencing / multiple bands — prefer rasterio,
# fall back to tifffile if rasterio is unavailable or the file isn't
# georeferenced.
_RASTER_EXTENSIONS = {".tif", ".tiff"}


import xml.etree.ElementTree as ET


def parse_pds4_xml(xml_path: Union[str, Path]) -> dict:
    """Parse a PDS4 XML label file for Chandrayaan-2 metadata."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return {}

    metadata = {"xml_path": str(xml_path)}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def get_tag_text(tag_names: list[str]) -> Optional[str]:
            for elem in root.iter():
                clean_tag = elem.tag.split('}', 1)[-1]
                if clean_tag in tag_names and elem.text:
                    return elem.text.strip()
            return None

        target = get_tag_text(["target_name", "name"])
        if target:
            metadata["target_name"] = target

        inst = get_tag_text(["instrument_id", "instrument_name"])
        if inst:
            metadata["instrument_id"] = inst

        pixel_res = get_tag_text(["pixel_resolution", "pixel_scale"])
        if pixel_res:
            try:
                metadata["pixel_resolution_m"] = float(pixel_res)
            except ValueError:
                metadata["pixel_resolution_raw"] = pixel_res

        start_t = get_tag_text(["start_date_time", "start_time"])
        if start_t:
            metadata["start_time"] = start_t

        stop_t = get_tag_text(["stop_date_time", "stop_time"])
        if stop_t:
            metadata["stop_time"] = stop_t

    except Exception as exc:
        metadata["xml_parse_error"] = str(exc)

    return metadata


def load_image(
    path: Union[str, Path],
    sensor: str,
    band: Optional[int] = None,
) -> ImageData:
    """
    Load an image product into a MoonVeil ImageData object.

    Parameters
    ----------
    path : str | Path
        Path to the image file.
    sensor : str
        One of "OHRC", "TMC", "IIRS" (case-sensitive, canonical form).
    band : int, optional
        For multi-band (e.g. IIRS hyperspectral) products, select a single
        band to load as the working 2D representation. If None and the
        source is multi-band, the full band stack is returned with shape
        (bands, height, width) — band selection then becomes preprocessing's
        job, per the IIRS data schema.

    Returns
    -------
    ImageData

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If `sensor` is not a recognised MoonVeil sensor, or `band` is out
        of range for the loaded product.
    """
    path = Path(path)

    if sensor not in _ALLOWED_SENSORS:
        raise ValueError(
            f"Unsupported sensor '{sensor}'. Must be one of {_ALLOWED_SENSORS}."
        )

    # Allow loading passing XML directly if image file has matching stem
    if path.suffix.lower() == ".xml":
        xml_sidecar = path
        possible_imgs = [path.with_suffix(ext) for ext in _RASTER_EXTENSIONS | _STANDARD_EXTENSIONS]
        found_img = next((p for p in possible_imgs if p.exists()), None)
        if found_img is None:
            raise FileNotFoundError(f"PDS4 XML passed but no matching image file found for {path}")
        path = found_img
    else:
        xml_sidecar = path.with_suffix(".xml")

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()

    if ext in _RASTER_EXTENSIONS:
        image = _load_raster(path)
    elif ext in _STANDARD_EXTENSIONS:
        image = _load_standard(path)
    else:
        # Attempt raster loader as a reasonable default for unknown
        # extensions before giving up.
        try:
            image = _load_raster(path)
        except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
            raise ValueError(
                f"Unrecognised or unreadable image format '{ext}' for {path}: {exc}"
            ) from exc

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        # Convention: (bands, height, width) for raster stacks.
        channels, height, width = image.shape
    else:
        raise ValueError(
            f"Loaded array for {path} has unexpected ndim={image.ndim}; "
            "expected 2 (H, W) or 3 (bands, H, W)."
        )

    if band is not None:
        if image.ndim != 3:
            raise ValueError(
                f"band={band} requested but loaded image for {path} is not "
                f"multi-band (ndim={image.ndim})."
            )
        if not (0 <= band < channels):
            raise ValueError(
                f"band={band} out of range for {path}; product has "
                f"{channels} bands (valid range 0-{channels - 1})."
            )
        image = image[band]
        channels = 1
        height, width = image.shape

    pds4_meta = parse_pds4_xml(xml_sidecar) if xml_sidecar.exists() else {}

    return ImageData(
        image=image,
        sensor=sensor,
        path=str(path),
        width=int(width),
        height=int(height),
        channels=int(channels),
        dtype=str(image.dtype),
        metadata={"band_selected": band, "pds4_metadata": pds4_meta},
    )


def _load_standard(path: Path) -> np.ndarray:
    """Load a plain image format (PNG/JPG/BMP) via Pillow, grayscale-safe."""
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        arr = np.array(img)
    return arr


def _load_raster(path: Path) -> np.ndarray:
    """
    Load a (Geo)TIFF via rasterio if available and the file is readable as
    a raster; otherwise fall back to tifffile for plain (non-georeferenced)
    TIFF stacks.

    Returns an array shaped (H, W) for single-band, or (bands, H, W) for
    multi-band products.
    """
    rasterio_arr = None
    try:
        import rasterio

        with rasterio.open(path) as src:
            rasterio_arr = src.read()  # shape: (bands, H, W)
    except Exception:
        pass  # fall through to tifffile-only path below

    # A TIFF can encode "many bands" two different ways: as true GDAL-style
    # bands within one page (SamplesPerPixel), which rasterio reads
    # correctly, OR as a multi-page/multi-IFD stack (e.g. what
    # tifffile.imwrite(path, 3d_array) produces), which rasterio sees as a
    # single-band file (only the first page). We cross-check page count via
    # tifffile and prefer whichever representation actually captures all
    # the data.
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tf:
            n_pages = len(tf.pages)
    except Exception:
        n_pages = 1

    rasterio_bands = rasterio_arr.shape[0] if rasterio_arr is not None else 0

    if rasterio_arr is not None and rasterio_bands >= n_pages and rasterio_bands > 0:
        arr = rasterio_arr
    else:
        # Either rasterio failed, or it under-counted bands relative to the
        # page stack — read the full multi-page stack via tifffile instead.
        arr = tifffile.imread(str(path))
        # tifffile may return (H, W), (H, W, bands) or (bands, H, W)
        # depending on how the file was written. Normalise
        # (H, W, bands) -> (bands, H, W) when the last axis looks like a
        # small band count rather than image width.
        if arr.ndim == 3 and arr.shape[-1] <= 32 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # collapse to (H, W) for single-band

    return arr

