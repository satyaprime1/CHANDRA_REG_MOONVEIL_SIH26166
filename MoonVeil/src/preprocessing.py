"""
MoonVeil MK1 — src/preprocessing.py

Responsibility: turn a raw loaded ImageData into a working, analysis-ready
2D grayscale representation, without ever mutating the original array.
Handles: invalid/no-data values, multi-band -> 2D reduction, intensity
normalization, optional contrast enhancement, optional illumination
normalization, optional resizing.
"""

from __future__ import annotations

import numpy as np
import cv2

from .types import ImageData, PreprocessedImage
from config import PreprocessingConfig


def preprocess_image(
    image_data: ImageData,
    config: PreprocessingConfig,
) -> PreprocessedImage:
    """
    Parameters
    ----------
    image_data : ImageData
        Raw loaded image (2D grayscale or 3D band-stack). Never modified.
    config : PreprocessingConfig

    Returns
    -------
    PreprocessedImage
    """
    original_shape = image_data.image.shape
    working = image_data.image.copy()  # never mutate the source array

    metadata: dict = dict(image_data.metadata) if image_data.metadata else {}
    steps_applied = []

    # --- 1. Multi-band -> single working 2D representation -----------------
    if working.ndim == 3:
        # (bands, H, W) -> band-derived representation. Mk1 default: mean
        # across bands. Real spectral-index-based representations for IIRS
        # can replace this later without touching downstream modules.
        working = working.mean(axis=0)
        metadata["band_derivation_method"] = "mean_across_bands"
        steps_applied.append("band_reduction(mean)")
    elif working.ndim != 2:
        raise ValueError(
            f"preprocess_image expects a 2D or 3D array, got ndim={working.ndim}"
        )

    # --- 2. Handle invalid / no-data values ---------------------------------
    working = working.astype(np.float64)
    invalid_mask = ~np.isfinite(working)
    if invalid_mask.any():
        finite_vals = working[~invalid_mask]
        fill_value = float(np.median(finite_vals)) if finite_vals.size else 0.0
        working[invalid_mask] = fill_value
        steps_applied.append(f"nodata_fill(median={fill_value:.2f})")

    # --- 3. Normalize to a suitable numerical range -------------------------
    normalization_method = "none"
    if config.normalize:
        # Robust normalization: clip to 1st/99th percentile before scaling,
        # so a handful of extreme outlier pixels don't wash out contrast.
        lo, hi = np.percentile(working, [1, 99])
        if hi <= lo:
            # Degenerate (near-constant) image — avoid divide-by-zero.
            hi = lo + 1.0
        working = np.clip(working, lo, hi)
        working = (working - lo) / (hi - lo) * 255.0
        working = working.astype(np.uint8)
        normalization_method = "percentile_clip_1_99_to_uint8"
        steps_applied.append(normalization_method)
    else:
        # Still need a concrete dtype for downstream OpenCV calls.
        working = np.clip(working, 0, 255).astype(np.uint8)

    # --- 4. Optional contrast enhancement ------------------------------------
    if config.contrast_method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        working = clahe.apply(working)
        steps_applied.append("CLAHE")
    elif config.contrast_method not in (None, "CLAHE"):
        raise ValueError(
            f"Unsupported contrast_method '{config.contrast_method}'. "
            "Use None or 'CLAHE'."
        )

    # --- 5. Optional illumination normalization ------------------------------
    if config.illumination_normalization:
        # Homomorphic-style correction: divide out a large-kernel blurred
        # (low-frequency / illumination) version, then renormalize. This is
        # a simple, well-understood technique appropriate for a Mk1
        # baseline rather than a full illumination model.
        blur_kernel = max(3, (min(working.shape) // 8) | 1)  # odd, scale-aware
        illumination = cv2.GaussianBlur(
            working.astype(np.float64), (blur_kernel, blur_kernel), 0
        )
        illumination[illumination == 0] = 1.0
        corrected = (working.astype(np.float64) / illumination) * illumination.mean()
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        working = corrected
        steps_applied.append(f"illumination_normalization(kernel={blur_kernel})")

    # --- 6. Optional resizing --------------------------------------------------
    scale_factor = 1.0
    if config.resize_factor != 1.0:
        if config.resize_factor <= 0:
            raise ValueError("resize_factor must be > 0")
        new_w = max(1, int(round(working.shape[1] * config.resize_factor)))
        new_h = max(1, int(round(working.shape[0] * config.resize_factor)))
        working = cv2.resize(working, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_factor = config.resize_factor
        steps_applied.append(f"resize(factor={config.resize_factor})")

    metadata["preprocessing_steps"] = steps_applied

    return PreprocessedImage(
        image=working,
        original_shape=original_shape,
        processed_shape=working.shape,
        sensor=image_data.sensor,
        normalization_method=normalization_method,
        scale_factor=scale_factor,
        metadata=metadata,
    )

