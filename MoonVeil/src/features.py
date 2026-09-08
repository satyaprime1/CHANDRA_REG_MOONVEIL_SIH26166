"""
MoonVeil MK1 — src/features.py

Responsibility: find local image structures (keypoints + descriptors) in a
preprocessed image. Mk1 baseline detector: SIFT. This module knows nothing
about matching, geometry, or reliability — it only detects and describes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import FeatureSet, PreprocessedImage
from config import FeatureConfig

_SUPPORTED_DETECTORS = {"SIFT"}


def refine_subpixel_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    win_size: tuple[int, int] = (3, 3),
) -> list[cv2.KeyPoint]:
    """Refine keypoint coordinates to sub-pixel accuracy using local intensity gradients."""
    if not keypoints:
        return keypoints

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    
    try:
        refined_pts = cv2.cornerSubPix(image, pts, win_size, (-1, -1), criteria)
        refined_pts = refined_pts.reshape(-1, 2)
        
        refined_kps = []
        for kp, (x_sub, y_sub) in zip(keypoints, refined_pts):
            refined_kp = cv2.KeyPoint(
                x=float(x_sub),
                y=float(y_sub),
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
            )
            refined_kps.append(refined_kp)
        return refined_kps
    except Exception:
        # Fall back to original keypoints if subpixel refinement fails on edge cases
        return keypoints


def detect_features(image: PreprocessedImage, config: FeatureConfig) -> FeatureSet:
    """
    Parameters
    ----------
    image : PreprocessedImage
        Must already be a 2D, uint8, analysis-ready image (output of
        preprocess_image).
    config : FeatureConfig

    Returns
    -------
    FeatureSet

    Raises
    ------
    ValueError
        If the detector is unsupported, the input isn't 2D, or no usable
        features are found.
    """
    if config.detector not in _SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unsupported detector '{config.detector}'. "
            f"Supported: {_SUPPORTED_DETECTORS}"
        )

    if image.image.ndim != 2:
        raise ValueError(
            f"detect_features expects a 2D image, got ndim={image.image.ndim}. "
            "Did preprocessing run first?"
        )

    working = image.image
    if working.dtype != np.uint8:
        # SIFT expects 8-bit input; be defensive even though preprocessing
        # should already guarantee this.
        working = cv2.normalize(working, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    detector = cv2.SIFT_create(
        nfeatures=config.nfeatures,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )

    keypoints, descriptors = detector.detectAndCompute(working, None)

    if descriptors is None or len(keypoints) == 0:
        raise ValueError(
            "No usable features detected. The image may be too uniform "
            "(e.g. flat lunar shadow region) or preprocessing may have "
            "destroyed useful contrast."
        )

    keypoints = list(keypoints)
    if getattr(config, "subpixel_refinement", True):
        keypoints = refine_subpixel_keypoints(working, keypoints)

    return FeatureSet(
        keypoints=keypoints,
        descriptors=descriptors,
        image_shape=working.shape,
        detector_name=config.detector,
    )

