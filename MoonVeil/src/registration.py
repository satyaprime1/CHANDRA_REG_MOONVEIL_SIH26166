"""
MoonVeil MK1 — src/registration.py

Responsibility: warp the source image into reference coordinates using an
already-estimated transformation (register_image), and gate whether that
warp should even be trusted enough to produce (should_register). The gate
is the mandatory checkpoint that prevents MoonVeil from silently
registering images on the back of weak evidence — it belongs here rather
than upstream because it's specifically the "should we register" question,
distinct from whether matches/geometry look individually reasonable.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import PreprocessedImage, RegistrationResult


def register_image(
    source: PreprocessedImage,
    transformation_matrix,
    output_shape: tuple,
) -> RegistrationResult:
    """
    Parameters
    ----------
    source : PreprocessedImage
        The image to be warped.
    transformation_matrix : np.ndarray | None
        (3, 3) matrix mapping source coordinates -> reference coordinates.
        If None, registration cannot proceed and a failed result is
        returned (never an exception, per the module contract).
    output_shape : tuple
        (height, width) of the target (reference) coordinate frame.

    Returns
    -------
    RegistrationResult
    """
    if transformation_matrix is None:
        return RegistrationResult(
            registered_image=None,
            transformation_matrix=None,
            output_shape=output_shape,
            success=False,
            failure_reason="No transformation matrix provided.",
        )

    transformation_matrix = np.asarray(transformation_matrix, dtype=np.float64)
    if transformation_matrix.shape != (3, 3):
        return RegistrationResult(
            registered_image=None,
            transformation_matrix=transformation_matrix,
            output_shape=output_shape,
            success=False,
            failure_reason=(
                f"Expected a (3, 3) transformation matrix, got shape "
                f"{transformation_matrix.shape}."
            ),
        )

    height, width = int(output_shape[0]), int(output_shape[1])

    try:
        registered = cv2.warpPerspective(
            source.image, transformation_matrix, (width, height)
        )
    except cv2.error as exc:
        return RegistrationResult(
            registered_image=None,
            transformation_matrix=transformation_matrix,
            output_shape=output_shape,
            success=False,
            failure_reason=f"OpenCV warp failed: {exc}",
        )

    return RegistrationResult(
        registered_image=registered,
        transformation_matrix=transformation_matrix,
        output_shape=output_shape,
        success=True,
        failure_reason=None,
    )


def should_register(
    trusted,
    geometry,
    config,
) -> tuple:
    """
    Explicit decision gate: should this pair actually be registered?

    Parameters
    ----------
    trusted : TrustedCorrespondenceSet
    geometry : GeometricResult
    config : RegistrationConfig

    Returns
    -------
    (bool, str) — (approved, reason)
    """
    if geometry is None or not geometry.success:
        reason = geometry.failure_reason if geometry else "No geometric result available."
        return False, f"Geometry not established: {reason}"

    if trusted.count < config.min_trusted_matches:
        return False, (
            f"Only {trusted.count} trusted (HIGH/MEDIUM reliability, inlier) "
            f"matches; minimum required is {config.min_trusted_matches}."
        )

    if geometry.inlier_ratio < config.min_inlier_ratio:
        return False, (
            f"Inlier ratio {geometry.inlier_ratio:.1%} is below the minimum "
            f"{config.min_inlier_ratio:.1%}."
        )

    if trusted.spatial_coverage < config.min_spatial_coverage:
        return False, (
            f"Spatial coverage {trusted.spatial_coverage:.1%} is below the "
            f"minimum {config.min_spatial_coverage:.1%} — matches are too "
            f"clustered to trust the registration across the full image."
        )

    if geometry.rmse > config.max_rmse:
        return False, (
            f"RMSE {geometry.rmse:.2f}px exceeds the maximum allowed "
            f"{config.max_rmse:.2f}px."
        )

    return True, "Registration criteria satisfied."
