"""
MoonVeil MK1 — src/geometry.py

Responsibility: given candidate correspondences, estimate the geometric
transformation that best explains them via RANSAC, separate inliers from
outliers, and compute reprojection error / RMSE. This module does NOT
decide whether the result is trustworthy for registration — that decision
belongs to should_register() downstream. It only reports what RANSAC found,
including an explicit failure state when there isn't enough evidence.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import GeometricResult, MatchCandidate
from config import GeometryConfig

_SUPPORTED_MODELS = {"homography"}


def calculate_reprojection_error(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Parameters
    ----------
    source_points : (N, 2)
    reference_points : (N, 2)
    transformation_matrix : (3, 3)

    Returns
    -------
    errors : (N,) Euclidean distance between predicted and actual
        reference coordinates.
    """
    n = source_points.shape[0]
    homogeneous = np.hstack([source_points, np.ones((n, 1))])  # (N, 3)
    projected = (transformation_matrix @ homogeneous.T).T  # (N, 3)

    # Guard against divide-by-zero on degenerate rows.
    w = projected[:, 2]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    projected_xy = projected[:, :2] / w_safe[:, None]

    errors = np.linalg.norm(projected_xy - reference_points, axis=1)
    return errors


def calculate_rmse(errors: np.ndarray) -> float:
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors ** 2)))


def estimate_geometry(
    matches: list,
    config: GeometryConfig,
) -> GeometricResult:
    """
    Parameters
    ----------
    matches : list[MatchCandidate]
    config : GeometryConfig

    Returns
    -------
    GeometricResult
        success=False with a failure_reason (not an exception) when there
        are too few matches or too few RANSAC inliers to trust the result.
    """
    if config.model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported geometry model '{config.model}'. Supported: {_SUPPORTED_MODELS}"
        )

    # A homography needs at least 4 correspondences in principle; require
    # a bit of margin before even attempting RANSAC.
    min_needed_for_attempt = max(4, config.min_inliers)
    if len(matches) < min_needed_for_attempt:
        return _failed_result(
            config,
            f"Only {len(matches)} candidate matches; need at least "
            f"{min_needed_for_attempt} to attempt {config.model} estimation.",
        )

    source_points = np.array([m.source_point for m in matches], dtype=np.float64)
    reference_points = np.array([m.reference_point for m in matches], dtype=np.float64)

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold,
        maxIters=config.max_iterations,
        confidence=config.confidence,
    )

    if H is None or mask is None:
        return _failed_result(
            config, "RANSAC failed to find a consistent homography (degenerate configuration)."
        )

    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())
    num_outliers = len(matches) - num_inliers

    if num_inliers < config.min_inliers:
        return _failed_result(
            config,
            f"Only {num_inliers} RANSAC inliers found; minimum is {config.min_inliers}.",
            transformation_matrix=H,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
        )

    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    outlier_matches = [m for m, keep in zip(matches, inlier_mask) if not keep]

    inlier_source_pts = source_points[inlier_mask]
    inlier_reference_pts = reference_points[inlier_mask]
    errors = calculate_reprojection_error(inlier_source_pts, inlier_reference_pts, H)
    rmse = calculate_rmse(errors)

    return GeometricResult(
        transformation_matrix=H,
        model_type=config.model,
        inlier_mask=inlier_mask,
        inlier_matches=inlier_matches,
        outlier_matches=outlier_matches,
        num_inliers=num_inliers,
        num_outliers=num_outliers,
        inlier_ratio=num_inliers / len(matches),
        reprojection_errors=errors,
        rmse=rmse,
        success=True,
        failure_reason=None,
    )


def _failed_result(
    config: GeometryConfig,
    reason: str,
    transformation_matrix=None,
    num_inliers=0,
    num_outliers=0,
) -> GeometricResult:
    return GeometricResult(
        transformation_matrix=transformation_matrix,
        model_type=config.model,
        inlier_mask=None,
        inlier_matches=[],
        outlier_matches=[],
        num_inliers=num_inliers,
        num_outliers=num_outliers,
        inlier_ratio=0.0,
        reprojection_errors=None,
        rmse=float("nan"),
        success=False,
        failure_reason=reason,
    )

