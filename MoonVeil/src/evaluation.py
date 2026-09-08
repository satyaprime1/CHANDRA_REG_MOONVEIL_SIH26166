"""
MoonVeil MK1 — src/evaluation.py

Responsibility: pure aggregation of metrics already computed elsewhere in
the pipeline. No image processing happens here. This module answers the
practical judge-facing question: "how good was this correspondence and
registration, overall?"
"""

from __future__ import annotations

import math

import numpy as np

from .types import EvaluationResult, GeometricResult, RegistrationResult, SpatialDistribution
from config import EvaluationConfig

_WEIGHT_TOLERANCE = 1e-6


def _validate_weights(config: EvaluationConfig) -> None:
    total = (
        config.weight_average_reliability
        + config.weight_inlier_ratio
        + config.weight_spatial_coverage
        + config.weight_rmse
    )
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"EvaluationConfig weights must sum to 1.0, got {total:.6f}"
        )


def evaluate(
    candidate_matches: list,
    geometry: GeometricResult,
    reliabilities: list,
    spatial: SpatialDistribution,
    registration: RegistrationResult,
    config: EvaluationConfig = None,
) -> EvaluationResult:
    """
    Parameters
    ----------
    candidate_matches : list[MatchCandidate]
    geometry : GeometricResult
    reliabilities : list[MatchReliability]
    spatial : SpatialDistribution
    registration : RegistrationResult
    config : EvaluationConfig, optional
        Defaults to EvaluationConfig() if not provided.

    Returns
    -------
    EvaluationResult
    """
    if config is None:
        config = EvaluationConfig()
    _validate_weights(config)

    n_candidates = len(candidate_matches)
    n_inliers = geometry.num_inliers if geometry else 0
    n_rejected = n_candidates - n_inliers

    inlier_ratio = geometry.inlier_ratio if geometry else 0.0
    rmse = geometry.rmse if (geometry and geometry.success) else float("nan")

    if geometry and geometry.success and geometry.reprojection_errors is not None and len(geometry.reprojection_errors) > 0:
        errors = geometry.reprojection_errors
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        max_err = float(np.max(errors))
    else:
        mean_err = float("nan")
        median_err = float("nan")
        max_err = float("nan")

    spatial_coverage = spatial.coverage_ratio if spatial else 0.0

    trusted_scores = [r.reliability_score for r in reliabilities if r.status != "REJECTED"]
    average_reliability = float(np.mean(trusted_scores)) if trusted_scores else 0.0

    # RMSE sub-score: 0 error -> 1.0, decaying toward 0 as error grows.
    # NaN (no successful geometry) contributes no credit.
    if math.isnan(rmse):
        rmse_score = 0.0
    else:
        rmse_score = math.exp(-rmse / config.rmse_scale_px)

    overall_composite = (
        config.weight_average_reliability * (average_reliability / 100.0)
        + config.weight_inlier_ratio * inlier_ratio
        + config.weight_spatial_coverage * spatial_coverage
        + config.weight_rmse * rmse_score
    )
    overall_reliability = overall_composite * 100.0

    if overall_reliability >= config.high_threshold:
        reliability_status = "HIGH"
    elif overall_reliability >= config.medium_threshold:
        reliability_status = "MEDIUM"
    else:
        reliability_status = "LOW"

    registration_success = bool(registration.success) if registration else False

    return EvaluationResult(
        candidate_matches=n_candidates,
        verified_inliers=n_inliers,
        rejected_matches=n_rejected,
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        mean_reprojection_error=mean_err,
        median_reprojection_error=median_err,
        max_reprojection_error=max_err,
        spatial_coverage=spatial_coverage,
        average_reliability=average_reliability,
        overall_reliability=overall_reliability,
        reliability_status=reliability_status,
        registration_success=registration_success,
    )

