"""
MoonVeil MK1 — src/reliability.py

Responsibility: the core MoonVeil-specific component. Turns already-
verified geometric information plus descriptor/spatial evidence into an
explainable 0-100 "MoonVeil Reliability Score" per match, with individual
components preserved (never just a bare number). This is explicitly NOT
presented as a statistically calibrated probability — it is a composite
confidence indicator built from measurable evidence, per the blueprint.
"""

from __future__ import annotations

import math

import numpy as np

from .types import GeometricResult, MatchReliability, SpatialDistribution
from config import ReliabilityConfig

_WEIGHT_TOLERANCE = 1e-6


def _validate_weights(config: ReliabilityConfig) -> None:
    total = (
        config.weight_descriptor
        + config.weight_ratio
        + config.weight_geometry
        + config.weight_reprojection
        + config.weight_spatial
    )
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"ReliabilityConfig weights must sum to 1.0, got {total:.6f}"
        )


def calculate_match_reliability(
    matches: list,
    geometry: GeometricResult,
    spatial: SpatialDistribution,
    config: ReliabilityConfig,
) -> list:
    """
    Parameters
    ----------
    matches : list[MatchCandidate]
        The full candidate set (both eventual inliers and outliers).
    geometry : GeometricResult
        Already-computed geometric verification (may be a failed result;
        every match is then treated as unverified/REJECTED).
    spatial : SpatialDistribution
        Spatial coverage computed over the trusted/inlier point set. Mk1
        applies this as a shared context score across all matches in this
        batch (see note below) rather than a fully per-match local density
        model — deliberately simple, per "do not over-engineer the first
        version."
    config : ReliabilityConfig

    Returns
    -------
    list[MatchReliability]

    Raises
    ------
    ValueError
        If the configured weights do not sum to 1.0.
    """
    _validate_weights(config)

    if len(matches) == 0:
        return []

    inlier_ids = {m.match_id for m in geometry.inlier_matches} if geometry else set()

    error_by_id = {}
    if geometry and geometry.success and geometry.reprojection_errors is not None:
        for m, err in zip(geometry.inlier_matches, geometry.reprojection_errors):
            error_by_id[m.match_id] = float(err)

    distances = np.array([m.descriptor_distance for m in matches], dtype=np.float64)
    d_min, d_max = float(distances.min()), float(distances.max())
    d_range = d_max - d_min if d_max > d_min else 1.0

    spatial_score = float(spatial.distribution_score) if spatial is not None else 0.0

    results = []
    for m, distance in zip(matches, distances):
        # Descriptor quality: relative min-max normalization within this
        # candidate set — lower L2 distance = stronger descriptor match.
        descriptor_score = float(np.clip(1.0 - (distance - d_min) / d_range, 0.0, 1.0))

        # Ratio test margin: lower Lowe ratio = more distinctive match.
        ratio_score = float(np.clip(1.0 - m.ratio, 0.0, 1.0))

        is_inlier = m.match_id in inlier_ids
        geometry_score = 1.0 if is_inlier else 0.0

        if is_inlier and m.match_id in error_by_id:
            err = error_by_id[m.match_id]
            reprojection_score = float(math.exp(-err / config.reprojection_scale_px))
        else:
            # Outliers, or inliers we have no recorded error for (shouldn't
            # normally happen), get no reprojection credit.
            reprojection_score = 0.0

        composite = (
            config.weight_descriptor * descriptor_score
            + config.weight_ratio * ratio_score
            + config.weight_geometry * geometry_score
            + config.weight_reprojection * reprojection_score
            + config.weight_spatial * spatial_score
        )
        score_0_100 = composite * 100.0

        if not is_inlier:
            # A RANSAC outlier is structurally rejected regardless of how
            # good its descriptor evidence looked in isolation — geometric
            # inconsistency overrides everything else.
            status = "REJECTED"
        elif score_0_100 >= config.high_threshold:
            status = "HIGH"
        elif score_0_100 >= config.medium_threshold:
            status = "MEDIUM"
        else:
            status = "LOW"

        results.append(
            MatchReliability(
                match_id=m.match_id,
                descriptor_score=descriptor_score,
                ratio_score=ratio_score,
                geometry_score=geometry_score,
                reprojection_score=reprojection_score,
                spatial_score=spatial_score,
                reliability_score=score_0_100,
                status=status,
            )
        )

    return results

