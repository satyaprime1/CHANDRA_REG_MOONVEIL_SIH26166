"""
MoonVeil MK1 — src/matching.py

Responsibility: turn two independent FeatureSets into a list of candidate
correspondences via nearest-neighbour matching + Lowe's ratio test. This
module knows nothing about geometry, RANSAC, or reliability — those are
downstream concerns. Only ratio-test-passing matches are returned, per the
MV_MK1 module contract.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import FeatureSet, MatchCandidate
from config import MatchingConfig

_SUPPORTED_MATCHERS = {"BF"}


def match_features(
    source: FeatureSet,
    reference: FeatureSet,
    config: MatchingConfig,
) -> list:
    """
    Parameters
    ----------
    source : FeatureSet
    reference : FeatureSet
    config : MatchingConfig

    Returns
    -------
    list[MatchCandidate]
        Only matches that pass the Lowe ratio test (is_ratio_pass=True).

    Raises
    ------
    ValueError
        If the matcher is unsupported, or either FeatureSet has no
        descriptors to work with.
    """
    if config.matcher not in _SUPPORTED_MATCHERS:
        raise ValueError(
            f"Unsupported matcher '{config.matcher}'. Supported: {_SUPPORTED_MATCHERS}"
        )

    if source.descriptors is None or len(source.descriptors) == 0:
        raise ValueError("Source FeatureSet has no descriptors to match.")
    if reference.descriptors is None or len(reference.descriptors) == 0:
        raise ValueError("Reference FeatureSet has no descriptors to match.")

    k = config.knn_k
    if k < 2:
        raise ValueError("knn_k must be >= 2 for the Lowe ratio test to apply.")

    # SIFT descriptors are floating point -> L2 norm is the correct metric.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Need at least k neighbours available in the reference set.
    effective_k = min(k, len(reference.descriptors))
    if effective_k < 2:
        raise ValueError(
            "Reference FeatureSet has fewer than 2 descriptors; "
            "cannot apply the ratio test."
        )

    raw_matches = bf.knnMatch(source.descriptors, reference.descriptors, k=effective_k)

    candidates = []
    match_id = 0
    for neighbours in raw_matches:
        if len(neighbours) < 2:
            continue  # can't compute a ratio without a second neighbour

        best, second_best = neighbours[0], neighbours[1]
        ratio = best.distance / second_best.distance if second_best.distance > 0 else 1.0
        is_pass = ratio < config.ratio_threshold

        if not is_pass:
            continue  # only ratio-test-passing matches proceed, per contract

        src_kp = source.keypoints[best.queryIdx]
        ref_kp = reference.keypoints[best.trainIdx]

        candidates.append(
            MatchCandidate(
                match_id=match_id,
                source_index=best.queryIdx,
                reference_index=best.trainIdx,
                source_point=(float(src_kp.pt[0]), float(src_kp.pt[1])),
                reference_point=(float(ref_kp.pt[0]), float(ref_kp.pt[1])),
                descriptor_distance=float(best.distance),
                ratio=float(ratio),
                is_ratio_pass=True,
            )
        )
        match_id += 1

    return candidates

