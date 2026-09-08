"""
MoonVeil MK1 — src/types.py

Shared data contracts used throughout the pipeline. This module contains
ONLY structural definitions (dataclasses, enums) — no image processing,
no I/O, no algorithms. Every other module imports from here; this module
imports from nothing else in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sensor identity
# ---------------------------------------------------------------------------

class SensorType(str, Enum):
    """Canonical sensor identifiers. Never pass raw strings like 'ohrc' or
    'OHRC_IMAGE' through the system — always resolve to one of these."""
    OHRC = "OHRC"
    TMC = "TMC"
    IIRS = "IIRS"


# ---------------------------------------------------------------------------
# Loaded image + metadata
# ---------------------------------------------------------------------------

@dataclass
class SensorMetadata:
    sensor: SensorType
    product_id: Optional[str] = None
    acquisition_time: Optional[str] = None
    spatial_resolution: Optional[float] = None
    coordinate_reference_system: Optional[str] = None
    width: int = 0
    height: int = 0
    bands: int = 1


@dataclass
class ImageData:
    image: np.ndarray
    sensor: str
    path: str
    width: int
    height: int
    channels: int
    dtype: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ImagePair:
    reference: ImageData
    source: ImageData
    pair_id: str
    same_sensor: bool


# ---------------------------------------------------------------------------
# Preprocessing output
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedImage:
    image: np.ndarray
    original_shape: tuple
    processed_shape: tuple
    sensor: str
    normalization_method: str
    scale_factor: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature detection output
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    keypoints: list
    descriptors: np.ndarray
    image_shape: tuple
    detector_name: str


# ---------------------------------------------------------------------------
# Matching output
# ---------------------------------------------------------------------------

@dataclass
class MatchCandidate:
    match_id: int
    source_index: int
    reference_index: int

    source_point: tuple  # (float, float)
    reference_point: tuple  # (float, float)

    descriptor_distance: float
    ratio: float

    is_ratio_pass: bool


# ---------------------------------------------------------------------------
# Geometric verification output
# ---------------------------------------------------------------------------

@dataclass
class GeometricResult:
    transformation_matrix: Optional[np.ndarray]
    model_type: str

    inlier_mask: Optional[np.ndarray]
    inlier_matches: list
    outlier_matches: list

    num_inliers: int
    num_outliers: int
    inlier_ratio: float

    reprojection_errors: Optional[np.ndarray]
    rmse: float

    success: bool = True
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Reliability output
# ---------------------------------------------------------------------------

@dataclass
class MatchReliability:
    match_id: int

    descriptor_score: float
    ratio_score: float
    geometry_score: float
    reprojection_score: float
    spatial_score: float

    reliability_score: float  # 0-100
    status: str  # "HIGH" | "MEDIUM" | "LOW" | "REJECTED"


# ---------------------------------------------------------------------------
# Spatial distribution output
# ---------------------------------------------------------------------------

@dataclass
class SpatialDistribution:
    grid_rows: int
    grid_cols: int

    counts: np.ndarray

    occupied_cells: int
    total_cells: int

    coverage_ratio: float
    distribution_score: float


# ---------------------------------------------------------------------------
# Trusted correspondence set (post-reliability-filtering)
# ---------------------------------------------------------------------------

@dataclass
class TrustedCorrespondenceSet:
    matches: list  # list[MatchCandidate]
    reliabilities: list  # list[MatchReliability]

    count: int
    average_reliability: float
    spatial_coverage: float


# ---------------------------------------------------------------------------
# Registration output
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    registered_image: Optional[np.ndarray]

    transformation_matrix: Optional[np.ndarray]

    output_shape: tuple

    success: bool
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    candidate_matches: int
    verified_inliers: int
    rejected_matches: int

    inlier_ratio: float

    rmse: float
    mean_reprojection_error: float
    median_reprojection_error: float
    max_reprojection_error: float

    spatial_coverage: float
    average_reliability: float

    overall_reliability: float
    reliability_status: str  # "HIGH" | "MEDIUM" | "LOW"

    registration_success: bool


# ---------------------------------------------------------------------------
# Pipeline status / failure signalling
# ---------------------------------------------------------------------------

@dataclass
class PipelineStatus:
    success: bool
    stage: str  # "LOAD" | "PREPROCESS" | "FEATURES" | "MATCHING" |
                # "GEOMETRY" | "RELIABILITY" | "REGISTRATION" |
                # "EVALUATION" | "COMPLETE"
    message: str


# ---------------------------------------------------------------------------
# Master pipeline output
# ---------------------------------------------------------------------------

@dataclass
class MoonVeilResult:
    pair_id: str
    sensor_reference: str
    sensor_source: str

    reference: ImageData
    source: ImageData

    reference_processed: PreprocessedImage
    source_processed: PreprocessedImage

    reference_features: FeatureSet
    source_features: FeatureSet

    candidate_matches: list  # list[MatchCandidate]

    geometry: GeometricResult

    match_reliabilities: list  # list[MatchReliability]
    spatial_distribution: SpatialDistribution

    registration: RegistrationResult
    evaluation: EvaluationResult

    status: PipelineStatus = field(
        default_factory=lambda: PipelineStatus(True, "COMPLETE", "OK")
    )

