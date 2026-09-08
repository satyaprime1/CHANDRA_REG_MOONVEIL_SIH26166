"""
MoonVeil MK1 — config.py

Holds all tunable parameters (feature detection, matching thresholds,
RANSAC settings, reliability weights, spatial grid size, preprocessing
options) so no magic numbers are scattered through the codebase. Config
dataclasses are added here incrementally as each module that needs one is
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessingConfig:
    normalize: bool = True
    contrast_method: Optional[str] = None  # None | "CLAHE"
    illumination_normalization: bool = False
    resize_factor: float = 1.0  # 1.0 = no resize


def default_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig()


@dataclass
class FeatureConfig:
    detector: str = "SIFT"
    nfeatures: int = 0  # 0 = no cap, let OpenCV decide
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    subpixel_refinement: bool = True


def default_feature_config() -> FeatureConfig:
    return FeatureConfig()


@dataclass
class AIMatcherConfig:
    enabled: bool = False
    model_name: str = "auto"  # "auto" | "dis_flow" | "lightglue"
    confidence_threshold: float = 0.2
    max_keypoints: int = 2048


def default_ai_matcher_config() -> AIMatcherConfig:
    return AIMatcherConfig()


@dataclass
class MatchingConfig:
    matcher: str = "BF"
    ratio_threshold: float = 0.75
    knn_k: int = 2


def default_matching_config() -> MatchingConfig:
    return MatchingConfig()


@dataclass
class GeometryConfig:
    model: str = "homography"
    ransac_threshold: float = 5.0  # pixels
    confidence: float = 0.99
    max_iterations: int = 2000
    min_inliers: int = 8


def default_geometry_config() -> GeometryConfig:
    return GeometryConfig()


@dataclass
class SpatialConfig:
    rows: int = 4
    cols: int = 4


def default_spatial_config() -> SpatialConfig:
    return SpatialConfig()


@dataclass
class ReliabilityConfig:
    weight_descriptor: float = 0.15
    weight_ratio: float = 0.15
    weight_geometry: float = 0.30
    weight_reprojection: float = 0.25
    weight_spatial: float = 0.15

    high_threshold: float = 80.0
    medium_threshold: float = 50.0
    reject_threshold: float = 0.0  # kept for schema parity; outliers are
    # rejected structurally (RANSAC status), not purely by score threshold

    # Distance (px) at which reprojection_score decays to ~0.37 (1/e).
    # Tied loosely to the geometry RANSAC threshold as a sensible default.
    reprojection_scale_px: float = 5.0


def default_reliability_config() -> ReliabilityConfig:
    return ReliabilityConfig()


@dataclass
class EvaluationConfig:
    # Scale (px) at which the RMSE-based sub-score decays to ~0.37 (1/e).
    rmse_scale_px: float = 5.0

    weight_average_reliability: float = 0.40
    weight_inlier_ratio: float = 0.30
    weight_spatial_coverage: float = 0.15
    weight_rmse: float = 0.15

    high_threshold: float = 80.0
    medium_threshold: float = 50.0


def default_evaluation_config() -> EvaluationConfig:
    return EvaluationConfig()


@dataclass
class RegistrationConfig:
    min_trusted_matches: int = 8
    min_inlier_ratio: float = 0.2
    min_spatial_coverage: float = 0.25
    max_rmse: float = 10.0


def default_registration_config() -> RegistrationConfig:
    return RegistrationConfig()


@dataclass
class MoonVeilConfig:
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    matching: MatchingConfig
    geometry: GeometryConfig
    reliability: ReliabilityConfig
    spatial: SpatialConfig
    evaluation: EvaluationConfig
    registration_gate: RegistrationConfig
    ai_matcher: AIMatcherConfig = field(default_factory=default_ai_matcher_config)


def default_moonveil_config() -> "MoonVeilConfig":
    return MoonVeilConfig(
        preprocessing=default_preprocessing_config(),
        features=default_feature_config(),
        matching=default_matching_config(),
        geometry=default_geometry_config(),
        reliability=default_reliability_config(),
        spatial=default_spatial_config(),
        evaluation=default_evaluation_config(),
        registration_gate=default_registration_config(),
        ai_matcher=default_ai_matcher_config(),
    )

