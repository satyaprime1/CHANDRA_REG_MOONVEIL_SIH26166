"""
MoonVeil MK1 — tests/test_types.py

Sanity test for src/types.py. Confirms every dataclass can be instantiated
with plausible dummy values with no errors. This module contains no
processing logic, so this is a structural smoke test, not a behavioural one.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import (
    SensorType,
    SensorMetadata,
    ImageData,
    ImagePair,
    PreprocessedImage,
    FeatureSet,
    MatchCandidate,
    GeometricResult,
    MatchReliability,
    SpatialDistribution,
    TrustedCorrespondenceSet,
    RegistrationResult,
    EvaluationResult,
    PipelineStatus,
    MoonVeilResult,
)


def test_all_types_instantiate():
    dummy_image = np.zeros((10, 10), dtype=np.uint8)

    sensor_meta = SensorMetadata(sensor=SensorType.OHRC, width=10, height=10)

    ref_image_data = ImageData(
        image=dummy_image, sensor="OHRC", path="ref.tif",
        width=10, height=10, channels=1, dtype="uint8",
    )
    src_image_data = ImageData(
        image=dummy_image, sensor="OHRC", path="src.tif",
        width=10, height=10, channels=1, dtype="uint8",
    )

    pair = ImagePair(
        reference=ref_image_data, source=src_image_data,
        pair_id="TEST_PAIR_01", same_sensor=True,
    )

    ref_processed = PreprocessedImage(
        image=dummy_image, original_shape=(10, 10), processed_shape=(10, 10),
        sensor="OHRC", normalization_method="minmax", scale_factor=1.0,
    )
    src_processed = PreprocessedImage(
        image=dummy_image, original_shape=(10, 10), processed_shape=(10, 10),
        sensor="OHRC", normalization_method="minmax", scale_factor=1.0,
    )

    feature_set = FeatureSet(
        keypoints=[], descriptors=np.zeros((0, 128), dtype=np.float32),
        image_shape=(10, 10), detector_name="SIFT",
    )

    match = MatchCandidate(
        match_id=0, source_index=0, reference_index=0,
        source_point=(1.0, 1.0), reference_point=(1.0, 1.0),
        descriptor_distance=10.0, ratio=0.7, is_ratio_pass=True,
    )

    geometry = GeometricResult(
        transformation_matrix=np.eye(3), model_type="homography",
        inlier_mask=np.array([True]), inlier_matches=[match],
        outlier_matches=[], num_inliers=1, num_outliers=0,
        inlier_ratio=1.0, reprojection_errors=np.array([0.1]), rmse=0.1,
    )

    reliability = MatchReliability(
        match_id=0, descriptor_score=0.9, ratio_score=0.9, geometry_score=1.0,
        reprojection_score=0.9, spatial_score=0.8,
        reliability_score=90.0, status="HIGH",
    )

    spatial = SpatialDistribution(
        grid_rows=4, grid_cols=4, counts=np.ones((4, 4), dtype=int),
        occupied_cells=16, total_cells=16, coverage_ratio=1.0,
        distribution_score=1.0,
    )

    trusted = TrustedCorrespondenceSet(
        matches=[match], reliabilities=[reliability],
        count=1, average_reliability=90.0, spatial_coverage=1.0,
    )

    registration = RegistrationResult(
        registered_image=dummy_image, transformation_matrix=np.eye(3),
        output_shape=(10, 10), success=True,
    )

    evaluation = EvaluationResult(
        candidate_matches=1, verified_inliers=1, rejected_matches=0,
        inlier_ratio=1.0, rmse=0.1, mean_reprojection_error=0.1,
        median_reprojection_error=0.1, max_reprojection_error=0.1,
        spatial_coverage=1.0, average_reliability=90.0,
        overall_reliability=90.0, reliability_status="HIGH",
        registration_success=True,
    )

    status = PipelineStatus(success=True, stage="COMPLETE", message="OK")

    result = MoonVeilResult(
        pair_id="TEST_PAIR_01", sensor_reference="OHRC", sensor_source="OHRC",
        reference=ref_image_data, source=src_image_data,
        reference_processed=ref_processed, source_processed=src_processed,
        reference_features=feature_set, source_features=feature_set,
        candidate_matches=[match], geometry=geometry,
        match_reliabilities=[reliability], spatial_distribution=spatial,
        registration=registration, evaluation=evaluation, status=status,
    )

    assert result.pair_id == "TEST_PAIR_01"
    assert result.evaluation.registration_success is True
    assert trusted.count == 1
    assert sensor_meta.sensor == SensorType.OHRC
    assert pair.same_sensor is True

    print("All MoonVeil types instantiated successfully.")


if __name__ == "__main__":
    test_all_types_instantiate()
