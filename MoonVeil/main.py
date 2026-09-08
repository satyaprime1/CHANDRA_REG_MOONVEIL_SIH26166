"""
MoonVeil MK1 — main.py

The single public entry point tying every module together:

    load -> preprocess -> detect features -> match -> candidate spatial
    distribution -> RANSAC geometry -> reliability -> trusted-match
    filtering -> final spatial distribution -> decision gate -> register
    -> evaluate -> MoonVeilResult

Error-handling philosophy (documented, not accidental):
  - Hard errors (bad file path, unsupported sensor, a truly featureless
    image) come from data_loader.py / features.py and are allowed to
    propagate as exceptions — these indicate the INPUT is unusable, not
    that correspondence is merely weak.
  - Soft failures (too few matches, too few inliers, poor spatial
    coverage, high RMSE) never raise. They flow through as a
    GeometricResult/RegistrationResult with success=False and an
    explicit reason, which run_moonveil surfaces via MoonVeilResult and
    a REGISTRATION NOT RELIABLE outcome — this is what makes Case C
    (failure/ambiguity) a demonstrable feature rather than a crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_loader import load_image
from src.preprocessing import preprocess_image
from src.features import detect_features
from src.matching import match_features
from src.spatial import analyze_spatial_distribution
from src.geometry import estimate_geometry
from src.reliability import calculate_match_reliability
from src.registration import register_image, should_register
from src.evaluation import evaluate
from src.types import MoonVeilResult, PipelineStatus, TrustedCorrespondenceSet

from config import MoonVeilConfig, default_moonveil_config


def _build_trusted_set(matches, reliabilities) -> TrustedCorrespondenceSet:
    """A match is 'trusted' only if it is a RANSAC inlier AND its
    reliability status is HIGH or MEDIUM — geometric consistency alone
    isn't enough if the supporting evidence is otherwise weak."""
    status_by_id = {r.match_id: r for r in reliabilities}
    trusted_matches = []
    trusted_reliabilities = []
    for m in matches:
        r = status_by_id.get(m.match_id)
        if r is not None and r.status in ("HIGH", "MEDIUM"):
            trusted_matches.append(m)
            trusted_reliabilities.append(r)

    count = len(trusted_matches)
    avg_reliability = (
        float(np.mean([r.reliability_score for r in trusted_reliabilities]))
        if count > 0 else 0.0
    )
    return TrustedCorrespondenceSet(
        matches=trusted_matches,
        reliabilities=trusted_reliabilities,
        count=count,
        average_reliability=avg_reliability,
        spatial_coverage=0.0,  # filled in by caller once final spatial is known
    )


def run_moonveil(
    reference_path,
    source_path,
    reference_sensor: str,
    source_sensor: str = None,
    config: MoonVeilConfig = None,
) -> MoonVeilResult:
    if config is None:
        config = default_moonveil_config()
    if source_sensor is None:
        source_sensor = reference_sensor

    # --- LOAD ---------------------------------------------------------
    reference_data = load_image(reference_path, sensor=reference_sensor)
    source_data = load_image(source_path, sensor=source_sensor)

    # --- PREPROCESS -----------------------------------------------------
    reference_processed = preprocess_image(reference_data, config.preprocessing)
    source_processed = preprocess_image(source_data, config.preprocessing)

    # --- FEATURES & MATCHING ------------------------------------------------
    if config.ai_matcher.enabled:
        from src.ai_matcher import match_features_ai
        from src.types import FeatureSet
        import cv2
        candidate_matches = match_features_ai(source_processed, reference_processed, config.ai_matcher)
        ref_kps = [cv2.KeyPoint(x=m.reference_point[0], y=m.reference_point[1], size=5.0) for m in candidate_matches]
        src_kps = [cv2.KeyPoint(x=m.source_point[0], y=m.source_point[1], size=5.0) for m in candidate_matches]
        dummy_desc = np.zeros((len(candidate_matches), 32), dtype=np.float32)
        reference_features = FeatureSet(ref_kps, dummy_desc, reference_processed.processed_shape, "AI_DIS_Flow")
        source_features = FeatureSet(src_kps, dummy_desc, source_processed.processed_shape, "AI_DIS_Flow")
    else:
        reference_features = detect_features(reference_processed, config.features)
        source_features = detect_features(source_processed, config.features)
        candidate_matches = match_features(source_features, reference_features, config.matching)

    # --- CANDIDATE SPATIAL DISTRIBUTION (before geometry, per spec §28) -----
    candidate_ref_pts = np.array([m.reference_point for m in candidate_matches]) \
        if candidate_matches else np.zeros((0, 2))
    candidate_spatial = analyze_spatial_distribution(
        candidate_ref_pts, reference_processed.processed_shape, config.spatial
    )

    # --- GEOMETRIC VERIFICATION (RANSAC) ------------------------------------
    geometry = estimate_geometry(candidate_matches, config.geometry)

    # --- RELIABILITY ---------------------------------------------------------
    reliabilities = calculate_match_reliability(
        candidate_matches, geometry, candidate_spatial, config.reliability
    )

    # --- TRUSTED-MATCH FILTERING + FINAL SPATIAL DISTRIBUTION ---------------
    trusted = _build_trusted_set(candidate_matches, reliabilities)
    trusted_ref_pts = np.array([m.reference_point for m in trusted.matches]) \
        if trusted.matches else np.zeros((0, 2))
    final_spatial = analyze_spatial_distribution(
        trusted_ref_pts, reference_processed.processed_shape, config.spatial
    )
    trusted.spatial_coverage = final_spatial.coverage_ratio

    # --- DECISION GATE ---------------------------------------------------------
    approved, reason = should_register(trusted, geometry, config.registration_gate)

    # --- REGISTRATION ---------------------------------------------------------
    if approved:
        registration = register_image(
            source_processed, geometry.transformation_matrix, reference_processed.processed_shape
        )
    else:
        from src.types import RegistrationResult
        registration = RegistrationResult(
            registered_image=None,
            transformation_matrix=geometry.transformation_matrix,
            output_shape=reference_processed.processed_shape,
            success=False,
            failure_reason=f"REGISTRATION NOT RELIABLE: {reason}",
        )

    # --- EVALUATION ---------------------------------------------------------
    evaluation = evaluate(
        candidate_matches, geometry, reliabilities, final_spatial, registration, config.evaluation
    )

    status = PipelineStatus(
        success=registration.success,
        stage="COMPLETE" if registration.success else "REGISTRATION",
        message="OK" if registration.success else registration.failure_reason,
    )

    pair_id = f"{Path(str(reference_path)).stem}__{Path(str(source_path)).stem}"

    return MoonVeilResult(
        pair_id=pair_id,
        sensor_reference=reference_sensor,
        sensor_source=source_sensor,
        reference=reference_data,
        source=source_data,
        reference_processed=reference_processed,
        source_processed=source_processed,
        reference_features=reference_features,
        source_features=source_features,
        candidate_matches=candidate_matches,
        geometry=geometry,
        match_reliabilities=reliabilities,
        spatial_distribution=final_spatial,
        registration=registration,
        evaluation=evaluation,
        status=status,
    )


def _result_to_metrics_dict(result: MoonVeilResult) -> dict:
    ev = result.evaluation
    return {
        "pair_id": result.pair_id,
        "reference_sensor": result.sensor_reference,
        "source_sensor": result.sensor_source,
        "candidate_matches": ev.candidate_matches,
        "verified_inliers": ev.verified_inliers,
        "rejected_matches": ev.rejected_matches,
        "inlier_ratio": ev.inlier_ratio,
        "rmse": ev.rmse,
        "mean_reprojection_error": ev.mean_reprojection_error,
        "median_reprojection_error": ev.median_reprojection_error,
        "max_reprojection_error": ev.max_reprojection_error,
        "spatial_coverage": ev.spatial_coverage,
        "average_reliability": ev.average_reliability,
        "overall_reliability": ev.overall_reliability,
        "reliability_status": ev.reliability_status,
        "registration_success": ev.registration_success,
    }


def main():
    parser = argparse.ArgumentParser(description="MoonVeil MK1 — lunar image correspondence & registration")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--sensor", required=True, help="OHRC | TMC | IIRS")
    parser.add_argument("--source-sensor", default=None, help="Override source sensor (for cross-sensor pairs)")
    parser.add_argument("--use-ai-matcher", action="store_true", help="Use AI/Dense Optical Flow matcher instead of standard SIFT")
    parser.add_argument("--outdir", default="outputs", help="Output directory root")
    args = parser.parse_args()

    cfg = default_moonveil_config()
    if args.use_ai_matcher:
        cfg.ai_matcher.enabled = True

    result = run_moonveil(args.reference, args.source, args.sensor, args.source_sensor, config=cfg)

    print("MOONVEIL MK1")
    print("============")
    print()
    print(f"Sensor: {result.sensor_reference}" + (
        f" -> {result.sensor_source}" if result.sensor_source != result.sensor_reference else ""
    ))
    print()
    ev = result.evaluation
    print(f"Candidate matches : {ev.candidate_matches}")
    print(f"Verified inliers  : {ev.verified_inliers}")
    print(f"Inlier ratio      : {ev.inlier_ratio:.1%}")
    print(f"RMSE              : {ev.rmse:.2f} px")
    print()
    print(f"Spatial coverage  : {ev.spatial_coverage:.1%}")
    print(f"Avg reliability   : {ev.average_reliability:.1f}")
    print(f"Overall status    : {ev.reliability_status}")
    print()
    print(f"Registration      : {'SUCCESS' if ev.registration_success else 'NOT RELIABLE — ' + (result.registration.failure_reason or '')}")
    print()

    out_dir = Path(args.outdir) / result.pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_result_to_metrics_dict(result), f, indent=2)
    print(f"Outputs:\n{out_dir}/")


if __name__ == "__main__":
    main()
