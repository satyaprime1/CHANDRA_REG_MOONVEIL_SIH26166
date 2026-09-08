import numpy as np
import pandas as pd
from src.preprocessing.synthetic_generator import create_synthetic_pair
from src.classical.sift_matcher import match_sift
from src.evaluation.metrics import evaluate_registration

def run_baseline_benchmark():
    print("=" * 80)
    print("           CHANDRA-REG: BASELINE SIFT REGISTRATION BENCHMARK")
    print("=" * 80)
    test_cases = [
        {"scale": 1.0,  "rotation": 0.0,  "illum_angle": 30.0, "name": "Identity"},
        {"scale": 1.25, "rotation": 15.0, "illum_angle": 45.0, "name": "Mild Scale + Rot"},
        {"scale": 1.5,  "rotation": 30.0, "illum_angle": 60.0, "name": "Moderate Scale + Rot + Illum"},
        {"scale": 2.0,  "rotation": 45.0, "illum_angle": 75.0, "name": "High Scale + Rot + Harsh Solar"},
        {"scale": 0.8,  "rotation": -20.0,"illum_angle": 50.0, "name": "Downscale + Neg Rot"}
    ]
    results = []
    for tc in test_cases:
        pair = create_synthetic_pair(scale=tc["scale"], rotation_deg=tc["rotation"], illumination_angle_deg=tc["illum_angle"])
        match_res = match_sift(pair["reference"], pair["moving"])
        inliers_pts = match_res["pts_ref"][match_res["inliers_mask"]] if match_res["success"] else np.array([])
        eval_res = evaluate_registration(
            H_gt=pair["H_GT"],
            H_est=match_res["H_est"],
            pts_ref_inliers=inliers_pts,
            num_raw_matches=match_res["num_raw_matches"],
            inlier_count=match_res["inlier_count"],
            img_shape=pair["reference"].shape
        )
        results.append({
            "Test Case": tc["name"],
            "Scale": tc["scale"],
            "Rotation": tc["rotation"],
            "Raw Matches": eval_res["num_raw_matches"],
            "Inliers": eval_res["inlier_count"],
            "Inlier Ratio": f"{eval_res['inlier_ratio']*100:.1f}%",
            "RMSE_px": round(eval_res["reprojection_rmse"], 2) if eval_res["reprojection_rmse"] != float("inf") else "FAIL",
            "SUI": round(eval_res["spatial_uniformity_index"], 3)
        })
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_baseline_benchmark()