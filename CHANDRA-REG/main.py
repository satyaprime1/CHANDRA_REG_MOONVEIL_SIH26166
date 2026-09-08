"""
CHANDRA-REG: Chandrayaan-2 Multi-Modal Image Correspondence & Registration Pipeline
CLI Entry Point
"""

import argparse
import sys
import os
import cv2
import subprocess
from src.preprocessing.synthetic_generator import create_synthetic_pair
from src.classical.sift_matcher import match_sift
from src.deep_matching.loftr_matcher import match_loftr
from src.evaluation.metrics import evaluate_registration
from src.evaluation.visualizer import save_registration_visualization
from benchmark_baseline import run_baseline_benchmark

def main():
    parser = argparse.ArgumentParser(description="Chandrayaan-2 Image Correspondence & Registration Pipeline")
    parser.add_argument("--mode", choices=["synthetic", "benchmark", "pair", "demo"], default="synthetic", help="Execution mode")
    parser.add_argument("--ref", type=str, help="Path to reference image")
    parser.add_argument("--moving", type=str, help="Path to moving image")
    parser.add_argument("--matcher", choices=["sift", "loftr"], default="loftr", help="Feature matching algorithm")
    parser.add_argument("--preprocessing", choices=["none", "phase_congruency", "wallis", "clahe"], default="none", help="Illumination preprocessing filter")
    parser.add_argument("--output", type=str, default="outputs/demo_result.png", help="Path to output visualization PNG")
    args = parser.parse_args()

    if args.mode == "demo":
        print("Launching CHANDRA-REG Interactive Streamlit Web Dashboard...")
        demo_path = os.path.join("app", "demo.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", demo_path])
        return

    if args.mode == "benchmark":
        run_baseline_benchmark()
    elif args.mode == "synthetic":
        print(f"Running synthetic lunar pair registration test (Matcher: {args.matcher.upper()}, Preprocessing: {args.preprocessing})...")
        pair = create_synthetic_pair(scale=1.25, rotation_deg=15.0, illumination_angle_deg=75.0)
        
        if args.matcher == "loftr":
            match_res = match_loftr(pair["reference"], pair["moving"])
        else:
            match_res = match_sift(pair["reference"], pair["moving"], preprocessing=args.preprocessing)
        
        inliers_pts = match_res["pts_ref"][match_res["inliers_mask"]] if match_res["success"] else []
        eval_res = evaluate_registration(
            H_gt=pair["H_GT"],
            H_est=match_res["H_est"],
            pts_ref_inliers=inliers_pts,
            num_raw_matches=match_res["num_raw_matches"],
            inlier_count=match_res["inlier_count"],
            img_shape=pair["reference"].shape
        )
        
        out_file = save_registration_visualization(pair["reference"], pair["moving"], match_res, eval_res, args.output)
        
        print("\n================================================================================")
        print("                  CHANDRA-REG REGISTRATION RESULTS                              ")
        print("================================================================================")
        print(f" Matcher Engine           : {args.matcher.upper()}")
        print(f" Preprocessing Filter    : {args.preprocessing}")
        print(f" Match Success            : {match_res['success']}")
        print(f" Raw Matches Extracted    : {match_res['num_raw_matches']}")
        print(f" Inliers Count            : {match_res['inlier_count']}")
        print(f" Inlier Ratio             : {eval_res['inlier_ratio']*100:.1f}%")
        print(f" Corner Reprojection RMSE : {eval_res['reprojection_rmse']:.2f} px")
        print(f" Spatial Uniformity Index : {eval_res['spatial_uniformity_index']:.3f} (Range: 0 to 1)")
        print(f" Visualization PNG Saved  : {out_file}")
        print("================================================================================\n")

    elif args.mode == "pair":
        if not args.ref or not args.moving:
            print("Error: --ref and --moving image paths are required for 'pair' mode.")
            sys.exit(1)
        ref_img = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
        moving_img = cv2.imread(args.moving, cv2.IMREAD_GRAYSCALE)
        if ref_img is None or moving_img is None:
            print("Error loading input images.")
            sys.exit(1)
            
        if args.matcher == "loftr":
            match_res = match_loftr(ref_img, moving_img)
        else:
            match_res = match_sift(ref_img, moving_img, preprocessing=args.preprocessing)
            
        inliers_pts = match_res["pts_ref"][match_res["inliers_mask"]] if match_res["success"] else []
        eval_res = evaluate_registration(
            H_gt=None,
            H_est=match_res["H_est"],
            pts_ref_inliers=inliers_pts,
            num_raw_matches=match_res["num_raw_matches"],
            inlier_count=match_res["inlier_count"],
            img_shape=ref_img.shape
        )
        out_file = save_registration_visualization(ref_img, moving_img, match_res, eval_res, args.output)
        print(f"Input Pair Registered (Matcher={args.matcher.upper()}): Success={match_res['success']}, Inliers={match_res['inlier_count']}, Viz={out_file}")

if __name__ == "__main__":
    main()