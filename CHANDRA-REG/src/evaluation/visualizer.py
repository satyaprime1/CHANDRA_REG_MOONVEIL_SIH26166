import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any

def save_registration_visualization(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    match_res: Dict[str, Any],
    eval_res: Dict[str, Any],
    output_path: str = "outputs/demo_result.png"
) -> str:
    """
    Renders and saves a comprehensive visual breakdown of registration results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    h, w = ref_img.shape[:2]
    
    # Warp moving image using estimated Homography H_est if successful
    if match_res.get('success', False) and match_res.get('H_est') is not None:
        registered_img = cv2.warpPerspective(moving_img, match_res['H_est'], (w, h))
        diff_map = cv2.absdiff(ref_img, registered_img)
        title_status = f"RMSE: {eval_res['reprojection_rmse']:.2f} px | Inliers: {eval_res['inlier_count']}/{eval_res['num_raw_matches']} ({eval_res['inlier_ratio']*100:.1f}%) | SUI: {eval_res['spatial_uniformity_index']:.3f}"
    else:
        registered_img = np.zeros_like(ref_img)
        diff_map = np.zeros_like(ref_img)
        title_status = "REGISTRATION FAILED (0 Inliers - Try --preprocessing clahe or none)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    filter_name = match_res.get('preprocessing', 'none')
    fig.suptitle(f"CHANDRA-REG Visualizer (Filter: {filter_name})\n{title_status}", fontsize=14, fontweight='bold', color='red' if not match_res.get('success', False) else 'black')
    
    # Panel 1: Reference Image
    axes[0, 0].imshow(ref_img, cmap='gray')
    axes[0, 0].set_title("1. Reference Image (e.g. LRO NAC / TMC)")
    axes[0, 0].axis('off')
    
    # Panel 2: Moving Image (Unregistered)
    axes[0, 1].imshow(moving_img, cmap='gray')
    axes[0, 1].set_title("2. Moving Image (Target OHRC / IIRS)")
    axes[0, 1].axis('off')
    
    # Panel 3: Registered Output Image
    axes[1, 0].imshow(registered_img, cmap='gray')
    if not match_res.get('success', False):
        axes[1, 0].text(w//2, h//2, "Registration Failed\nNo valid inliers found", color='white', fontsize=14, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.7))
    axes[1, 0].set_title("3. Registered / Aligned Output Image")
    axes[1, 0].axis('off')
    
    # Panel 4: Registration Difference Map
    im = axes[1, 1].imshow(diff_map, cmap='inferno')
    axes[1, 1].set_title("4. Alignment Absolute Difference (Error Map)")
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path