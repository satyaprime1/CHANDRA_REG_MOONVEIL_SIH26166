import numpy as np
from typing import Dict, Any

def compute_reprojection_rmse(H_gt: np.ndarray, H_est: np.ndarray, img_shape: tuple = (800, 800)) -> float:
    if H_est is None or H_gt is None:
        return float('inf')
        
    h, w = img_shape[:2]
    grid_x, grid_y = np.meshgrid(np.linspace(0, w, 20), np.linspace(0, h, 20))
    pts = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)])
    
    # Project with Ground Truth
    proj_gt = H_gt @ pts
    proj_gt /= (proj_gt[2, :] + 1e-8)
    
    # Project with Estimated Homography
    proj_est = H_est @ pts
    proj_est /= (proj_est[2, :] + 1e-8)
    
    # Euclidean distance error across test grid
    errors = np.sqrt(np.sum((proj_gt[:2, :] - proj_est[:2, :])**2, axis=0))
    rmse = float(np.mean(errors))
    return rmse

def compute_spatial_uniformity_index(pts: np.ndarray, img_shape: tuple = (800, 800), grid_bins: int = 8) -> float:
    if len(pts) == 0:
        return 0.0
        
    h, w = img_shape[:2]
    x_edges = np.linspace(0, w, grid_bins + 1)
    y_edges = np.linspace(0, h, grid_bins + 1)
    
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    
    total_pts = len(pts)
    probs = hist.ravel() / total_pts
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(grid_bins * grid_bins)
    
    sui = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return sui

def evaluate_registration(
    H_gt: np.ndarray,
    H_est: np.ndarray,
    pts_ref_inliers: np.ndarray,
    num_raw_matches: int,
    inlier_count: int,
    img_shape: tuple = (800, 800)
) -> Dict[str, Any]:
    inlier_ratio = float(inlier_count / num_raw_matches) if num_raw_matches > 0 else 0.0
    rmse = compute_reprojection_rmse(H_gt, H_est, img_shape)
    sui = compute_spatial_uniformity_index(pts_ref_inliers, img_shape)
    
    return {
        'num_raw_matches': num_raw_matches,
        'inlier_count': inlier_count,
        'inlier_ratio': inlier_ratio,
        'reprojection_rmse': rmse,
        'spatial_uniformity_index': sui
    }
