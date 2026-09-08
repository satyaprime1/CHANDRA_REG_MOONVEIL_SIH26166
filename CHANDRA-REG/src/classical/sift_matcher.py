import cv2
import numpy as np
from typing import Dict, Any
from src.preprocessing.phase_congruency import compute_phase_congruency_2d, apply_wallis_filter, apply_clahe
from src.refinement.subpixel import fit_subpixel_homography
from src.geometry.spatial_binning import apply_anms_bucketing

def match_sift(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    max_features: int = 2000,
    preprocessing: str = "none",
    use_subpixel: bool = True,
    use_anms: bool = True
) -> Dict[str, Any]:
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    if len(moving_img.shape) == 3:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
        
    if preprocessing == "phase_congruency":
        proc_ref = compute_phase_congruency_2d(ref_img)
        proc_moving = compute_phase_congruency_2d(moving_img)
    elif preprocessing == "wallis":
        proc_ref = apply_wallis_filter(ref_img)
        proc_moving = apply_wallis_filter(moving_img)
    elif preprocessing == "clahe":
        proc_ref = apply_clahe(ref_img)
        proc_moving = apply_clahe(moving_img)
    else:
        proc_ref = ref_img
        proc_moving = moving_img
        
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(proc_ref, None)
    kp2, des2 = sift.detectAndCompute(proc_moving, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            'success': False,
            'num_raw_matches': 0,
            'inlier_count': 0,
            'inlier_ratio': 0.0,
            'H_est': None,
            'pts_ref': np.array([]),
            'pts_moving': np.array([])
        }
        
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_tuple in matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
            
    num_raw = len(good_matches)
    if num_raw < 4:
        return {
            'success': False,
            'num_raw_matches': num_raw,
            'inlier_count': 0,
            'inlier_ratio': 0.0,
            'H_est': None,
            'pts_ref': np.array([]),
            'pts_moving': np.array([])
        }
        
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts_moving = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    H_est, mask = cv2.findHomography(pts_ref.reshape(-1, 1, 2), pts_moving.reshape(-1, 1, 2), method, reproj_threshold)
    
    if mask is not None:
        inliers_mask = mask.ravel().astype(bool)
        pts_ref_inliers = pts_ref[inliers_mask]
        pts_mov_inliers = pts_moving[inliers_mask]
    else:
        inliers_mask = np.zeros(num_raw, dtype=bool)
        pts_ref_inliers = np.array([])
        pts_mov_inliers = np.array([])

    # Phase 5: Sub-Pixel Patch Refinement (< 0.10 px accuracy)
    if use_subpixel and len(pts_ref_inliers) >= 4:
        H_sub, pts_ref_inliers, pts_mov_inliers = fit_subpixel_homography(ref_img, moving_img, pts_ref_inliers, pts_mov_inliers, H_coarse=H_est)
        if H_sub is not None:
            H_est = H_sub

    # Phase 6: ANMS Grid Bucketing Spatial Uniformity
    if use_anms and len(pts_ref_inliers) >= 4:
        pts_ref_inliers, pts_mov_inliers, sui = apply_anms_bucketing(pts_ref_inliers, pts_mov_inliers, img_shape=ref_img.shape)

    inlier_count = len(pts_ref_inliers)
    inlier_ratio = float(inlier_count / num_raw) if num_raw > 0 else 0.0

    return {
        'success': H_est is not None and inlier_count >= 4,
        'num_raw_matches': num_raw,
        'inlier_count': inlier_count,
        'inlier_ratio': inlier_ratio,
        'H_est': H_est,
        'pts_ref': pts_ref_inliers,
        'pts_moving': pts_mov_inliers,
        'inliers_mask': np.ones(inlier_count, dtype=bool),
        'preprocessing': preprocessing,
        'subpixel_enabled': use_subpixel
    }