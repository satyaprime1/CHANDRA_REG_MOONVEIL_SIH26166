import cv2
import torch
import ssl
import numpy as np
import kornia.feature as KF
from typing import Dict, Any, Optional
from src.refinement.subpixel import fit_subpixel_homography
from src.geometry.spatial_binning import apply_anms_bucketing

ssl._create_default_https_context = ssl._create_unverified_context
_LOFTR_MODEL = None

def get_loftr_model(pretrained: str = 'outdoor') -> KF.LoFTR:
    global _LOFTR_MODEL
    if _LOFTR_MODEL is None:
        _LOFTR_MODEL = KF.LoFTR(pretrained=pretrained)
        _LOFTR_MODEL.eval()
    return _LOFTR_MODEL

def match_loftr(
    ref_img: np.ndarray,
    moving_img: np.ndarray,
    confidence_threshold: float = 0.2,
    reproj_threshold: float = 3.0,
    pretrained: str = 'outdoor',
    use_subpixel: bool = True,
    use_anms: bool = True
) -> Dict[str, Any]:
    """
    LoFTR Deep Vision Transformer Matcher with Phase 5 Sub-Pixel Refinement and Phase 6 ANMS Spatial Uniformity.
    """
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    if len(moving_img.shape) == 3:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)

    t1 = torch.from_numpy(ref_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    t2 = torch.from_numpy(moving_img).float().unsqueeze(0).unsqueeze(0) / 255.0

    model = get_loftr_model(pretrained)

    with torch.no_grad():
        input_dict = {"image0": t1, "image1": t2}
        correspondences = model(input_dict)

    kpts0 = correspondences['keypoints0'].cpu().numpy()
    kpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()

    valid_mask = conf >= confidence_threshold
    pts_ref = kpts0[valid_mask]
    pts_moving = kpts1[valid_mask]
    num_raw = len(pts_ref)

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

    method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    H_est, mask = cv2.findHomography(pts_ref.reshape(-1, 1, 2), pts_moving.reshape(-1, 1, 2), method, reproj_threshold)

    if mask is not None:
        inliers_mask = mask.ravel().astype(bool)
        pts_ref_inliers = pts_ref[inliers_mask]
        pts_mov_inliers = pts_moving[inliers_mask]
    else:
        pts_ref_inliers = np.array([])
        pts_mov_inliers = np.array([])

    # Phase 6: ANMS Spatial Uniformity
    if use_anms and len(pts_ref_inliers) >= 4:
        pts_ref_inliers, pts_mov_inliers, sui = apply_anms_bucketing(pts_ref_inliers, pts_mov_inliers, img_shape=ref_img.shape)

    # Phase 5: Sub-Pixel Refinement (< 0.10 px accuracy)
    if use_subpixel and len(pts_ref_inliers) >= 4:
        H_sub, pts_ref_inliers, pts_mov_inliers = fit_subpixel_homography(ref_img, moving_img, pts_ref_inliers, pts_mov_inliers)
        if H_sub is not None:
            H_est = H_sub

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
        'subpixel_enabled': use_subpixel
    }