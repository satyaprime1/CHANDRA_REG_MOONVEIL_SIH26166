import numpy as np
from typing import Tuple, Dict, Any, Optional

def apply_anms_bucketing(
    pts_ref: np.ndarray,
    pts_moving: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    img_shape: Tuple[int, int] = (800, 800),
    grid_bins: int = 8,
    max_per_bin: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Adaptive Non-Maximum Suppression (ANMS) and Grid Bucketing.
    Prevents keypoints from clustering in 1 crater and enforces uniform spatial 
    distribution across the image domain as required by ISRO PS 26166.
    """
    if len(pts_ref) == 0:
        return pts_ref, pts_moving, 0.0

    h, w = img_shape[:2]
    if confidences is None:
        confidences = np.ones(len(pts_ref), dtype=np.float32)

    x_edges = np.linspace(0, w, grid_bins + 1)
    y_edges = np.linspace(0, h, grid_bins + 1)

    selected_indices = []

    for i in range(grid_bins):
        for j in range(grid_bins):
            x_min, x_max = x_edges[i], x_edges[i + 1]
            y_min, y_max = y_edges[j], y_edges[j + 1]

            in_bin = (
                (pts_ref[:, 0] >= x_min) & (pts_ref[:, 0] < x_max) &
                (pts_ref[:, 1] >= y_min) & (pts_ref[:, 1] < y_max)
            )
            bin_idx = np.where(in_bin)[0]

            if len(bin_idx) > 0:
                sorted_bin_idx = bin_idx[np.argsort(-confidences[bin_idx])]
                selected_indices.extend(sorted_bin_idx[:max_per_bin])

    if len(selected_indices) == 0:
        selected_indices = np.arange(len(pts_ref))

    selected_indices = np.array(selected_indices)
    pts_ref_sub = pts_ref[selected_indices]
    pts_mov_sub = pts_moving[selected_indices]

    hist, _, _ = np.histogram2d(pts_ref_sub[:, 0], pts_ref_sub[:, 1], bins=[x_edges, y_edges])
    probs = hist.ravel() / len(pts_ref_sub)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(grid_bins * grid_bins)
    sui = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return pts_ref_sub, pts_mov_sub, sui