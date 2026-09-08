"""
MoonVeil MK1 — src/spatial.py

Responsibility: divide the image into a grid and report how evenly a set
of match points is distributed across it. This directly supports the SIH
problem statement's requirement for uniformly distributed match points,
and feeds the reliability engine's spatial component.
"""

from __future__ import annotations

import numpy as np

from .types import SpatialDistribution
from config import SpatialConfig


def analyze_spatial_distribution(
    points: np.ndarray,
    image_shape: tuple,
    config: SpatialConfig,
) -> SpatialDistribution:
    """
    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        (x, y) coordinates, e.g. reference-image coordinates of accepted
        matches.
    image_shape : tuple
        (height, width) of the image the points were measured in.
    config : SpatialConfig

    Returns
    -------
    SpatialDistribution
    """
    if config.rows <= 0 or config.cols <= 0:
        raise ValueError("SpatialConfig.rows and .cols must both be > 0")

    height, width = image_shape[0], image_shape[1]
    rows, cols = config.rows, config.cols
    counts = np.zeros((rows, cols), dtype=int)

    points = np.asarray(points, dtype=np.float64)

    if points.size > 0:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {points.shape}")

        xs = points[:, 0]
        ys = points[:, 1]

        col_idx = np.clip((xs / max(width, 1) * cols).astype(int), 0, cols - 1)
        row_idx = np.clip((ys / max(height, 1) * rows).astype(int), 0, rows - 1)

        for r, c in zip(row_idx, col_idx):
            counts[r, c] += 1

    total_cells = rows * cols
    occupied_cells = int(np.count_nonzero(counts))
    coverage_ratio = occupied_cells / total_cells if total_cells > 0 else 0.0

    total_points = counts.sum()
    if total_points > 0:
        probs = counts.flatten() / total_points
        nonzero = probs[probs > 0]
        entropy = -np.sum(nonzero * np.log(nonzero))
        max_entropy = np.log(total_cells) if total_cells > 1 else 1.0
        distribution_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        distribution_score = 0.0

    return SpatialDistribution(
        grid_rows=rows,
        grid_cols=cols,
        counts=counts,
        occupied_cells=occupied_cells,
        total_cells=total_cells,
        coverage_ratio=coverage_ratio,
        distribution_score=distribution_score,
    )

