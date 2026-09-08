"""
MoonVeil MK1 — tests/test_spatial.py

Sanity tests for src/spatial.py:
  - evenly spread points -> high coverage_ratio and high distribution_score
  - all points clustered in one cell -> low coverage_ratio and low score
  - no points -> zeroed-out result, no crash
  - bad grid config -> ValueError
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spatial import analyze_spatial_distribution
from config import SpatialConfig


def test_evenly_spread_points():
    # One point roughly centered in each of the 4x4 cells of a 400x400 image.
    points = []
    for r in range(4):
        for c in range(4):
            x = c * 100 + 50
            y = r * 100 + 50
            points.append([x, y])
    points = np.array(points, dtype=np.float64)

    result = analyze_spatial_distribution(points, (400, 400), SpatialConfig(rows=4, cols=4))

    assert result.occupied_cells == 16
    assert result.coverage_ratio == 1.0
    assert result.distribution_score > 0.95
    print(f"Even spread: coverage={result.coverage_ratio:.2f}, score={result.distribution_score:.2f} — OK")


def test_clustered_points():
    # All 20 points crammed into one corner cell.
    points = np.array([[5 + i * 0.1, 5 + i * 0.1] for i in range(20)])

    result = analyze_spatial_distribution(points, (400, 400), SpatialConfig(rows=4, cols=4))

    assert result.occupied_cells == 1
    assert result.coverage_ratio == pytest_approx(1 / 16)
    assert result.distribution_score < 0.1
    print(f"Clustered: coverage={result.coverage_ratio:.3f}, score={result.distribution_score:.3f} — OK")


def pytest_approx(x, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_no_points():
    result = analyze_spatial_distribution(np.zeros((0, 2)), (400, 400), SpatialConfig())
    assert result.occupied_cells == 0
    assert result.coverage_ratio == 0.0
    assert result.distribution_score == 0.0
    print("No points -> zeroed result: OK")


def test_bad_grid_raises():
    try:
        analyze_spatial_distribution(np.zeros((0, 2)), (100, 100), SpatialConfig(rows=0, cols=4))
        raise AssertionError("Expected ValueError")
    except ValueError:
        print("Bad grid config -> ValueError: OK")


if __name__ == "__main__":
    test_evenly_spread_points()
    test_clustered_points()
    test_no_points()
    test_bad_grid_raises()
    print("\nAll spatial tests passed.")
