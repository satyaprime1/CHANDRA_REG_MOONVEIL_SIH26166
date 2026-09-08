"""
MoonVeil MK1 — tests/_helpers.py

Shared synthetic-data helpers for tests that need textured images (so SIFT
has real structure to find) and known-ground-truth transformed pairs (so
geometry/registration correctness can be checked against a known answer,
per the blueprint's synthetic ground-truth testing strategy).
"""

import cv2
import numpy as np


def make_textured_image(size=300, seed=42):
    """Draw circles/lines to simulate crater-like/rock-like texture —
    gives SIFT plenty of real corners and blobs, unlike pure noise."""
    img = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(80):
        center = tuple(rng.integers(15, size - 15, size=2).tolist())
        radius = int(rng.integers(4, 20))
        color = int(rng.integers(60, 255))
        cv2.circle(img, center, radius, color, -1)
    for _ in range(30):
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple(rng.integers(0, size, size=2).tolist())
        cv2.line(img, pt1, pt2, int(rng.integers(40, 255)), 2)
    return img


def make_known_homography(
    size, rotation_deg=8.0, scale=1.05, tx=15.0, ty=-10.0, perspective=0.0002
):
    """Build a small, realistic known homography: rotation + scale +
    translation + a touch of perspective skew."""
    cx, cy = size / 2.0, size / 2.0
    theta = np.deg2rad(rotation_deg)

    # Rotation + scale about image center, then translate.
    cos_t, sin_t = np.cos(theta) * scale, np.sin(theta) * scale
    A = np.array(
        [
            [cos_t, -sin_t, (1 - cos_t) * cx + sin_t * cy + tx],
            [sin_t, cos_t, -sin_t * cx + (1 - cos_t) * cy + ty],
            [0.0, 0.0, 1.0],
        ]
    )

    # Mild perspective component so it's a true homography, not just affine.
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective, perspective * 0.5, 1.0]])

    H = P @ A
    H = H / H[2, 2]
    return H


def make_synthetic_pair(size=300, seed=42):
    """Returns (reference_image, source_image, H_ref_to_source) where
    source = warp(reference, H). i.e. reference_point @ H ~ source_point.
    Downstream code that estimates 'source -> reference' should recover
    approximately inverse(H)."""
    reference = make_textured_image(size=size, seed=seed)
    H = make_known_homography(size)
    source = cv2.warpPerspective(reference, H, (size, size))
    return reference, source, H
