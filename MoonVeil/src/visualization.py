"""
MoonVeil MK1 — src/visualization.py

Responsibility: turn pipeline results into inspectable images. Every
function here RETURNS an image (BGR np.ndarray) — none of them save to
disk. Saving is the integration layer's job (main.py), per the module
contract. Nothing here computes metrics; it only draws what's handed to it.
"""

from __future__ import annotations

import cv2
import numpy as np

_STATUS_COLORS_BGR = {
    "all": (0, 255, 255),       # yellow
    "inliers": (0, 200, 0),     # green
    "outliers": (0, 0, 255),    # red
    "rejected": (0, 0, 255),    # red
    "candidate": (0, 255, 255),  # yellow
}

_RELIABILITY_COLORS_BGR = {
    "HIGH": (0, 200, 0),      # green
    "MEDIUM": (0, 200, 255),  # orange/yellow
    "LOW": (0, 100, 255),     # darker orange
    "REJECTED": (0, 0, 255),  # red
}


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _side_by_side_canvas(reference: np.ndarray, source: np.ndarray):
    """Compose reference (left) and source (right) into one BGR canvas.
    Returns (canvas, ref_bgr_height, ref_bgr_width) so callers can offset
    source-side coordinates by ref width when drawing."""
    ref_bgr = _to_bgr(reference)
    src_bgr = _to_bgr(source)

    h = max(ref_bgr.shape[0], src_bgr.shape[0])
    w = ref_bgr.shape[1] + src_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: ref_bgr.shape[0], : ref_bgr.shape[1]] = ref_bgr
    canvas[: src_bgr.shape[0], ref_bgr.shape[1] : ref_bgr.shape[1] + src_bgr.shape[1]] = src_bgr
    return canvas, ref_bgr.shape[0], ref_bgr.shape[1]


def draw_matches(
    reference: np.ndarray,
    source: np.ndarray,
    matches: list,
    status: str = "all",
) -> np.ndarray:
    """
    Draw correspondences between reference (left) and source (right) as
    connecting lines. `status` only controls the color/label used for
    THIS call's matches — callers pass in already-filtered match lists
    (e.g. just inliers, or just outliers) to build up a full picture
    across multiple calls if desired.
    """
    canvas, ref_h, ref_w = _side_by_side_canvas(reference, source)
    color = _STATUS_COLORS_BGR.get(status, (255, 255, 255))

    for m in matches:
        ref_pt = (int(round(m.reference_point[0])), int(round(m.reference_point[1])))
        src_pt = (int(round(m.source_point[0])) + ref_w, int(round(m.source_point[1])))
        cv2.circle(canvas, ref_pt, 3, color, -1)
        cv2.circle(canvas, src_pt, 3, color, -1)
        cv2.line(canvas, ref_pt, src_pt, color, 1, lineType=cv2.LINE_AA)

    label = f"{status.upper()}: {len(matches)} matches"
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def draw_reliability_matches(
    reference: np.ndarray,
    source: np.ndarray,
    matches: list,
    reliabilities: list,
) -> np.ndarray:
    """
    Same as draw_matches, but colors each correspondence by its MoonVeil
    Reliability status (HIGH/MEDIUM/LOW green->orange->red, REJECTED red),
    so the explainability of the reliability engine is visible directly.
    """
    canvas, ref_h, ref_w = _side_by_side_canvas(reference, source)
    status_by_id = {r.match_id: r.status for r in reliabilities}

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "REJECTED": 0}
    for m in matches:
        status = status_by_id.get(m.match_id, "REJECTED")
        counts[status] = counts.get(status, 0) + 1
        color = _RELIABILITY_COLORS_BGR.get(status, (255, 255, 255))

        ref_pt = (int(round(m.reference_point[0])), int(round(m.reference_point[1])))
        src_pt = (int(round(m.source_point[0])) + ref_w, int(round(m.source_point[1])))
        cv2.circle(canvas, ref_pt, 3, color, -1)
        cv2.circle(canvas, src_pt, 3, color, -1)
        cv2.line(canvas, ref_pt, src_pt, color, 1, lineType=cv2.LINE_AA)

    legend = " | ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(canvas, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_registration_overlay(
    reference: np.ndarray,
    registered: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Red/green channel overlay: reference in the red channel, registered
    source in the green channel. Well-aligned regions appear yellowish;
    misaligned regions show visible red/green fringing/ghosting — a
    standard, intuitive way to visualize registration quality at a glance.
    `alpha` blends this color-fringe view with a plain grayscale average
    (alpha=1.0 -> pure color-fringe view; alpha=0.0 -> plain grayscale
    blend).
    """
    if reference.shape != registered.shape:
        registered = cv2.resize(registered, (reference.shape[1], reference.shape[0]))

    ref_gray = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reg_gray = registered if registered.ndim == 2 else cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)

    fringe = np.zeros((*ref_gray.shape, 3), dtype=np.uint8)
    fringe[:, :, 2] = ref_gray  # R = reference
    fringe[:, :, 1] = reg_gray  # G = registered

    gray_avg = ((ref_gray.astype(np.float64) + reg_gray.astype(np.float64)) / 2.0).astype(np.uint8)
    gray_bgr = cv2.cvtColor(gray_avg, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(fringe, alpha, gray_bgr, 1.0 - alpha, 0)
    return blended


def draw_spatial_distribution(image: np.ndarray, spatial) -> np.ndarray:
    """
    Overlay the spatial grid used for distribution analysis, with the
    match count per cell, so uneven coverage is visible directly on the
    image rather than only as a number.
    """
    canvas = _to_bgr(image)
    h, w = canvas.shape[:2]
    rows, cols = spatial.grid_rows, spatial.grid_cols

    for r in range(1, rows):
        y = int(round(r * h / rows))
        cv2.line(canvas, (0, y), (w, y), (0, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(round(c * w / cols))
        cv2.line(canvas, (x, 0), (x, h), (0, 255, 255), 1, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            cell_w, cell_h = w / cols, h / rows
            cx = int(round(c * cell_w + cell_w / 2))
            cy = int(round(r * cell_h + cell_h / 2))
            count = int(spatial.counts[r, c])
            cv2.putText(
                canvas, str(count), (cx - 8, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

    label = f"Coverage: {spatial.coverage_ratio:.0%}  Distribution score: {spatial.distribution_score:.2f}"
    cv2.putText(canvas, label, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

