"""
MoonVeil MK1 — src/ai_matcher.py

Responsibility: Advanced/AI feature matching for low-contrast lunar terrain
where standard SIFT descriptors struggle.

Provides two matching engines:
1. OpenCV DIS Dense Optical Flow (always available, fast, zero external dependencies).
2. PyTorch LightGlue / SuperPoint deep matcher (optional, with automatic fallback if not installed).
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import MatchCandidate, PreprocessedImage
from config import AIMatcherConfig


def match_features_ai(
    source: PreprocessedImage,
    reference: PreprocessedImage,
    config: AIMatcherConfig,
) -> list[MatchCandidate]:
    """
    Perform dense optical flow or deep neural feature matching between source and reference.

    Parameters
    ----------
    source : PreprocessedImage
    reference : PreprocessedImage
    config : AIMatcherConfig

    Returns
    -------
    list[MatchCandidate]
    """
    model_name = config.model_name.lower()

    if model_name in ("lightglue", "superpoint", "auto"):
        try:
            return _match_lightglue(source, reference, config)
        except (ImportError, Exception):
            # Fall back seamlessly to DIS Optical Flow if PyTorch/LightGlue is not available
            pass

    return _match_dis_flow(source, reference, config)


def _match_dis_flow(
    source: PreprocessedImage,
    reference: PreprocessedImage,
    config: AIMatcherConfig,
) -> list[MatchCandidate]:
    """Dense Optical Flow matching using OpenCV DIS."""
    img_src = source.image
    img_ref = reference.image

    if img_src.dtype != np.uint8:
        img_src = cv2.normalize(img_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img_ref.dtype != np.uint8:
        img_ref = cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect Good Features To Track on reference
    max_corners = config.max_keypoints if config.max_keypoints > 0 else 1000
    pts_ref = cv2.goodFeaturesToTrack(
        img_ref,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )

    if pts_ref is None or len(pts_ref) == 0:
        return []

    # Calculate DIS Optical Flow (forward and backward for validation)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow_fw = dis.calc(img_ref, img_src, None)
    flow_bw = dis.calc(img_src, img_ref, None)

    candidates = []
    h, w = img_ref.shape

    for idx, pt in enumerate(pts_ref):
        rx, ry = pt[0]
        irx, iry = int(round(rx)), int(round(ry))
        if 0 <= irx < w and 0 <= iry < h:
            flow_vec = flow_fw[iry, irx]
            sx = rx + flow_vec[0]
            sy = ry + flow_vec[1]

            # Forward-backward error check
            isx, isy = int(round(sx)), int(round(sy))
            if 0 <= isx < w and 0 <= isy < h:
                back_vec = flow_bw[isy, isx]
                rec_x = sx + back_vec[0]
                rec_y = sy + back_vec[1]
                fb_err = np.sqrt((rx - rec_x) ** 2 + (ry - rec_y) ** 2)

                if fb_err <= 3.0:  # Valid optical flow match
                    mc = MatchCandidate(
                        match_id=idx,
                        source_index=idx,
                        reference_index=idx,
                        source_point=(float(sx), float(sy)),
                        reference_point=(float(rx), float(ry)),
                        descriptor_distance=float(fb_err),
                        ratio=float(1.0 - (fb_err / 3.0)),
                        is_ratio_pass=True,
                    )
                    candidates.append(mc)

    return candidates


def _match_lightglue(
    source: PreprocessedImage,
    reference: PreprocessedImage,
    config: AIMatcherConfig,
) -> list[MatchCandidate]:
    """LightGlue / SuperPoint deep matcher wrapper (requires torch + lightglue)."""
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=config.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    t_src = torch.from_numpy(source.image).float() / 255.0
    t_ref = torch.from_numpy(reference.image).float() / 255.0

    if t_src.ndim == 2:
        t_src = t_src.unsqueeze(0).unsqueeze(0)
    if t_ref.ndim == 2:
        t_ref = t_ref.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        feats_src = extractor({"image": t_src.to(device)})
        feats_ref = extractor({"image": t_ref.to(device)})
        matches01 = matcher({"image0": feats_src, "image1": feats_ref})

    feats_src, feats_ref, matches01 = [rbd(x) for x in (feats_src, feats_ref, matches01)]
    kpts_src = feats_src["keypoints"].cpu().numpy()
    kpts_ref = feats_ref["keypoints"].cpu().numpy()
    matches = matches01["matches"].cpu().numpy()
    scores = matches01["scores"].cpu().numpy()

    candidates = []
    match_counter = 0
    for (idx_src, idx_ref), score in zip(matches, scores):
        if score >= config.confidence_threshold:
            sx, sy = kpts_src[idx_src]
            rx, ry = kpts_ref[idx_ref]

            mc = MatchCandidate(
                match_id=match_counter,
                source_index=int(idx_src),
                reference_index=int(idx_ref),
                source_point=(float(sx), float(sy)),
                reference_point=(float(rx), float(ry)),
                descriptor_distance=float(1.0 - score),
                ratio=float(score),
                is_ratio_pass=True,
            )
            candidates.append(mc)
            match_counter += 1

    return candidates
