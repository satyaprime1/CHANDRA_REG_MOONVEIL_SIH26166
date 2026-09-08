import pytest
import numpy as np
from src.preprocessing.synthetic_generator import create_synthetic_pair
from src.preprocessing.phase_congruency import compute_phase_congruency_2d, apply_wallis_filter, apply_clahe
from src.classical.sift_matcher import match_sift
from src.deep_matching.loftr_matcher import match_loftr
from src.evaluation.metrics import evaluate_registration

def test_synthetic_pair_generator():
    pair = create_synthetic_pair(scale=1.2, rotation_deg=10.0)
    assert 'reference' in pair and 'moving' in pair and 'H_GT' in pair
    assert pair['reference'].shape == (800, 800)

def test_preprocessing_filters():
    pair = create_synthetic_pair()
    ref = pair['reference']
    pc = compute_phase_congruency_2d(ref)
    wallis = apply_wallis_filter(ref)
    clahe = apply_clahe(ref)
    assert pc.shape == ref.shape and wallis.shape == ref.shape and clahe.shape == ref.shape

def test_sift_matcher_with_preprocessing():
    pair = create_synthetic_pair(scale=1.15, rotation_deg=10.0, illumination_angle_deg=60.0)
    match_res = match_sift(pair['reference'], pair['moving'], preprocessing="wallis")
    assert match_res['success'] is True
    assert match_res['inlier_count'] > 0

def test_loftr_deep_matcher():
    pair = create_synthetic_pair(scale=1.25, rotation_deg=15.0, illumination_angle_deg=75.0)
    match_res = match_loftr(pair['reference'], pair['moving'])
    assert match_res['success'] is True
    assert match_res['inlier_count'] > 100