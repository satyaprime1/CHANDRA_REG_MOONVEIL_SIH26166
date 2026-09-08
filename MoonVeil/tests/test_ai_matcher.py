"""
MoonVeil MK1 — tests/test_ai_matcher.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_matcher import match_features_ai
from src.types import PreprocessedImage
from config import AIMatcherConfig
from tests._helpers import make_synthetic_pair


class TestAIMatcher(unittest.TestCase):
    def test_dis_flow_matcher(self):
        ref_arr, src_arr, _ = make_synthetic_pair(size=200, seed=42)
        ref_img = PreprocessedImage(
            image=ref_arr,
            original_shape=ref_arr.shape,
            processed_shape=ref_arr.shape,
            sensor="OHRC",
            normalization_method="minmax",
            scale_factor=1.0,
        )
        src_img = PreprocessedImage(
            image=src_arr,
            original_shape=src_arr.shape,
            processed_shape=src_arr.shape,
            sensor="OHRC",
            normalization_method="minmax",
            scale_factor=1.0,
        )

        config = AIMatcherConfig(enabled=True, model_name="dis_flow", max_keypoints=500)
        candidates = match_features_ai(src_img, ref_img, config)

        self.assertGreater(len(candidates), 0)
        for c in candidates:
            self.assertIsNotNone(c.source_point)
            self.assertIsNotNone(c.reference_point)
            self.assertGreaterEqual(c.ratio, 0.0)

    def test_auto_fallback(self):
        ref_arr, src_arr, _ = make_synthetic_pair(size=200, seed=42)
        ref_img = PreprocessedImage(
            image=ref_arr,
            original_shape=ref_arr.shape,
            processed_shape=ref_arr.shape,
            sensor="OHRC",
            normalization_method="minmax",
            scale_factor=1.0,
        )
        src_img = PreprocessedImage(
            image=src_arr,
            original_shape=src_arr.shape,
            processed_shape=src_arr.shape,
            sensor="OHRC",
            normalization_method="minmax",
            scale_factor=1.0,
        )

        # "auto" will try LightGlue, fail/fallback if not installed, and succeed with DIS
        config = AIMatcherConfig(enabled=True, model_name="auto", max_keypoints=500)
        candidates = match_features_ai(src_img, ref_img, config)
        self.assertGreater(len(candidates), 0)


if __name__ == "__main__":
    unittest.main()
