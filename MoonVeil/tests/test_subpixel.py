"""
MoonVeil MK1 — tests/test_subpixel.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import refine_subpixel_keypoints
from tests._helpers import make_textured_image


class TestSubpixel(unittest.TestCase):
    def test_refine_subpixel_keypoints(self):
        img = make_textured_image(size=200, seed=42)
        detector = cv2.SIFT_create()
        kps, _ = detector.detectAndCompute(img, None)
        self.assertGreater(len(kps), 0)

        refined = refine_subpixel_keypoints(img, list(kps))
        self.assertEqual(len(refined), len(kps))

        # Coordinates should be float and close to original keypoints
        for orig, ref in zip(kps, refined):
            self.assertIsInstance(ref.pt[0], float)
            self.assertIsInstance(ref.pt[1], float)
            self.assertLess(abs(orig.pt[0] - ref.pt[0]), 5.0)
            self.assertLess(abs(orig.pt[1] - ref.pt[1]), 5.0)


if __name__ == "__main__":
    unittest.main()
