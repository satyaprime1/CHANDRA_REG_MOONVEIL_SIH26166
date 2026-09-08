"""
MoonVeil MK1 — tests/test_pds4.py
"""

import sys
import unittest
import tempfile
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import parse_pds4_xml, load_image
from tests._helpers import make_textured_image


class TestPDS4(unittest.TestCase):
    def test_parse_pds4_xml(self):
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
            <Target_Identification>
                <name>Moon</name>
            </Target_Identification>
            <Instrument>
                <instrument_id>OHRC</instrument_id>
            </Instrument>
            <pixel_resolution>0.25</pixel_resolution>
            <start_date_time>2021-01-01T00:00:00Z</start_date_time>
        </Product_Observational>
        """

        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "test_product.xml"
            xml_path.write_text(sample_xml, encoding="utf-8")

            img_path = Path(tmp) / "test_product.png"
            img = make_textured_image(size=100)
            cv2.imwrite(str(img_path), img)

            metadata = parse_pds4_xml(xml_path)
            self.assertEqual(metadata.get("target_name"), "Moon")
            self.assertEqual(metadata.get("instrument_id"), "OHRC")
            self.assertEqual(metadata.get("pixel_resolution_m"), 0.25)

            img_data = load_image(img_path, sensor="OHRC")
            pds4_meta = img_data.metadata.get("pds4_metadata", {})
            self.assertEqual(pds4_meta.get("target_name"), "Moon")


if __name__ == "__main__":
    unittest.main()
