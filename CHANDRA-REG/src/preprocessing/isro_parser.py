import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

def parse_pds4_xml(xml_content: str) -> Dict[str, Any]:
    """
    Parses ISRO Chandrayaan-2 PDS4 XML metadata labels.
    """
    root = ET.fromstring(xml_content)
    meta = {
        'lines': 0,
        'samples': 0,
        'data_type': 'UnsignedByte',
        'sensor': 'CH2',
        'start_time': '',
        'stop_time': ''
    }

    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == 'lines':
            meta['lines'] = int(elem.text)
        elif tag == 'elements' or tag == 'line_samples':
            if meta['samples'] == 0:
                meta['samples'] = int(elem.text)
        elif tag == 'data_type':
            meta['data_type'] = elem.text
        elif tag == 'start_date_time':
            meta['start_time'] = elem.text
        elif tag == 'stop_date_time':
            meta['stop_time'] = elem.text

    return meta

def ingest_isro_zip(zip_path: str, extract_root: str = "data/raw/isro_extracted") -> Dict[str, Any]:
    """
    Ingests any raw ISRO Chandrayaan-2 ZIP data product (OHRC, TMC-2, IIRS).
    Extracts XML metadata, browse PNG, and binary .img paths.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ISRO Zip file not found: {zip_path}")

    zip_filename = os.path.splitext(os.path.basename(zip_path))[0]
    target_dir = os.path.join(extract_root, zip_filename)
    os.makedirs(target_dir, exist_ok=True)

    browse_png_path = None
    xml_label_path = None
    raw_img_path = None
    metadata = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()

        # Locate XML label, Browse PNG, and Binary .img
        xml_files = [f for f in namelist if f.endswith('.xml') and 'data/calibrated' in f]
        png_files = [f for f in namelist if f.endswith('.png') and 'browse' in f]
        img_files = [f for f in namelist if f.endswith('.img') and 'data/calibrated' in f]

        if xml_files:
            xml_label_path = zf.extract(xml_files[0], target_dir)
            xml_content = zf.read(xml_files[0]).decode('utf-8', errors='ignore')
            metadata = parse_pds4_xml(xml_content)

        if png_files:
            browse_png_path = zf.extract(png_files[0], target_dir)

        if img_files:
            raw_img_path = zf.extract(img_files[0], target_dir)

    # Determine sensor type from filename
    if 'ohr' in zip_filename.lower():
        sensor = 'OHRC'
    elif 'tmc' in zip_filename.lower():
        sensor = 'TMC-2'
    elif 'iir' in zip_filename.lower():
        sensor = 'IIRS'
    else:
        sensor = 'Chandrayaan-2'

    metadata['sensor'] = sensor
    metadata['zip_path'] = zip_path
    metadata['browse_png'] = browse_png_path
    metadata['raw_img'] = raw_img_path
    metadata['xml_label'] = xml_label_path

    return metadata

def load_isro_browse_image(metadata: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Loads pre-rendered browse PNG image from ISRO PDS4 package.
    """
    png_path = metadata.get('browse_png')
    if png_path and os.path.exists(png_path):
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        return img
    return None

def extract_tile_from_strip(img: np.ndarray, center_y_ratio: float = 0.5, tile_size: int = 800) -> np.ndarray:
    """
    Extracts a square spatial tile crop from long orbital strip images.
    """
    h, w = img.shape[:2]
    center_y = int(h * center_y_ratio)

    half = tile_size // 2
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)

    tile = img[y1:y2, :]
    if tile.shape[0] != tile.shape[1]:
        # Resize to square tile for consistent matching
        tile = cv2.resize(tile, (tile_size, tile_size))

    return tile