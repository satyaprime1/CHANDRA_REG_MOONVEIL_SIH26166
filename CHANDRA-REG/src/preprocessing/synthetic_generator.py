import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def generate_lunar_surface(width: int = 800, height: int = 800, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    base = np.random.normal(120, 15, (height, width)).astype(np.float32)
    base = cv2.GaussianBlur(base, (15, 15), 5.0)
    num_craters = np.random.randint(15, 30)
    for _ in range(num_craters):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        radius = np.random.randint(10, 80)
        y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
        dist_sq = x*x + y*y
        floor_mask = dist_sq <= radius*radius
        base[floor_mask] = np.clip(base[floor_mask] * 0.6 + 20, 0, 255)
        rim_mask = (dist_sq > (radius*0.8)**2) & (dist_sq <= (radius*1.2)**2)
        shadow_mask = rim_mask & (x > 0)
        highlight_mask = rim_mask & (x <= 0)
        base[highlight_mask] = np.clip(base[highlight_mask] * 1.5 + 40, 0, 255)
        base[shadow_mask] = np.clip(base[shadow_mask] * 0.4, 0, 255)
    surface = cv2.GaussianBlur(base, (5, 5), 1.0)
    return np.clip(surface, 0, 255).astype(np.uint8)

def apply_illumination_change(image: np.ndarray, light_angle_deg: float = 45.0, contrast_scale: float = 1.2) -> np.ndarray:
    h, w = image.shape[:2]
    rad = np.radians(light_angle_deg)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    gradient = xx * np.cos(rad) + yy * np.sin(rad)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    illumination_field = 0.5 + 0.8 * gradient
    illuminated = image.astype(np.float32) * illumination_field * contrast_scale
    return np.clip(illuminated, 0, 255).astype(np.uint8)

def create_synthetic_pair(
    image: Optional[np.ndarray] = None,
    scale: float = 1.25,
    rotation_deg: float = 25.0,
    translation: Tuple[float, float] = (30.0, -15.0),
    perspective_skew: Tuple[float, float] = (0.0001, -0.0002),
    illumination_angle_deg: float = 60.0
) -> Dict[str, Any]:
    if image is None:
        ref_img = generate_lunar_surface(800, 800)
    else:
        ref_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = ref_img.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0
    rad = np.radians(rotation_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    T_center = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype=np.float32)
    R_S = np.array([[scale * cos_a, -scale * sin_a, 0], [scale * sin_a, scale * cos_a, 0], [0, 0, 1]], dtype=np.float32)
    T_trans = np.array([[1, 0, center_x + translation[0]], [0, 1, center_y + translation[1]], [0, 0, 1]], dtype=np.float32)
    P_skew = np.array([[1, 0, 0], [0, 1, 0], [perspective_skew[0], perspective_skew[1], 1]], dtype=np.float32)
    H_GT = T_trans @ R_S @ P_skew @ T_center
    moving_img = cv2.warpPerspective(ref_img, H_GT, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    moving_img = apply_illumination_change(moving_img, light_angle_deg=illumination_angle_deg)
    return {'reference': ref_img, 'moving': moving_img, 'H_GT': H_GT, 'scale': scale, 'rotation_deg': rotation_deg, 'translation': translation}

if __name__ == '__main__':
    pair = create_synthetic_pair()
    print("Synthetic pair successfully created!")
    print("H_GT shape:", pair["H_GT"].shape)