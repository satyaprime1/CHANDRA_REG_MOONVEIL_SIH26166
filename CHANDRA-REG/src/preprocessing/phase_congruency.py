import cv2
import numpy as np
from typing import Tuple

def compute_phase_congruency_2d(
    img: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55
) -> np.ndarray:
    """
    Computes 2D Phase Congruency using Log-Gabor filter banks in Frequency Domain.
    Phase Congruency produces a dimensionless structural feature map invariant to 
    solar illumination, contrast, and shading variations.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = img.astype(np.float32)
    rows, cols = img.shape
    
    # Fast Fourier Transform of input image
    image_fft = np.fft.fft2(img)
    
    # Create frequency grid coordinates
    x = np.ones((rows, 1)) * np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)[:, np.newaxis] * np.ones((1, cols))
    
    # Polar coordinates: radius (frequency) and theta (orientation angle)
    radius = np.sqrt(x**2 + y**2)
    radius[rows // 2, cols // 2] = 1.0  # Avoid zero division at DC
    theta = np.arctan2(-y, x)
    
    # Shift center to (0,0) for FFT alignment
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    
    energy_sum = np.zeros((rows, cols), dtype=np.float32)
    amplitude_sum = np.zeros((rows, cols), dtype=np.float32)
    
    # Filter bank across orientations and scales
    for o in range(norient):
        angle = o * np.pi / norient
        # Angular filter component
        ds = np.sin(theta - angle)
        dc = np.cos(theta - angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        # Spread function (Gaussian angular spread)
        angular_filter = np.exp(- (dtheta**2) / (2 * (np.pi / norient / 1.2)**2))
        
        for s in range(nscale):
            wavelength = min_wave_length * (mult**s)
            fo = 1.0 / wavelength
            
            # Log-Gabor radial filter
            log_gabor = np.exp(- (np.log(radius / fo)**2) / (2 * np.log(sigma_on_f)**2))
            log_gabor[radius == 0] = 0  # Zero DC component
            
            filter_2d = log_gabor * angular_filter
            
            # Convolve in Frequency Domain
            filtered_fft = image_fft * filter_2d
            spatial_response = np.fft.ifft2(filtered_fft)
            
            real_part = np.real(spatial_response)
            imag_part = np.imag(spatial_response)
            amp = np.sqrt(real_part**2 + imag_part**2)
            
            amplitude_sum += amp
            energy_sum += np.sqrt(np.maximum(0, real_part**2 + imag_part**2 - 0.01))
            
    # Phase Congruency calculation
    epsilon = 1e-4
    phase_congruency = energy_sum / (amplitude_sum + epsilon)
    
    # Normalize output to 0 - 255 uint8 range
    pc_normalized = cv2.normalize(phase_congruency, None, 0, 255, cv2.NORM_MINMAX)
    return pc_normalized.astype(np.uint8)

def apply_wallis_filter(img: np.ndarray, window_size: int = 15, target_mean: float = 127.0, target_std: float = 50.0) -> np.ndarray:
    """
    Applies Wallis filter for local contrast and brightness normalization.
    Decouples terrain topography from strong directional solar shading.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_f = img.astype(np.float32)
    kernel = (window_size, window_size)
    
    mean = cv2.blur(img_f, kernel)
    mean_sq = cv2.blur(img_f**2, kernel)
    var = np.maximum(0, mean_sq - mean**2)
    std = np.sqrt(var)
    
    # Wallis transformation: (I - mean) * (target_std / std) + target_mean
    normalized = (img_f - mean) * (target_std / (std + 1e-4)) + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)

def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)