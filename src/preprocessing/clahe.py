"""
CLAHE (Contrast-Limited Adaptive Histogram Equalization) for low-light /
low-contrast imagery (e.g., space or night-time drone footage).

Process:
  1. Convert BGR → LAB colour space (or keep grayscale).
  2. Apply CLAHE to the L (luminance) channel only — preserves hue/saturation.
  3. Clip limit prevents noise amplification by redistributing histogram excess.
  4. Tile grid divides the image into blocks; bilinear interpolation smooths
     block boundaries so there are no visible seams.
  5. Convert back to BGR (or return grayscale).
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from .tracker import PreprocessingTracker


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    tracker: Optional[PreprocessingTracker] = None,
    image_id: Optional[str] = None,
) -> np.ndarray:
    """
    Apply CLAHE to a BGR or grayscale image.

    For BGR input the filter is applied to the L channel of LAB colour space
    so colour balance is not disturbed — only local luminance contrast is lifted.

    Args:
        image:          BGR (H×W×3) or grayscale (H×W) uint8 image.
        clip_limit:     Threshold for histogram clipping. 2.0 is conservative;
                        raise to 4–8 for very dark images, but expect more noise.
        tile_grid_size: (cols, rows) of non-overlapping tiles. Smaller tiles →
                        more localised enhancement. (8,8) is the standard default.
        tracker:        Optional PreprocessingTracker; records the operation.
        image_id:       Identifier string logged in the tracker entry.

    Returns:
        Enhanced image, same shape and dtype as input.
    """
    if image is None or image.size == 0:
        raise ValueError("apply_clahe: received empty image")

    original_dtype = image.dtype
    if original_dtype != np.uint8:
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    is_color = img_u8.ndim == 3 and img_u8.shape[2] == 3

    if is_color:
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        result_u8 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        result_u8 = clahe.apply(img_u8)

    result = result_u8.astype(original_dtype)

    if tracker is not None:
        # RMS contrast: std of the luminance channel
        def _rms_contrast(img: np.ndarray) -> float:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return float(np.std(gray.astype(np.float32)))

        before_contrast = _rms_contrast(img_u8)
        after_contrast = _rms_contrast(result_u8)
        tracker.record(
            operation="clahe",
            image_id=image_id,
            params={
                "clip_limit": clip_limit,
                "tile_grid_size": list(tile_grid_size),
                "colour_mode": "LAB-L" if is_color else "grayscale",
            },
            metrics={
                "rms_contrast_before": round(before_contrast, 4),
                "rms_contrast_after": round(after_contrast, 4),
                "contrast_gain": round(after_contrast - before_contrast, 4),
                "shape": list(image.shape),
            },
        )

    return result
