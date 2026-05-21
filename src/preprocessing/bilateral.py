"""
Bilateral filtering for sensor-noise removal while preserving edges and textures.

Bilateral filter considers both spatial distance (sigma_space) and pixel value
similarity (sigma_color), so it smooths flat regions without blurring edges —
important for weapon silhouette detail in drone imagery.
"""

import cv2
import numpy as np
from typing import Optional
from .tracker import PreprocessingTracker


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    tracker: Optional[PreprocessingTracker] = None,
    image_id: Optional[str] = None,
) -> np.ndarray:
    """
    Apply bilateral filter to an image.

    Args:
        image:       BGR or grayscale uint8 image.
        d:           Diameter of each pixel neighbourhood. 9 is a good
                     balance between noise removal and speed; use 5 for
                     real-time, 15 for offline quality.
        sigma_color: Filter sigma in colour space. Larger values mix more
                     distant colours, increasing smoothing range.
        sigma_space: Filter sigma in coordinate space. Larger values mean
                     farther pixels influence each other when their colours
                     are close enough.
        tracker:     Optional PreprocessingTracker; records the operation if
                     provided.
        image_id:    Identifier string logged in the tracker entry.

    Returns:
        Filtered image (same dtype and channel count as input).
    """
    if image is None or image.size == 0:
        raise ValueError("apply_bilateral_filter: received empty image")

    original_dtype = image.dtype
    if original_dtype != np.uint8:
        # bilateral filter requires uint8; convert, filter, restore
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    filtered = cv2.bilateralFilter(img_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    result = filtered.astype(original_dtype)

    if tracker is not None:
        before_mean = float(np.mean(image))
        after_mean = float(np.mean(result))
        tracker.record(
            operation="bilateral_filter",
            image_id=image_id,
            params={"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space},
            metrics={
                "mean_before": round(before_mean, 4),
                "mean_after": round(after_mean, 4),
                "mean_delta": round(after_mean - before_mean, 4),
                "shape": list(image.shape),
            },
        )

    return result
