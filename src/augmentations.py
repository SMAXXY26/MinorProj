"""
augmentations.py  —  Image-level augmentations for the detector dataset.
==========================================================================
Reads the `augmentation` block from hyperparams.yaml and applies:
  - Motion blur  (camera shake / fast movement)
  - Gaussian blur (out-of-focus)
  - Gaussian noise (sensor noise)
  - JPEG compression artefacts
  - Downscale + upscale (low-res simulation)
  - Color / brightness / contrast jitter
  - Horizontal flip (with label adjustment)

All transforms operate on **uint8 BGR images** (OpenCV convention) and
return the augmented image + adjusted labels.
"""

import random
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Individual transforms (operate on uint8 BGR numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _motion_blur(img, cfg):
    """Apply random directional motion blur."""
    lo, hi = cfg["kernel_size"]
    ksize = random.randint(lo, hi)
    if ksize % 2 == 0:
        ksize += 1  # must be odd
    angle = random.uniform(0, 360)
    # Build motion-blur kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def _gaussian_blur(img, cfg):
    """Apply Gaussian blur with random sigma."""
    lo, hi = cfg["sigma"]
    sigma = random.uniform(lo, hi)
    ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _gaussian_noise(img, cfg):
    """Add Gaussian noise (pixel-level)."""
    lo, hi = cfg["std"]
    std = random.uniform(lo, hi)
    noise = np.random.normal(cfg.get("mean", 0.0), std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _jpeg_compression(img, cfg):
    """Encode/decode with random JPEG quality."""
    lo, hi = cfg["quality"]
    quality = random.randint(lo, hi)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _downscale(img, cfg):
    """Downscale then upscale to simulate low resolution."""
    lo, hi = cfg["scale"]
    scale = random.uniform(lo, hi)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _random_rotate(img, labels, max_degrees):
    """Random rotation for aerial views — weapons appear at all angles."""
    angle = random.uniform(-max_degrees, max_degrees)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if labels is not None and len(labels) > 0:
        # Rotate bounding box centres; use enclosing box (conservative)
        rad   = np.deg2rad(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx_px = (labels[:, 1] - 0.5) * w
        cy_px = (labels[:, 2] - 0.5) * h
        rx    = cos_a * cx_px - sin_a * cy_px
        ry    = sin_a * cx_px + cos_a * cy_px
        labels[:, 1] = np.clip(rx / w + 0.5, 0.01, 0.99)
        labels[:, 2] = np.clip(ry / h + 0.5, 0.01, 0.99)

    return img, labels


def _color_jitter(img, cfg):
    """Random brightness, contrast, saturation, hue adjustments."""
    img = img.astype(np.float32)

    # Brightness
    if cfg.get("brightness", 0) > 0:
        delta = random.uniform(-cfg["brightness"], cfg["brightness"]) * 255
        img = np.clip(img + delta, 0, 255)

    # Contrast
    if cfg.get("contrast", 0) > 0:
        factor = random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0, 255)

    img = img.astype(np.uint8)

    # Saturation + hue in HSV space
    if cfg.get("saturation", 0) > 0 or cfg.get("hue", 0) > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        if cfg.get("saturation", 0) > 0:
            factor = random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"])
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        if cfg.get("hue", 0) > 0:
            delta = random.uniform(-cfg["hue"], cfg["hue"]) * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def _coarse_dropout(img, cfg):
    """Drop num_holes random rectangular patches to black (simulates occlusion)."""
    h, w = img.shape[:2]
    num_holes  = cfg.get("num_holes", 8)
    hole_h_min = int(h * cfg.get("min_height", 0.05))
    hole_h_max = int(h * cfg.get("max_height", 0.15))
    hole_w_min = int(w * cfg.get("min_width",  0.05))
    hole_w_max = int(w * cfg.get("max_width",  0.15))
    img = img.copy()
    for _ in range(num_holes):
        ph = random.randint(hole_h_min, max(hole_h_min, hole_h_max))
        pw = random.randint(hole_w_min, max(hole_w_min, hole_w_max))
        y1 = random.randint(0, h - ph)
        x1 = random.randint(0, w - pw)
        img[y1:y1 + ph, x1:x1 + pw] = 0
    return img


def _hsv_augment(img, h_gain, s_gain, v_gain):
    """Classic YOLOv5-style HSV augmentation."""
    if h_gain == 0 and s_gain == 0 and v_gain == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] * r[0]) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Main augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentations(img, labels, aug_cfg):
    """
    Apply configured augmentations to an image + YOLO labels.

    Args:
        img:      uint8 BGR numpy array (H, W, 3)
        labels:   numpy array of shape (N, 5+) with [cls, cx, cy, w, h, ...]
        aug_cfg:  the `augmentation` dict from hyperparams.yaml. If None or
                  disabled, returns image and labels unchanged.

    Returns:
        img, labels  (augmented)
    """
    if aug_cfg is None or not aug_cfg.get("enabled", False):
        return img, labels

    # Global gate — skip augmentation for some images entirely
    if random.random() > aug_cfg.get("apply_prob", 1.0):
        return img, labels

    # --- Degradation augmentations (order matters: blur before noise) ---
    mb = aug_cfg.get("motion_blur", {})
    if mb.get("enabled") and random.random() < mb.get("prob", 0):
        img = _motion_blur(img, mb)

    gb = aug_cfg.get("gaussian_blur", {})
    if gb.get("enabled") and random.random() < gb.get("prob", 0):
        img = _gaussian_blur(img, gb)

    gn = aug_cfg.get("gaussian_noise", {})
    if gn.get("enabled") and random.random() < gn.get("prob", 0):
        img = _gaussian_noise(img, gn)

    jp = aug_cfg.get("jpeg_compression", {})
    if jp.get("enabled") and random.random() < jp.get("prob", 0):
        img = _jpeg_compression(img, jp)

    ds = aug_cfg.get("downscale", {})
    if ds.get("enabled") and random.random() < ds.get("prob", 0):
        img = _downscale(img, ds)

    # --- Occlusion simulation ---
    cd = aug_cfg.get("coarse_dropout", {})
    if cd.get("enabled") and random.random() < cd.get("prob", 0):
        img = _coarse_dropout(img, cd)

    # --- Color augmentations ---
    cj = aug_cfg.get("color_jitter", {})
    if cj.get("enabled") and random.random() < cj.get("prob", 0):
        img = _color_jitter(img, cj)

    # HSV augmentation (YOLOv5 style)
    img = _hsv_augment(
        img,
        aug_cfg.get("hsv_h", 0),
        aug_cfg.get("hsv_s", 0),
        aug_cfg.get("hsv_v", 0),
    )

    # --- Spatial augmentations ---
    # Horizontal flip
    if random.random() < aug_cfg.get("fliplr", 0):
        img = np.fliplr(img).copy()
        if labels is not None and len(labels) > 0:
            labels[:, 1] = 1.0 - labels[:, 1]  # mirror cx

    # Vertical flip (important for aerial — no canonical "up" direction)
    if random.random() < aug_cfg.get("flipud", 0):
        img = np.flipud(img).copy()
        if labels is not None and len(labels) > 0:
            labels[:, 2] = 1.0 - labels[:, 2]  # mirror cy

    # Random rotation for aerial view
    aerial = aug_cfg.get("aerial", {})
    if aerial.get("degrees", 0) > 0:
        img, labels = _random_rotate(img, labels, aerial["degrees"])

    return img, labels
