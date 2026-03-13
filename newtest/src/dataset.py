"""
dataset.py — Weapon Detection Dataset & Augmentation Pipeline
Handles both detector (OBB labels) and classifier (crop) datasets.
Augmentation is domain-tuned for cinematic / John Wick–style footage:
  - motion blur, low-light, mosaic, occlusion, colour jitter
"""

import os
import cv2
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


# ─────────────────────────────────────────────────────────────────────────────
#  Label I/O  (OBB format: class cx cy w h θ  — θ in radians)
# ─────────────────────────────────────────────────────────────────────────────

def load_obb_labels(label_path: str) -> np.ndarray:
    """
    Load YOLO-OBB labels from a .txt file.
    Each line: <class_id> <cx> <cy> <w> <h> <theta>
    All values normalised to [0,1] except theta (radians ∈ [-π/2, π/2]).
    Returns ndarray shape (N, 6) or empty (0, 6).
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 6), dtype=np.float32)
    labels = []
    with open(label_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 6:
                labels.append([float(v) for v in vals])
    return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 6), dtype=np.float32)


def obb_to_corners(cx, cy, w, h, theta, img_w, img_h):
    """
    Convert OBB (cx, cy, w, h, θ) in normalised coords to 4 corner points
    in pixel coords.  θ is rotation angle in radians.
    Returns ndarray shape (4, 2) — corners in pixel space.
    """
    cx_px, cy_px = cx * img_w, cy * img_h
    w_px,  h_px  = w  * img_w, h  * img_h
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Half extents
    hw, hh = w_px / 2, h_px / 2
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float32)
    # Rotation matrix
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    corners = corners @ R.T + np.array([cx_px, cy_px])
    return corners  # (4, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_motion_blur_transform(kernel_range: Tuple[int, int] = (5, 25),
                                 prob: float = 0.4):
    """Simulate weapon motion during fast combat sequences."""
    return A.OneOf([
        A.MotionBlur(blur_limit=kernel_range, p=1.0),
        A.Blur(blur_limit=(3, 9), p=1.0),
    ], p=prob)


def build_low_light_transform(prob: float = 0.3,
                               gamma_range: Tuple[float, float] = (0.3, 0.7)):
    """
    Simulate John Wick club / tunnel / night scenes.
    Applies random gamma + Gaussian noise to mimic underexposed footage.
    """
    return A.OneOf([
        A.RandomGamma(gamma_limit=(int(gamma_range[0]*100), int(gamma_range[1]*100)), p=1.0),
        A.Compose([
            A.RandomBrightness(limit=(-0.5, -0.1), p=1.0),
            A.GaussNoise(var_limit=(30, 100), p=0.5),
        ]),
    ], p=prob)


def build_train_transform(cfg: dict, img_size: int = 640):
    """
    Full training augmentation for the detector dataset.
    All transforms are OBB-safe (we handle label transform separately
    for rotation; albumentations handles flips/crops).
    """
    aug = cfg["augmentation"]
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=aug["random_flip_lr"]),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=aug["scale"] - 1.0,
            rotate_limit=aug["random_rotate_deg"],
            border_mode=cv2.BORDER_CONSTANT,
            value=114,
            p=0.5,
        ),
        A.Perspective(scale=(0.0, aug["perspective"]), p=0.1),

        # Photometric — heavy for cinematic domain
        A.ColorJitter(
            brightness=aug["random_brightness"],
            contrast=aug["random_contrast"],
            saturation=0.3,
            hue=aug["hsv_h"],
            p=0.8,
        ),
        build_motion_blur_transform(
            kernel_range=tuple(aug["motion_blur_kernel"]),
            prob=aug["motion_blur_prob"],
        ),
        build_low_light_transform(
            prob=aug["low_light_prob"],
            gamma_range=tuple(aug["low_light_gamma"]),
        ),

        # Occlusion — simulate hands blocking weapon during combat grips
        A.CoarseDropout(
            max_holes=aug["random_erase_count"][1],
            min_holes=aug["random_erase_count"][0],
            max_height=int(img_size * 0.15),
            max_width=int(img_size * 0.15),
            fill_value=0,
            p=aug["random_erase_prob"],
        ),

        # Normalise to ImageNet stats + convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_val_transform(img_size: int = 640):
    """Validation / test transform — deterministic."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_classifier_transform(img_size: int = 224, train: bool = True):
    """Augmentation for the EfficientNet-B5 crop classifier."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size,
                                scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.02, p=0.7),
            A.MotionBlur(blur_limit=(3, 15), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  Mosaic augmentation (4-image mosaic — YOLOv8 style)
# ─────────────────────────────────────────────────────────────────────────────

class MosaicMixer:
    """
    Stitch 4 images into a 2×2 mosaic at a random cut point.
    Labels (OBB) are translated to the correct mosaic quadrant.
    Handles edge case where an image has no labels.
    """

    def __init__(self, dataset, img_size: int = 640):
        self.dataset   = dataset
        self.img_size  = img_size
        self.s         = img_size

    def __call__(self, main_idx: int):
        # Pick 3 additional random images
        indices = [main_idx] + random.sample(
            range(len(self.dataset)), 3
        )
        s = self.s
        # Random cut point
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)
        mosaic_img   = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, idx in enumerate(indices):
            img, labels = self.dataset.load_raw(idx)  # (H,W,3), (N,6)
            h, w = img.shape[:2]

            # Placement offset for each quadrant
            if   i == 0: x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
            elif i == 1: x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
            elif i == 2: x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(cy + h, s * 2)
            else:        x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(cy + h, s * 2)

            x1b = 0
            y1b = 0
            x2b = x2a - x1a
            y2b = y2a - y1a

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Shift OBB labels to mosaic coords (normalised)
            if labels.shape[0]:
                shifted = labels.copy()
                # cx, cy shift (still in normalised space of 2s×2s canvas)
                shifted[:, 1] = (labels[:, 1] * w + x1a - x1b) / (2 * s)
                shifted[:, 2] = (labels[:, 2] * h + y1a - y1b) / (2 * s)
                shifted[:, 3] = labels[:, 3] * w  / (2 * s)
                shifted[:, 4] = labels[:, 4] * h  / (2 * s)
                # θ is unchanged by translation
                mosaic_labels.append(shifted)

        mosaic_img = mosaic_img[cy - s//2: cy + s//2,
                                cx - s//2: cx + s//2]
        mosaic_img = cv2.resize(mosaic_img, (s, s))

        labels_out = (np.concatenate(mosaic_labels, axis=0)
                      if mosaic_labels else np.zeros((0, 6), dtype=np.float32))

        # Clip cx,cy to [0,1]
        labels_out[:, 1:3] = labels_out[:, 1:3].clip(0, 1)
        return mosaic_img, labels_out


# ─────────────────────────────────────────────────────────────────────────────
#  Detector Dataset (OBB)
# ─────────────────────────────────────────────────────────────────────────────

class WeaponDetectorDataset(Dataset):
    """
    Dataset for Stage 1 — YOLOv8x-OBB detector training.

    Directory layout expected:
        data/
          images/train/*.jpg
          images/val/*.jpg
          labels/train/*.txt   (YOLO-OBB format)
          labels/val/*.txt
    """

    def __init__(
        self,
        cfg:      dict,
        split:    str  = "train",
        img_size: int  = 640,
    ):
        self.cfg      = cfg
        self.split    = split
        self.img_size = img_size
        self.train    = split == "train"

        root  = Path(cfg["dataset"]["root"])
        img_dir   = root / "images" / split
        label_dir = root / "labels" / split

        self.img_paths   = sorted(img_dir.glob("*.jpg")) + \
                           sorted(img_dir.glob("*.png"))
        self.label_paths = [
            label_dir / (p.stem + ".txt") for p in self.img_paths
        ]

        # Mosaic mixer (training only)
        self.mosaic     = MosaicMixer(self, img_size) if self.train else None
        self.mosaic_prob = cfg["augmentation"]["mosaic"]

        # Transforms
        self.transform   = (build_train_transform(cfg, img_size)
                            if self.train else build_val_transform(img_size))

        print(f"[Dataset] {split} — {len(self.img_paths)} images loaded.")

    # ── internal helpers ─────────────────────────────────────────────────────

    def load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load image (BGR→RGB) + OBB labels without any augmentation."""
        img    = cv2.imread(str(self.img_paths[idx]))
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w   = img.shape[:2]
        # Letterbox resize
        img, ratio, (dw, dh) = letterbox(img, self.img_size)
        labels = load_obb_labels(str(self.label_paths[idx]))
        # Adjust labels for letterbox
        if labels.shape[0]:
            labels = adjust_obb_for_letterbox(labels, ratio, dw, dh, w, h)
        return img, labels

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # Apply mosaic with probability
        if self.train and random.random() < self.mosaic_prob:
            img, labels = self.mosaic(idx)
        else:
            img, labels = self.load_raw(idx)

        # Albumentations (pixel-level transforms only — labels handled above)
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]          # (3, H, W) float32 tensor

        # Convert labels to torch tensor
        labels_tensor = torch.from_numpy(labels)  # (N, 6)

        return img_tensor, labels_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier Dataset  (crops from detector output)
# ─────────────────────────────────────────────────────────────────────────────

class WeaponCropDataset(Dataset):
    """
    Dataset for Stage 2 — EfficientNet-B5 classifier.
    Expects crops already saved to  data/crops/<class_name>/<img>.jpg
    (generated by a preprocessing script that runs the detector on raw data).
    """

    CLASSES = ["pistol", "revolver", "rifle", "shotgun",
               "smg", "knife", "blunt_weapon"]

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 224, padding: float = 0.15):
        self.root     = Path(root) / "crops" / split
        self.img_size = img_size
        self.padding  = padding
        self.train    = split == "train"
        self.transform = build_classifier_transform(img_size, train=self.train)

        self.samples: List[Tuple[Path, int]] = []
        for cls_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for p in cls_dir.glob("*.jpg"):
                    self.samples.append((p, cls_idx))
                for p in cls_dir.glob("*.png"):
                    self.samples.append((p, cls_idx))

        # Class distribution for weighted sampling
        counts = np.bincount([s[1] for s in self.samples],
                             minlength=len(self.CLASSES))
        weights = 1.0 / (counts + 1e-6)
        self.sample_weights = torch.tensor(
            [weights[s[1]] for s in self.samples], dtype=torch.float
        )
        print(f"[CropDataset] {split} — {len(self.samples)} crops | "
              f"class counts: {dict(zip(self.CLASSES, counts.tolist()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return aug["image"], torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Dataset  (sliding-window sequences for BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalDetectionDataset(Dataset):
    """
    Wraps per-frame detection features into sliding windows of length T.
    Each sample: feature_window (T, input_size), label_window (T,)

    feature vector per frame (14-dim):
      [conf, cx, cy, w, h, theta, p0..p6, motion_magnitude]
      where p0..p6 are classifier softmax scores
    """

    def __init__(self, sequences: List[Dict], window_size: int = 8):
        """
        sequences: list of dicts, each with keys
            'features': ndarray (F, 14)
            'labels':   ndarray (F,) — 0=no weapon, 1..7=weapon class
        """
        self.window   = window_size
        self.samples  = []
        for seq in sequences:
            feats  = seq["features"]
            labels = seq["labels"]
            F = len(feats)
            for start in range(0, F - window_size + 1):
                self.samples.append((
                    feats[start: start + window_size],
                    labels[start: start + window_size],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, labels = self.samples[idx]
        return (
            torch.tensor(feats,  dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: letterbox resize
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int = 640,
              fill: int = 114) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """
    Resize keeping aspect ratio, then pad to target×target.
    Returns (padded_img, scale_ratio, (dw, dh)) where dw/dh are padding offsets.
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (target - new_w) / 2
    dh = (target - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img, scale, (dw, dh)


def adjust_obb_for_letterbox(labels: np.ndarray,
                              ratio: float,
                              dw: float, dh: float,
                              orig_w: int, orig_h: int) -> np.ndarray:
    """
    Recompute normalised OBB coordinates after letterbox resize.
    Input  labels  cols: [class, cx_norm, cy_norm, w_norm, h_norm, theta]
    Output labels cols: same format, adjusted for letterboxed image.
    """
    out = labels.copy()
    target_size = orig_w * ratio + 2 * dw  # = orig_h * ratio + 2*dh = img_size
    # cx, cy in pixels in original image → adjust for letterbox
    out[:, 1] = (labels[:, 1] * orig_w * ratio + dw) / target_size
    out[:, 2] = (labels[:, 2] * orig_h * ratio + dh) / target_size
    out[:, 3] = labels[:, 3] * orig_w * ratio / target_size
    out[:, 4] = labels[:, 4] * orig_h * ratio / target_size
    # theta unchanged by scaling (no shear)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_detector_loaders(cfg: dict):
    """Return (train_loader, val_loader, test_loader) for the OBB detector."""
    det_cfg  = cfg["detector"]
    dat_cfg  = cfg["dataset"]

    train_ds = WeaponDetectorDataset(cfg, split="train", img_size=det_cfg["img_size"])
    val_ds   = WeaponDetectorDataset(cfg, split="val",   img_size=det_cfg["img_size"])
    test_ds  = WeaponDetectorDataset(cfg, split="test",  img_size=det_cfg["img_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size  = det_cfg["batch_size"],
        shuffle     = True,
        num_workers = dat_cfg["num_workers"],
        pin_memory  = dat_cfg["pin_memory"],
        collate_fn  = obb_collate_fn,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = det_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = dat_cfg["num_workers"],
        pin_memory  = dat_cfg["pin_memory"],
        collate_fn  = obb_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = det_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = dat_cfg["num_workers"],
        collate_fn  = obb_collate_fn,
    )
    return train_loader, val_loader, test_loader


def build_classifier_loaders(cfg: dict):
    """Return (train_loader, val_loader) for the crop classifier."""
    cls_cfg = cfg["classifier"]
    dat_cfg = cfg["dataset"]
    root    = cfg["dataset"]["root"]

    train_ds = WeaponCropDataset(root, "train", cls_cfg.get("img_size", 224))
    val_ds   = WeaponCropDataset(root, "val",   cls_cfg.get("img_size", 224))

    # Weighted sampler to handle class imbalance
    sampler = torch.utils.data.WeightedRandomSampler(
        weights     = train_ds.sample_weights,
        num_samples = len(train_ds),
        replacement = True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cls_cfg["batch_size"],
        sampler     = sampler,
        num_workers = dat_cfg["num_workers"],
        pin_memory  = dat_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cls_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = dat_cfg["num_workers"],
    )
    return train_loader, val_loader


def obb_collate_fn(batch):
    """
    Custom collate for OBB detection batches.
    Images → stacked tensor.  Labels → list of tensors (variable N per image).
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)
