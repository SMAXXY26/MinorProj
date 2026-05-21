"""
small_object_aug.py — Loss-guided online small-object augmentation.

Two complementary techniques:

1. SmallObjectBank
   Pre-indexes training images that contain small weapon instances
   (sqrt(w·h) < small_thresh in normalised bbox coords). Used as the
   injection pool during training.

2. LossGuidedInjector
   Tracks a per-batch EMA of the small-object GT fraction. When the EMA
   drops below target_frac (i.e. the network is seeing too few small
   objects per batch), it replaces an increasing fraction of batch slots
   with small-object-rich images from the bank.  Injection rate decays
   when the EMA recovers, so the intervention is truly adaptive.

3. small_object_copy_paste (offline helper)
   Pastes small weapon crops onto background images to generate new
   training samples that can be added to the dataset before training.

Integration (distill.py WeaponKDTrainer):
    See attach_small_object_aug(trainer, cfg, dataset_yaml) below.

Config keys (student: section of hyperparams.yaml):
    small_object_aug:          true
    small_object_thresh:       0.04   # sqrt(w·h) below which a box is "small"
    small_object_target_frac:  0.15   # min desired fraction of small boxes/batch
    small_object_max_rate:     0.40   # max fraction of batch slots to replace
    small_object_ema_window:   100    # EMA smoothing window (batches)
"""

import random
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize + pad to square while preserving aspect ratio (grey pad 114)."""
    h0, w0 = img.shape[:2]
    r = size / max(h0, w0)
    if r != 1.0:
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    h, w = img.shape[:2]
    top    = (size - h) // 2
    bottom = size - h - top
    left   = (size - w) // 2
    right  = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR HWC → float32 RGB CHW / 255."""
    return torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0


# ══════════════════════════════════════════════════════════════════════════════
#  SmallObjectBank
# ══════════════════════════════════════════════════════════════════════════════

class SmallObjectBank:
    """
    Pre-indexed pool of training images containing small objects.

    Args:
        img_dir:       Path to images/train
        lbl_dir:       Path to labels/train
        small_thresh:  A box is "small" when sqrt(norm_w * norm_h) < thresh.
                       0.04 ≈ 32/800 (the MS-COCO small-object cutoff scaled
                       to a ~800 px normalised space).
        max_cache:     Number of pre-decoded images to keep in RAM.
                       Set to 0 to disable RAM cache.
    """

    def __init__(
        self,
        img_dir: str,
        lbl_dir: str,
        small_thresh: float = 0.04,
        max_cache: int = 256,
    ):
        self.img_dir     = Path(img_dir)
        self.lbl_dir     = Path(lbl_dir)
        self.thresh      = small_thresh
        self.max_cache   = max_cache

        # entries: list of (img_path, list[(cls, cx, cy, w, h)], n_small)
        self.entries: list = []
        self._cache: dict  = {}   # img_path → np.ndarray BGR

        self._build_index()

    # ── index ────────────────────────────────────────────────────────────────

    def _build_index(self):
        thresh = self.thresh
        for img_path in self.img_dir.iterdir():
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            boxes = []
            n_small = 0
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                boxes.append((cls, cx, cy, w, h))
                if (w * h) ** 0.5 < thresh:
                    n_small += 1
            if n_small > 0:
                self.entries.append((img_path, boxes, n_small))

        # Highest small-object count first so random.choices naturally
        # prefers the richest images.
        self.entries.sort(key=lambda e: e[2], reverse=True)
        print(f"  [SmallObjectBank] {len(self.entries)} images with small "
              f"objects indexed  (thresh √(wh) < {self.thresh:.3f})")

    def __len__(self) -> int:
        return len(self.entries)

    # ── sampling ─────────────────────────────────────────────────────────────

    def _load_img(self, img_path: Path) -> np.ndarray | None:
        if img_path in self._cache:
            return self._cache[img_path]
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        if self.max_cache and len(self._cache) < self.max_cache:
            self._cache[img_path] = img
        return img

    def sample(self, n: int, img_size: int = 640) -> list:
        """
        Return a list of n dicts, each with:
            "img"    : torch.Tensor [3, img_size, img_size] float32/255
            "boxes"  : list of (cls, cx, cy, w, h) — original normalised coords
        Failed loads are silently skipped (list may be shorter than n).
        """
        if not self.entries:
            return []
        chosen = random.choices(self.entries, k=n)
        results = []
        for img_path, boxes, _ in chosen:
            raw = self._load_img(img_path)
            if raw is None:
                continue
            lb  = _letterbox(raw, img_size)
            results.append({"img": _to_tensor(lb), "boxes": boxes})
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  LossGuidedInjector
# ══════════════════════════════════════════════════════════════════════════════

class LossGuidedInjector:
    """
    Monitors the EMA of the small-object GT fraction in each training batch.
    When the EMA falls below *target_frac*, replaces an increasing share of
    batch image slots with small-object-rich images from *bank*.

    Call  injector.process_batch(batch, img_size, device)  from inside the
    trainer's preprocess_batch override — **after** the parent normalisation
    so that `batch["img"]` is already a float32 [B,C,H,W] tensor on `device`.
    """

    def __init__(
        self,
        bank:         SmallObjectBank,
        target_frac:  float = 0.15,
        small_thresh: float = 0.04,
        ema_window:   int   = 100,
        step_up:      float = 0.05,
        step_down:    float = 0.025,
        max_rate:     float = 0.40,
    ):
        self.bank         = bank
        self.target_frac  = target_frac
        self.thresh       = small_thresh
        self.step_up      = step_up
        self.step_down    = step_down
        self.max_rate     = max_rate

        alpha       = 2.0 / (ema_window + 1)
        self._alpha = alpha
        self._ema   = target_frac   # initialise at target (no cold-start shock)
        self.injection_rate: float = 0.0

        # Public telemetry (read by trainer callbacks if desired)
        self.last_batch_frac:  float = 0.0
        self.last_n_replaced:  int   = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _small_frac(self, batch: dict) -> float:
        bboxes = batch.get("bboxes")
        if bboxes is None or bboxes.numel() == 0:
            return 0.0
        w = bboxes[:, 2]
        h = bboxes[:, 3]
        return float(((w * h).clamp(min=0.0) ** 0.5 < self.thresh).float().mean())

    # ── main API ──────────────────────────────────────────────────────────────

    def process_batch(self, batch: dict, img_size: int, device) -> dict:
        """
        Potentially replace some batch slots with small-object-rich images.
        Must be called after parent preprocess_batch normalises images.
        Mutates and returns batch in place.
        """
        frac = self._small_frac(batch)
        self.last_batch_frac = frac

        # EMA update
        self._ema = self._alpha * frac + (1 - self._alpha) * self._ema

        # Adjust injection rate
        if self._ema < self.target_frac:
            self.injection_rate = min(self.max_rate,
                                      self.injection_rate + self.step_up)
        else:
            self.injection_rate = max(0.0,
                                      self.injection_rate - self.step_down)

        if self.injection_rate < 1e-3 or len(self.bank) == 0:
            self.last_n_replaced = 0
            return batch

        B = batch["img"].shape[0]
        n_replace = max(1, int(B * self.injection_rate))
        slots     = random.sample(range(B), min(n_replace, B))

        replacements = self.bank.sample(len(slots), img_size=img_size)
        if not replacements:
            self.last_n_replaced = 0
            return batch

        bboxes    = batch["bboxes"]    # [N, 4]
        cls_t     = batch["cls"]       # [N, 1]
        batch_idx = batch["batch_idx"] # [N]

        new_bb, new_cls, new_bidx = [], [], []

        # Keep boxes for non-replaced slots
        for bi in range(B):
            if bi in slots:
                continue
            mask = batch_idx == bi
            if mask.any():
                new_bb.append(bboxes[mask])
                new_cls.append(cls_t[mask])
                new_bidx.append(torch.full((int(mask.sum()),), bi,
                                           dtype=torch.long, device=device))

        # Inject replacement images
        actual_replaced = 0
        for slot, rep in zip(slots, replacements):
            batch["img"][slot] = rep["img"].to(device)
            boxes = rep["boxes"]
            if boxes:
                bb  = torch.tensor([[cx, cy, w, h]
                                     for (_, cx, cy, w, h) in boxes],
                                   dtype=torch.float32, device=device)
                cl  = torch.tensor([[c] for (c, *_) in boxes],
                                   dtype=torch.float32, device=device)
                bi  = torch.full((len(boxes),), slot,
                                 dtype=torch.long, device=device)
                new_bb.append(bb)
                new_cls.append(cl)
                new_bidx.append(bi)
            actual_replaced += 1

        if new_bb:
            batch["bboxes"]    = torch.cat(new_bb,   dim=0)
            batch["cls"]       = torch.cat(new_cls,  dim=0)
            batch["batch_idx"] = torch.cat(new_bidx, dim=0)

        self.last_n_replaced = actual_replaced
        return batch

    # ── logging ───────────────────────────────────────────────────────────────

    def status_line(self) -> str:
        return (f"small_obj  ema={self._ema:.3f}  "
                f"inject={self.injection_rate:.2f}  "
                f"replaced={self.last_n_replaced}")


# ══════════════════════════════════════════════════════════════════════════════
#  Offline copy-paste helper
# ══════════════════════════════════════════════════════════════════════════════

def small_object_copy_paste(
    bg_img:      np.ndarray,
    crop_dir:    str | Path,
    n_pastes:    int   = 3,
    scale_range: tuple = (0.02, 0.06),
    min_iou:     float = 0.0,
) -> tuple[np.ndarray, list[tuple]]:
    """
    Paste *n_pastes* small weapon crops onto *bg_img*.

    Crops are randomly sampled from *crop_dir* (flat directory of JPG/PNG
    files, e.g. data/classifier_crops/).  Scale is chosen uniformly from
    *scale_range* as a fraction of the shorter bg_img dimension, ensuring
    the pasted object appears "small" in the scene.

    Returns:
        img_out  : augmented image (uint8 BGR, same shape as bg_img)
        new_boxes: list of (cls_id, cx, cy, w_norm, h_norm) in YOLO format
                   cls_id is inferred from the crop's parent-folder name
                   (knife→0, pistol→1, rifle→2); defaults to 0 if unknown.
    """
    crop_dir  = Path(crop_dir)
    CLASS_MAP = {"knife": 0, "pistol": 1, "rifle": 2}

    all_crops = []
    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = crop_dir / cls_name
        if cls_dir.is_dir():
            for p in cls_dir.iterdir():
                if p.suffix.lower() in _IMG_EXTS:
                    all_crops.append((p, cls_id))

    if not all_crops:
        return bg_img.copy(), []

    H, W = bg_img.shape[:2]
    short = min(H, W)
    img_out   = bg_img.copy()
    new_boxes = []

    for _ in range(n_pastes):
        crop_path, cls_id = random.choice(all_crops)
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        scale   = random.uniform(*scale_range)
        tgt_len = max(4, int(short * scale))
        ch, cw  = crop.shape[:2]
        r       = tgt_len / max(ch, cw)
        crop    = cv2.resize(crop, (max(1, int(cw * r)), max(1, int(ch * r))),
                             interpolation=cv2.INTER_AREA)
        ph, pw  = crop.shape[:2]

        if ph >= H or pw >= W:
            continue

        x1 = random.randint(0, W - pw)
        y1 = random.randint(0, H - ph)

        # Simple paste (alpha=1; blending not needed for small objects)
        img_out[y1:y1 + ph, x1:x1 + pw] = crop

        cx    = (x1 + pw / 2) / W
        cy    = (y1 + ph / 2) / H
        w_n   = pw / W
        h_n   = ph / H
        new_boxes.append((cls_id, cx, cy, w_n, h_n))

    return img_out, new_boxes


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer attachment helper
# ══════════════════════════════════════════════════════════════════════════════

def build_injector_from_cfg(cfg: dict, dataset_yaml: str) -> "LossGuidedInjector | None":
    """
    Build a LossGuidedInjector from the project config dict.
    Returns None if small_object_aug is disabled or bank has no entries.

    Args:
        cfg:          Full config dict (hyperparams.yaml).
        dataset_yaml: Path to dataset.yaml (to locate train img/lbl dirs).
    """
    st = cfg.get("student", {})
    if not st.get("small_object_aug", False):
        return None

    # Resolve train directories from dataset yaml
    with open(dataset_yaml) as f:
        ds = yaml.safe_load(f)

    ds_root = Path(dataset_yaml).parent
    if "path" in ds:
        ds_root = Path(ds["path"])

    train_rel = ds.get("train", "images/train")
    img_dir   = (ds_root / train_rel).resolve()
    lbl_dir   = img_dir.parent.parent / "labels" / img_dir.name

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"  [SmallObjectAug] WARN: img_dir={img_dir} or "
              f"lbl_dir={lbl_dir} not found — disabling")
        return None

    thresh = st.get("small_object_thresh",      0.04)
    target = st.get("small_object_target_frac", 0.15)
    window = st.get("small_object_ema_window",  100)
    maxr   = st.get("small_object_max_rate",    0.40)

    bank = SmallObjectBank(img_dir, lbl_dir, small_thresh=thresh)
    if len(bank) == 0:
        print("  [SmallObjectAug] No small-object images found — disabling")
        return None

    return LossGuidedInjector(
        bank=bank,
        target_frac=target,
        small_thresh=thresh,
        ema_window=window,
        max_rate=maxr,
    )
