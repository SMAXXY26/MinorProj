"""
part2.py  —  Generate crops + Train the Classifier
===================================================
Step A: Runs logs/detector/best.pt (YOLOv8n) over training images
        and saves cropped weapon boxes to:
        data/classifier_crops/knife/
        data/classifier_crops/pistol/
        data/classifier_crops/rifle/

Step B: Trains WeaponClassifier (EfficientNet-B5) on those crops
        Output: logs/classifier/best.pt

Usage:
    python part2.py
    python part2.py --skip-crops   # if crops already exist
"""
from __future__ import annotations

import os
import sys
import cv2
import random
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model   import WeaponClassifier
from src.dataset import build_classifier_loaders
from src.losses  import FocalLoss

# Ultralytics for crop generation
from ultralytics import YOLO

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# =============================================================================
#  Helpers
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path="config/hyperparams.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [Saved] {path}")


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema   = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# =============================================================================
#  Step A — Generate classifier crops using YOLOv8n
# =============================================================================

def generate_crops(cfg, det_ckpt="logs/detector/best.pt"):
    print(f"\n{'═'*60}")
    print(f"  Step A — Generating classifier crops (YOLOv8n)")
    print(f"{'═'*60}\n")

    class_names = cfg["dataset"]["class_names"]
    conf_thresh = 0.30
    img_exts    = {".jpg", ".jpeg", ".png", ".bmp"}

    # Output dirs
    crop_root = cfg["dataset"].get(
        "classifier_root",
        os.path.join(cfg["dataset"]["root"], "classifier_crops"),
    )
    for name in class_names:
        os.makedirs(os.path.join(crop_root, name), exist_ok=True)

    # Load YOLOv8n with trained weights
    model = YOLO(det_ckpt)
    print(f"  Detector loaded : {det_ckpt}")
    print(f"  Conf threshold  : {conf_thresh}")

    # Collect all images
    data_yaml = os.path.join(cfg["dataset"]["root"], "data.yaml")
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    root      = cfg["dataset"]["root"]
    train_img = os.path.normpath(os.path.join(root, data_cfg["train"]))
    val_img   = os.path.normpath(os.path.join(root, data_cfg["val"]))

    all_images = []
    for folder in [train_img, val_img]:
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if Path(f).suffix.lower() in img_exts:
                    all_images.append(os.path.join(folder, f))

    print(f"  Images to scan  : {len(all_images)}\n")

    crop_counts = {name: 0 for name in class_names}
    no_det      = 0

    for i, img_path in enumerate(all_images):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        # Run YOLOv8 inference
        results = model(
            img_path,
            conf    = conf_thresh,
            verbose = False,
            device  = 0,
        )

        dets = results[0].boxes
        if dets is None or len(dets) == 0:
            no_det += 1
            continue

        stem = Path(img_path).stem

        for j, box in enumerate(dets):
            # xyxy coords in original image space
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cls_idx  = int(box.cls[0].item())
            cls_idx  = min(cls_idx, len(class_names) - 1)
            cls_name = class_names[cls_idx]

            crop     = img_bgr[y1:y2, x1:x2]
            out_path = os.path.join(
                crop_root, cls_name, f"{stem}_det{j:03d}.jpg"
            )
            cv2.imwrite(out_path, crop)
            crop_counts[cls_name] += 1

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>5}/{len(all_images)}]  " +
                  "  ".join(f"{k}:{v}" for k, v in crop_counts.items()))

    print(f"\n  Crop generation complete:")
    for name, count in crop_counts.items():
        print(f"    {name:10s} → {count} crops")
    print(f"    No detection   → {no_det} images")

    total = sum(crop_counts.values())
    if total == 0:
        print("\n  [WARNING] No crops generated — falling back to GT labels")
        _fallback_crops_from_labels(cfg, crop_root, class_names)

    return crop_root


def _fallback_crops_from_labels(cfg, crop_root, class_names):
    """Crop directly from ground truth labels if detector finds nothing."""
    print("\n  [Fallback] Cropping from ground truth labels ...")
    data_yaml = os.path.join(cfg["dataset"]["root"], "data.yaml")
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    root      = cfg["dataset"]["root"]
    train_img = os.path.normpath(os.path.join(root, data_cfg["train"]))
    train_lbl = train_img.replace("images", "labels")
    counts    = {name: 0 for name in class_names}

    for fname in sorted(os.listdir(train_img)):
        if Path(fname).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img_path = os.path.join(train_img, fname)
        lbl_path = os.path.join(train_lbl, Path(fname).stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        box_idx = 0
        with open(lbl_path) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) < 5:
                    continue
                cls_idx      = int(vals[0])
                cx, cy, bw, bh = map(float, vals[1:5])
                x1 = int((cx - bw/2) * W)
                y1 = int((cy - bh/2) * H)
                x2 = int((cx + bw/2) * W)
                y2 = int((cy + bh/2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_name = class_names[min(cls_idx, len(class_names) - 1)]
                crop     = img[y1:y2, x1:x2]
                out_path = os.path.join(
                    crop_root, cls_name,
                    f"{Path(fname).stem}_gt{cls_idx}_{box_idx:03d}.jpg"
                )
                cv2.imwrite(out_path, crop)
                counts[cls_name] += 1
                box_idx += 1

    print("  Fallback crops:")
    for name, count in counts.items():
        print(f"    {name:10s} → {count} crops")


# =============================================================================
#  Step B — Train classifier
# =============================================================================

@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion):
    model.eval()
    correct = total = val_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits    = model(imgs)
        loss      = criterion(logits, labels)
        correct  += (logits.argmax(dim=-1) == labels).sum().item()
        total    += len(labels)
        val_loss += float(loss)
    return correct / max(total, 1), val_loss / max(len(loader), 1)


def train_classifier(cfg, device):
    cls_cfg  = cfg["classifier"]
    backbone = cls_cfg.get("backbone", "efficientnet_b5")

    print(f"\n{'═'*60}")
    print(f"  Step B — Classifier Training ({backbone})")
    print(f"{'═'*60}\n")

    save_dir = Path(cfg["logging"]["save_dir"]) / "classifier"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = WeaponClassifier(
        num_classes = cfg["dataset"]["num_classes"],
        dropout     = cls_cfg["dropout_rate"],
        pretrained  = cls_cfg["pretrained"],
        backbone    = backbone,
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model] Total: {n_total/1e6:.1f}M   Trainable: {n_trainable/1e6:.1f}M")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_classifier_loaders(cfg)
    print(f"  [Data]  Train: {len(train_loader)} batches   Val: {len(val_loader)} batches")

    # ── Class weights to handle imbalance (knife 2144 vs pistol 12339) ───
    class_counts = []
    crop_root    = cfg["dataset"].get("classifier_root", "data/classifier_crops")
    for name in cfg["dataset"]["class_names"]:
        cls_dir = os.path.join(crop_root, name)
        count   = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 1
        class_counts.append(count)
        print(f"  [Class] {name:10s} → {count} crops")

    total_crops   = sum(class_counts)
    class_weights = torch.tensor(
        [total_crops / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32,
    ).to(device)
    print(f"  [Weights] {[round(w.item(),2) for w in class_weights]}\n")

    if len(train_loader) == 0:
        print("  [ERROR] No classifier training data found.")
        print(f"  Check : {cfg['dataset'].get('classifier_root', 'data/classifier_crops')}")
        return None

    # ── Optimiser — two param groups ─────────────────────────────────────────
    model.freeze_backbone()
    optimizer = torch.optim.AdamW([
        {"params": list(model.head.parameters()) +
                   list(model.pool.parameters()),     "lr": cls_cfg["lr"]},
        {"params": list(model.backbone.parameters()), "lr": cls_cfg["lr"] * 0.1},
    ], weight_decay=cls_cfg["weight_decay"])

    scheduler  = CosineAnnealingWarmRestarts(
        optimizer, T_0=cls_cfg["T_0"], T_mult=cls_cfg["T_mult"],
        eta_min=cls_cfg["eta_min"],
    )
    criterion  = FocalLoss(
        gamma=cls_cfg["focal_gamma"], alpha=cls_cfg["focal_alpha"],
        label_smoothing=cls_cfg["label_smoothing"],
        num_classes=cfg["dataset"]["num_classes"],
        class_weights=class_weights,
    )
    scaler        = GradScaler("cuda", enabled=cfg["hardware"]["amp"])
    ema           = ModelEMA(model, cfg["hardware"]["ema_decay"])
    best_acc      = 0.0
    no_improve    = 0
    patience      = cfg["logging"]["patience"]
    epochs        = cls_cfg["epochs"]
    unfreeze_done = False

    print(f"  Epochs  : {epochs}   Patience : {patience}")
    print(f"  Phase 1 : backbone frozen for first 5 epochs\n")

    for epoch in range(epochs):

        # Unfreeze at epoch 5
        if epoch == 5 and not unfreeze_done:
            model.unfreeze_backbone()
            unfreeze_done = True
            print("  [Phase 2] Backbone unfrozen — full fine-tune\n")

        model.train()
        running_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if random.random() < cls_cfg["mixup_prob"] and len(imgs) > 1:
                imgs, la, lb, lam = mixup_data(imgs, labels, cls_cfg["mixup_alpha"])
                with autocast("cuda", enabled=cfg["hardware"]["amp"]):
                    logits = model(imgs)
                    loss   = lam * criterion(logits, la) + \
                             (1 - lam) * criterion(logits, lb)
            else:
                with autocast("cuda", enabled=cfg["hardware"]["amp"]):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()           # ← scaler first (Task 5d)
            optimizer.zero_grad()
            ema.update(model)         # ← EMA after scaler finalises

            running_loss += float(loss)
            correct      += (logits.argmax(dim=-1) == labels).sum().item()
            total        += len(labels)

        scheduler.step()
        val_acc, val_loss = evaluate_classifier(ema.ema, val_loader, device, criterion)
        train_acc         = correct / max(total, 1)

        print(
            f"  Epoch [{epoch+1:>3}/{epochs}]  "
            f"loss={running_loss/len(train_loader):.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        state = {
            "epoch":       epoch + 1,
            "model":       ema.ema.state_dict(),
            "acc":         val_acc,
            "num_classes": cfg["dataset"]["num_classes"],
            "backbone":    backbone,
        }
        save_checkpoint(state, str(save_dir / "last.pt"))

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            save_checkpoint(state, str(save_dir / "best.pt"))
            print(f"  ✓ New best val_acc: {best_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [Early Stop] No improvement for {patience} epochs.")
                break

    print(f"\n{'═'*60}")
    print(f"  Classifier training complete.")
    print(f"  Best val_acc : {best_acc:.4f}")
    print(f"  Weights      : {save_dir}/best.pt")
    print(f"{'═'*60}")
    print(f"\n  Next step → python part3.py")
    return str(save_dir / "best.pt")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 — Crops + Classifier")
    parser.add_argument("--config",     type=str, default="config/hyperparams.yaml")
    parser.add_argument("--det-ckpt",   type=str, default="logs/detector/best.pt")
    parser.add_argument("--skip-crops", action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)

    print(f"\n{'═'*60}")
    print(f"  Part 2 — Generate Crops + Train Classifier")
    print(f"{'═'*60}")

    if not args.skip_crops:
        crop_root = generate_crops(cfg, args.det_ckpt)
        cfg["dataset"]["classifier_root"] = crop_root
    else:
        print("\n  [Skipping crop generation]")

    train_classifier(cfg, device)