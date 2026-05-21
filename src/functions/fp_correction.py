"""
fp_correction.py — False Positive Correction Pipeline.
=======================================================
Refactored from part4.py.

Runs 4 sub-steps automatically:

  Sub-step 1: mine_hard_negatives()
    Runs detector at low threshold on weapon-free images
    Saves every FP box as a hard negative example

  Sub-step 2: find_optimal_threshold()
    Sweeps conf thresholds on val set
    Finds threshold where precision >= 0.90
    Writes result back to config automatically

  Sub-step 3: finetune_detector()
    Short fine-tune of detector on original + hard negatives
    Backbone frozen — only head updates

  Sub-step 4: finetune_classifier_with_background()
    Adds background class to classifier
    Re-trains with FP crops as background examples

Output:
    logs/fp_correction/detector_ft_best.pt
    logs/fp_correction/classifier_bg_best.pt
    logs/fp_correction/threshold_sweep.json
    config/hyperparams.yaml  (updated threshold)

Functions:
    mine_hard_negatives(cfg, fp_cfg, det_ckpt) → (str, str)
    find_optimal_threshold(cfg, fp_cfg, config_path, det_ckpt) → float
    finetune_detector(cfg, fp_cfg, neg_images_dir, det_ckpt) → str
    finetune_classifier_with_background(cfg, fp_cfg, neg_crops_dir, cls_ckpt) → str
    run_fp_correction(cfg, config_path, skip_mining) → None
"""
from __future__ import annotations

import os
import sys
import cv2
import json
import shutil
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from ultralytics import YOLO

from src.model   import WeaponClassifier
from src.losses  import FocalLoss
from src.functions.common import (
    save_config, save_checkpoint, ModelEMA, mixup_data,
)

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# =============================================================================
#  Sub-step 1 — Hard Negative Mining
# =============================================================================

def mine_hard_negatives(cfg, fp_cfg, det_ckpt="logs/detector/best.pt"):
    """
    Run detector at low threshold on weapon-free images and save
    every false positive box as a hard negative example.

    Args:
        cfg:      Full config dict.
        fp_cfg:   fp_correction section of config.
        det_ckpt: Path to detector weights.

    Returns:
        tuple: (neg_images_dir, neg_crops_dir)
    """
    print(f"\n{'═'*60}")
    print(f"  Sub-step 1 — Hard Negative Mining")
    print(f"{'═'*60}\n")

    save_root  = Path(cfg["logging"]["save_dir"]) / "fp_correction"
    neg_images = save_root / "hard_negatives" / "images"
    neg_labels = save_root / "hard_negatives" / "labels"
    neg_crops  = save_root / "hard_negatives" / "crops"
    for d in [neg_images, neg_labels, neg_crops]:
        d.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 detector
    model        = YOLO(det_ckpt)
    mine_thresh  = fp_cfg.get("mine_conf_threshold", 0.20)
    neg_img_dir  = fp_cfg.get(
        "negative_image_dir", "data/negatives/images"
    )
    img_exts     = {".jpg", ".jpeg", ".png", ".bmp"}

    img_paths = [
        os.path.join(neg_img_dir, f)
        for f in os.listdir(neg_img_dir)
        if Path(f).suffix.lower() in img_exts
    ]

    print(f"  Detector      : {det_ckpt}")
    print(f"  Neg images    : {len(img_paths)}")
    print(f"  Mine threshold: {mine_thresh}\n")

    fp_count  = 0
    clean     = 0

    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None or img_bgr.size == 0:
            continue

        # image checker
        try:
            from PIL import Image
            Image.open(img_path).verify()
        except Exception:
            print(f'  [Skip] Corrupted: {Path(img_path).name}')
            continue

        try:
            results = model(
                img_path, conf=mine_thresh, verbose=False
            )
        except Exception as e:
            print(f'  [Skip] Inference error: {Path(img_path).name} — {e}')
            continue
        dets = results[0].boxes

        fname_stem = Path(img_path).stem

        if dets is None or len(dets) == 0:
            clean += 1
            continue

        dst_img = neg_images / Path(img_path).name
        shutil.copy2(img_path, dst_img)
        (neg_labels / f"{fname_stem}.txt").write_text("")

        # Crop FP box
        H, W = img_bgr.shape[:2]
        for i, box in enumerate(dets):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop      = img_bgr[y1:y2, x1:x2]
            crop_path = neg_crops / f"{fname_stem}_fp{i:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            fp_count += 1

    print(f"  Clean images (no FP) : {clean}")
    print(f"  FP boxes collected   : {fp_count}")
    print(f"  Hard neg images      → {neg_images}")
    print(f"  FP crop images       → {neg_crops}")

    return str(neg_images), str(neg_crops)


# =============================================================================
#  Sub-step 2 — Precision-Recall Threshold Sweep
# =============================================================================

def find_optimal_threshold(cfg, fp_cfg, config_path,
                            det_ckpt="logs/detector/best.pt"):
    """
    Sweep confidence thresholds on val set and find the threshold
    where precision >= target_precision.

    Args:
        cfg:         Full config dict.
        fp_cfg:      fp_correction section of config.
        config_path: Path to config YAML (for write-back).
        det_ckpt:    Path to detector weights.

    Returns:
        float: Selected optimal threshold.
    """
    print(f"\n{'═'*60}")
    print(f"  Sub-step 2 — Threshold Sweep")
    print(f"{'═'*60}\n")

    model            = YOLO(det_ckpt)
    target_precision = fp_cfg.get("target_precision", 0.90)
    img_size         = cfg["detector"]["img_size"]

    # Load val images + labels
    data_yaml = os.path.join(cfg["dataset"]["root"], "data.yaml")
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    root    = cfg["dataset"]["root"]
    val_img = os.path.normpath(os.path.join(root, data_cfg["val"]))
    val_lbl = val_img.replace("images", "labels")

    img_exts = {".jpg", ".jpeg", ".png"}
    val_imgs = [
        os.path.join(val_img, f)
        for f in os.listdir(val_img)
        if Path(f).suffix.lower() in img_exts
    ][:500]   # limit to 500 for speed

    print(f"  Val images    : {len(val_imgs)}")
    print(f"  Target P      : {target_precision}\n")

    sweep_thresholds = np.arange(0.25, 0.92, 0.05).tolist()
    results          = []

    print(f"  {'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*45}")

    for thresh in sweep_thresholds:
        tp = fp = fn = 0

        for img_path in val_imgs:
            lbl_path = os.path.join(
                val_lbl, Path(img_path).stem + ".txt"
            )
            # Parse ground truth boxes
            gt_boxes = []
            if os.path.exists(lbl_path):
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    ih, iw = img_bgr.shape[:2]
                    with open(lbl_path) as f:
                        for line in f:
                            vals = line.strip().split()
                            if len(vals) < 5:
                                continue
                            cx, cy, bw, bh = map(float, vals[1:5])
                            gx1 = (cx - bw/2) * iw
                            gy1 = (cy - bh/2) * ih
                            gx2 = (cx + bw/2) * iw
                            gy2 = (cy + bh/2) * ih
                            gt_boxes.append([gx1, gy1, gx2, gy2])
            n_gt = len(gt_boxes)

            det_results = model(img_path, conf=thresh, verbose=False)
            pred_boxes = []
            if det_results[0].boxes is not None:
                for box in det_results[0].boxes:
                    pred_boxes.append(box.xyxy[0].cpu().numpy().tolist())
            n_pred = len(pred_boxes)

            # IoU-based matching (IoU > 0.5 = TP)
            matched = 0
            used_gt = set()
            for pb in pred_boxes:
                best_iou = 0.0
                best_gt  = -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in used_gt:
                        continue
                    ix1 = max(pb[0], gb[0])
                    iy1 = max(pb[1], gb[1])
                    ix2 = min(pb[2], gb[2])
                    iy2 = min(pb[3], gb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_p = (pb[2]-pb[0]) * (pb[3]-pb[1])
                    area_g = (gb[2]-gb[0]) * (gb[3]-gb[1])
                    iou = inter / max(area_p + area_g - inter, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt  = gi
                if best_iou >= 0.5 and best_gt >= 0:
                    matched += 1
                    used_gt.add(best_gt)

            tp += matched
            fp += max(0, n_pred - matched)
            fn += max(0, n_gt   - matched)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        results.append({
            "thresh":    round(thresh, 2),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        })
        marker = " ◄" if precision >= target_precision else ""
        print(f"  {thresh:>7.2f}  {precision:>10.4f}  "
              f"{recall:>8.4f}  {f1:>8.4f}{marker}")

    # Select best threshold
    candidates = [r for r in results if r["precision"] >= target_precision]
    if candidates:
        best = max(candidates, key=lambda r: r["recall"])
    else:
        best = max(results, key=lambda r: r["f1"])
        print(f"\n  [WARN] Could not reach precision={target_precision} "
              f"— using best F1 instead")

    selected = best["thresh"]
    print(f"\n  Selected threshold : {selected}")
    print(f"  P={best['precision']}  R={best['recall']}  F1={best['f1']}")

    # Save sweep JSON
    sweep_path = Path(cfg["logging"]["save_dir"]) / \
                 "fp_correction" / "threshold_sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({"sweep": results, "selected": best}, f, indent=2)
    print(f"  Sweep saved → {sweep_path}")

    # Write back to config
    cfg["detector"]["conf_threshold"] = selected
    save_config(cfg, config_path)

    return selected


# =============================================================================
#  Sub-step 3 — Detector Fine-tune with Hard Negatives
# =============================================================================

def finetune_detector(cfg, fp_cfg, neg_images_dir,
                      det_ckpt="logs/detector/best.pt"):
    """
    Short fine-tune of detector on original + hard negative images.
    Backbone frozen — only head updates.

    Args:
        cfg:            Full config dict.
        fp_cfg:         fp_correction section of config.
        neg_images_dir: Path to hard negative images.
        det_ckpt:       Path to detector weights.

    Returns:
        str: Path to fine-tuned detector weights.
    """
    print(f"\n{'═'*60}")
    print(f"  Sub-step 3 — Detector Fine-tune with Hard Negatives")
    print(f"{'═'*60}\n")

    save_dir   = Path(cfg["logging"]["save_dir"]) / "fp_correction"
    ft_epochs  = fp_cfg.get("finetune_epochs",  25)
    ft_lr      = fp_cfg.get("finetune_lr",    1e-4)

    # Build merged dataset YAML
    data_yaml    = os.path.join(cfg["dataset"]["root"], "data.yaml")
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    root      = cfg["dataset"]["root"]
    train_img = os.path.normpath(os.path.join(root, data_cfg["train"]))
    val_img   = os.path.normpath(os.path.join(root, data_cfg["val"]))

    # Merge hard negative images into a combined train dir
    merged_img = save_dir / "merged" / "images" / "train"
    merged_lbl = save_dir / "merged" / "labels" / "train"
    merged_val_img = save_dir / "merged" / "images" / "val"
    merged_val_lbl = save_dir / "merged" / "labels" / "val"

    for d in [merged_img, merged_lbl, merged_val_img, merged_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Symlink original train images
    print("  Building merged dataset...")
    orig_lbl = train_img.replace("images", "labels")
    copied   = 0
    for f in os.listdir(train_img):
        if not Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        src_img = Path(train_img) / f
        dst_img = merged_img / f
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        src_lbl = Path(orig_lbl) / (Path(f).stem + ".txt")
        dst_lbl = merged_lbl / (Path(f).stem + ".txt")
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        copied += 1

    # Add hard negative images (empty labels)
    neg_lbl_dir = Path(neg_images_dir).parent / "labels"
    hn_count    = 0
    for f in os.listdir(neg_images_dir):
        if not Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(
            os.path.join(neg_images_dir, f),
            merged_img / f,
        )
        # Empty label
        (merged_lbl / (Path(f).stem + ".txt")).write_text("")
        hn_count += 1

    # Copy val set
    val_lbl_dir = val_img.replace("images", "labels")
    for f in os.listdir(val_img):
        if not Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(Path(val_img) / f, merged_val_img / f)
        src_lbl = Path(val_lbl_dir) / (Path(f).stem + ".txt")
        if src_lbl.exists():
            shutil.copy2(src_lbl, merged_val_lbl / (Path(f).stem + ".txt"))

    print(f"  Original train : {copied} images")
    print(f"  Hard negatives : {hn_count} images")

    # Write merged dataset YAML
    merged_yaml_path = save_dir / "merged" / "dataset.yaml"
    merged_data = {
        "path":  str((save_dir / "merged").resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    cfg["dataset"]["num_classes"],
        "names": cfg["dataset"]["class_names"],
    }
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged_data, f, default_flow_style=False)

    # Fine-tune with frozen backbone
    print(f"\n  Fine-tuning for {ft_epochs} epochs at lr={ft_lr}")
    model   = YOLO(det_ckpt)
    results = model.train(
        data         = str(merged_yaml_path),
        epochs       = ft_epochs,
        imgsz        = cfg["detector"]["img_size"],
        batch        = cfg["detector"]["batch_size"],
        lr0          = ft_lr,
        lrf          = 0.01,
        warmup_epochs= 1,
        freeze       = 10,        # freeze first 10 layers (backbone)
        patience     = cfg["logging"]["patience"],
        project      = str(cfg["logging"]["save_dir"]),
        name         = "fp_correction",
        exist_ok     = True,
        device       = 0,
        verbose      = True,
        plots        = False,
    )

    # Copy best weights
    ft_best_src = Path(cfg["logging"]["save_dir"]) / \
                  "fp_correction" / "weights" / "best.pt"

    # Search for it
    for p in Path(cfg["logging"]["save_dir"]).rglob("best.pt"):
        if "fp_correction" in str(p):
            ft_best_src = p
            break

    ft_best_dst = save_dir / "detector_ft_best.pt"
    if ft_best_src.exists():
        shutil.copy2(str(ft_best_src), str(ft_best_dst))
        print(f"\n  FT weights → {ft_best_dst}")
    else:
        # Fallback — use original
        shutil.copy2(det_ckpt, str(ft_best_dst))
        print(f"\n  [WARN] FT weights not found — using original detector")

    return str(ft_best_dst)


# =============================================================================
#  Background Classifier Dataset
# =============================================================================

class HardNegativeDataset(Dataset):
    """Dataset combining weapon crops + hard negative (background) crops."""

    def __init__(self, weapon_root, negative_root,
                 class_names, crop_size=224):
        self.samples   = []
        self.crop_size = crop_size
        bg_label       = len(class_names)

        # Weapon crops
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = os.path.join(weapon_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({
                        "path":  os.path.join(cls_dir, fname),
                        "label": cls_idx,
                    })

        # Hard negative crops
        if os.path.isdir(negative_root):
            for fname in os.listdir(negative_root):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({
                        "path":  os.path.join(negative_root, fname),
                        "label": bg_label,
                    })

        n_weapon = sum(1 for s in self.samples if s["label"] < bg_label)
        n_bg     = sum(1 for s in self.samples if s["label"] == bg_label)
        print(f"  [Dataset] weapon={n_weapon}  background={n_bg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img  = cv2.imread(item["path"])
        if img is None:
            img = np.zeros((self.crop_size, self.crop_size, 3), np.uint8)
        img = cv2.resize(img, (self.crop_size, self.crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        return (
            torch.from_numpy(img.transpose(2, 0, 1)).float(),
            item["label"],
        )


# =============================================================================
#  Sub-step 4 — Classifier Background Fine-tune
# =============================================================================

def finetune_classifier_with_background(
    cfg, fp_cfg, neg_crops_dir,
    cls_ckpt="logs/classifier/best.pt"
):
    """
    Add background class to classifier and re-train with FP crops.

    Args:
        cfg:           Full config dict.
        fp_cfg:        fp_correction section of config.
        neg_crops_dir: Path to hard negative crop images.
        cls_ckpt:      Path to original classifier weights.

    Returns:
        str: Path to fine-tuned classifier weights.
    """
    print(f"\n{'═'*60}")
    print(f"  Sub-step 4 — Classifier Background Fine-tune")
    print(f"{'═'*60}\n")

    cls_cfg             = cfg["classifier"]
    num_weapons         = cfg["dataset"]["num_classes"]
    num_classes_with_bg = num_weapons + 1
    save_dir            = Path(cfg["logging"]["save_dir"]) / "fp_correction"
    device              = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build expanded model
    model = WeaponClassifier(
        num_classes = num_classes_with_bg,
        dropout     = cls_cfg["dropout_rate"],
        pretrained  = False,
    ).to(device)

    ckpt       = torch.load(cls_ckpt, map_location=device)
    state_dict = ckpt["model"]

    # Manually copy weights — skip head layers (size mismatch)
    # New head has 4 outputs, old has 3 — copy backbone only
    model_dict    = model.state_dict()
    pretrained    = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    skipped = [
        k for k, v in state_dict.items()
        if k not in model_dict or v.shape != model_dict[k].shape
    ]
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    print(f"  Loaded classifier : {cls_ckpt}")
    print(f"  Copied layers     : {len(pretrained)}")
    print(f"  Skipped (resized) : {skipped}")
    print(f"  Head expanded     : {num_weapons} → {num_classes_with_bg} classes")
    print(f"  New head neuron   : randomly initialised")

    model.freeze_backbone()

    weapon_root = cfg["dataset"].get(
        "classifier_root",
        os.path.join(cfg["dataset"]["root"], "classifier_crops"),
    )
    class_names = cfg["dataset"]["class_names"]

    dataset = HardNegativeDataset(
        weapon_root   = weapon_root,
        negative_root = neg_crops_dir,
        class_names   = class_names,
        crop_size     = cls_cfg.get("img_size", 224),
    )

    if len(dataset) == 0:
        print("  [ERROR] No samples found — skipping classifier fine-tune")
        return cls_ckpt

    n_train  = int(0.85 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    _workers = fp_cfg.get("workers", 2)
    train_loader = DataLoader(
        train_ds, cls_cfg["batch_size"],
        shuffle=True, num_workers=_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, cls_cfg["batch_size"] * 2,
        shuffle=False, num_workers=_workers,
    )

    # Up-weight background class
    bg_weight     = fp_cfg.get("background_class_weight", 2.0)
    class_weights = torch.ones(num_classes_with_bg, device=device)
    class_weights[-1] = bg_weight

    criterion = FocalLoss(
        gamma           = cls_cfg["focal_gamma"],
        alpha           = cls_cfg["focal_alpha"],
        label_smoothing = 0.05,
        num_classes     = num_classes_with_bg,
        class_weights   = class_weights,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = fp_cfg.get("classifier_finetune_lr", 3e-4),
        weight_decay = cls_cfg["weight_decay"],
    )
    ft_epochs = fp_cfg.get("classifier_finetune_epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=1e-6)
    scaler    = GradScaler("cuda", enabled=True)
    ema       = ModelEMA(model, 0.9999)
    best_acc  = 0.0

    print(f"  Fine-tuning for {ft_epochs} epochs")
    print(f"  Background weight : {bg_weight}x\n")

    for epoch in range(ft_epochs):
        model.train()
        running_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast("cuda", enabled=True):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

            running_loss += float(loss)
            correct      += (logits.argmax(-1) == labels).sum().item()
            total        += len(labels)

        scheduler.step()

        # Validate
        ema.ema.eval()
        val_correct = val_total = 0
        bg_correct  = bg_total  = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds        = ema.ema(imgs).argmax(-1)
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)
                bg_mask      = labels >= num_weapons
                if bg_mask.any():
                    bg_correct += (preds[bg_mask] >= num_weapons).sum().item()
                    bg_total   += bg_mask.sum().item()

        val_acc        = val_correct / max(val_total, 1)
        bg_suppression = bg_correct  / max(bg_total,  1)
        train_acc      = correct     / max(total,      1)

        print(
            f"  Epoch [{epoch+1:>3}/{ft_epochs}]  "
            f"loss={running_loss/len(train_loader):.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"bg_suppression={bg_suppression:.4f}"
        )

        # Use combined score — both weapon acc AND bg suppression matter
        combined = val_acc * 0.4 + bg_suppression * 0.6

        if combined > best_acc:
            best_acc = combined
            save_checkpoint(
                {
                    "epoch":          epoch + 1,
                    "model":          ema.ema.state_dict(),
                    "acc":            val_acc,
                    "bg_suppression": bg_suppression,
                    "num_classes":    num_classes_with_bg,
                },
                str(save_dir / "classifier_bg_best.pt"),
            )
            print(f"  ✓ New best val_acc: {best_acc:.4f}  "
                  f"bg_suppression={bg_suppression:.4f}")

    print(f"\n  Best val_acc     : {best_acc:.4f}")
    print(f"  Weights          : {save_dir}/classifier_bg_best.pt")
    return str(save_dir / "classifier_bg_best.pt")


# =============================================================================
#  Main orchestrator
# =============================================================================

def run_fp_correction(cfg, config_path, skip_mining=False):
    """
    Run the complete 4-step false positive correction pipeline.

    Args:
        cfg:          Full config dict.
        config_path:  Path to config YAML.
        skip_mining:  Skip hard negative mining (reuse existing).
    """
    save_dir = Path(cfg["logging"]["save_dir"])
    fp_cfg   = cfg.get("fp_correction", {})

    det_ckpt = "logs/detector/best.pt"
    cls_ckpt = "logs/classifier/best.pt"

    if not Path(det_ckpt).exists():
        print(f"[ERROR] {det_ckpt} not found — run part1.py first")
        sys.exit(1)
    if not Path(cls_ckpt).exists():
        print(f"[ERROR] {cls_ckpt} not found — run part2.py first")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Part 4 — False Positive Correction")
    print(f"{'═'*60}")

    # ── Sub-step 1 ────────────────────────────────────────────────────────
    if skip_mining:
        neg_images_dir = str(
            save_dir / "fp_correction" / "hard_negatives" / "images"
        )
        neg_crops_dir  = str(
            save_dir / "fp_correction" / "hard_negatives" / "crops"
        )
        print("\n  [Skip] Using existing hard negatives")
    else:
        neg_images_dir, neg_crops_dir = mine_hard_negatives(
            cfg, fp_cfg, det_ckpt
        )

    # ── Sub-step 2 ────────────────────────────────────────────────────────
    optimal_thresh = find_optimal_threshold(
        cfg, fp_cfg, config_path, det_ckpt
    )

    # ── Sub-step 3 ────────────────────────────────────────────────────────
    ft_det_ckpt = finetune_detector(
        cfg, fp_cfg, neg_images_dir, det_ckpt
    )

    # ── Sub-step 4 ────────────────────────────────────────────────────────
    ft_cls_ckpt = finetune_classifier_with_background(
        cfg, fp_cfg, neg_crops_dir, cls_ckpt
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Part 4 Complete — Summary")
    print(f"{'═'*60}")
    print(f"  Optimal threshold    : {optimal_thresh}")
    print(f"  Finetuned detector   : {ft_det_ckpt}")
    print(f"  Finetuned classifier : {ft_cls_ckpt}")
    print(f"\n  Update detection.py to use:")
    print(f"    --det-weights {ft_det_ckpt}")
    print(f"    --cls-weights {ft_cls_ckpt}")
    print(f"    --det-conf   {optimal_thresh}")
    print(f"\n  Next step → python detection.py \\")
    print(f"    --source your_video.mp4 \\")
    print(f"    --det-weights {ft_det_ckpt} \\")
    print(f"    --cls-weights {ft_cls_ckpt} \\")
    print(f"    --det-conf {optimal_thresh}")

    # Save manifest
    manifest = {
        "optimal_threshold":   optimal_thresh,
        "finetuned_detector":  ft_det_ckpt,
        "finetuned_classifier":ft_cls_ckpt,
    }
    out = save_dir / "fp_correction" / "outputs.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest → {out}")
