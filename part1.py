"""
part1.py  —  Train YOLOv8n detector (Ultralytics)
==================================================
Uses the official YOLOv8 implementation which is:
  - Proven to work out of the box
  - Exports to NCNN for Pi 5 deployment
  - Real mAP50-95 metrics built in

Usage:
    python part1.py
    python part1.py --resume
"""

import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path

# ── Ultralytics YOLOv8 ───────────────────────────────────────────────────────
from ultralytics import YOLO


def load_config(path="config/hyperparams.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_dataset_yaml(cfg):
    """
    Ultralytics needs a dataset YAML in a specific format.
    Writes it to data/dataset.yaml.
    """
    root        = Path(cfg["dataset"]["root"]).resolve()
    class_names = cfg["dataset"]["class_names"]

    dataset = {
        "path":  str(root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }

    out_path = root / "dataset.yaml"
    with open(out_path, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False)

    print(f"  [Dataset] Written → {out_path}")
    print(f"  [Dataset] Classes : {class_names}")
    print(f"  [Dataset] Train   : {root}/images/train")
    print(f"  [Dataset] Val     : {root}/images/val")
    return str(out_path)


def train(cfg, resume=False):
    det_cfg      = cfg["detector"]
    save_dir     = Path(cfg["logging"]["save_dir"]) / "detector"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare dataset YAML ─────────────────────────────────────────────
    dataset_yaml = prepare_dataset_yaml(cfg)

    # ── Load YOLOv8n (nano — best for Pi 5) ──────────────────────────────
    if resume and (save_dir / "weights" / "last.pt").exists():
        model_path = str(save_dir / "weights" / "last.pt")
        print(f"  [Resume] Loading from {model_path}")
    else:
        model_path = "yolov8n.pt"   # downloads automatically if not present
        print(f"  [Model] YOLOv8n — nano variant for Pi 5 deployment")

    model = YOLO(model_path)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Part 1 — YOLOv8n Detector Training")
    print(f"  Epochs   : {det_cfg['epochs']}")
    print(f"  img_size : {det_cfg['img_size']}")
    print(f"  batch    : {det_cfg['batch_size']}")
    print(f"  Save dir : {save_dir}")
    print(f"{'═'*60}\n")

    # ── Read augmentation settings ────────────────────────────────────────
    aug = cfg.get("augmentation", {})

    results = model.train(
        data         = dataset_yaml,
        epochs       = det_cfg["epochs"],
        imgsz        = det_cfg["img_size"],
        batch        = det_cfg["batch_size"],
        lr0          = det_cfg["lr0"],
        lrf          = det_cfg["lrf"],
        momentum     = det_cfg["momentum"],
        weight_decay = det_cfg["weight_decay"],
        warmup_epochs= det_cfg["warmup_epochs"],
        patience     = cfg["logging"]["patience"],
        project      = str(cfg["logging"]["save_dir"]),
        name         = "detector",
        exist_ok     = True,
        device       = 0,
        workers      = 4,
        amp          = cfg["hardware"]["amp"],
        cos_lr       = True,
        close_mosaic = 10,

        # Spatial augmentations from config
        hsv_h        = aug.get("hsv_h",     0.015),
        hsv_s        = aug.get("hsv_s",     0.5),
        hsv_v        = aug.get("hsv_v",     0.3),
        fliplr       = aug.get("fliplr",    0.5),
        flipud       = aug.get("flipud",    0.0),
        scale        = aug.get("scale",     0.5),
        translate    = aug.get("translate", 0.1),
        mosaic       = aug.get("mosaic",    0.0),
        mixup        = aug.get("mixup",     0.0),

        verbose      = True,
        plots        = True,
    )

    # ── Copy best weights to expected path ───────────────────────────────
    # Ultralytics saves to runs/detect/<name>/weights/best.pt
    possible_paths = [
        save_dir / "weights" / "best.pt",
        Path("runs") / "detect" / "logs" / "detector" / "weights" / "best.pt",
        Path("runs") / "detect" / "detector" / "weights" / "best.pt",
    ]

    yolo_best = None
    for p in possible_paths:
        if p.exists():
            yolo_best = p
            break

    our_best = save_dir / "best.pt"
    save_dir.mkdir(parents=True, exist_ok=True)

    if yolo_best:
        shutil.copy2(str(yolo_best), str(our_best))
        print(f"\n  [Copied] {yolo_best} → {our_best}")
    else:
        print(f"\n  [WARN] Could not find best.pt — copying manually")

    # ── Print final metrics ───────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Training Complete")
    try:
        metrics = results.results_dict
        print(f"  mAP50    : {metrics.get('metrics/mAP50(B)',    0):.4f}")
        print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall   : {metrics.get('metrics/recall(B)',    0):.4f}")
    except Exception:
        pass
    print(f"  Weights  : {our_best}")
    print(f"{'═'*60}")
    print(f"\n  Next step → python part2.py")

    return str(our_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 1 — Train YOLOv8n")
    parser.add_argument("--config", type=str, default="config/hyperparams.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    train(cfg, resume=args.resume)