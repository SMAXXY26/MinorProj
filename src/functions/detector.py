"""
detector.py — Train YOLOv8n detector (Ultralytics).
=====================================================
Refactored from part1.py.

Uses the official YOLOv8 implementation which is:
  - Proven to work out of the box
  - Exports to NCNN for Pi 5 deployment
  - Real mAP50-95 metrics built in

Functions:
    prepare_dataset_yaml(cfg) → str
    train_detector(cfg, resume=False) → str
"""

import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO

from src.functions.common import load_config
from src.functions.monitor import TrainingMonitor


# =============================================================================
#  Dataset YAML
# =============================================================================

def prepare_dataset_yaml(cfg):
    """
    Ultralytics needs a dataset YAML in a specific format.
    Writes it to <data_root>/dataset.yaml.

    Returns:
        str: Path to the generated dataset YAML.
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


# =============================================================================
#  Train YOLOv8n
# =============================================================================

def train_detector(cfg, resume=False):
    """
    Train a YOLOv8n detector using Ultralytics API.

    Args:
        cfg:    Full config dict from hyperparams.yaml.
        resume: If True, resume from last checkpoint.

    Returns:
        str: Path to best weights.
    """
    det_cfg  = cfg["detector"]
    save_dir = Path(cfg["logging"]["save_dir"]) / "detector"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare dataset YAML ─────────────────────────────────────────────
    dataset_yaml = prepare_dataset_yaml(cfg)

    # ── Load model (from config or last checkpoint) ──────────────────────
    pretrained_weights = det_cfg.get("pretrained_weights", "logs/detector/best.pt")
    if resume and (save_dir / "weights" / "last.pt").exists():
        model_path = str(save_dir / "weights" / "last.pt")
        print(f"  [Resume] Loading from {model_path}")
    else:
        model_path = pretrained_weights
        print(f"  [Model] {pretrained_weights}")

    model = YOLO(model_path)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Part 1 — Detector Training ({pretrained_weights})")
    print(f"  Epochs   : {det_cfg['epochs']}")
    print(f"  img_size : {det_cfg['img_size']}")
    print(f"  batch    : {det_cfg['batch_size']}")
    print(f"  Save dir : {save_dir}")
    print(f"{'═'*60}\n")

    # ── Training monitor ─────────────────────────────────────────────────
    mon = TrainingMonitor(log_dir=str(save_dir), step_name="detector")
    mon.log_event(
        f"Start: {det_cfg['epochs']}ep  {pretrained_weights}  "
        f"{det_cfg['img_size']}px  bs={det_cfg['batch_size']}"
    )

    def _on_epoch_start(trainer):
        mon.start_epoch(trainer.epoch, trainer.epochs)

    def _on_epoch_end(trainer):
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k.split("/")[-1]] = round(float(v), 4)
        if hasattr(trainer, "loss") and trainer.loss is not None:
            try:
                metrics["loss"] = round(float(trainer.loss), 4)
            except Exception:
                pass
        mon.end_epoch(metrics)

    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_train_epoch_end",   _on_epoch_end)

    # ── Read augmentation settings ────────────────────────────────────────
    aug    = cfg.get("augmentation", {})
    aerial = aug.get("aerial", {})

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
        workers      = det_cfg.get("workers", 2),
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

        # Aerial / drone augmentations
        degrees      = aerial.get("degrees",     0.0),
        perspective  = aerial.get("perspective", 0.0),
        shear        = aerial.get("shear",       0.0),

        verbose      = True,
        plots        = True,
    )

    # ── Copy best weights to expected path ───────────────────────────────
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

    mon.log_event("Training complete")
    mon.close()

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

    return str(our_best)
