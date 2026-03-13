"""
STEP 3: Train YOLOv8 on Weapon Detection Dataset
==================================================
Trains a YOLOv8 model on a single-class weapon detection dataset.
  - Class 0: weapon

Dataset stats:
  - Training images:   19,970
  - Validation images:  1,023

Hardware: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM)

Model: YOLOv8s (small) — best precision/speed tradeoff for this dataset size.
       Nano was too weak for ~20k images; medium risks VRAM pressure at batch 16.

After training, best weights saved to: runs/<PROJECT_NAME>/weights/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import logging

# ── Config ────────────────────────────────────────────────────────────────────
FOLD         = 1                     # Which fold to train (1 to 5)
DATASET_YAML = "newtest/data/dataset_split/data.yaml"
MODEL_SIZE   = "yolov8n.pt"          # small model — strong accuracy on ~20k images
PROJECT_NAME = "gun_model_v3"
EPOCHS       = 80                   # large dataset benefits from more epochs
IMAGE_SIZE   = 416                   # standard YOLO input size
BATCH_SIZE   = 16                    # RTX 4070 8GB handles batch 16 with FP16
DEVICE       = "0" if torch.cuda.is_available() else "cpu"

# Single-class dataset — no class filtering needed
TARGET_CLASSES = [0]  # weapon

# ── Precision-focused hyperparameters ─────────────────────────────────────────
HYPERPARAMS = dict(
    # ── Optimiser ──
    lr0=0.01,               # standard initial LR for SGD (YOLOv8 default)
    lrf=0.01,               # final LR = lr0 * lrf = 0.0001
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5.0,
    warmup_momentum=0.8,

    # ── Loss weights ──
    box=7.5,                # bounding-box regression loss gain (default)
    cls=0.5,                # class loss gain (low since single-class)

    # ── NMS / confidence ──
    conf=0.30,              # slightly lower conf → catch more detections, NMS cleans up
    iou=0.65,               # IoU threshold for NMS — higher → fewer duplicate boxes

    # ── Regularisation ──
    label_smoothing=0.05,   # prevents overconfidence on single class
    dropout=0.0,

    # ── Augmentation (strong — large dataset can handle it) ──
    hsv_h=0.015,            # hue jitter
    hsv_s=0.7,              # saturation jitter
    hsv_v=0.4,              # brightness jitter
    degrees=5.0,            # slight rotation
    translate=0.1,          # slight translation
    scale=0.5,              # scale jitter
    shear=2.0,              # minor shear
    flipud=0.0,             # no vertical flip (weapons have orientation)
    fliplr=0.5,             # horizontal flip
    mosaic=1.0,             # full mosaic — big dataset handles it well
    mixup=0.1,              # light mixup regularisation
    copy_paste=0.1,         # synthetic paste for small-object recall
)


def train():
    print(f"Device:  {DEVICE}")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Model:   {MODEL_SIZE}")
    print(f"Images:  ~19,970 train / ~1,023 val")
    print(f"Classes: {TARGET_CLASSES}  → weapon (single class)")
    print(f"Epochs:  {EPOCHS}  |  Batch: {BATCH_SIZE}  |  ImgSz: {IMAGE_SIZE}")

    # Load pretrained YOLOv8-small (downloads automatically on first run)
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs",
        name=PROJECT_NAME,
        save=True,
        plots=True,
        classes=TARGET_CLASSES,
        patience=25,          # early-stop after 25 epochs w/o improvement
        cos_lr=True,          # cosine annealing LR schedule
        amp=True,             # FP16 mixed precision — saves VRAM
        cache=False,          # disk-only; avoids RAM pressure on large dataset
        workers=6,            # good balance for 8-core+ CPUs without I/O bottleneck
        close_mosaic=15,      # disable mosaic for last 15 epochs → refine detections
        **HYPERPARAMS,
    )

    # Evaluate on validation set
    metrics = model.val(classes=TARGET_CLASSES)
    print("\n── Validation Metrics ──")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    best_weights = Path("runs") / PROJECT_NAME / "weights/best.pt"

    # Setup logging to track training metrics
    log_file = Path("runs") / PROJECT_NAME / f"metrics_fold_{FOLD}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
    
    logging.info(f"--- Training Completed for Fold {FOLD} ---")
    logging.info(f"mAP50: {metrics.box.map50:.3f}")
    logging.info(f"mAP50-95: {metrics.box.map:.3f}")
    logging.info(f"Precision: {metrics.box.mp:.3f}")
    logging.info(f"Recall: {metrics.box.mr:.3f}")
    logging.info(f"Best weights stored at: {best_weights.absolute()}")

    if metrics.box.mp < 0.80:
        print("⚠️  Precision is below 0.80. Consider: more epochs, higher conf, or more data.")
    else:
        print("✅ Precision target (>0.80) achieved!")

    print(f"\n✅ Training complete! Best model: {best_weights}")
    print(f"📄 Metrics logged to: {log_file.absolute()}")
    return str(best_weights)


if __name__ == "__main__":
    train()