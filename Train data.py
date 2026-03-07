"""
STEP 3: Train YOLOv8 on Weapon Detection Dataset (Knives, Pistols, Rifles)
===========================================================================
Trains a YOLOv8 model using your prepared dataset, filtered to 3 classes:
  - Knife   (original class index 1)
  - Pistol  (original class index 3)
  - Rifle   (original class index 4)

Hyperparameters are tuned to push Precision > 0.80.

Model size options (speed vs accuracy tradeoff):
  yolov8n  → nano   (fastest, least accurate — good for low-end hardware)
  yolov8s  → small  (good balance for most use cases)
  yolov8m  → medium (recommended for precision) ← used here
  yolov8l  → large
  yolov8x  → xlarge (most accurate, needs GPU)

After training, best weights saved to: runs/detect/gun_model3cls/weights/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML = "weapon_detection/data.yaml"  # original 5-class dataset yaml
MODEL_SIZE    = "yolov8m.pt"                 # medium model — best precision/speed tradeoff
PROJECT_NAME  = "gun_model3cls"
EPOCHS        = 80                           # more epochs → better generalisation
IMAGE_SIZE    = 640                          # standard YOLO input size
BATCH_SIZE    = 8                            # safe for 8GB VRAM; increase to 16 once stable
DEVICE        = "0" if torch.cuda.is_available() else "cpu"

# ── Only train on these class indices from the original dataset ───────────────
# Original classes: 0=Grenade, 1=Knife, 2=Missile, 3=Pistol, 4=Rifle
TARGET_CLASSES = [1, 3, 4]   # Knife, Pistol, Rifle

# ── Precision-focused hyperparameters ─────────────────────────────────────────
# - Higher conf threshold at inference time filters weak predictions → fewer FPs
# - Higher iou threshold means boxes must overlap more to be suppressed → cleaner NMS
# - label_smoothing regularises over-confident predictions
# - Reduced mosaic/mixup to preserve clean bounding box signals
HYPERPARAMS = dict(
    # Optimiser
    lr0=0.005,          # initial learning rate (lower → steadier convergence)
    lrf=0.01,           # final LR as fraction of lr0
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5.0,
    # Loss weights — boost box & cls losses to tighten localisation
    box=9.0,            # bounding-box regression loss gain (default 7.5)
    cls=1.0,            # class loss gain (default 0.5 — raised to sharpen class scores)
    # NMS / confidence
    conf=0.35,          # minimum confidence threshold during val/inference
    iou=0.6,            # IoU threshold for NMS (higher → fewer, cleaner boxes)
    # Regularisation
    label_smoothing=0.05,   # prevents overconfidence, improves precision
    dropout=0.0,
    # Augmentation — conservative to keep clean bounding box signal
    hsv_h=0.010,
    hsv_s=0.5,
    hsv_v=0.3,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.8,         # slightly reduced to preserve label quality
    mixup=0.05,
    copy_paste=0.1,     # helps small-object recall without hurting precision
)


def train():
    print(f"Device:  {DEVICE}")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Classes: {TARGET_CLASSES}  → Knife, Pistol, Rifle")

    # Load pretrained YOLOv8 (downloads automatically on first run)
    model = YOLO(MODEL_SIZE)

    # Train — pass 'classes' to filter the dataset to only the 3 target classes
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs",
        name=PROJECT_NAME,
        save=True,
        plots=True,           # saves training curves
        classes=TARGET_CLASSES,  # ← filters to Knife / Pistol / Rifle only
        patience=20,          # early-stop if no improvement for 20 epochs
        cos_lr=True,          # cosine LR schedule → smoother convergence
        amp=True,             # FP16 mixed precision (keeps VRAM usage low)
        cache=False,          # ← disabled: RAM caching was causing system crashes
        workers=4,            # ← reduced from 8 to ease CPU/RAM pressure
        **HYPERPARAMS,
    )

    # Evaluate on validation set
    metrics = model.val(classes=TARGET_CLASSES)
    print("\n── Validation Metrics ──")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    if metrics.box.mp < 0.80:
        print("⚠️  Precision is below 0.80. Consider: more epochs, higher conf, or more training data.")
    else:
        print("✅ Precision target (>0.80) achieved!")

    best_weights = Path("runs/detect") / PROJECT_NAME / "weights/best.pt"
    print(f"\n✅ Training complete! Best model: {best_weights}")
    return str(best_weights)


if __name__ == "__main__":
    train()