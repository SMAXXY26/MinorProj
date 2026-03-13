# Weapon Detection — YOLOv8s

YOLOv8s model trained to detect weapons with high precision.

---

## Dataset

| Split      | Images  |
|------------|---------|
| Train      | ~19,970 |
| Validation | ~1,023  |

**Class:** `Weapon` (index 0)

---

## Training Config

| Parameter        | Value       |
|------------------|-------------|
| Model            | yolov8s.pt  |
| Epochs           | 50          |
| Image Size       | 416         |
| Batch Size       | 16          |
| LR (`lr0`)       | 0.01        |
| Box Loss (`box`) | 7.5         |
| Class Loss (`cls`) | 0.5       |
| Label Smoothing  | 0.05        |
| Conf Threshold   | 0.30        |
| IoU Threshold    | 0.65        |

---

## Results (Epoch 50)

| Metric    | Score |
|-----------|-------|
| Precision | 0.850 |
| Recall    | 0.744 |
| mAP50     | 0.833 |
| mAP50-95  | 0.622 |

### Training Curves
![Results](./assets/results.png)

### Validation Predictions
![Val Batch 0](./assets/val_batch0_pred.jpg)
![Val Batch 1](./assets/val_batch1_pred.jpg)

---

## Dataset Credits

| Dataset | Workspace | License |
|---------|-----------|---------|
| [Weapon Detection v1](https://universe.roboflow.com/test-7awfy/weapon-detection-f1lih/dataset/1) — 5-class (Grenade, Knife, Missile, Pistol, Rifle) | test-7awfy | CC BY 4.0 |
| [Weapon Detection v2](https://universe.roboflow.com/atmai/weapon-detection-j5ehm/dataset/2) — 1-class (Weapon) | atmai | Public Domain |
| [Pistols v1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1) — 1-class (Pistol)| joseph-nelson | Public Domain |