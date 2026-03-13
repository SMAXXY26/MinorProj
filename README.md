# Weapon Detection — YOLOv8s & Advanced 3-Stage Pipeline

A comprehensive weapon detection project utilizing YOLOv8 for real-time baseline performance and an experimental 3-stage architecture for high-fidelity cinematic analysis.

---

## 🚀 Current Status: Baseline Training (v3)
The baseline **YOLOv8s** model is currently undergoing an optimized training run for **80 epochs** on a expanded dataset of ~23,000 images to maximize recall and precision for security applications.

| Parameter        | Value       |
|------------------|-------------|
| Model            | yolov8s.pt  |
| Epochs           | 80 (Ongoing)|
| Image Size       | 416         |
| Batch Size       | 16          |
| LR (`lr0`)       | 0.01        |
| Target Precision | > 0.85      |

---

## 🧪 Experimental Branch: `experimental-newtest`
The `experimental-newtest` branch features a **State-of-the-Art 3-Stage Workflow** designed for cinematic footage (e.g., John Wick) where motion blur and complex lighting are prevalent.

### 🛡️ Architecture Highlights
1.  **Stage 1: Oriented Bounding Box (OBB) Detector**
    *   **Core:** YOLOv8x-OBB.
    *   **Logic:** Predicts rotated boxes $(\text{cx}, \text{cy}, w, h, \theta)$. This allows the model to "follow" the angle of a weapon in a shooter's hand.
2.  **Stage 2: Fine-Grained Classifier**
    *   **Core:** EfficientNet-B5 with GeM (Generalized Mean) Pooling.
    *   **Logic:** Re-examines the YOLO crop to distinguish between 7 classes: `pistol`, `revolver`, `rifle`, `shotgun`, `smg`, `knife`, `blunt_weapon`.
3.  **Stage 3: Temporal BiLSTM Smoother**
    *   **Core:** Bidirectional LSTM.
    *   **Logic:** Analyzes a sliding window of 8 frames to smooth detection confidence and eliminate flickering "ghost" detections common in fast-cut footage.

### 📐 Geometry & Metadata extraction
The pipeline extracts:
*   **Rotation Angle ($\theta$):** Normalized tilt.
*   **Eccentricity & Aspect Ratio:** Shape descriptors for secondary validation.
*   **Keypoints:** Barrel tips, triggers, and grips.

---

## 📊 Dataset Stats (Combined)

| Split      | Images  |
|------------|---------|
| Train      | ~22,370 |
| Validation | ~1,623  |

**Target Class:** `Weapon` (index 0)

---

## 🏁 Results (Baseline v2)

| Metric    | Score |
|-----------|-------|
| Precision | 0.850 |
| Recall    | 0.744 |
| mAP50     | 0.833 |
| mAP50-95  | 0.622 |

---

## 📚 Dataset Credits & Citations

This project aggregates high-quality weapon datasets from several open-source and academic providers:

| Source | Description | License |
|:-------|:------------|:--------|
| **University of Granada (SCI2S)** | [Weapons Detection Dataset](https://sci2s.ugr.es/weapons-detection#RP) - Original pistol/handgun imagery. | Public Domain |
| **Roboflow Universe (test-7awfy)** | [Weapon Detection v1](https://universe.roboflow.com/test-7awfy/weapon-detection-f1lih/dataset/1) - 5-class multi-weapon data. | CC BY 4.0 |
| **Roboflow Universe (atmai)** | [Weapon Detection v2](https://universe.roboflow.com/atmai/weapon-detection-j5ehm/dataset/2) - Unified single-class detection. | Public Domain |
| **Roboflow Universe (joseph-nelson)**| [Pistols v1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1) - Clean hand-weapon crops. | Public Domain |

---
*Developed for research and security analysis applications.*