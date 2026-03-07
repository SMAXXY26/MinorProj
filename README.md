# Weapon Detection Project

This project trains a YOLOv8 medium (`yolov8m`) model to detect specific weapons, prioritizing high precision. 

## 📊 Dataset Details

The dataset contains a total of **9,633 images** annotated in YOLO format.

**Data Splits:**
- **Train**: 7,182 images
- **Validation**: 1,815 images
- **Test**: 636 images

**Target Classes (Filtered for Training):**
- **Knife** (Class Index 1)
- **Pistol** (Class Index 3)
- **Rifle** (Class Index 4)

---

## ⚙️ Hyperparameter Configuration

The configuration is tuned to push the model's **Precision above 0.80** by tightening bounding box rules and minimizing false positives.

**Core Settings:**
- **Model:** `yolov8m.pt` (Medium size for optimal precision vs. speed)
- **Epochs:** 80
- **Image Size:** 640
- **Batch Size:** 8 (Safe for 8GB VRAM)

**Loss & Optimization (Precision Focused):**
- **Initial Learning Rate (`lr0`):** `0.005` (Lower rate prioritizes steady, stable convergence)
- **Box Loss Gain (`box`):** `9.0` (Heavily penalizes loose bounding boxes to tighten localization)
- **Class Loss Gain (`cls`):** `1.0` (Raised from default to sharpen class confidence)
- **Label Smoothing:** `0.05` (Slightly regularizes confidence to prevent overfitting)

**NMS & Regularization:**
- **Confidence Threshold (`conf`):** `0.35` (Filters out weak predictions immediately)
- **IoU Threshold (`iou`):** `0.6` (Ensures clustered/overlapping boxes are cleanly suppressed)
- **Mosaic Augmentation:** `0.80` (Kept slightly conservative to maintain clean bounding box signals)

---

## 📈 Performance Metrics Guide

Understanding the metrics printed during training and evaluation:

**Training Losses(Lower is better)**
*   **`box_loss`**: How accurate the bounding boxes are drawn around the objects.
*   **`cls_loss`**: How accurate the model is at guessing the correct object strictly (Knife vs. Pistol).
*   **`dfl_loss`**: How precise the model is with the fine pixel edges/boundaries of the box.

**Training Status:**
*   **`GPU_mem`**: The amount of graphic memory (VRAM) currently used by the batch.
*   **`Instances`**: Total count of objects (weapons) processed within the current batch.

**Validation Metrics (Higher is better)**
*   **`Precision`**: When the model yells "Gun!", how often is it actually a gun? (Minimizes false alarms).
*   **`Recall`**: Out of all the real guns in the picture, how many did the model find? (Minimizes missed detections).
*   **`mAP50`**: Mean Average Precision at 50% overlap. Assesses general detection success.
*   **`mAP50-95`**: The strictest and most important metric. Averages precision across 50% to 95% bounding box overlaps. Ensures the model not only finds the object but draws a perfectly tight box around it.
