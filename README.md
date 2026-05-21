# WeaponDetection V2

Real-time aerial weapon detection pipeline for drone-mounted platforms. Detects **knives**, **pistols**, and **rifles** via a cascaded multi-stage deep learning architecture targeting the **NVIDIA Jetson Orin Nano** (primary edge target) and **Raspberry Pi 5 AI HAT** (Hailo-8L).

**Published results (62,266-image dataset, 3 classes):**

| Model | mAP50 | Size | Latency |
|---|---|---|---|
| Teacher (YOLOv8s FP-corrected) | 0.843 | 44.6 MB | 1.0× |
| Edge student (YOLOv11n, KD) | 0.748 | 5.9 MB | 4.2× faster |

---

## Pipeline Architecture

Training runs four sequential steps, each producing weights consumed by the next:

```
Step 1  part1.py   → YOLOv8s detector          logs/detector/best.pt
Step 2  part2.py   → EfficientNet-B5 classifier logs/classifier/best.pt
Step 3  part3.py   → BiLSTM temporal smoother   logs/temporal/best.pt
Step 4  part4.py   → FP correction              logs/fp_correction/best.pt
Step 9  distill.py → YOLOv11n edge student (KD) logs/student/best.pt
```

Inference chains these as a two-gate pipeline:

```
Frame → [Gate 1] YOLOv11n detector → candidate crops
      → [Gate 2] EfficientNet-B5 classifier → class probabilities
      → [Blend]  0.6 × classifier + 0.4 × temporal (BiLSTM, 16-frame window)
      → [Track]  ByteTrack multi-object tracker
      → [Alert]  conf ≥ 0.70, cooldown 90 frames → JSONL / webhook
      → [Geo]    BboxToGeoProjector → GPS ground coordinates
```

---

## Quick Start

### 1. Install dependencies

**Dev / desktop (conda):**
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ml_env
pip install -r requirements.txt
```

**Jetson Orin Nano** — do not use conda; use `requirements_jetson.txt`:
```bash
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
pip install -r requirements_jetson.txt
```

**HPC cluster (Rachel):**
```bash
source /apps/anaconda3/bin/activate deeplearning
pip install --user -r requirements_hpc.txt
```

### 2. Prepare data

```bash
python3 download_roboflow_data.py    # Download main dataset
python3 download_negatives.py        # Download hard-negative images
python3 split_dataset.py             # Train / val split
python3 scripts/make_sequences.py    # Generate video sequences (Step 3)
```

### 3. Train

```bash
# Full pipeline (Steps 1-4)
bash run_pipeline.sh

# Include BiLSTM temporal smoother
bash run_pipeline.sh --temporal

# Skip completed steps
bash run_pipeline.sh --skip-step1
bash run_pipeline.sh --skip-step1 --skip-step2

# Individual steps
python3 part1.py --config config/hyperparams.yaml
python3 part2.py --config config/hyperparams.yaml --det-ckpt logs/detector/best.pt
python3 part3.py --config config/hyperparams.yaml --sequences data/sequences/
python3 part4.py --config config/hyperparams.yaml

# Knowledge distillation (Step 9) — desktop or HPC
python3 train_all.py --only 9
```

### 4. Run inference

```bash
# Webcam
python3 inference.py --source 0

# Video file
python3 inference.py --source test_video.mp4

# Jetson with TensorRT INT8
python3 inference.py --source 0 --trt

# Check alert log
cat logs/alerts/alerts.jsonl | python3 -m json.tool | head -20
```

---

## Configuration

All hyperparameters are in `config/hyperparams.yaml`. Key sections:

| Section | Controls |
|---|---|
| `detector` | YOLOv8 training — epochs, batch, img_size, conf_threshold |
| `classifier` | EfficientNet-B5 — focal loss, mixup, cosine annealing |
| `temporal` | BiLSTM — window_size=16, hidden_size=128 |
| `fp_correction` | Hard-negative mining thresholds and fine-tuning LRs |
| `student` | KD distillation — temperature, alpha, warm-up epochs |
| `tracking` | ByteTrack — activation threshold, lost_track_buffer |
| `alert` | Confidence threshold (0.70), cooldown frames (90), webhook URL |
| `geo` | GPS serial port, camera FOV, mock GPS coords |

---

## TensorRT Export (Jetson only)

```bash
bash run_pipeline.sh --export-trt
# or individually:
python3 export_trt.py --detector-only
python3 export_trt.py --classifier-only
```

Set Jetson to maximum performance before inference:
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

---

## Hailo-8L Export (Pi 5)

```bash
python3 export_hailo.py
python3 compile_hailo.py
python3 inference_pi5.py --source 0
```

---

## Code Structure

```
config/
  hyperparams.yaml        — all training and inference hyperparameters
  yolo11n_p2.yaml         — student architecture definition

src/
  model.py                — WeaponDetector (YOLOv8-style), WeaponClassifier
  dataset.py              — build_classifier_loaders(), augmentation pipeline
  losses.py               — FocalLoss with label smoothing and class weights
  augmentations.py        — aerial-domain augmentations
  functions/
    detector.py           — YOLOv8 detector training wrapper
    classifier.py         — EfficientNet-B5 training wrapper
    temporal.py           — BiLSTM temporal smoother
    fp_correction.py      — hard-negative mining and FP fine-tuning
    distill.py            — multi-teacher knowledge distillation (Step 9)
    export.py             — TensorRT / ONNX export helpers
    monitor.py            — TrainingMonitor — live metric logging
    validation.py         — mAP evaluation utilities
    small_object_aug.py   — small-object oversampling augmentation
  preprocessing/
    bilateral.py          — bilateral filter pre-processing
    clahe.py              — CLAHE contrast enhancement
    pipeline.py           — composable pre-processing pipeline

utils/
  tracker.py              — WeaponTracker (ByteTrack wrapper), TrackedWeapon
  alert.py                — AlertManager, AlertPayload, AlertState
  dispatchers.py          — ConsoleDispatcher, FileLogDispatcher, WebhookDispatcher
  threat_scorer.py        — multi-factor threat score
  geo.py                  — GpsReader, MavlinkTelemetry, BboxToGeoProjector

scripts/
  make_sequences.py       — generate video sequences for temporal training
  setup_jetson.sh         — Jetson environment setup
  setup_pi5.sh            — Pi 5 + Hailo-8L environment setup

part1.py                  — Step 1: detector training entry point
part2.py                  — Step 2: classifier training entry point
part3.py                  — Step 3: temporal smoother training entry point
part4.py                  — Step 4: FP correction entry point
train_all.py              — run all steps (or --only N for one step)
inference.py              — full inference pipeline (PyTorch or TRT)
inference_pi5.py          — inference on Pi 5 / Hailo-8L
benchmark.py              — latency and mAP benchmarking suite
run_pipeline.sh           — shell wrapper for the full training pipeline
transfer_to_hpc.sh        — sync project to HPC cluster for Step 9
```

---

## Dataset Format

YOLO format in `data/images/{train,val}/` and `data/labels/{train,val}/`.  
Label files: `class_id cx cy w h` (normalized 0–1).  
Classes: `0=knife  1=pistol  2=rifle`

`part1.py` auto-generates `data/dataset.yaml` on first run.

**Label hygiene** — before any training run, validate labels:
```bash
python3 - <<'EOF'
from pathlib import Path
for split in ("train", "val"):
    for f in Path(f"data/labels/{split}").glob("*.txt"):
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        clean = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5: continue
            try:
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                continue
            if all(0.0 <= v <= 1.0 for v in vals):
                clean.append(ln)
        clean = list(dict.fromkeys(clean))
        f.write_text("\n".join(clean) + ("\n" if clean else ""))
# Delete stale cache after editing labels:
# rm data/labels/train.cache data/labels/val.cache
EOF
```

---

## Temporal Feature Vector

The 8-dim feature fed to the BiLSTM per frame:
```
[p_knife, p_pistol, p_rifle, x_c_norm, y_c_norm, w_norm, h_norm, aspect_ratio]
```

---

## Docker

```bash
# Full training pipeline
docker compose up train

# Inference from webcam
docker compose run --rm infer --source 0

# Interactive shell
docker compose run --rm --entrypoint bash train
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Benchmarking

```bash
python3 benchmark.py
```

Outputs latency (ms/frame), mAP50, per-class AP, and FP-rate curves.

---

## Requirements Files

| File | Use |
|---|---|
| `requirements.txt` | Dev / desktop (full) |
| `requirements_jetson.txt` | Jetson Orin Nano (minimal) |
| `requirements_hpc.txt` | HPC cluster — Step 9 distillation only |
| `docker/requirements_train.txt` | Docker training container |
| `docker/requirements_infer.txt` | Docker inference container |

---

## Citation

See `CITATIONS.md` for references to YOLOv8, EfficientNet, ByteTrack, BiLSTM, Focal Loss, and the datasets used.
