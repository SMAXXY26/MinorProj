"""
temporal.py — Temporal Smoother Training (BiLSTM).
====================================================
Refactored from part3.py.

Extracts per-frame detection features from video sequences
and trains a BiLSTM to smooth detections across time.

This fixes:
  - Flickering detections (weapon appears/disappears every few frames)
  - Brief false positives (single-frame FP gets suppressed)
  - Missed detections (weapon present but confidence dips below threshold)

Pipeline:
  Video sequences → YOLOv8n features → BiLSTM smoother → stable detections

Classes:
    TemporalSmoother — BiLSTM model
    TemporalLoss     — combined confidence + classification + smoothness loss
    TemporalDataset  — sliding window dataset

Functions:
    extract_features_from_video(...)  → list[dict]
    extract_features_from_folder(...) → list[dict]
    evaluate_temporal(...)            → (loss, conf_acc, cls_acc)
    train_temporal(...)               → str
    demo_smoother(...)                → None
"""

import os
import time
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from ultralytics import YOLO

from src.functions.common import save_checkpoint


# =============================================================================
#  Temporal Smoother Model
# =============================================================================

class TemporalSmoother(nn.Module):
    """
    BiLSTM that takes a sequence of per-frame detection features
    and outputs smoothed confidence + class predictions.

    Input features per frame (8 values):
        [conf, x1_norm, y1_norm, x2_norm, y2_norm, cls_one_hot_0, cls_one_hot_1, cls_one_hot_2]
        → padded to input_size with zeros if no detection

    Output per frame:
        conf_pred  : (B, T)       smoothed confidence 0-1
        cls_logits : (B, T, nc)   class logits
    """
    def __init__(self, input_size=8, hidden_size=128,
                 num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.input_size  = input_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.norm      = nn.LayerNorm(hidden_size * 2)
        self.conf_head = nn.Linear(hidden_size * 2, 1)
        self.cls_head  = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """x: (B, T, input_size)"""
        out, _     = self.lstm(x)
        out        = self.norm(out)
        conf       = torch.sigmoid(self.conf_head(out)).squeeze(-1)  # (B,T)
        cls_logits = self.cls_head(out)                               # (B,T,nc)
        return conf, cls_logits


# =============================================================================
#  Loss
# =============================================================================

class TemporalLoss(nn.Module):
    """Combined confidence, classification and smoothness loss."""

    def __init__(self, smoothness_weight=0.1, num_classes=3):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.num_classes       = num_classes

    def forward(self, conf_pred, cls_logits, labels):
        """
        conf_pred  : (B, T)       predicted confidence
        cls_logits : (B, T, nc)   class logits
        labels     : (B, T)       0=background  1..nc=weapon class
        """
        # Confidence target — 1 if weapon present, 0 if background
        conf_target = (labels > 0).float()
        conf_loss   = F.binary_cross_entropy(conf_pred, conf_target)

        # Class loss — only on frames with weapon present
        weapon_mask = labels > 0
        cls_loss    = torch.tensor(0.0, device=conf_pred.device)
        if weapon_mask.any():
            cls_target = (labels - 1).clamp(0)   # shift: 1..nc → 0..nc-1
            cls_loss   = F.cross_entropy(
                cls_logits[weapon_mask],
                cls_target[weapon_mask],
            )

        # Temporal smoothness — penalise sudden conf jumps
        smoothness = torch.mean(
            torch.abs(conf_pred[:, 1:] - conf_pred[:, :-1])
        )

        total = conf_loss + cls_loss + self.smoothness_weight * smoothness
        return total, {
            "conf":   float(conf_loss),
            "cls":    float(cls_loss),
            "smooth": float(smoothness),
        }


# =============================================================================
#  Dataset
# =============================================================================

class TemporalDataset(Dataset):
    """Slices sequences into fixed-length windows for training."""

    def __init__(self, sequences: list, window_size: int = 16, stride: int = 8):
        self.windows     = []
        self.window_size = window_size

        for seq in sequences:
            feats  = seq["features"]   # (T, 8)
            lbls   = seq["labels"]     # (T,)
            T      = len(feats)

            if T < window_size:
                # Pad short sequences
                pad_f  = np.zeros((window_size - T, feats.shape[1]), np.float32)
                pad_l  = np.zeros(window_size - T, np.int64)
                feats  = np.concatenate([feats, pad_f], 0)
                lbls   = np.concatenate([lbls,  pad_l], 0)
                self.windows.append((feats, lbls))
            else:
                # Sliding window
                for start in range(0, T - window_size + 1, stride):
                    self.windows.append((
                        feats[start:start + window_size],
                        lbls[ start:start + window_size],
                    ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        feats, lbls = self.windows[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(lbls,  dtype=torch.long),
        )


# =============================================================================
#  Feature Extraction
# =============================================================================

def extract_features_from_video(
    video_path: str,
    detector:   YOLO,
    class_names: list,
    conf_thresh: float = 0.15,
    img_size:    int   = 416,
    max_frames:  int   = 500,
) -> list:
    """
    Run YOLOv8 on each frame and extract a fixed-size feature vector.

    Feature vector per frame (8 values):
        [conf, x1, y1, x2, y2, cls==0, cls==1, cls==2]
        all normalised 0-1. If no detection → all zeros.

    Returns:
        list of dicts: {"features": np.array(T, 8), "labels": np.array(T,), "video": str}
    """
    nc  = len(class_names)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}")
        return []

    features = []
    labels   = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = detector(
            frame,
            conf    = conf_thresh,
            verbose = False,
            imgsz   = img_size,
        )
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # No detection — zero vector
            features.append(np.zeros(8, np.float32))
            labels.append(0)   # background
        else:
            # Take highest-confidence detection
            confs   = boxes.conf.cpu().numpy()
            best    = int(confs.argmax())
            box     = boxes.xyxy[best].cpu().numpy()
            conf    = float(confs[best])
            cls_idx = int(boxes.cls[best].item())

            H, W = frame.shape[:2]
            x1 = box[0] / W
            y1 = box[1] / H
            x2 = box[2] / W
            y2 = box[3] / H

            # One-hot class encoding
            cls_vec = np.zeros(nc, np.float32)
            cls_vec[min(cls_idx, nc - 1)] = 1.0

            feat = np.array([conf, x1, y1, x2, y2], np.float32)
            feat = np.concatenate([feat, cls_vec])   # (5 + nc,) = 8
            features.append(feat)
            labels.append(cls_idx + 1)   # 0=background, 1..nc=weapon classes

    cap.release()

    if len(features) < 10:
        return []

    return [{
        "features": np.array(features, np.float32),   # (T, 8)
        "labels":   np.array(labels,   np.int64),      # (T,)
        "video":    video_path,
    }]


def extract_features_from_folder(
    seq_dir:     str,
    detector:    YOLO,
    class_names: list,
    cfg:         dict,
) -> list:
    """
    Extract features from all videos in seq_dir.
    Also handles image sequences (folders of frames).

    Args:
        seq_dir:     Path to directory of video files.
        detector:    Loaded YOLO model.
        class_names: List of class names.
        cfg:         Full config dict.

    Returns:
        list: List of sequence dicts.
    """
    tmp_cfg  = cfg["temporal"]
    img_size = cfg["detector"]["img_size"]
    sequences = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    seq_path = Path(seq_dir)
    if not seq_path.exists():
        print(f"  [WARN] Sequence directory not found: {seq_dir}")
        print(f"  Creating placeholder — add videos and re-run.")
        seq_path.mkdir(parents=True, exist_ok=True)
        return []

    # ── Video files ───────────────────────────────────────────────────────
    videos = [
        str(p) for p in seq_path.rglob("*")
        if p.suffix.lower() in video_exts
    ]

    print(f"  Found {len(videos)} video files in {seq_dir}")

    for vid in videos:
        print(f"  Extracting: {Path(vid).name}")
        seqs = extract_features_from_video(
            vid, detector, class_names,
            conf_thresh = 0.15,
            img_size    = img_size,
            max_frames  = tmp_cfg.get("max_frames_per_video", 300),
        )
        sequences.extend(seqs)
        print(f"    → {sum(len(s['features']) for s in seqs)} frames extracted")

    return sequences


# =============================================================================
#  Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_temporal(model, loader, device, criterion):
    """
    Evaluate temporal smoother on a dataloader.

    Returns:
        tuple: (avg_loss, conf_accuracy, cls_accuracy)
    """
    model.eval()
    total_loss   = 0
    conf_correct = 0
    conf_total   = 0
    cls_correct  = 0
    cls_total    = 0
    n_batches    = 0

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        conf_pred, cls_logits = model(feats)

        loss, _ = criterion(conf_pred, cls_logits, labels)
        total_loss += float(loss)
        n_batches  += 1

        # Confidence accuracy — sample-level
        conf_target   = (labels > 0).float()
        conf_correct += ((conf_pred > 0.5) == conf_target.bool()).float().sum().item()
        conf_total   += conf_target.numel()

        # Class accuracy (weapon frames only) — sample-level
        weapon_mask = labels > 0
        if weapon_mask.any():
            cls_target   = (labels - 1).clamp(0)
            cls_pred     = cls_logits[weapon_mask].argmax(-1)
            cls_correct += (cls_pred == cls_target[weapon_mask]).sum().item()
            cls_total   += weapon_mask.sum().item()

    return (
        total_loss   / max(n_batches, 1),
        conf_correct / max(conf_total, 1),
        cls_correct  / max(cls_total,  1),
    )


# =============================================================================
#  Training
# =============================================================================

def train_temporal(cfg, sequences, device, resume=False):
    """
    Train the BiLSTM temporal smoother.

    Args:
        cfg:       Full config dict.
        sequences: List of sequence dicts from extract_features_from_folder.
        device:    torch.device.
        resume:    Resume from last checkpoint.

    Returns:
        str: Path to best weights.
    """
    tmp_cfg  = cfg["temporal"]
    save_dir = Path(cfg["logging"]["save_dir"]) / "temporal"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    nc    = cfg["dataset"]["num_classes"]
    model = TemporalSmoother(
        input_size   = 5 + nc,      # [conf, x1,y1,x2,y2] + one-hot class
        hidden_size  = tmp_cfg["hidden_size"],
        num_layers   = tmp_cfg["num_layers"],
        num_classes  = nc,
        dropout      = tmp_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [Model] TemporalSmoother — {n_params/1e3:.1f}K parameters")

    # ── Data ─────────────────────────────────────────────────────────────────
    window_size = tmp_cfg["window_size"]
    stride      = window_size // 2

    random.shuffle(sequences)
    split      = int(len(sequences) * 0.85)
    train_seqs = sequences[:split]
    val_seqs   = sequences[split:]

    train_ds = TemporalDataset(train_seqs, window_size, stride)
    val_ds   = TemporalDataset(val_seqs,   window_size, stride)

    train_loader = DataLoader(
        train_ds, tmp_cfg["batch_size"],
        shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, tmp_cfg["batch_size"] * 2,
        shuffle=False, num_workers=2,
    )

    print(f"  [Data]  Train windows: {len(train_ds)}   Val windows: {len(val_ds)}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tmp_cfg["lr"],
        weight_decay = tmp_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tmp_cfg["epochs"], eta_min=tmp_cfg["lr"] * 0.01,
    )
    criterion = TemporalLoss(smoothness_weight=0.1, num_classes=nc)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")

    if resume:
        ckpt_path = save_dir / "last.pt"
        if ckpt_path.exists():
            ckpt        = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt.get("epoch", 0)
            best_loss   = ckpt.get("loss", float("inf"))
            print(f"  [Resume] Epoch {start_epoch}  best_loss={best_loss:.4f}")

    # ── Loop ─────────────────────────────────────────────────────────────────
    epochs     = tmp_cfg["epochs"]
    patience   = cfg["logging"]["patience"]
    no_improve = 0

    print(f"\n{'═'*60}")
    print(f"  Part 3 — Temporal Smoother Training")
    print(f"  Epochs      : {epochs}")
    print(f"  Window size : {window_size} frames")
    print(f"  Sequences   : {len(sequences)} videos")
    print(f"{'═'*60}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0           = time.time()
        running_loss = 0.0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            conf_pred, cls_logits = model(feats)
            loss, loss_dict = criterion(conf_pred, cls_logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss)

        scheduler.step()
        val_loss, conf_acc, cls_acc = evaluate_temporal(
            model, val_loader, device, criterion
        )
        avg_train = running_loss / max(len(train_loader), 1)
        elapsed   = time.time() - t0

        print(
            f"  Epoch [{epoch+1:>3}/{epochs}]  "
            f"train={avg_train:.4f}  val={val_loss:.4f}  "
            f"conf_acc={conf_acc:.4f}  cls_acc={cls_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"t={elapsed:.0f}s"
        )

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "loss":  val_loss,
            "conf_acc": conf_acc,
            "cls_acc":  cls_acc,
            "input_size":  5 + nc,
            "hidden_size": tmp_cfg["hidden_size"],
            "num_layers":  tmp_cfg["num_layers"],
            "num_classes": nc,
        }
        save_checkpoint(state, str(save_dir / "last.pt"))

        if val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0
            save_checkpoint(state, str(save_dir / "best.pt"))
            print(f"  ✓ New best val_loss: {best_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [Early Stop] No improvement for {patience} epochs.")
                break

    print(f"\n{'═'*60}")
    print(f"  Temporal training complete.")
    print(f"  Best val_loss : {best_loss:.4f}")
    print(f"  Weights       : {save_dir}/best.pt")
    print(f"{'═'*60}")
    return str(save_dir / "best.pt")


# =============================================================================
#  Demo mode
# =============================================================================

def demo_smoother(source, cfg, device):
    """
    Run the full 3-stage pipeline on a video and show smoothed detections.
    Requires trained temporal smoother at logs/temporal/best.pt
    """
    import torchvision.transforms as T
    from src.model import WeaponClassifier

    CLASS_NAMES  = cfg["dataset"]["class_names"]
    nc           = len(CLASS_NAMES)

    # Load all three models
    detector   = YOLO("logs/detector/best.pt")

    cls_ckpt   = torch.load("logs/classifier/best.pt", map_location=device)
    classifier = WeaponClassifier(
        num_classes=nc, dropout=0.0, pretrained=False
    ).to(device)
    classifier.load_state_dict(cls_ckpt["model"], strict=False)
    classifier.eval()

    tmp_ckpt   = torch.load("logs/temporal/best.pt", map_location=device)
    smoother   = TemporalSmoother(
        input_size   = tmp_ckpt["input_size"],
        hidden_size  = tmp_ckpt["hidden_size"],
        num_layers   = tmp_ckpt["num_layers"],
        num_classes  = tmp_ckpt["num_classes"],
    ).to(device)
    smoother.load_state_dict(tmp_ckpt["model"])
    smoother.eval()

    cls_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(
        int(source) if str(source).isdigit() else source
    )
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {source}")
        return

    # Rolling window of features
    window_size = cfg["temporal"]["window_size"]
    feat_buffer = []

    print(f"[Demo] Running 3-stage pipeline on {source}")
    print(f"[Demo] Q=quit  P=pause")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            H, W = frame.shape[:2]

            # Gate 1 — detector
            results  = detector(frame, conf=0.15, verbose=False)
            boxes    = results[0].boxes
            raw_conf = 0.0
            raw_cls  = 0
            feat     = np.zeros(5 + nc, np.float32)

            if boxes is not None and len(boxes) > 0:
                best    = int(boxes.conf.argmax())
                raw_conf = float(boxes.conf[best])
                raw_cls  = int(boxes.cls[best])
                box      = boxes.xyxy[best].cpu().numpy()
                feat[0]  = raw_conf
                feat[1]  = box[0] / W
                feat[2]  = box[1] / H
                feat[3]  = box[2] / W
                feat[4]  = box[3] / H
                feat[5 + min(raw_cls, nc-1)] = 1.0

            feat_buffer.append(feat)
            if len(feat_buffer) > window_size:
                feat_buffer.pop(0)

            # Pad buffer to window size
            buf = feat_buffer.copy()
            while len(buf) < window_size:
                buf.insert(0, np.zeros(5 + nc, np.float32))

            # Gate 3 — temporal smoother
            with torch.no_grad():
                x = torch.tensor(
                    np.array(buf), dtype=torch.float32
                ).unsqueeze(0).to(device)
                smooth_conf, cls_logits = smoother(x)
                final_conf = float(smooth_conf[0, -1])
                final_cls  = int(cls_logits[0, -1].argmax())

            # Draw result
            if final_conf > 0.5 and boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.argmax())
                box  = boxes.xyxy[best].cpu().numpy().astype(int)
                name = CLASS_NAMES[min(final_cls, nc-1)]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{name} {final_conf:.2f}",
                            (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 180), -1)
                cv2.putText(frame, f"WEAPON: {name.upper()}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

            # Confidence bar
            bar_w = int(final_conf * 300)
            color = (0, 255, 0) if final_conf > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (10, H-30), (10+bar_w, H-10), color, -1)
            cv2.putText(frame, f"Smooth conf: {final_conf:.2f}",
                        (10, H-35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)

        cv2.imshow("Part 3 — Temporal Smoother Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
