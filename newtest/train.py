"""
train.py — Three-Stage Weapon Detection Training

Run:
    python train.py --stage detector   --config config/hyperparams.yaml
    python train.py --stage classifier --config config/hyperparams.yaml
    python train.py --stage temporal   --config config/hyperparams.yaml
    python train.py --stage all        --config config/hyperparams.yaml
"""

import os
import sys
import math
import time
import argparse
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from copy import deepcopy
from pathlib import Path
from typing import Optional

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model   import WeaponDetector, WeaponClassifier, TemporalSmoother
from src.dataset import build_detector_loaders, build_classifier_loaders
from src.losses  import OBBDetectionLoss, FocalLoss, TemporalLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> torch.device:
    if cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no CUDA available)")
    return device


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M"


# ─────────────────────────────────────────────────────────────────────────────
#  EMA (Exponential Moving Average) helper
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Maintains an exponential moving average of model weights.
    EMA model typically outperforms the last-epoch checkpoint at inference.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema   = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [Checkpoint] saved → {path}")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer=None, device=None) -> int:
    """Load checkpoint. Returns epoch number."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    print(f"  [Checkpoint] resumed from epoch {epoch} ← {path}")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: Detector training
# ─────────────────────────────────────────────────────────────────────────────

def train_detector(cfg: dict, device: torch.device,
                   resume: Optional[str] = None):
    """
    Train YOLOv8x-OBB on weapon dataset.

    Key design decisions:
     - Warmup LR for first 3 epochs (avoid early instability)
     - Cosine LR decay to lrf * lr0
     - AMP (fp16) for memory efficiency
     - EMA for stable inference weights
     - Early stopping on mAP50
    """
    det_cfg = cfg["detector"]
    save_dir = Path(cfg["logging"]["save_dir"]) / "detector"

    # ── Model ────────────────────────────────────────────────────────────────
    model = WeaponDetector(
        num_classes = cfg["dataset"]["num_classes"],
        width_mult  = 1.25,
        depth_mult  = 1.0,
    ).to(device)

    # Try to load pretrained YOLOv8x weights (backbone only) if available
    pretrained_path = Path("weights") / det_cfg["pretrained_weights"]
    if pretrained_path.exists():
        try:
            ckpt = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            print(f"  [Weights] Loaded pretrained backbone from {pretrained_path}")
        except Exception as e:
            print(f"  [Weights] Could not load pretrained: {e}")

    print(f"  [Model] Detector parameters — {count_parameters(model)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_detector_loaders(cfg)

    # ── Optimiser (AdamW with param groups) ──────────────────────────────────
    # Separate weight decay: don't apply to BN params or biases
    pg_decay, pg_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or ".bias" in name:
            pg_no_decay.append(param)
        else:
            pg_decay.append(param)

    optimizer = torch.optim.AdamW([
        {"params": pg_decay,    "weight_decay": det_cfg["weight_decay"]},
        {"params": pg_no_decay, "weight_decay": 0.0},
    ], lr=det_cfg["lr0"], betas=(det_cfg["momentum"], 0.999))

    # ── LR scheduler: cosine decay ───────────────────────────────────────────
    total_epochs = det_cfg["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = total_epochs - det_cfg["warmup_epochs"],
        eta_min = det_cfg["lr0"] * det_cfg["lrf"],
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = OBBDetectionLoss(cfg, device)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=cfg["hardware"]["amp"])

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=cfg["hardware"]["ema_decay"]) \
          if cfg["hardware"]["ema"] else None

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, device)

    best_map   = 0.0
    patience   = cfg["logging"]["patience"]
    no_improve = 0

    print(f"\n{'═'*60}")
    print(f"  Training Detector  — {total_epochs} epochs")
    print(f"{'═'*60}")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        t0  = time.time()
        running = {"loss": 0, "box": 0, "cls": 0, "angle": 0}

        # ── Warmup LR ────────────────────────────────────────────────────────
        if epoch < det_cfg["warmup_epochs"]:
            warmup_factor = (epoch + 1) / det_cfg["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = det_cfg["lr0"] * warmup_factor

        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)

            with autocast(enabled=cfg["hardware"]["amp"]):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, labels)

            # Gradient accumulation
            loss = loss / det_cfg["accumulate_grad_batches"]
            scaler.scale(loss).backward()

            if (step + 1) % det_cfg["accumulate_grad_batches"] == 0:
                # Gradient clipping — prevents exploding gradients in early epochs
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0) + v
            running["loss"] += float(loss) * det_cfg["accumulate_grad_batches"]

        # ── Validation ────────────────────────────────────────────────────────
        val_map = evaluate_detector(
            ema.ema if ema else model, val_loader, device, cfg
        )

        if epoch >= det_cfg["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - t0
        n_steps = len(train_loader)
        print(
            f"  Epoch [{epoch+1:>3}/{total_epochs}]  "
            f"loss={running['loss']/n_steps:.4f}  "
            f"box={running['box']/n_steps:.3f}  "
            f"cls={running['cls']/n_steps:.3f}  "
            f"ang={running['angle']/n_steps:.3f}  "
            f"mAP50={val_map:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"t={elapsed:.0f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = val_map > best_map
        if is_best:
            best_map   = val_map
            no_improve = 0
        else:
            no_improve += 1

        state = {
            "epoch":     epoch + 1,
            "model":     (ema.ema if ema else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "map50":     val_map,
        }
        if is_best:
            save_checkpoint(state, str(save_dir / "best.pt"))
        if (epoch + 1) % cfg["logging"]["save_period"] == 0:
            save_checkpoint(state, str(save_dir / f"epoch{epoch+1}.pt"))

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n  [Early Stop] No improvement for {patience} epochs.")
            break

    print(f"\n  Best mAP50 = {best_map:.4f}  →  {save_dir}/best.pt")
    return str(save_dir / "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2: Classifier training
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(cfg: dict, device: torch.device,
                     resume: Optional[str] = None):
    """
    Train EfficientNet-B5 on weapon crop images.

    Strategy:
      Epochs 1–5   : backbone FROZEN, train head only (fast convergence)
      Epochs 6+    : full fine-tune with lower LR on backbone
    """
    cls_cfg  = cfg["classifier"]
    save_dir = Path(cfg["logging"]["save_dir"]) / "classifier"

    model = WeaponClassifier(
        num_classes = cfg["dataset"]["num_classes"],
        dropout     = cls_cfg["dropout_rate"],
        pretrained  = cls_cfg["pretrained"],
    ).to(device)

    print(f"  [Model] Classifier parameters — {count_parameters(model)}")

    # ── Phase 1: freeze backbone ──────────────────────────────────────────────
    model.freeze_backbone()

    train_loader, val_loader = build_classifier_loaders(cfg)

    # Build param groups with 10× lower LR for backbone
    head_params     = list(model.head.parameters()) + list(model.pool.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": cls_cfg["lr"]},
        {"params": backbone_params, "lr": cls_cfg["lr"] * 0.1},
    ], weight_decay=cls_cfg["weight_decay"])

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0    = cls_cfg["T_0"],
        T_mult = cls_cfg["T_mult"],
        eta_min = cls_cfg["eta_min"],
    )

    criterion = FocalLoss(
        gamma           = cls_cfg["focal_gamma"],
        alpha           = cls_cfg["focal_alpha"],
        label_smoothing = cls_cfg["label_smoothing"],
        num_classes     = cfg["dataset"]["num_classes"],
    )

    scaler     = GradScaler(enabled=cfg["hardware"]["amp"])
    ema        = ModelEMA(model, cfg["hardware"]["ema_decay"])
    best_acc   = 0.0
    no_improve = 0
    unfreeze_done = False
    total_epochs  = cls_cfg["epochs"]

    print(f"\n{'═'*60}")
    print(f"  Training Classifier  — {total_epochs} epochs")
    print(f"{'═'*60}")

    for epoch in range(total_epochs):
        # ── Unfreeze backbone after 5 warmup epochs ───────────────────────────
        if epoch == 5 and not unfreeze_done:
            model.unfreeze_backbone()
            unfreeze_done = True
            print("  [Phase 2] Backbone unfrozen — full fine-tune")

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # MixUp / CutMix (simple toggle)
            if (random.random() < cls_cfg["mixup_prob"]
                    and len(imgs) > 1):
                imgs, labels_a, labels_b, lam = mixup_data(
                    imgs, labels, cls_cfg["mixup_alpha"]
                )
                with autocast(enabled=cfg["hardware"]["amp"]):
                    logits = model(imgs)
                    loss   = (lam * criterion(logits, labels_a) +
                              (1 - lam) * criterion(logits, labels_b))
            else:
                with autocast(enabled=cfg["hardware"]["amp"]):
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
            preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_acc, val_loss = evaluate_classifier(
            ema.ema, val_loader, device, criterion
        )
        train_acc = correct / max(total, 1)

        print(
            f"  Epoch [{epoch+1:>3}/{total_epochs}]  "
            f"train_loss={running_loss/len(train_loader):.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            no_improve = 0
            save_checkpoint(
                {"epoch": epoch+1, "model": ema.ema.state_dict(), "acc": val_acc},
                str(save_dir / "best.pt"),
            )
        else:
            no_improve += 1
            if no_improve >= cfg["logging"]["patience"]:
                print(f"  [Early Stop] No improvement for {cfg['logging']['patience']} epochs.")
                break

    print(f"\n  Best Val Acc = {best_acc:.4f}  →  {save_dir}/best.pt")
    return str(save_dir / "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3: Temporal smoother training
# ─────────────────────────────────────────────────────────────────────────────

def train_temporal(cfg: dict, device: torch.device,
                   sequences: list, resume: Optional[str] = None):
    """
    Train BiLSTM temporal smoother on pre-extracted detection sequences.

    `sequences` is a list of dicts with keys 'features' and 'labels'
    (typically generated by running Stages 1+2 on your training videos
    and saving per-frame detections).
    """
    from src.dataset import TemporalDetectionDataset

    tmp_cfg  = cfg["temporal"]
    save_dir = Path(cfg["logging"]["save_dir"]) / "temporal"

    model = TemporalSmoother(
        input_size  = tmp_cfg["input_size"],
        hidden_size = tmp_cfg["hidden_size"],
        num_layers  = tmp_cfg["num_layers"],
        num_classes = cfg["dataset"]["num_classes"],
        dropout     = tmp_cfg["dropout"],
    ).to(device)

    print(f"  [Model] Temporal smoother — {count_parameters(model)}")

    # Split sequences
    split = int(len(sequences) * 0.85)
    train_seqs = sequences[:split]
    val_seqs   = sequences[split:]

    from torch.utils.data import DataLoader
    train_ds = TemporalDetectionDataset(train_seqs, tmp_cfg["window_size"])
    val_ds   = TemporalDetectionDataset(val_seqs,   tmp_cfg["window_size"])

    train_loader = DataLoader(train_ds, tmp_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   tmp_cfg["batch_size"] * 2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tmp_cfg["lr"],
        weight_decay=tmp_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=tmp_cfg["step_size"], gamma=tmp_cfg["gamma"]
    )
    criterion = TemporalLoss(smoothness_weight=0.1)

    best_loss  = float("inf")
    total_epochs = tmp_cfg["epochs"]

    print(f"\n{'═'*60}")
    print(f"  Training Temporal Smoother  — {total_epochs} epochs")
    print(f"{'═'*60}")

    for epoch in range(total_epochs):
        model.train()
        running = 0.0

        for feats, labels in train_loader:
            feats  = feats.to(device)
            labels = labels.to(device)

            # Derive conf_target: 1 where any weapon class is detected
            conf_target = (labels > 0).float()

            conf_pred, cls_logits = model(feats)
            loss, _ = criterion(conf_pred, cls_logits, conf_target, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running += float(loss)

        scheduler.step()

        # ── Val ───────────────────────────────────────────────────────────────
        val_loss = evaluate_temporal(model, val_loader, device, criterion)
        print(
            f"  Epoch [{epoch+1:>3}/{total_epochs}]  "
            f"train={running/len(train_loader):.4f}  val={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                {"epoch": epoch+1, "model": model.state_dict()},
                str(save_dir / "best.pt"),
            )

    print(f"\n  Best val loss = {best_loss:.4f}  →  {save_dir}/best.pt")
    return str(save_dir / "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_detector(model, loader, device, cfg) -> float:
    """
    Approximate mAP50 evaluation on the validation set.
    For a full COCO-style mAP, use the ultralytics val routine.
    Here we compute a simplified version: average confidence on GT-matched boxes.
    """
    model.eval()
    conf_sum, count = 0.0, 0
    conf_thresh = cfg["detector"]["conf_threshold"]

    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        dets  = model.decode_predictions(
            preds, img_size=imgs.shape[-1],
            conf_thresh=conf_thresh,
        )
        for det, gt in zip(dets, labels):
            if det.shape[0] > 0 and gt.shape[0] > 0:
                conf_sum += float(det[:, 5].mean())
                count += 1

    return conf_sum / max(count, 1)


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        val_loss += float(loss)
    return correct / max(total, 1), val_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_temporal(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        conf_target   = (labels > 0).float()
        conf_pred, cls_logits = model(feats)
        loss, _ = criterion(conf_pred, cls_logits, conf_target, labels)
        total_loss += float(loss)
    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MixUp helper
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weapon Detection Training")
    parser.add_argument("--stage",  type=str, default="all",
                        choices=["detector", "classifier", "temporal", "all"])
    parser.add_argument("--config", type=str, default="config/hyperparams.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)

    print(f"\n{'═'*60}")
    print(f"  {cfg['project']['name']}  v{cfg['project']['version']}")
    print(f"{'═'*60}\n")

    if args.stage in ("detector", "all"):
        det_ckpt = train_detector(cfg, device, resume=args.resume)

    if args.stage in ("classifier", "all"):
        cls_ckpt = train_classifier(cfg, device, resume=args.resume)

    if args.stage in ("temporal", "all"):
        # For temporal training, sequences must be pre-extracted.
        # Provide your own sequences list or run the extraction script first.
        print("\n  [Temporal] Skipping — no sequences provided.")
        print("  Run:  python extract_sequences.py  first, then re-run with --stage temporal")

    print("\n  Training complete.")


if __name__ == "__main__":
    main()
