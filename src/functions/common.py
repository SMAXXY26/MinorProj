"""
common.py — Shared utilities for the weapon detection training pipeline.
=========================================================================
Deduplicated from part1–part4. Every training module imports from here.
"""

import os
import random
import yaml
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path


# =============================================================================
#  Config I/O
# =============================================================================

def load_config(path="config/hyperparams.yaml"):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg, path="config/hyperparams.yaml"):
    """Save configuration back to YAML file."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  [Config] Saved → {path}")


# =============================================================================
#  Reproducibility
# =============================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
#  Device
# =============================================================================

def get_device(cfg):
    """Get torch device from config, with GPU name logging."""
    if cfg["hardware"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device


# =============================================================================
#  Checkpointing
# =============================================================================

def save_checkpoint(state, path):
    """Save a training checkpoint, creating parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [Saved] {path}")


# =============================================================================
#  Exponential Moving Average
# =============================================================================

class ModelEMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.9999):
        self.ema   = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# =============================================================================
#  MixUp Data Augmentation
# =============================================================================

def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# =============================================================================
#  Display Helpers
# =============================================================================

def fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_banner(step, title, status="STARTING"):
    """Print a step header banner."""
    print(f"\n{'═'*70}")
    print(f"  Step {step} — {title}")
    print(f"  Status: {status}")
    print(f"{'═'*70}\n")


def print_summary(step, title, elapsed, status="DONE"):
    """Print a step completion summary."""
    print(f"\n{'─'*70}")
    print(f"  Step {step} — {title} [{status}] ({fmt_time(elapsed)})")
    print(f"{'─'*70}\n")
