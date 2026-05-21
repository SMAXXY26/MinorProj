"""
train_synthetic_laptop.py  —  Student distillation with synthetic aerial data
==============================================================================
Waits for BlenderProc rendering to finish, builds the merged dataset, then
runs Step 9 student distillation tuned for RTX 4070 Laptop (8 GB VRAM).

System: RTX 4070 Laptop 8GB | i7-14650HX 24-thread | 15GB RAM

Laptop-specific overrides vs HPC config:
  batch_size : 128 → 32   (8GB VRAM limit; yolo11n@416 safe at 32)
  epochs     : 200 → 75   (as requested)
  patience   : 60  → 30   (proportional to epoch budget)
  workers    : 6          (unchanged — 24 threads available)
  device     : cuda       (RTX 4070)

Dataset: data/synthetic_merged/ (original train + 4,500 synthetic aerial images)
         Uses data/dataset_with_synthetic.yaml

Usage:
    python train_synthetic_laptop.py           # waits for render, then trains
    python train_synthetic_laptop.py --no-wait # skip wait, assume render done
"""

import argparse
import copy
import sys
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Laptop overrides ──────────────────────────────────────────────────────────
LAPTOP_OVERRIDES = {
    "student": {
        "batch_size":      32,    # safe for 8GB VRAM with yolo11n@416
        "epochs":          75,
        "patience":        30,    # proportional to 75-epoch budget
        "workers":         6,     # 24 threads, I/O bound at 6
        "lr0":             0.005, # keep R2 gentle lr
        "lrf":             0.001,
        "kd_warmup_epochs": 5,
        "save_period":     5,     # checkpoint every 5 epochs (shorter run)
    }
}

SYNTHETIC_YAML = ROOT / "data" / "dataset_with_synthetic.yaml"
SYN_IMAGES     = ROOT / "data" / "synthetic" / "images"
TARGET_IMAGES  = 4500


def wait_for_render(poll_interval=30):
    """Block until data/synthetic/images/ has >= TARGET_IMAGES files."""
    print(f"\n  Waiting for render to finish ({TARGET_IMAGES} images required)...")
    while True:
        n = len(list(SYN_IMAGES.glob("*.jpg"))) if SYN_IMAGES.exists() else 0
        pct = n / TARGET_IMAGES * 100
        print(f"  [{n}/{TARGET_IMAGES}  {pct:.0f}%]  polling again in {poll_interval}s...")
        if n >= TARGET_IMAGES:
            print(f"  Render complete — {n} images found.\n")
            return
        time.sleep(poll_interval)


def build_merged_dataset():
    """Run generate_dataset.py --merge-only to symlink and write YAML."""
    import subprocess
    script = ROOT / "scripts" / "blenderproc_pipeline" / "generate_dataset.py"
    print("  Building merged dataset (symlinks + dataset_with_synthetic.yaml)...")
    result = subprocess.run(
        [sys.executable, str(script), "--merge-only"],
        cwd=ROOT, check=False
    )
    if result.returncode != 0:
        print("  [WARN] generate_dataset.py --merge-only returned non-zero exit code")
    if not SYNTHETIC_YAML.exists():
        raise RuntimeError(f"Expected {SYNTHETIC_YAML} to exist after merge step")
    print(f"  Merged dataset YAML: {SYNTHETIC_YAML}\n")


def patch_config_for_laptop(cfg: dict) -> dict:
    """Return a deep copy of cfg with laptop overrides applied."""
    patched = copy.deepcopy(cfg)
    for section, overrides in LAPTOP_OVERRIDES.items():
        if section not in patched:
            patched[section] = {}
        patched[section].update(overrides)
    # Point dataset root to the merged dataset yaml
    # distill.py builds: Path(cfg["dataset"]["root"]) / "dataset.yaml"
    # So we write a "dataset.yaml" symlink in data/ that points to the synthetic one
    patched["dataset"]["root"] = str(ROOT / "data")
    return patched


def swap_dataset_yaml(restore=False):
    """
    distill.py hardcodes <dataset.root>/dataset.yaml.
    We swap data/dataset.yaml → data/dataset_with_synthetic.yaml for the run,
    then restore afterwards.
    """
    orig  = ROOT / "data" / "dataset.yaml"
    synth = ROOT / "data" / "dataset_with_synthetic.yaml"
    backup = ROOT / "data" / "dataset_original.yaml.bak"

    if restore:
        if backup.exists():
            orig.unlink(missing_ok=True)
            backup.rename(orig)
            print("  Restored data/dataset.yaml")
        return

    if not synth.exists():
        raise RuntimeError(f"Synthetic YAML not found: {synth}")

    # Back up original
    if orig.exists() and not backup.exists():
        import shutil
        shutil.copy2(orig, backup)
        print(f"  Backed up data/dataset.yaml → dataset_original.yaml.bak")

    # Overwrite with synthetic merged yaml content
    import shutil
    shutil.copy2(synth, orig)
    print(f"  Swapped data/dataset.yaml → dataset_with_synthetic.yaml content")


def run_training(cfg: dict):
    """
    Import and run Step 9 distillation directly using the patched config.
    Avoids subprocess so GPU memory isn't fragmented by a child process.
    """
    sys.path.insert(0, str(ROOT))
    from src.functions.distill import train_student

    st_cfg    = cfg["student"]
    teachers  = st_cfg.get("teachers", [])

    # Filter to existing teacher checkpoints
    existing_teachers = [t for t in teachers if Path(t).exists()]
    if not existing_teachers:
        raise RuntimeError(
            f"No teacher checkpoints found. Expected one of:\n"
            + "\n".join(f"  {t}" for t in teachers)
        )

    print(f"\n{'═'*60}")
    print(f"  Student Distillation — Laptop Run")
    print(f"{'═'*60}")
    print(f"  Dataset  : data/dataset_with_synthetic.yaml (merged)")
    print(f"  Model    : {st_cfg['model']}")
    print(f"  Epochs   : {st_cfg['epochs']}")
    print(f"  Batch    : {st_cfg['batch_size']}")
    print(f"  Teachers : {len(existing_teachers)}")
    for t in existing_teachers:
        print(f"    {t}")
    print(f"{'═'*60}\n")

    train_student(cfg, existing_teachers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip render wait; assume rendering is done")
    parser.add_argument("--config", default="config/hyperparams.yaml")
    args = parser.parse_args()

    # Load base config
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Apply laptop overrides
    cfg = patch_config_for_laptop(cfg)

    print(f"\n  Laptop training config:")
    for k, v in cfg["student"].items():
        print(f"    student.{k}: {v}")

    # Wait for render
    if not args.no_wait:
        n = len(list(SYN_IMAGES.glob("*.jpg"))) if SYN_IMAGES.exists() else 0
        if n < TARGET_IMAGES:
            wait_for_render()
        else:
            print(f"  Render already done ({n} images). Proceeding.\n")
    else:
        print("  --no-wait: skipping render check.\n")

    # Build merged dataset
    build_merged_dataset()

    # Swap dataset.yaml to point at synthetic merged set
    try:
        swap_dataset_yaml(restore=False)
        run_training(cfg)
    finally:
        # Always restore original dataset.yaml
        swap_dataset_yaml(restore=True)


if __name__ == "__main__":
    main()
