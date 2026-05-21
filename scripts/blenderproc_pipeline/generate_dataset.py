"""
generate_dataset.py  —  Orchestrator for full synthetic aerial weapon dataset
=============================================================================
Calls render_aerial_weapons.py via BlenderProc for each weapon class,
then merges the synthetic images into the existing training set and
generates a YOLO dataset.yaml pointing to the merged data.

Usage:
    # First run download_assets.py, then put .obj models in models/
    python scripts/blenderproc_pipeline/generate_dataset.py

    # Render fewer images for a quick test:
    python scripts/blenderproc_pipeline/generate_dataset.py --n 100

    # Dry run — show what would be rendered, don't actually render:
    python scripts/blenderproc_pipeline/generate_dataset.py --dry-run

Output:
    data/synthetic/images/      ← rendered JPEGs
    data/synthetic/labels/      ← YOLO .txt files
    data/synthetic_merged/      ← symlinks: original train + synthetic merged
    data/dataset_with_synthetic.yaml  ← drop-in replacement for dataset.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent   # project root
SCRIPT_DIR = Path(__file__).resolve().parent
SYN_DIR    = ROOT / "data" / "synthetic"
MERGED_DIR = ROOT / "data" / "synthetic_merged"

# Each class: (class_id, model_filename, n_images_to_render)
#
# Data-driven targets (see step1_baseline analysis):
#   - pistol:  SKIP — already 56% aerial in training (14,431 instances). Distillation R2 fixes this.
#   - knife:   1,500 images — currently only 1.2% aerial (188 instances), need ~10% for impact
#   - rifle:   3,000 images — ZERO aerial instances in training, highest priority
#
# Multiple models per class to avoid overfitting to one geometry:
#   knife  → knife_01.obj, knife_02.obj, knife_03.obj  (500 each)
#   rifle  → rifle_ak47.obj, rifle_ar15.obj, rifle_bolt.obj  (1000 each)
#
# If you only have one model per class, the render count is split across
# the single model and you get less geometric diversity.
CLASSES = [
    # (class_id, model_filename,     n_images)
    (0, "knife_01.obj",   500),   # fixed-blade knife
    (0, "knife_02.obj",   500),   # folding knife
    (0, "knife_03.obj",   500),   # hunting knife
    (2, "rifle_ak47.obj", 1000),  # AK-style rifle
    (2, "rifle_ar15.obj", 1000),  # AR-style rifle
    (2, "rifle_bolt.obj", 1000),  # bolt-action rifle
    # pistol intentionally omitted — already 56% aerial in training
]


def check_models():
    missing = []
    for cls_id, fname, _ in CLASSES:
        p = SCRIPT_DIR / "models" / fname
        if not p.exists():
            missing.append(str(p))
    return missing


def run_blenderproc(cls_id: int, model_file: str, n: int, dry_run: bool):
    model_path = SCRIPT_DIR / "models" / model_file
    cmd = [
        "blenderproc", "run",
        str(SCRIPT_DIR / "render_aerial_weapons.py"),
        "--",
        "--model",    str(model_path),
        "--class-id", str(cls_id),
        "--n",        str(n),
        "--out",      str(SYN_DIR),
        "--img-size", "416",
    ]
    print(f"\n  {'[DRY-RUN] Would run' if dry_run else 'Running'}:")
    print(f"    {' '.join(cmd)}\n")
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  [WARN] BlenderProc exited with code {result.returncode}")


def build_merged_dataset():
    """
    Create data/synthetic_merged/ with symlinks to both original train images
    and synthetic images, then write a new dataset.yaml pointing to it.
    """
    for split in ("images", "labels"):
        (MERGED_DIR / split).mkdir(parents=True, exist_ok=True)

    orig_img = ROOT / "data" / "images" / "train"
    orig_lbl = ROOT / "data" / "labels" / "train"
    syn_img  = SYN_DIR / "images"
    syn_lbl  = SYN_DIR / "labels"

    # Symlink original train images
    n_orig = 0
    for f in orig_img.glob("*.jpg"):
        dest = MERGED_DIR / "images" / f.name
        if not dest.exists():
            dest.symlink_to(f)
            n_orig += 1
    for f in orig_lbl.glob("*.txt"):
        dest = MERGED_DIR / "labels" / f.name
        if not dest.exists():
            dest.symlink_to(f)

    # Symlink synthetic images (prefixed with syn_ to avoid name collisions)
    n_syn = 0
    for f in syn_img.glob("*.jpg"):
        dest = MERGED_DIR / "images" / ("syn_" + f.name)
        if not dest.exists():
            dest.symlink_to(f)
            n_syn += 1
    for f in syn_lbl.glob("*.txt"):
        dest = MERGED_DIR / "labels" / ("syn_" + f.name)
        if not dest.exists():
            dest.symlink_to(f)

    # Write dataset.yaml
    import yaml
    yaml_path = ROOT / "data" / "dataset_with_synthetic.yaml"
    content = {
        "path":  str(ROOT / "data"),
        "train": str(MERGED_DIR / "images"),
        "val":   "images/val",
        "nc":    3,
        "names": ["knife", "pistol", "rifle"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)

    print(f"\n  Merged dataset:")
    print(f"    Original train : {n_orig} images (+ {len(list((MERGED_DIR/'images').glob('*.jpg'))) - n_orig - n_syn} already linked)")
    print(f"    Synthetic      : {n_syn} new images")
    print(f"    Total          : {len(list((MERGED_DIR/'images').glob('*.jpg')))} images")
    print(f"    Dataset YAML   : {yaml_path}")
    print(f"\n  To use in training, set in config/hyperparams.yaml:")
    print(f"    dataset:")
    print(f"      root: data")
    print(f"    And point student.data: data/dataset_with_synthetic.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=None,
                        help="Override n_images for all classes (default: per-class in script)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run, don't actually render")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip rendering, just rebuild the merged dataset dir")
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print(f"  Synthetic Aerial Weapon Dataset Generator")
    print(f"{'═'*65}")

    if not args.merge_only:
        # Check blenderproc is installed
        if subprocess.run(["which", "blenderproc"], capture_output=True).returncode != 0:
            print("\n  ERROR: blenderproc not found.")
            print("  Install with:  pip install blenderproc")
            print("  BlenderProc will download Blender automatically on first run.")
            sys.exit(1)

        # Check models exist
        missing = check_models()
        if missing:
            print(f"\n  Missing 3D models ({len(missing)} files):")
            for m in missing:
                print(f"    {m}")
            print(f"\n  Download free CC0 weapon .obj files and place them in:")
            print(f"    {SCRIPT_DIR / 'models'}/")
            print(f"\n  Good sources:")
            print(f"    https://sketchfab.com  (filter: Free license, format: OBJ)")
            print(f"    https://polyhaven.com/models")
            if not args.dry_run:
                sys.exit(1)

        # Render each class
        for cls_id, model_file, default_n in CLASSES:
            n = args.n if args.n is not None else default_n
            run_blenderproc(cls_id, model_file, n, args.dry_run)

    if not args.dry_run:
        build_merged_dataset()
    else:
        print("\n  [DRY-RUN] Skipping merge step.")

    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()
