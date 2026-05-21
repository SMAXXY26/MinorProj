"""
download_roboflow_data.py — Download knife & rifle datasets from Roboflow
=========================================================================
Downloads additional knife and rifle images from public Roboflow Universe
datasets to balance the training data.

Current imbalance:
  knife:  1,622 annotations (1.0x)
  pistol: 9,841 annotations (6.1x)
  rifle:  3,066 annotations (1.9x)

Target: ~6,000+ knife and ~5,000+ rifle annotations

Usage:
    1. Paste your Roboflow API key in .env
    2. python download_roboflow_data.py
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    print("[ERROR] Set your ROBOFLOW_API_KEY in .env first")
    sys.exit(1)

from roboflow import Roboflow

DATA_ROOT   = Path("data")
TRAIN_IMG   = DATA_ROOT / "images" / "train"
TRAIN_LBL   = DATA_ROOT / "labels" / "train"
VAL_IMG     = DATA_ROOT / "images" / "val"
VAL_LBL     = DATA_ROOT / "labels" / "val"
TEMP_DIR    = Path("/tmp/roboflow_downloads")

# Class name mapping — map source dataset class names to our classes
# Our classes: 0=knife, 1=pistol, 2=rifle
OUR_CLASSES = ["knife", "pistol", "rifle"]

# ═══════════════════════════════════════════════════════════════════════
#  Roboflow datasets to download
#  Format: (workspace, project, version, class_mapping)
#  class_mapping: {source_class_name: our_class_index}
# ═══════════════════════════════════════════════════════════════════════

DATASETS = [
    # ── Knife datasets ────────────────────────────────────────────────
    {
        "workspace": "laserworkspace",
        "project":   "knife_bottle_cup",
        "version":   1,
        "class_map": {"knife": 0, "Knife": 0},
        "target":    "knife",
    },
    {
        "workspace": "cao-jkghk",
        "project":   "knife-aqe0g",
        "version":   3,
        "class_map": {"knife": 0, "Knife": 0},
        "target":    "knife",
    },
    {
        "workspace": "motos-and-cars",
        "project":   "knife-sd3xq",
        "version":   3,
        "class_map": {"knife": 0, "Knife": 0},
        "target":    "knife",
    },
    {
        "workspace": "labelling-7k1aj",
        "project":   "knife-skqnq",
        "version":   1,
        "class_map": {"knife": 0, "Knife": 0, "knife_handle": 0},
        "target":    "knife",
    },
    # ── Rifle datasets ────────────────────────────────────────────────
    {
        "workspace": "doken-edgar",
        "project":   "rifle-xr2aa",
        "version":   1,
        "class_map": {"rifle": 2, "Rifle": 2},
        "target":    "rifle",
    },
    {
        "workspace": "model-training-iwx9e",
        "project":   "handgun-longgun_surv_v2-o0rky",
        "version":   2,
        "class_map": {"rifle": 2, "Rifle": 2, "shotgun": 2, "Shotgun": 2},
        "target":    "rifle",
    },
    {
        "workspace": "weapon-detection-frwne",
        "project":   "rifle-1b8vx",
        "version":   2,
        "class_map": {"heavygun": 2, "rifle": 2, "Rifle": 2},
        "target":    "rifle",
    },
]


def count_current_annotations():
    """Count current per-class annotations in training set."""
    counts = {name: 0 for name in OUR_CLASSES}
    lbl_dir = TRAIN_LBL
    if not lbl_dir.exists():
        return counts
    for f in lbl_dir.iterdir():
        if f.suffix != ".txt":
            continue
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_idx = int(parts[0])
                    if cls_idx < len(OUR_CLASSES):
                        counts[OUR_CLASSES[cls_idx]] += 1
    return counts


def download_dataset(ds_info, rf):
    """Download a single Roboflow dataset and return the path."""
    ws   = ds_info["workspace"]
    proj = ds_info["project"]
    ver  = ds_info["version"]

    print(f"\n  Downloading: {ws}/{proj} v{ver}")
    try:
        project = rf.workspace(ws).project(proj)
        dataset = project.version(ver).download(
            "yolov8",
            location=str(TEMP_DIR / f"{proj}_v{ver}"),
            overwrite=True,
        )
        return Path(dataset.location)
    except Exception as e:
        print(f"  [WARN] Failed to download {proj}: {e}")
        return None


def merge_dataset(ds_path, class_map, prefix):
    """
    Merge downloaded dataset into our data structure.
    Remaps class indices according to class_map.
    """
    added_imgs = 0
    added_anns = 0

    # Find train/valid dirs in the downloaded dataset
    for split_name in ["train", "valid", "test"]:
        img_dir = ds_path / split_name / "images"
        lbl_dir = ds_path / split_name / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        # Read the dataset's data.yaml to understand class mapping
        data_yaml = ds_path / "data.yaml"
        src_names = {}
        if data_yaml.exists():
            with open(data_yaml) as f:
                dc = yaml.safe_load(f)
                if "names" in dc:
                    if isinstance(dc["names"], list):
                        src_names = {i: n for i, n in enumerate(dc["names"])}
                    elif isinstance(dc["names"], dict):
                        src_names = dc["names"]

        # Determine target split (80% train, 20% val)
        img_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        for i, img_file in enumerate(img_files):
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Remap labels
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    src_cls = int(parts[0])
                    src_name = src_names.get(src_cls, "").lower()

                    # Find our class index
                    our_cls = None
                    for map_name, map_idx in class_map.items():
                        if src_name == map_name.lower():
                            our_cls = map_idx
                            break
                    # Also try matching by source class index directly
                    if our_cls is None:
                        for map_name, map_idx in class_map.items():
                            if map_name.lower() in src_name:
                                our_cls = map_idx
                                break

                    if our_cls is not None:
                        new_lines.append(
                            f"{our_cls} {' '.join(parts[1:])}"
                        )

            if not new_lines:
                continue

            # Decide train vs val (80/20)
            if i % 5 == 0:
                dst_img_dir = VAL_IMG
                dst_lbl_dir = VAL_LBL
            else:
                dst_img_dir = TRAIN_IMG
                dst_lbl_dir = TRAIN_LBL

            # Copy with unique prefix to avoid name collisions
            new_name = f"{prefix}_{img_file.name}"
            dst_img = dst_img_dir / new_name
            dst_lbl = dst_lbl_dir / (f"{prefix}_{img_file.stem}.txt")

            if not dst_img.exists():
                shutil.copy2(str(img_file), str(dst_img))
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                added_imgs += 1
                added_anns += len(new_lines)

    return added_imgs, added_anns


def main():
    print(f"\n{'═'*60}")
    print(f"  Roboflow Data Downloader — Knife & Rifle Augmentation")
    print(f"{'═'*60}")

    # Current counts
    counts = count_current_annotations()
    print(f"\n  Current annotations:")
    for name, count in counts.items():
        print(f"    {name:10s}: {count}")

    # Init Roboflow
    rf = Roboflow(api_key=API_KEY)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_anns = 0

    for ds in DATASETS:
        ds_path = download_dataset(ds, rf)
        if ds_path is None:
            continue

        prefix = f"rf_{ds['project'][:15]}"
        imgs, anns = merge_dataset(ds_path, ds["class_map"], prefix)
        total_imgs += imgs
        total_anns += anns
        print(f"  → Added {imgs} images, {anns} annotations ({ds['target']})")

    # Updated counts
    new_counts = count_current_annotations()
    print(f"\n{'═'*60}")
    print(f"  Done!")
    print(f"  Images added  : {total_imgs}")
    print(f"  Annotations   : {total_anns}")
    print(f"\n  Updated counts:")
    for name in OUR_CLASSES:
        delta = new_counts[name] - counts[name]
        print(f"    {name:10s}: {counts[name]} → {new_counts[name]}  (+{delta})")
    print(f"{'═'*60}")

    # Cleanup temp
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

    print(f"\n  Next: Re-run python part1.py to retrain with balanced data")


if __name__ == "__main__":
    main()
