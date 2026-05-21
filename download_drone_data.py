"""
download_drone_data.py — Download aerial/drone datasets for fine-tuning
========================================================================
Downloads two types of data to improve drone-mounted weapon detection:

1. VisDrone dataset (aerial perspective — people/vehicles from above)
   - Teaches the model drone geometry, small-object scale, top-down views
   - We extract only "pedestrian" and "people" classes as background context

2. Additional weapon datasets from Roboflow with varied angles
   - CCTV weapon detection (overhead cameras — closest to drone angle)
   - General weapon datasets for more knife/pistol/rifle variety

Usage:
    1. Set ROBOFLOW_API_KEY in .env (needed for Roboflow datasets)
    2. python download_drone_data.py
    3. python download_drone_data.py --visdrone-only    # skip Roboflow
    4. python download_drone_data.py --roboflow-only    # skip VisDrone
"""

import os
import sys
import shutil
import argparse
import random
import yaml
from pathlib import Path

DATA_ROOT   = Path("data")
TRAIN_IMG   = DATA_ROOT / "images" / "train"
TRAIN_LBL   = DATA_ROOT / "labels" / "train"
VAL_IMG     = DATA_ROOT / "images" / "val"
VAL_LBL     = DATA_ROOT / "labels" / "val"
TEMP_DIR    = Path("/tmp/drone_data_downloads")

OUR_CLASSES = ["knife", "pistol", "rifle"]


# ═══════════════════════════════════════════════════════════════════════
#  VisDrone Download & Processing
# ═══════════════════════════════════════════════════════════════════════

def download_visdrone():
    """
    Download VisDrone-DET dataset using Ultralytics built-in support.
    VisDrone has 10 classes — we use the images as-is for aerial perspective
    but generate empty label files (no weapon labels) so YOLO treats them
    as hard negatives / background aerial images.

    Alternatively, we can use the full VisDrone annotations to first
    pre-train on aerial object detection, then fine-tune on weapons.
    """
    print("\n" + "=" * 60)
    print("  Downloading VisDrone dataset (aerial drone imagery)")
    print("=" * 60)

    visdrone_dir = TEMP_DIR / "visdrone"
    visdrone_dir.mkdir(parents=True, exist_ok=True)

    # VisDrone-DET URLs (official mirrors)
    urls = {
        "train_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "val_images":   "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    }

    import urllib.request
    import zipfile

    for name, url in urls.items():
        zip_path = visdrone_dir / f"{name}.zip"
        if zip_path.exists():
            print(f"  {name} already downloaded, skipping")
            continue

        print(f"  Downloading {name}...")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            print(f"  Done: {zip_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"  [ERROR] Failed to download {name}: {e}")
            print(f"  You can manually download from: {url}")
            return 0

    # Extract
    for name in urls:
        zip_path = visdrone_dir / f"{name}.zip"
        extract_dir = visdrone_dir / name
        if extract_dir.exists():
            print(f"  {name} already extracted")
            continue
        print(f"  Extracting {name}...")
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extractall(str(visdrone_dir))
        print(f"  Extracted")

    return merge_visdrone(visdrone_dir)


def merge_visdrone(visdrone_dir, max_images=2000):
    """
    Take a random sample of VisDrone aerial images and add them
    as background/negative images with empty labels.

    This teaches the model:
    - What drone-view scenes look like (so it doesn't hallucinate weapons)
    - Small object scale from aerial perspective
    - Top-down geometry
    """
    print(f"\n  Merging VisDrone images as aerial backgrounds (max {max_images})...")

    added = 0

    # Find all VisDrone images across train/val
    all_images = []
    for subdir in visdrone_dir.rglob("images"):
        for img in subdir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                all_images.append(img)

    if not all_images:
        # Try finding images directly in VisDrone extracted dirs
        for subdir in visdrone_dir.iterdir():
            if subdir.is_dir():
                for img in subdir.rglob("*"):
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        all_images.append(img)

    print(f"  Found {len(all_images)} VisDrone images")

    if not all_images:
        print("  [WARN] No VisDrone images found after extraction")
        return 0

    # Random sample
    random.seed(42)
    sample = random.sample(all_images, min(max_images, len(all_images)))

    for img_path in sample:
        new_name = f"visdrone_{img_path.name}"

        # 80/20 train/val split
        if random.random() < 0.2:
            dst_img = VAL_IMG / new_name
            dst_lbl = VAL_LBL / f"visdrone_{img_path.stem}.txt"
        else:
            dst_img = TRAIN_IMG / new_name
            dst_lbl = TRAIN_LBL / f"visdrone_{img_path.stem}.txt"

        if dst_img.exists():
            continue

        shutil.copy2(str(img_path), str(dst_img))
        # Empty label file = background image (no weapons present)
        dst_lbl.write_text("")
        added += 1

    print(f"  Added {added} VisDrone aerial background images")
    return added


# ═══════════════════════════════════════════════════════════════════════
#  Roboflow Weapon Datasets (overhead/CCTV angles)
# ═══════════════════════════════════════════════════════════════════════

ROBOFLOW_DATASETS = [
    # CCTV overhead weapon detection — closest to drone perspective
    {
        "workspace": "weapon-detection-cctv",
        "project":   "weapon-detection-cctv-v3-dataset",
        "version":   1,
        "class_map": {
            "knife": 0, "Knife": 0,
            "pistol": 1, "Pistol": 1, "handgun": 1, "Handgun": 1,
            "rifle": 2, "Rifle": 2, "shotgun": 2, "Shotgun": 2,
        },
        "desc": "CCTV overhead weapon detection (11 classes)",
    },
    # Weapon detection with multiple versions — varied angles
    {
        "workspace": "yolov7test-u13vc",
        "project":   "weapon-detection-m7qso",
        "version":   16,
        "class_map": {
            "knife": 0, "Knife": 0,
            "pistol": 1, "Pistol": 1, "gun": 1, "Gun": 1, "handgun": 1,
            "rifle": 2, "Rifle": 2, "shotgun": 2, "Shotgun": 2,
            "weapon": 1,
        },
        "desc": "Weapon detection (multi-class, varied angles)",
    },
    # Gun detection dataset
    {
        "workspace": "em2023",
        "project":   "gun-detection-s5poj",
        "version":   1,
        "class_map": {
            "pistol": 1, "Pistol": 1, "gun": 1, "Gun": 1, "handgun": 1,
            "Handgun": 1, "rifle": 2, "Rifle": 2,
        },
        "desc": "Gun detection from varied angles",
    },
]


def download_roboflow_weapons():
    """Download weapon datasets from Roboflow."""
    print("\n" + "=" * 60)
    print("  Downloading Roboflow weapon datasets")
    print("=" * 60)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("  [ERROR] Set ROBOFLOW_API_KEY in .env first")
        print("  Get a free key at: https://app.roboflow.com/settings/api")
        return 0

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_anns = 0

    for ds in ROBOFLOW_DATASETS:
        ws   = ds["workspace"]
        proj = ds["project"]
        ver  = ds["version"]

        print(f"\n  Downloading: {ws}/{proj} v{ver}")
        print(f"  ({ds['desc']})")

        try:
            project = rf.workspace(ws).project(proj)
            dataset = project.version(ver).download(
                "yolov8",
                location=str(TEMP_DIR / f"{proj}_v{ver}"),
                overwrite=True,
            )
            ds_path = Path(dataset.location)
        except Exception as e:
            print(f"  [WARN] Failed to download {proj}: {e}")
            continue

        imgs, anns = merge_roboflow_dataset(ds_path, ds["class_map"], f"drone_{proj[:12]}")
        total_imgs += imgs
        total_anns += anns
        print(f"  -> Added {imgs} images, {anns} annotations")

    return total_imgs


def merge_roboflow_dataset(ds_path, class_map, prefix):
    """Merge a downloaded Roboflow dataset into our data structure."""
    added_imgs = 0
    added_anns = 0

    for split_name in ["train", "valid", "test"]:
        img_dir = ds_path / split_name / "images"
        lbl_dir = ds_path / split_name / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        # Read source data.yaml for class names
        data_yaml = ds_path / "data.yaml"
        src_names = {}
        if data_yaml.exists():
            with open(data_yaml) as f:
                dc = yaml.safe_load(f)
                if "names" in dc:
                    if isinstance(dc["names"], list):
                        src_names = {i: n for i, n in enumerate(dc["names"])}
                    elif isinstance(dc["names"], dict):
                        src_names = {int(k): v for k, v in dc["names"].items()}

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

                    our_cls = None
                    for map_name, map_idx in class_map.items():
                        if src_name == map_name.lower():
                            our_cls = map_idx
                            break
                    if our_cls is None:
                        for map_name, map_idx in class_map.items():
                            if map_name.lower() in src_name:
                                our_cls = map_idx
                                break

                    if our_cls is not None:
                        new_lines.append(f"{our_cls} {' '.join(parts[1:])}")

            if not new_lines:
                continue

            # 80/20 train/val split
            if i % 5 == 0:
                dst_img_dir = VAL_IMG
                dst_lbl_dir = VAL_LBL
            else:
                dst_img_dir = TRAIN_IMG
                dst_lbl_dir = TRAIN_LBL

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


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download drone/aerial datasets")
    parser.add_argument("--visdrone-only", action="store_true",
                        help="Only download VisDrone (skip Roboflow)")
    parser.add_argument("--roboflow-only", action="store_true",
                        help="Only download Roboflow weapons (skip VisDrone)")
    parser.add_argument("--max-visdrone", type=int, default=2000,
                        help="Max VisDrone images to sample (default: 2000)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Drone Data Downloader — Aerial Perspective Fine-tuning")
    print(f"{'=' * 60}")

    # Ensure dirs exist
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    total_added = 0

    if not args.roboflow_only:
        total_added += download_visdrone()

    if not args.visdrone_only:
        total_added += download_roboflow_weapons()

    # Cleanup
    print(f"\n  Cleaning up temp files...")
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

    # Delete stale caches
    for cache in [
        DATA_ROOT / "labels" / "train.cache",
        DATA_ROOT / "labels" / "val.cache",
    ]:
        if cache.exists():
            cache.unlink()
            print(f"  Deleted stale cache: {cache}")

    print(f"\n{'=' * 60}")
    print(f"  Done! Total images added: {total_added}")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. Boost aerial augmentations in config/hyperparams.yaml")
    print(f"    2. Re-train: python part1.py --config config/hyperparams.yaml")
    print(f"    3. Or re-run full pipeline: bash run_pipeline.sh")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
