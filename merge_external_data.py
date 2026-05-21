"""
merge_external_data.py — Merge OD-WeaponDetection into our dataset
====================================================================
Merges images from /home/vinesh/ML/assets/OD-WeaponDetection/ into
our data/images/{train,val} and data/labels/{train,val}.

Data sources used:
  1. Knife_detection/   — 2078 knife images + Pascal VOC XML annotations
                          → Converted to YOLO format, mapped to class 0 (knife)
  2. Sohas_weapon-Detection-YOLOv5/ — 5002 images already in YOLO format
                          → Classes remapped: 0→pistol(1), 2→knife(0)
                            Only pistol, knife classes taken; rest skipped
  3. Knife classification/Knife + ak47 — ~800 classification images
                          → Copied as negatives (no bbox) are not useful for
                            detection, but background folders ARE useful
  4. Background/negative images from classification folders
                          → 458 + 467 BACKGROUND_Google + other categories
                            Copied to data/negatives/images/ for FP correction

Usage:
    python merge_external_data.py
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

EXTERNAL_ROOT = Path("/home/vinesh/ML/assets/OD-WeaponDetection")
DATA_ROOT     = Path("data")
TRAIN_IMG     = DATA_ROOT / "images" / "train"
TRAIN_LBL     = DATA_ROOT / "labels" / "train"
VAL_IMG       = DATA_ROOT / "images" / "val"
VAL_LBL       = DATA_ROOT / "labels" / "val"
NEG_DIR       = DATA_ROOT / "negatives" / "images"

# Our class mapping: 0=knife, 1=pistol, 2=rifle
OUR_CLASSES = ["knife", "pistol", "rifle"]

# Sohas YOLO class mapping: their idx → our idx
# Their classes: ['pistol', 'smartphone', 'knife', 'monedero', 'billete', 'tarjeta']
SOHAS_MAP = {
    0: 1,  # pistol → pistol
    2: 0,  # knife  → knife
    # 1,3,4,5 are smartphone, wallet, bill, card → skip
}

random.seed(42)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dirs():
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, NEG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def count_existing():
    """Count current per-class annotations."""
    counts = {n: 0 for n in OUR_CLASSES}
    for f in TRAIN_LBL.iterdir():
        if f.suffix != ".txt":
            continue
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    ci = int(parts[0])
                    if ci < len(OUR_CLASSES):
                        counts[OUR_CLASSES[ci]] += 1
    return counts


def copy_image(src, dst_dir, new_name):
    """Copy image file, return True if successful."""
    dst = dst_dir / new_name
    if dst.exists():
        return False
    shutil.copy2(str(src), str(dst))
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Source 1: Knife_detection (Pascal VOC → YOLO)
# ═══════════════════════════════════════════════════════════════════════

def merge_knife_detection():
    """Convert Pascal VOC knife annotations to YOLO and merge."""
    src_img = EXTERNAL_ROOT / "Knife_detection" / "Images"
    src_ann = EXTERNAL_ROOT / "Knife_detection" / "annotations"

    if not src_img.exists() or not src_ann.exists():
        print("  [SKIP] Knife_detection not found")
        return 0, 0

    added_imgs = 0
    added_anns = 0

    img_files = sorted([
        f for f in src_img.iterdir()
        if f.suffix.lower() in IMG_EXTS
    ])

    for i, img_file in enumerate(img_files):
        xml_file = src_ann / (img_file.stem + ".xml")
        if not xml_file.exists():
            continue

        # Parse VOC XML
        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Get image dimensions
        size_el = root.find("size")
        if size_el is None:
            continue
        W = int(size_el.find("width").text)
        H = int(size_el.find("height").text)
        if W <= 0 or H <= 0:
            continue

        # Convert bounding boxes
        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if "knife" not in name and "blade" not in name:
                continue

            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Clamp
            xmin = max(0, min(xmin, W))
            ymin = max(0, min(ymin, H))
            xmax = max(0, min(xmax, W))
            ymax = max(0, min(ymax, H))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert to YOLO format (cx, cy, w, h) normalized
            cx = ((xmin + xmax) / 2) / W
            cy = ((ymin + ymax) / 2) / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        # Decide split (80/20)
        if i % 5 == 0:
            img_dst, lbl_dst = VAL_IMG, VAL_LBL
        else:
            img_dst, lbl_dst = TRAIN_IMG, TRAIN_LBL

        new_name = f"kdet_{img_file.name}"
        if copy_image(img_file, img_dst, new_name):
            lbl_file = lbl_dst / f"kdet_{img_file.stem}.txt"
            with open(lbl_file, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            added_imgs += 1
            added_anns += len(yolo_lines)

    return added_imgs, added_anns


# ═══════════════════════════════════════════════════════════════════════
#  Source 2: Sohas YOLO dataset (remap classes)
# ═══════════════════════════════════════════════════════════════════════

def merge_sohas_yolo():
    """Merge the Sohas YOLOv5 dataset with class remapping."""
    base = (EXTERNAL_ROOT / "Weapons and similar handled objects"
            / "Sohas_weapon-Detection-YOLOv5" / "obj_train_data")

    added_imgs = 0
    added_anns = 0

    for split_name in ["train", "test"]:
        src_img_dir = base / "images" / split_name
        src_lbl_dir = base / "labels" / split_name

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        img_files = sorted([
            f for f in src_img_dir.iterdir()
            if f.suffix.lower() in IMG_EXTS
        ])

        for i, img_file in enumerate(img_files):
            lbl_file = src_lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Remap labels — only keep pistol and knife
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    src_cls = int(parts[0])
                    if src_cls in SOHAS_MAP:
                        our_cls = SOHAS_MAP[src_cls]
                        new_lines.append(
                            f"{our_cls} {' '.join(parts[1:])}"
                        )

            if not new_lines:
                continue

            # 80/20 split
            if i % 5 == 0:
                img_dst, lbl_dst = VAL_IMG, VAL_LBL
            else:
                img_dst, lbl_dst = TRAIN_IMG, TRAIN_LBL

            new_name = f"sohas_{img_file.name}"
            if copy_image(img_file, img_dst, new_name):
                out_lbl = lbl_dst / f"sohas_{img_file.stem}.txt"
                with open(out_lbl, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                added_imgs += 1
                added_anns += len(new_lines)

    return added_imgs, added_anns


# ═══════════════════════════════════════════════════════════════════════
#  Source 3: Negative images from classification datasets
# ═══════════════════════════════════════════════════════════════════════

# Folders that are clearly NON-weapon — good negatives
NEGATIVE_FOLDERS = [
    "BACKGROUND_Google", "Faces", "Faces_easy", "Leopards", "Motorbikes",
    "airplanes", "barrel", "bass", "beaver", "binocular", "bonsai",
    "brain", "buddha", "butterfly", "camera", "car_side", "ceiling_fan",
    "chair", "chandelier", "cup", "dalmatian", "dollar_bill", "dolphin",
    "dragonfly", "electric_guitar", "elephant", "emu", "ferry",
    "flamingo", "garfield", "gramophone", "grand_piano", "hawksbill",
    "headphone", "hedgehog", "helicopter", "ibis", "joshua_tree",
    "kangaroo", "ketch", "lamp", "laptop", "llama", "lobster", "lotus",
    "mandolin", "mayfly", "menorah", "metronome", "minaret", "nautilus",
    "octopus", "okapi", "pagoda", "panda", "pigeon", "pizza",
    "platypus", "pyramid", "rhino", "rooster", "saxophone", "schooner",
    "scorpion", "sea_horse", "snoopy", "soccer_ball", "stapler",
    "starfish", "stegosaurus", "stop_sign", "strawberry", "sunflower",
    "trilobite", "umbrella", "watch", "water_lilly", "wheelchair",
    "wild_cat", "windsor_chair", "wrench", "yin_yang",
    # from Pistol classification extras
    "accordion", "anchor", "ant", "brontosaurus", "cellphone",
    "crocodile", "crocodile_head", "inline_skate", "tick",
]

# Skip weapon-related folders
WEAPON_FOLDERS = {"Knife", "ak47", "Pistol", "cannon", "scissors",
                  "baseball-bat", "smartphone", "cigar", "pen", "keys"}


def merge_negatives():
    """Copy negative images from classification datasets."""
    added = 0
    max_per_folder = 15  # limit per folder to avoid flooding

    for classification_dir in [
        "Knife classification",
        "Pistol classification",
    ]:
        base = EXTERNAL_ROOT / classification_dir
        if not base.exists():
            continue

        for folder_name in sorted(os.listdir(base)):
            folder = base / folder_name
            if not folder.is_dir():
                continue
            if folder_name in WEAPON_FOLDERS:
                continue
            if folder_name not in NEGATIVE_FOLDERS:
                continue

            imgs = [
                f for f in folder.iterdir()
                if f.suffix.lower() in IMG_EXTS
            ]
            random.shuffle(imgs)

            for img_file in imgs[:max_per_folder]:
                new_name = f"neg_{classification_dir[:5]}_{folder_name}_{img_file.name}"
                dst = NEG_DIR / new_name
                if not dst.exists():
                    shutil.copy2(str(img_file), str(dst))
                    added += 1

    return added


def main():
    print(f"\n{'═'*60}")
    print(f"  Merging OD-WeaponDetection → our dataset")
    print(f"{'═'*60}")

    ensure_dirs()

    # Current counts
    before = count_existing()
    print(f"\n  Before:")
    for name, count in before.items():
        print(f"    {name:10s}: {count}")

    neg_before = len(list(NEG_DIR.iterdir())) if NEG_DIR.exists() else 0
    print(f"    {'negatives':10s}: {neg_before}")

    # ── Source 1: Knife detection (VOC → YOLO) ───────────────────────
    print(f"\n  [1/3] Knife_detection (Pascal VOC → YOLO)...")
    imgs1, anns1 = merge_knife_detection()
    print(f"        → Added {imgs1} images, {anns1} knife annotations")

    # ── Source 2: Sohas YOLO (remap classes) ──────────────────────────
    print(f"\n  [2/3] Sohas_weapon-Detection-YOLOv5 (class remap)...")
    imgs2, anns2 = merge_sohas_yolo()
    print(f"        → Added {imgs2} images, {anns2} annotations")

    # ── Source 3: Negatives ───────────────────────────────────────────
    print(f"\n  [3/3] Negative images from classification folders...")
    neg_added = merge_negatives()
    print(f"        → Added {neg_added} negative images")

    # Final counts
    after = count_existing()
    neg_after = len(list(NEG_DIR.iterdir())) if NEG_DIR.exists() else 0

    print(f"\n{'═'*60}")
    print(f"  Results:")
    print(f"{'─'*60}")
    print(f"  {'Class':<12} {'Before':>8} {'After':>8} {'Added':>8}")
    print(f"{'─'*60}")
    for name in OUR_CLASSES:
        delta = after[name] - before[name]
        marker = " ✓" if delta > 0 else ""
        print(f"  {name:<12} {before[name]:>8} {after[name]:>8} {'+' + str(delta):>8}{marker}")
    print(f"  {'negatives':<12} {neg_before:>8} {neg_after:>8} {'+'+ str(neg_added):>8}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
