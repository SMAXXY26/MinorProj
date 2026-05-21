"""
split_dataset.py  —  Creates train/val split from a flat data directory.
Usage:
    python split_dataset.py --data data --ratio 0.8
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def find_pairs(data_dir: str):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs    = []

    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")

    if os.path.isdir(img_dir):
        for fname in sorted(os.listdir(img_dir)):
            if Path(fname).suffix.lower() not in img_exts:
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, Path(fname).stem + ".txt")
            lbl_path = lbl_path if os.path.exists(lbl_path) else None
            pairs.append((img_path, lbl_path))
        print(f"[Layout A] Found {len(pairs)} images in {img_dir}")
        return pairs

    for fname in sorted(os.listdir(data_dir)):
        if Path(fname).suffix.lower() not in img_exts:
            continue
        img_path = os.path.join(data_dir, fname)
        lbl_path = os.path.join(data_dir, Path(fname).stem + ".txt")
        lbl_path = lbl_path if os.path.exists(lbl_path) else None
        pairs.append((img_path, lbl_path))

    print(f"[Layout B] Found {len(pairs)} images in {data_dir}")
    return pairs


def split_and_copy(data_dir: str, ratio: float, seed: int = 42):
    pairs = find_pairs(data_dir)

    if len(pairs) == 0:
        print("ERROR: No images found. Check your data directory path.")
        return

    random.seed(seed)
    random.shuffle(pairs)

    n_train     = int(len(pairs) * ratio)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    print(f"Split → train: {len(train_pairs)}  |  val: {len(val_pairs)}")

    out_dirs = {
        "train_img": os.path.join(data_dir, "images", "train"),
        "train_lbl": os.path.join(data_dir, "labels", "train"),
        "val_img":   os.path.join(data_dir, "images", "val"),
        "val_lbl":   os.path.join(data_dir, "labels", "val"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    def copy_pair(img_path, lbl_path, img_dst_dir, lbl_dst_dir):
        shutil.copy2(img_path, os.path.join(img_dst_dir, Path(img_path).name))
        lbl_dst = os.path.join(lbl_dst_dir, Path(img_path).stem + ".txt")
        if lbl_path and os.path.exists(lbl_path):
            shutil.copy2(lbl_path, lbl_dst)
        else:
            open(lbl_dst, "w").close()

    print("Copying train set ...")
    for img_path, lbl_path in train_pairs:
        copy_pair(img_path, lbl_path, out_dirs["train_img"], out_dirs["train_lbl"])

    print("Copying val set ...")
    for img_path, lbl_path in val_pairs:
        copy_pair(img_path, lbl_path, out_dirs["val_img"], out_dirs["val_lbl"])

    print(f"\nDone!")
    print(f"  data/images/train/ → {len(os.listdir(out_dirs['train_img']))} images")
    print(f"  data/images/val/   → {len(os.listdir(out_dirs['val_img']))} images")
    print(f"\nNext step: python train.py --stage detector --config config/hyperparams.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str,   default="data")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()
    split_and_copy(args.data, args.ratio, args.seed)
