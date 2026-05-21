"""
make_sequences.py  —  Create synthetic video sequences from images
Converts your training images into short video clips for temporal training.
Usage: python make_sequences.py
"""
import os
import cv2
import random
import numpy as np
from pathlib import Path
import yaml


def make_weapon_sequences(data_root, out_dir, seq_len=60, fps=15, n_seqs=50):
    """
    Creates synthetic weapon video sequences by:
    1. Picking a random weapon image
    2. Applying motion blur / zoom / brightness variation over frames
    3. Saving as a short video clip
    """
    img_dir  = os.path.join(data_root, "images", "train")
    img_exts = {".jpg", ".jpeg", ".png"}
    images   = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in img_exts
    ]

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Creating {n_seqs} weapon sequences from {len(images)} images")

    for seq_idx in range(n_seqs):
        # Pick random image
        img_path = random.choice(images)
        img      = cv2.imread(img_path)
        if img is None:
            continue

        H, W     = img.shape[:2]
        out_path = os.path.join(out_dir, f"weapon_seq_{seq_idx:03d}.mp4")
        writer   = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )

        for frame_idx in range(seq_len):
            frame = img.copy()

            # Simulate camera motion — slight random crop + resize
            crop_margin = random.randint(0, 20)
            if crop_margin > 0 and H > crop_margin*2 and W > crop_margin*2:
                dx = random.randint(0, crop_margin)
                dy = random.randint(0, crop_margin)
                frame = frame[dy:H-crop_margin+dy, dx:W-crop_margin+dx]
                frame = cv2.resize(frame, (W, H))

            # Brightness variation
            beta  = random.randint(-20, 20)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

            # Occasional motion blur
            if random.random() < 0.2:
                k     = random.choice([3, 5])
                frame = cv2.GaussianBlur(frame, (k, k), 0)

            writer.write(frame)

        writer.release()

    print(f"  Weapon sequences saved to {out_dir}")


def make_negative_sequences_from_val(data_root, out_dir,
                                      seq_len=60, fps=15, n_seqs=30):
    """
    Creates synthetic negative sequences using val images
    that have NO weapon labels (empty label files).
    """
    img_dir = os.path.join(data_root, "images", "val")
    lbl_dir = os.path.join(data_root, "labels", "val")
    img_exts = {".jpg", ".jpeg", ".png"}

    # Find images with empty label files
    neg_images = []
    for f in os.listdir(img_dir):
        if Path(f).suffix.lower() not in img_exts:
            continue
        lbl = os.path.join(lbl_dir, Path(f).stem + ".txt")
        if os.path.exists(lbl):
            content = open(lbl).read().strip()
            if content == "":
                neg_images.append(os.path.join(img_dir, f))

    if len(neg_images) == 0:
        print(f"  [WARN] No empty label images found in val set")
        print(f"  All val images have weapon labels — skipping synthetic negatives")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Creating {min(n_seqs, len(neg_images))} negative sequences "
          f"from {len(neg_images)} weapon-free val images")

    for seq_idx in range(min(n_seqs, len(neg_images))):
        img_path = neg_images[seq_idx]
        img      = cv2.imread(img_path)
        if img is None:
            continue

        H, W     = img.shape[:2]
        out_path = os.path.join(out_dir, f"negative_seq_{seq_idx:03d}.mp4")
        writer   = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )

        for _ in range(seq_len):
            frame = img.copy()
            beta  = random.randint(-15, 15)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
            writer.write(frame)

        writer.release()

    print(f"  Negative sequences saved to {out_dir}")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/hyperparams.yaml"))
    root = os.path.join("..", cfg["dataset"]["root"])

    print("\n  Creating synthetic sequences for temporal training\n")

    # Weapon sequences
    make_weapon_sequences(
        data_root = root,
        out_dir   = "../data/sequences/weapon_clips",
        n_seqs    = 50,
    )

    # Negative sequences from val images
    make_negative_sequences_from_val(
        data_root = root,
        out_dir   = "../data/sequences/negative_clips",
        n_seqs    = 30,
    )

    print("\n  Done! Now run: python part3.py --sequences data/sequences/")
