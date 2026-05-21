import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.augmentations import apply_augmentations

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DetectorDataset(Dataset):
    """
    Expects:
        root/images/  — .jpg/.png files
        root/labels/  — YOLO-format .txt files (class cx cy w h [angle])
    """
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=False, aug_cfg=None):
        self.img_size = img_size
        self.augment  = augment
        self.aug_cfg  = aug_cfg  # from hyperparams.yaml -> augmentation
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples  = [
            (os.path.join(img_dir, f),
             os.path.join(lbl_dir, Path(f).stem + ".txt"))
            for f in sorted(os.listdir(img_dir))
            if Path(f).suffix.lower() in exts
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)

        img = cv2.resize(img, (self.img_size, self.img_size))

        # --- Load labels early so augmentation can adjust them ---
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) >= 5:
                        labels.append(vals)
        lbl_np = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        # --- Apply augmentations (on uint8 BGR, before normalisation) ---
        if self.augment and self.aug_cfg is not None:
            img, lbl_np = apply_augmentations(img, lbl_np, self.aug_cfg)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        labels_t = torch.from_numpy(lbl_np).float() if lbl_np.shape[0] > 0 \
                   else torch.zeros((0, 5))
        return tensor, labels_t


class ClassifierDataset(Dataset):
    """
    Folder structure:  root/<class_name>/<image_files>
    """
    def __init__(self, root, class_names, img_size=224):
        self.img_size    = img_size
        self.class_names = class_names
        self.samples     = []
        exts = {".jpg", ".jpeg", ".png"}
        for idx, name in enumerate(class_names):
            cls_dir = os.path.join(root, name)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if Path(f).suffix.lower() in exts:
                    self.samples.append((os.path.join(cls_dir, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), label


class TemporalDetectionDataset(Dataset):
    def __init__(self, sequences, window_size=16):
        self.sequences   = sequences
        self.window_size = window_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        feats  = torch.tensor(seq["features"], dtype=torch.float32)
        labels = torch.tensor(seq["labels"],   dtype=torch.long)
        T = self.window_size
        if feats.shape[0] >= T:
            feats  = feats[:T]
            labels = labels[:T]
        else:
            pad = T - feats.shape[0]
            feats  = torch.cat([feats,  torch.zeros(pad, feats.shape[1])], 0)
            labels = torch.cat([labels, torch.zeros(pad, dtype=torch.long)], 0)
        return feats, labels


def collate_fn(batch):
    """
    Custom collate that keeps labels as a list instead of stacking them.
    This handles images with different numbers of bounding boxes.
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)   # stack images normally — all same size
    # labels stay as a tuple/list — each item is a different-sized tensor
    return imgs, list(labels)


def build_detector_loaders(cfg, extra_neg_dir=None):
    import yaml

    data_yaml = os.path.join(cfg["dataset"]["root"], "data.yaml")

    if os.path.exists(data_yaml):
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)

        root      = cfg["dataset"]["root"]
        train_img = os.path.normpath(os.path.join(root, data_cfg["train"]))
        val_img   = os.path.normpath(os.path.join(root, data_cfg["val"]))
        train_lbl = train_img.replace("images", "labels")
        val_lbl   = val_img.replace("images", "labels")

        if "names" in data_cfg:
            cfg["dataset"]["class_names"] = data_cfg["names"]
            cfg["dataset"]["num_classes"] = len(data_cfg["names"])

        print(f"  [Dataset] Loaded paths from {data_yaml}")
        print(f"  [Dataset] train → {train_img}  ({len(os.listdir(train_img))} images)")
        print(f"  [Dataset] val   → {val_img}  ({len(os.listdir(val_img))} images)")
    else:
        root      = cfg["dataset"]["root"]
        train_img = os.path.join(root, "images", "train")
        train_lbl = os.path.join(root, "labels", "train")
        val_img   = os.path.join(root, "images", "val")
        val_lbl   = os.path.join(root, "labels", "val")

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    img_size = cfg["detector"]["img_size"]
    bs       = cfg["detector"]["batch_size"]

    aug_cfg  = cfg.get("augmentation", None)
    train_ds = DetectorDataset(train_img, train_lbl, img_size, augment=True,  aug_cfg=aug_cfg)
    val_ds   = DetectorDataset(val_img,   val_lbl,   img_size, augment=False, aug_cfg=None)

    train_loader = DataLoader(
        train_ds, bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,      # ← fix applied here
    )
    val_loader = DataLoader(
        val_ds, bs * 2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,      # ← fix applied here
    )

    return train_loader, val_loader, None

def build_classifier_loaders(cfg):
    cls_root = cfg["dataset"].get(
        "classifier_root",
        os.path.join(cfg["dataset"]["root"], "classifier_crops"),
    )
    class_names = cfg["dataset"]["class_names"]
    img_size    = cfg["classifier"].get("img_size", 224)
    bs          = cfg["classifier"]["batch_size"]

    full_ds = ClassifierDataset(cls_root, class_names, img_size)
    n_train = int(0.85 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, bs,     shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   bs * 2, shuffle=False, num_workers=4)
    return train_loader, val_loader
