"""
validation.py — Pre-flight checks, data validation, and model evaluation.
===========================================================================
Industry-ready validation steps for the weapon detection training pipeline.

Functions:
    preflight_checks(cfg)            → bool   (environment + GPU + disk + deps)
    validate_dataset(cfg)            → dict   (integrity, balance, corruption scan)
    evaluate_model(cfg, device)      → dict   (full val metrics: mAP, P/R/F1, confusion)
"""

import os
import sys
import time
import json
import platform
import shutil
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from collections import Counter


# =============================================================================
#  Step 0 — Pre-flight Checks
# =============================================================================

def preflight_checks(cfg):
    """
    Run environment validation before training begins.
    Checks: Python, CUDA, GPU memory, disk space, dependencies, config sanity.

    Returns:
        bool: True if all checks pass, False if critical failure.
    """
    print(f"\n{'═'*70}")
    print(f"  Pre-flight Checks")
    print(f"{'═'*70}\n")

    all_ok = True

    # ── Python version ────────────────────────────────────────────────────
    py_ver = platform.python_version()
    py_ok  = sys.version_info >= (3, 9)
    status = "✓" if py_ok else "⚠"
    print(f"  {status} Python        : {py_ver}")
    if not py_ok:
        print(f"    → Recommend Python ≥ 3.9")

    # ── PyTorch + CUDA ────────────────────────────────────────────────────
    torch_ver = torch.__version__
    cuda_ok   = torch.cuda.is_available()
    status    = "✓" if cuda_ok else "✗"
    print(f"  {status} PyTorch       : {torch_ver}")
    print(f"  {status} CUDA          : {'available' if cuda_ok else 'NOT AVAILABLE'}")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated(0)) / 1e9
        print(f"  ✓ GPU           : {gpu_name}")
        print(f"  ✓ GPU Memory    : {gpu_mem:.1f} GB total, ~{free_mem:.1f} GB free")

        if gpu_mem < 4.0:
            print(f"    ⚠ Low GPU memory — reduce batch_size if OOM")
    else:
        print(f"    ✗ CRITICAL: No GPU — training will be extremely slow")
        all_ok = False

    # ── Disk space ────────────────────────────────────────────────────────
    data_root = cfg["dataset"]["root"]
    if os.path.exists(data_root):
        disk_usage = shutil.disk_usage(data_root)
        free_gb    = disk_usage.free / 1e9
        status     = "✓" if free_gb > 5 else "⚠"
        print(f"  {status} Disk space    : {free_gb:.1f} GB free")
        if free_gb < 5:
            print(f"    → Need ≥5 GB for training artefacts")
    else:
        print(f"  ✗ Data root     : {data_root} NOT FOUND")
        all_ok = False

    # ── Dependencies ──────────────────────────────────────────────────────
    deps = {
        "ultralytics": "YOLO detector",
        "cv2":         "Image processing",
        "yaml":        "Config parsing",
        "numpy":       "Numerical ops",
        "torchvision": "Pretrained models",
    }
    missing = []
    for mod, desc in deps.items():
        try:
            __import__(mod)
            # print(f"  ✓ {mod:14s} : {desc}")
        except ImportError:
            print(f"  ✗ {mod:14s} : {desc} — MISSING")
            missing.append(mod)

    if missing:
        print(f"    Install: pip install {' '.join(missing)}")
        all_ok = False
    else:
        print(f"  ✓ Dependencies  : all {len(deps)} packages found")

    # ── Config sanity ─────────────────────────────────────────────────────
    issues = []
    det = cfg.get("detector", {})
    if det.get("batch_size", 16) > 64:
        issues.append(f"detector.batch_size={det['batch_size']} is very high")
    if det.get("epochs", 100) < 10:
        issues.append(f"detector.epochs={det['epochs']} is very low")
    if det.get("img_size", 640) not in [320, 416, 512, 640, 1280]:
        issues.append(f"detector.img_size={det['img_size']} is unusual")

    cls = cfg.get("classifier", {})
    if cls.get("lr", 0.001) > 0.01:
        issues.append(f"classifier.lr={cls['lr']} is very high")

    if issues:
        print(f"  ⚠ Config issues :")
        for iss in issues:
            print(f"    → {iss}")
    else:
        print(f"  ✓ Config        : all hyperparams in sane ranges")

    # ── Dataset paths ─────────────────────────────────────────────────────
    required_dirs = [
        os.path.join(data_root, "images", "train"),
        os.path.join(data_root, "images", "val"),
        os.path.join(data_root, "labels", "train"),
        os.path.join(data_root, "labels", "val"),
    ]
    dirs_ok = True
    for d in required_dirs:
        if not os.path.isdir(d):
            print(f"  ✗ Missing dir   : {d}")
            dirs_ok = False
            all_ok  = False
    if dirs_ok:
        print(f"  ✓ Dataset dirs  : all required directories exist")

    # ── Summary ───────────────────────────────────────────────────────────
    if all_ok:
        print(f"\n  ✓ All pre-flight checks passed")
    else:
        print(f"\n  ✗ Some checks FAILED — review above before training")

    return all_ok


# =============================================================================
#  Step 1 — Data Validation
# =============================================================================

def validate_dataset(cfg):
    """
    Validate dataset integrity, class balance, and image quality.

    Returns:
        dict: Validation report with stats and issues.
    """
    print(f"\n{'═'*70}")
    print(f"  Data Validation")
    print(f"{'═'*70}\n")

    data_root   = cfg["dataset"]["root"]
    class_names = cfg["dataset"]["class_names"]
    report      = {
        "valid": True,
        "train_images": 0, "val_images": 0,
        "train_annotations": 0, "val_annotations": 0,
        "class_distribution": {},
        "corrupted": [],
        "orphaned_labels": [],
        "missing_labels": [],
        "issues": [],
    }

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "val"]:
        img_dir = os.path.join(data_root, "images", split)
        lbl_dir = os.path.join(data_root, "labels", split)

        if not os.path.isdir(img_dir):
            report["issues"].append(f"Missing: {img_dir}")
            report["valid"] = False
            continue

        # ── Count images and validate ─────────────────────────────────────
        img_files = [f for f in os.listdir(img_dir)
                     if Path(f).suffix.lower() in img_exts]
        lbl_files = {Path(f).stem for f in os.listdir(lbl_dir)
                     if f.endswith(".txt")} if os.path.isdir(lbl_dir) else set()

        n_images = len(img_files)
        report[f"{split}_images"] = n_images

        # ── Check for corrupted images (sample up to 200) ────────────────
        sample_size = min(200, n_images)
        sampled = np.random.choice(img_files, sample_size, replace=False) \
                  if n_images > 0 else []
        corrupted = 0
        for f in sampled:
            img = cv2.imread(os.path.join(img_dir, f))
            if img is None or img.size == 0:
                corrupted += 1
                report["corrupted"].append(os.path.join(img_dir, f))

        if corrupted > 0:
            est_corrupt = int(corrupted / sample_size * n_images)
            report["issues"].append(
                f"{split}: ~{est_corrupt} corrupted images "
                f"(sampled {corrupted}/{sample_size})"
            )

        # ── Missing / orphaned labels ─────────────────────────────────────
        img_stems = {Path(f).stem for f in img_files}
        missing = img_stems - lbl_files
        orphaned = lbl_files - img_stems

        if len(missing) > 10:
            report["issues"].append(
                f"{split}: {len(missing)} images without labels"
            )
            report["missing_labels"].extend(list(missing)[:10])
        if len(orphaned) > 5:
            report["issues"].append(
                f"{split}: {len(orphaned)} orphaned label files"
            )

        # ── Class distribution ────────────────────────────────────────────
        class_counts = Counter()
        total_annots = 0
        bbox_issues  = 0

        for lbl_file in lbl_files:
            lbl_path = os.path.join(lbl_dir, lbl_file + ".txt")
            if not os.path.exists(lbl_path):
                continue
            with open(lbl_path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Validate bbox values
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and
                            0 < w <= 1 and 0 < h <= 1):
                        bbox_issues += 1
                        continue

                    if cls_idx < len(class_names):
                        class_counts[class_names[cls_idx]] += 1
                    total_annots += 1

        report[f"{split}_annotations"] = total_annots

        if bbox_issues > 0:
            report["issues"].append(
                f"{split}: {bbox_issues} invalid bounding boxes"
            )

        # Store class distribution
        for name in class_names:
            key = f"{split}_{name}"
            report["class_distribution"][key] = class_counts.get(name, 0)

        # ── Print split summary ───────────────────────────────────────────
        print(f"  {split.upper()}")
        print(f"    Images      : {n_images:,}")
        print(f"    Annotations : {total_annots:,}")
        if corrupted > 0:
            print(f"    Corrupted   : ~{int(corrupted/sample_size*n_images)} "
                  f"(sampled {corrupted}/{sample_size})")
        for name in class_names:
            cnt = class_counts.get(name, 0)
            bar = "█" * min(int(cnt / max(total_annots, 1) * 40), 40)
            print(f"    {name:10s}  : {cnt:>6,}  {bar}")
        print()

    # ── Class imbalance check ─────────────────────────────────────────────
    train_counts = [
        report["class_distribution"].get(f"train_{name}", 0)
        for name in class_names
    ]
    if train_counts and max(train_counts) > 0:
        ratio = max(train_counts) / max(min(train_counts), 1)
        if ratio > 5:
            report["issues"].append(
                f"Severe class imbalance: {ratio:.1f}x ratio "
                f"(max/min in train)"
            )
            print(f"  ⚠ Class imbalance : {ratio:.1f}x ratio")
        elif ratio > 3:
            print(f"  ⚠ Moderate imbalance : {ratio:.1f}x ratio")
        else:
            print(f"  ✓ Class balance   : {ratio:.1f}x ratio (acceptable)")

    # ── Train/val split ratio ─────────────────────────────────────────────
    total = report["train_images"] + report["val_images"]
    if total > 0:
        val_pct = report["val_images"] / total * 100
        if val_pct < 10 or val_pct > 40:
            report["issues"].append(
                f"Unusual val split: {val_pct:.0f}% (recommend 15-25%)"
            )
        print(f"  ✓ Split ratio   : {100-val_pct:.0f}% train / {val_pct:.0f}% val")

    # ── Final verdict ─────────────────────────────────────────────────────
    if report["issues"]:
        print(f"\n  Issues found ({len(report['issues'])}):")
        for iss in report["issues"]:
            print(f"    ⚠ {iss}")
        report["valid"] = len([i for i in report["issues"]
                               if "Missing" in i or "corrupted" in i]) == 0
    else:
        print(f"\n  ✓ Dataset validation passed — no issues found")

    return report


# =============================================================================
#  Post-Training — Model Evaluation
# =============================================================================

def evaluate_model(cfg, device="cuda"):
    """
    Full post-training evaluation on the validation set.
    Computes: mAP@50, per-class P/R/F1, confusion matrix.

    Returns:
        dict: Full evaluation results.
    """
    from ultralytics import YOLO

    print(f"\n{'═'*70}")
    print(f"  Model Evaluation — Full Val Set Metrics")
    print(f"{'═'*70}\n")

    # ── Find best weights ─────────────────────────────────────────────────
    det_weights = None
    for path in [
        "logs/fp_correction/detector_ft_best.pt",
        "logs/detector/best.pt",
    ]:
        if Path(path).exists():
            det_weights = path
            break

    if det_weights is None:
        print("  [ERROR] No detector weights found — skipping evaluation")
        return {"error": "no_weights"}

    # ── Prepare dataset YAML ──────────────────────────────────────────────
    data_yaml = os.path.join(cfg["dataset"]["root"], "dataset.yaml")
    if not Path(data_yaml).exists():
        root        = Path(cfg["dataset"]["root"]).resolve()
        class_names = cfg["dataset"]["class_names"]
        dataset = {
            "path":  str(root),
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(class_names),
            "names": class_names,
        }
        with open(data_yaml, "w") as f:
            yaml.dump(dataset, f, default_flow_style=False)

    # ── Run YOLO val ──────────────────────────────────────────────────────
    print(f"  Detector weights : {det_weights}")
    model = YOLO(det_weights)

    results = model.val(
        data    = data_yaml,
        imgsz   = cfg["detector"]["img_size"],
        batch   = cfg["detector"]["batch_size"],
        device  = 0 if torch.cuda.is_available() else "cpu",
        verbose = False,
        plots   = True,
    )

    # ── Extract metrics ───────────────────────────────────────────────────
    metrics = {}
    try:
        rd = results.results_dict
        metrics["mAP50"]     = round(rd.get("metrics/mAP50(B)", 0), 4)
        metrics["mAP50_95"]  = round(rd.get("metrics/mAP50-95(B)", 0), 4)
        metrics["precision"] = round(rd.get("metrics/precision(B)", 0), 4)
        metrics["recall"]    = round(rd.get("metrics/recall(B)", 0), 4)
    except Exception:
        metrics["mAP50"]     = 0
        metrics["mAP50_95"]  = 0
        metrics["precision"] = 0
        metrics["recall"]    = 0

    f1 = (2 * metrics["precision"] * metrics["recall"] /
          max(metrics["precision"] + metrics["recall"], 1e-8))
    metrics["f1"] = round(f1, 4)

    # ── Per-class metrics ─────────────────────────────────────────────────
    class_names = cfg["dataset"]["class_names"]
    per_class   = {}
    try:
        for i, name in enumerate(class_names):
            p = float(results.box.p[i]) if i < len(results.box.p) else 0
            r = float(results.box.r[i]) if i < len(results.box.r) else 0
            f = 2*p*r / max(p+r, 1e-8)
            ap50 = float(results.box.ap50[i]) if i < len(results.box.ap50) else 0
            per_class[name] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4),
                "ap50":      round(ap50, 4),
            }
    except Exception:
        pass

    metrics["per_class"] = per_class

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n  {'Metric':<15s}  {'Value':>8s}")
    print(f"  {'─'*28}")
    print(f"  {'mAP@50':<15s}  {metrics['mAP50']:>8.4f}")
    print(f"  {'mAP@50-95':<15s}  {metrics['mAP50_95']:>8.4f}")
    print(f"  {'Precision':<15s}  {metrics['precision']:>8.4f}")
    print(f"  {'Recall':<15s}  {metrics['recall']:>8.4f}")
    print(f"  {'F1':<15s}  {metrics['f1']:>8.4f}")

    if per_class:
        print(f"\n  Per-class breakdown:")
        print(f"  {'Class':<10s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}  {'AP50':>7s}")
        print(f"  {'─'*42}")
        for name, m in per_class.items():
            print(f"  {name:<10s}  {m['precision']:>7.4f}  "
                  f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['ap50']:>7.4f}")

    # ── Quality gates ─────────────────────────────────────────────────────
    gates = []
    if metrics["mAP50"] < 0.5:
        gates.append(f"mAP@50 = {metrics['mAP50']:.4f} (target ≥ 0.50)")
    if metrics["precision"] < 0.7:
        gates.append(f"Precision = {metrics['precision']:.4f} (target ≥ 0.70)")
    if metrics["recall"] < 0.5:
        gates.append(f"Recall = {metrics['recall']:.4f} (target ≥ 0.50)")

    if gates:
        print(f"\n  ⚠ Quality gates NOT met:")
        for g in gates:
            print(f"    → {g}")
        metrics["quality_gates_passed"] = False
    else:
        print(f"\n  ✓ All quality gates passed")
        metrics["quality_gates_passed"] = True

    # ── Save metrics ──────────────────────────────────────────────────────
    eval_dir = Path(cfg["logging"]["save_dir"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {eval_dir}/metrics.json")

    return metrics
