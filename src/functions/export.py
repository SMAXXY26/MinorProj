"""
export.py — Model export for deployment (ONNX).
=================================================
Lightweight ONNX export for the detector and classifier.
For full TensorRT INT8 export, use export_trt.py separately.

Functions:
    export_detector_onnx(cfg)    → str
    export_classifier_onnx(cfg)  → str
    export_models(cfg)           → dict
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path


# =============================================================================
#  ONNX Export — Detector
# =============================================================================

def export_detector_onnx(cfg):
    """
    Export YOLOv8 detector to ONNX format for portable deployment.

    Returns:
        str: Path to exported ONNX file, or None if failed.
    """
    from ultralytics import YOLO

    # Find best detector weights
    det_weights = None
    for path in [
        "logs/fp_correction/detector_ft_best.pt",
        "logs/detector/best.pt",
    ]:
        if Path(path).exists():
            det_weights = path
            break

    if det_weights is None:
        print("  [SKIP] No detector weights found")
        return None

    export_dir = Path(cfg["logging"]["save_dir"]) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Detector ONNX export:")
    print(f"    Weights  : {det_weights}")

    model = YOLO(det_weights)
    img_size = cfg["detector"]["img_size"]

    t0 = time.time()
    onnx_path = model.export(
        format = "onnx",
        imgsz  = img_size,
        opset  = 18,
        simplify = True,
        dynamic  = True,
    )
    elapsed = time.time() - t0

    # Copy to our export dir
    if onnx_path and Path(str(onnx_path)).exists():
        import shutil
        dst = export_dir / "detector.onnx"
        shutil.copy2(str(onnx_path), str(dst))
        size_mb = dst.stat().st_size / 1e6
        print(f"    Output   : {dst}")
        print(f"    Size     : {size_mb:.1f} MB")
        print(f"    Time     : {elapsed:.1f}s")
        return str(dst)
    else:
        print(f"    ✗ Export failed")
        return None


# =============================================================================
#  ONNX Export — Classifier
# =============================================================================

def export_classifier_onnx(cfg):
    """
    Export EfficientNet-B5 classifier to ONNX format.

    Returns:
        str: Path to exported ONNX file, or None if failed.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import WeaponClassifier

    cls_weights = None
    for path in [
        "logs/fp_correction/classifier_bg_best.pt",
        "logs/classifier/best.pt",
    ]:
        if Path(path).exists():
            cls_weights = path
            break

    if cls_weights is None:
        print("  [SKIP] No classifier weights found")
        return None

    export_dir = Path(cfg["logging"]["save_dir"]) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Classifier ONNX export:")
    print(f"    Weights  : {cls_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(cls_weights, map_location=device)
    nc     = ckpt.get("num_classes", cfg["dataset"]["num_classes"])

    model = WeaponClassifier(num_classes=nc, dropout=0.0, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(device)

    img_size    = cfg["classifier"].get("img_size", 224)
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    onnx_path   = str(export_dir / "classifier.onnx")

    t0 = time.time()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version       = 18,
        input_names         = ["input"],
        output_names        = ["logits"],
        dynamic_axes        = {
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding = True,
    )
    elapsed = time.time() - t0

    if Path(onnx_path).exists():
        size_mb = Path(onnx_path).stat().st_size / 1e6
        print(f"    Output   : {onnx_path}")
        print(f"    Size     : {size_mb:.1f} MB")
        print(f"    Classes  : {nc}")
        print(f"    Time     : {elapsed:.1f}s")
        return onnx_path
    else:
        print(f"    ✗ Export failed")
        return None


# =============================================================================
#  Combined Export
# =============================================================================

def export_models(cfg):
    """
    Export all trained models to ONNX format.

    Returns:
        dict: Paths to exported models.
    """
    print(f"\n{'═'*70}")
    print(f"  Model Export — ONNX")
    print(f"{'═'*70}\n")

    exports = {}

    det_onnx = export_detector_onnx(cfg)
    if det_onnx:
        exports["detector_onnx"] = det_onnx

    cls_onnx = export_classifier_onnx(cfg)
    if cls_onnx:
        exports["classifier_onnx"] = cls_onnx

    # ── Summary ───────────────────────────────────────────────────────────
    if exports:
        print(f"\n  Exported models:")
        for name, path in exports.items():
            print(f"    ✓ {name:20s} → {path}")
    else:
        print(f"\n  ⚠ No models exported — train first")

    print(f"\n  For TensorRT INT8 export (Jetson/edge), run:")
    print(f"    python export_trt.py")
    print(f"\n  For Pi 5 AI HAT (Hailo-8L), run step 9 to distil+export NCNN/ONNX:")
    print(f"    python train_all.py --only 9")

    return exports
