"""
export_hailo.py — Export teacher + student models to Hailo-8L HEF for Pi 5
===========================================================================
Target hardware: Raspberry Pi 5 + Hailo-8L AI HAT (26 TOPS, PCIe)

This script runs on your DEV MACHINE (not on the Pi) and produces .hef files
that you then copy to the Pi 5.

Prerequisites (dev machine):
  1. Hailo Dataflow Compiler (DFC) >= 3.27
     Register and download from https://hailo.ai/developer-zone/
     pip install hailo_dataflow_compiler  (via hailo's private PyPI)
  2. ultralytics >= 8.3  (for built-in hailo export of YOLO models)

If DFC is not installed, use --onnx-only to export ONNX files first,
then run the printed DFC shell commands on a machine where DFC is installed.

Outputs (in logs/hailo/):
  detector.hef      — YOLOv8s teacher (fp_correction) for Hailo-8L
  student.hef       — YOLOv11n student (distilled, smaller/faster)
  classifier.hef    — EfficientNet classifier for Hailo-8L
  *.onnx            — intermediate ONNX files (kept for debugging)

Usage:
    python3 export_hailo.py                              # All three models
    python3 export_hailo.py --detector-only              # Teacher detector only
    python3 export_hailo.py --student-only               # Student only
    python3 export_hailo.py --classifier-only
    python3 export_hailo.py --onnx-only                  # ONNX only, no DFC
    python3 export_hailo.py --pi-host 192.168.1.100      # Custom Pi IP
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HAILO_OUT_DIR = Path("logs/hailo")
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/hyperparams.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_teacher_weights() -> str:
    """Return the best available teacher detector checkpoint."""
    for p in [
        "runs/detect/logs/fp_correction/weights/best.pt",
        "logs/fp_correction/detector_ft_best.pt",
        "logs/detector/best.pt",
    ]:
        if Path(p).exists():
            logger.info(f"  Teacher detector : {p}")
            return p
    logger.error("No teacher detector weights found. Run training first.")
    sys.exit(1)


def _find_student_weights() -> str:
    """Return the best available student checkpoint."""
    for p in [
        "logs/student/best.pt",
        "logs/student/weights/best.pt",
    ]:
        if Path(p).exists():
            logger.info(f"  Student detector : {p}")
            return p
    logger.error("No student weights found. Run train_all.py step 9 first.")
    sys.exit(1)


def _find_classifier_weights() -> str:
    for p in [
        "logs/fp_correction/classifier_bg_best.pt",
        "logs/classifier/best.pt",
    ]:
        if Path(p).exists():
            logger.info(f"  Classifier       : {p}")
            return p
    logger.error("No classifier weights found. Run part2.py first.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Detector export
# ─────────────────────────────────────────────────────────────────────────────

def export_detector(cfg: dict, onnx_only: bool = False) -> str:
    """
    Export YOLOv8s teacher detector to HEF for Hailo-8L.

    Uses Ultralytics built-in hailo export which:
      - Converts to ONNX internally (opset 11)
      - Runs Hailo DFC: parse → optimize (INT8 calibration) → compile
      - Embeds NMS post-processing inside the HEF

    With --onnx-only: exports ONNX only and prints manual DFC commands.
    Returns path to .hef (or .onnx if onnx-only/DFC not available).
    """
    from ultralytics import YOLO

    det_weights = _find_teacher_weights()
    img_size    = cfg["detector"].get("img_size", 416)

    HAILO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'═'*60}")
    logger.info("  Detector → Hailo-8L HEF")
    logger.info(f"  Weights  : {det_weights}")
    logger.info(f"  img_size : {img_size}")
    logger.info(f"  hw_arch  : hailo8l")
    logger.info(f"{'═'*60}")

    model = YOLO(det_weights)

    if onnx_only:
        onnx_path = str(HAILO_OUT_DIR / "detector.onnx")
        t0 = time.perf_counter()
        exported = model.export(
            format   = "onnx",
            imgsz    = img_size,
            opset    = 11,       # Hailo DFC is most reliable with opset 11
            simplify = True,
            dynamic  = False,    # Static batch=1 required for Hailo
            batch    = 1,
        )
        elapsed = time.perf_counter() - t0
        if exported and Path(str(exported)).exists():
            shutil.copy2(str(exported), onnx_path)
            size_mb = Path(onnx_path).stat().st_size / 1e6
            logger.info(f"  ONNX : {onnx_path}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")
        else:
            logger.error("  ONNX export failed")
            sys.exit(1)
        _print_detector_dfc_commands(onnx_path, img_size)
        return onnx_path

    # Full Hailo export (requires hailo_dataflow_compiler)
    try:
        t0       = time.perf_counter()
        exported = model.export(
            format  = "hailo",
            hw_arch = "hailo8l",
            imgsz   = img_size,
            batch   = 1,
        )
        elapsed = time.perf_counter() - t0
    except ImportError as exc:
        logger.error(f"\n  Hailo DFC not installed: {exc}")
        logger.error("  Get it from https://hailo.ai/developer-zone/")
        logger.error("  Or re-run with --onnx-only to export ONNX for manual DFC.\n")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"\n  Hailo export failed: {exc}")
        logger.error("  Try --onnx-only and compile manually with DFC.\n")
        sys.exit(1)

    # Locate and copy the produced .hef
    hef_dst = HAILO_OUT_DIR / "detector.hef"
    exported_path = Path(str(exported))
    alt_hef = Path(det_weights).with_suffix(".hef")

    src = None
    for candidate in [exported_path, alt_hef]:
        if candidate.exists() and candidate.suffix == ".hef":
            src = candidate
            break

    if src is None:
        logger.error(f"  HEF not found at {exported_path} or {alt_hef}")
        sys.exit(1)

    shutil.copy2(str(src), str(hef_dst))
    size_mb = hef_dst.stat().st_size / 1e6
    logger.info(f"  HEF  : {hef_dst}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")
    return str(hef_dst)


def export_student(cfg: dict, onnx_only: bool = False) -> str:
    """
    Export YOLOv11n student model to HEF for Hailo-8L.
    Same flow as the teacher detector — both are YOLO, so ultralytics
    built-in hailo export handles everything.
    Returns path to .hef (or .onnx if onnx-only/DFC not available).
    """
    from ultralytics import YOLO

    student_weights = _find_student_weights()
    img_size        = cfg.get("student", {}).get("imgsz", 416)

    HAILO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'═'*60}")
    logger.info("  Student (YOLOv11n) → Hailo-8L HEF")
    logger.info(f"  Weights  : {student_weights}")
    logger.info(f"  img_size : {img_size}")
    logger.info(f"  hw_arch  : hailo8l")
    logger.info(f"{'═'*60}")

    model = YOLO(student_weights)

    if onnx_only:
        onnx_path = str(HAILO_OUT_DIR / "student.onnx")
        t0 = time.perf_counter()
        exported = model.export(
            format   = "onnx",
            imgsz    = img_size,
            opset    = 11,
            simplify = True,
            dynamic  = False,
            batch    = 1,
        )
        elapsed = time.perf_counter() - t0
        if exported and Path(str(exported)).exists():
            shutil.copy2(str(exported), onnx_path)
            size_mb = Path(onnx_path).stat().st_size / 1e6
            logger.info(f"  ONNX : {onnx_path}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")
        else:
            logger.error("  ONNX export failed")
            sys.exit(1)
        _print_yolo_dfc_commands(onnx_path, img_size, "student")
        return onnx_path

    try:
        t0       = time.perf_counter()
        exported = model.export(
            format  = "hailo",
            hw_arch = "hailo8l",
            imgsz   = img_size,
            batch   = 1,
        )
        elapsed = time.perf_counter() - t0
    except ImportError as exc:
        logger.error(f"\n  Hailo DFC not installed: {exc}")
        logger.error("  Re-run with --onnx-only to export ONNX for manual DFC.\n")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"\n  Hailo export failed: {exc}")
        logger.error("  Try --onnx-only and compile manually with DFC.\n")
        sys.exit(1)

    hef_dst = HAILO_OUT_DIR / "student.hef"
    exported_path = Path(str(exported))
    alt_hef = Path(student_weights).with_suffix(".hef")

    src = None
    for candidate in [exported_path, alt_hef]:
        if candidate.exists() and candidate.suffix == ".hef":
            src = candidate
            break

    if src is None:
        logger.error(f"  HEF not found at {exported_path} or {alt_hef}")
        sys.exit(1)

    shutil.copy2(str(src), str(hef_dst))
    size_mb = hef_dst.stat().st_size / 1e6
    logger.info(f"  HEF  : {hef_dst}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")
    return str(hef_dst)


def _print_yolo_dfc_commands(onnx_path: str, img_size: int, name: str) -> None:
    logger.info(f"\n  Manual DFC commands for {name}:")
    logger.info(f"  {'─'*56}")
    logger.info(f"  hailo parser onnx {onnx_path} \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --output-har logs/hailo/{name}.har")
    logger.info(f"")
    logger.info(f"  mkdir -p logs/hailo/calib_{name}")
    logger.info(f"  ls data/images/train/*.jpg | shuf | head -100 \\")
    logger.info(f"      | xargs -I{{}} cp {{}} logs/hailo/calib_{name}/")
    logger.info(f"")
    logger.info(f"  hailo optimize logs/hailo/{name}.har \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --calib-path logs/hailo/calib_{name}/ \\")
    logger.info(f"      --output-har logs/hailo/{name}_optimized.har")
    logger.info(f"")
    logger.info(f"  hailo compiler logs/hailo/{name}_optimized.har \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --output-dir logs/hailo/")
    logger.info(f"  # Produces: logs/hailo/{name}.hef")


def _print_detector_dfc_commands(onnx_path: str, img_size: int) -> None:
    _print_yolo_dfc_commands(onnx_path, img_size, "detector")


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier export
# ─────────────────────────────────────────────────────────────────────────────

def export_classifier(cfg: dict, onnx_only: bool = False) -> str:
    """
    Export EfficientNet classifier to HEF for Hailo-8L.
    Path: PyTorch → ONNX (opset 11, static batch=1) → Hailo DFC → HEF.
    Returns path to .hef, or .onnx if onnx_only / DFC not available.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.model import WeaponClassifier

    cls_weights = _find_classifier_weights()
    img_size    = cfg["classifier"].get("img_size", 224)
    num_classes = cfg["dataset"]["num_classes"]
    crop_dir    = cfg["dataset"].get("classifier_root", "data/classifier_crops")

    HAILO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = str(HAILO_OUT_DIR / "classifier.onnx")

    logger.info(f"\n{'═'*60}")
    logger.info("  Classifier → ONNX (Hailo-compatible)")
    logger.info(f"  Weights  : {cls_weights}")
    logger.info(f"  ONNX     : {onnx_path}")
    logger.info(f"  img_size : {img_size}")
    logger.info(f"{'═'*60}")

    # Load model
    ckpt     = torch.load(cls_weights, map_location="cpu")
    nc       = ckpt.get("num_classes", num_classes)
    backbone = ckpt.get("backbone", "efficientnet_b5")
    model    = WeaponClassifier(num_classes=nc, dropout=0.0, pretrained=False,
                                backbone=backbone)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    logger.info(f"  Backbone : {backbone}  (nc={nc})")

    # ONNX export — static batch=1, opset 11 (most Hailo DFC versions)
    dummy = torch.randn(1, 3, img_size, img_size)
    t0    = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version       = 11,
        input_names         = ["input"],
        output_names        = ["logits"],
        do_constant_folding = True,
        # No dynamic_axes — Hailo requires static shapes
    )
    elapsed = time.perf_counter() - t0
    size_mb = Path(onnx_path).stat().st_size / 1e6
    logger.info(f"  ONNX : {onnx_path}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")

    # Sanity-check the ONNX
    try:
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        logger.info("  ONNX check : ✓")
    except ImportError:
        pass   # onnx package optional here
    except Exception as exc:
        logger.warning(f"  ONNX check warning: {exc}")

    if onnx_only:
        _print_classifier_dfc_commands(onnx_path, img_size, crop_dir)
        return onnx_path

    # Attempt DFC compilation via Python SDK
    hef_path = str(HAILO_OUT_DIR / "classifier.hef")
    try:
        hef_path = _compile_classifier_dfc(
            onnx_path   = onnx_path,
            hef_path    = hef_path,
            img_size    = img_size,
            crop_dir    = crop_dir,
        )
    except ImportError:
        logger.warning("\n  Hailo SDK not found — ONNX saved but HEF not compiled.")
        logger.warning("  Install hailo_dataflow_compiler and re-run, OR use:")
        _print_classifier_dfc_commands(onnx_path, img_size, crop_dir)
        return onnx_path
    except Exception as exc:
        logger.error(f"\n  DFC compilation failed: {exc}")
        logger.error("  ONNX is available. Try manual DFC commands:")
        _print_classifier_dfc_commands(onnx_path, img_size, crop_dir)
        return onnx_path

    return hef_path


def _compile_classifier_dfc(
    onnx_path: str,
    hef_path: str,
    img_size: int,
    crop_dir: str,
) -> str:
    """Compile classifier ONNX → HEF using the Hailo DFC Python SDK."""
    from hailo_sdk_client import ClientRunner    # from hailo_dataflow_compiler

    import cv2

    har_path     = str(HAILO_OUT_DIR / "classifier.har")
    har_opt_path = str(HAILO_OUT_DIR / "classifier_optimized.har")

    logger.info("\n  Compiling classifier via Hailo DFC ...")
    runner = ClientRunner(hw_arch="hailo8l")

    # 1. Parse ONNX → HAR
    logger.info("  [1/3] Parsing ONNX ...")
    runner.translate_onnx_model(
        onnx_path,
        "classifier",
        start_node_names = ["input"],
        end_node_names   = ["logits"],
        net_input_shapes = {"input": [1, 3, img_size, img_size]},
    )
    runner.save_har(har_path)
    logger.info(f"  HAR : {har_path}")

    # 2. Collect calibration crops for INT8 quantization
    logger.info("  [2/3] Optimizing (INT8 calibration) ...")
    calib_images = []
    crop_root = Path(crop_dir)
    if crop_root.exists():
        all_imgs = list(crop_root.rglob("*.jpg")) + list(crop_root.rglob("*.png"))
        random.shuffle(all_imgs)
        for p in all_imgs[:200]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD
            calib_images.append(img.transpose(2, 0, 1))  # CHW

    if not calib_images:
        logger.warning(
            f"  No crops in {crop_dir} — using random calibration data "
            "(run part2.py --skip-crops=False first for better accuracy)"
        )
        calib_images = [
            np.random.randn(3, img_size, img_size).astype(np.float32)
            for _ in range(64)
        ]

    calib_data = np.stack(calib_images)  # (N, 3, H, W)
    runner.load_har(har_path)
    runner.optimize_full_precision(calib_set={"input": calib_data})
    runner.save_har(har_opt_path)
    logger.info(f"  HAR optimized : {har_opt_path}")

    # 3. Compile → HEF
    logger.info("  [3/3] Compiling to HEF ...")
    runner.load_har(har_opt_path)
    hef_bytes = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef_bytes)

    size_mb = Path(hef_path).stat().st_size / 1e6
    logger.info(f"  HEF : {hef_path}  ({size_mb:.1f} MB)")
    return hef_path


def _print_classifier_dfc_commands(onnx_path: str, img_size: int, crop_dir: str) -> None:
    logger.info("\n  Manual DFC commands for classifier:")
    logger.info(f"  {'─'*56}")
    logger.info(f"  # 1. Parse")
    logger.info(f"  hailo parser onnx {onnx_path} \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --output-har logs/hailo/classifier.har \\")
    logger.info(f"      --start-node-names input \\")
    logger.info(f"      --end-node-names logits \\")
    logger.info(f"      --net-input-shapes 'input=[1,3,{img_size},{img_size}]'")
    logger.info(f"")
    logger.info(f"  # 2. Collect 100 calibration crops")
    logger.info(f"  mkdir -p logs/hailo/calib_cls")
    logger.info(f"  find {crop_dir} -name '*.jpg' | shuf | head -100 \\")
    logger.info(f"      | xargs -I{{}} cp {{}} logs/hailo/calib_cls/")
    logger.info(f"")
    logger.info(f"  # 3. Optimize (INT8 quantization)")
    logger.info(f"  hailo optimize logs/hailo/classifier.har \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --calib-path logs/hailo/calib_cls/ \\")
    logger.info(f"      --output-har logs/hailo/classifier_optimized.har")
    logger.info(f"")
    logger.info(f"  # 4. Compile to HEF")
    logger.info(f"  hailo compiler logs/hailo/classifier_optimized.har \\")
    logger.info(f"      --hw-arch hailo8l \\")
    logger.info(f"      --output-dir logs/hailo/")
    logger.info(f"  # Produces: logs/hailo/classifier.hef")


# ─────────────────────────────────────────────────────────────────────────────
#  Transfer instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_transfer_commands(pi_user: str, pi_host: str) -> None:
    logger.info(f"\n{'═'*60}")
    logger.info("  Transfer to Raspberry Pi 5")
    logger.info(f"{'═'*60}")
    target = f"{pi_user}@{pi_host}:~/weapon_detection"
    logger.info(f"  # HEF models (teacher detector, student, classifier)")
    logger.info(f"  rsync -avz logs/hailo/*.hef {target}/logs/hailo/")
    logger.info(f"")
    logger.info(f"  # Runtime scripts and config")
    logger.info(f"  rsync -avz inference_pi5.py {target}/")
    logger.info(f"  rsync -avz config/ {target}/config/")
    logger.info(f"  rsync -avz utils/ {target}/utils/")
    logger.info(f"  rsync -avz src/ {target}/src/")
    logger.info(f"")
    logger.info(f"  # Optional: LSTM temporal smoother (CPU, ~1 MB)")
    logger.info(f"  rsync -avz logs/temporal/best.pt {target}/logs/temporal/")
    logger.info(f"")
    logger.info(f"  # Then on the Pi 5:")
    logger.info(f"  ssh {pi_user}@{pi_host}")
    logger.info(f"  cd ~/weapon_detection")
    logger.info(f"  bash scripts/setup_pi5.sh           # first time only")
    logger.info(f"  python3 inference_pi5.py --source 0 --hailo")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export teacher models to Hailo-8L HEF for Raspberry Pi 5"
    )
    parser.add_argument("--config",          default="config/hyperparams.yaml")
    parser.add_argument("--detector-only",   action="store_true")
    parser.add_argument("--student-only",    action="store_true")
    parser.add_argument("--classifier-only", action="store_true")
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Export ONNX only (no DFC required). "
             "Use the printed DFC commands to produce HEF files separately.",
    )
    parser.add_argument("--pi-user", default="pi",
                        help="Pi 5 SSH username (for transfer commands)")
    parser.add_argument("--pi-host", default="raspberrypi.local",
                        help="Pi 5 hostname or IP (for transfer commands)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    HAILO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    only_one = args.detector_only or args.student_only or args.classifier_only
    do_det = args.detector_only or not only_one
    do_stu = args.student_only  or not only_one
    do_cls = args.classifier_only or not only_one

    outputs = {}
    if do_det:
        outputs["detector"] = export_detector(cfg, onnx_only=args.onnx_only)
        logger.info(f"\n  [OK] Detector  : {outputs['detector']}")

    if do_stu:
        outputs["student"] = export_student(cfg, onnx_only=args.onnx_only)
        logger.info(f"\n  [OK] Student   : {outputs['student']}")

    if do_cls:
        outputs["classifier"] = export_classifier(cfg, onnx_only=args.onnx_only)
        logger.info(f"\n  [OK] Classifier: {outputs['classifier']}")

    print_transfer_commands(args.pi_user, args.pi_host)

    logger.info(f"\n{'═'*60}")
    if args.onnx_only:
        logger.info("  Next: install Hailo DFC → run the DFC commands above → copy HEFs to Pi 5")
    else:
        hef_files = [v for v in outputs.values() if v.endswith(".hef")]
        if hef_files:
            logger.info("  HEFs ready. Transfer to Pi 5 and run inference_pi5.py --hailo")
        else:
            logger.info("  ONNX files ready. Compile with DFC to get HEFs.")
    logger.info(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
