"""
compile_hailo.py — Compile ONNX models to Hailo-8L HEF using DFC Python SDK.

Run inside the hailo_dfc conda env:
  conda activate hailo_dfc
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 compile_hailo.py

Outputs:
  logs/hailo/detector.hef
  logs/hailo/student.hef
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np

HAILO_DIR = Path("logs/hailo")
DATA_DIR  = Path("data/images/train")
IMG_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_CALIB   = 100     # calibration images for INT8 quantization


def compile_yolo(onnx_path: Path, hef_name: str, img_size: int = 416):
    """Compile a YOLO ONNX to HEF via DFC: parse → optimize → compile."""
    from hailo_sdk_client import ClientRunner

    har_path     = HAILO_DIR / f"{hef_name}.har"
    har_opt_path = HAILO_DIR / f"{hef_name}_optimized.har"
    hef_path     = HAILO_DIR / f"{hef_name}.hef"

    print(f"\n{'='*60}")
    print(f"  Compiling {onnx_path.name} → {hef_name}.hef")
    print(f"{'='*60}")

    runner = ClientRunner(hw_arch="hailo8l")

    # 1. Parse ONNX → HAR
    # End nodes cut off the DFL decoder (unsupported by Hailo) — post-processing
    # (decode + NMS) runs on CPU. This is the standard YOLO→Hailo approach.
    print("  [1/3] Parsing ONNX ...")
    runner.translate_onnx_model(
        str(onnx_path),
        hef_name,
        net_input_shapes={"images": [1, 3, img_size, img_size]},
        end_node_names=["/model.22/Sigmoid", "/model.22/Concat"],
    )
    runner.save_har(str(har_path))
    print(f"        HAR saved: {har_path}")

    # 2. Collect calibration images and optimize (INT8 quantization)
    print(f"  [2/3] Collecting {N_CALIB} calibration images ...")
    calib_images = []
    if DATA_DIR.exists():
        imgs = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))
        random.shuffle(imgs)
        for p in imgs[:N_CALIB]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD
            calib_images.append(img)   # HWC — Hailo expects (H, W, C)

    if len(calib_images) < 10:
        print(f"  WARNING: only {len(calib_images)} calibration images found in {DATA_DIR}")
        print("  Using random data — accuracy may be lower. Run part1.py to get training images.")
        calib_images += [
            np.random.randn(img_size, img_size, 3).astype(np.float32)
            for _ in range(N_CALIB - len(calib_images))
        ]

    calib = np.stack(calib_images[:N_CALIB])
    print(f"  [2/3] Optimizing (INT8) with {len(calib)} images ...")
    # runner already in hailo_model state after translate — no load_har needed
    runner.optimize(calib)
    runner.save_har(str(har_opt_path))
    print(f"        Optimized HAR: {har_opt_path}")

    # 3. Compile → HEF
    print("  [3/3] Compiling to HEF ...")
    hef_bytes = runner.compile()
    hef_path.write_bytes(hef_bytes)
    size_mb = hef_path.stat().st_size / 1e6
    print(f"  [OK]  {hef_path}  ({size_mb:.1f} MB)")
    return hef_path


def export_student_onnx(img_size: int = 416) -> Path:
    """Export student .pt → ONNX if not already done."""
    onnx_path = HAILO_DIR / "student.onnx"
    if onnx_path.exists():
        print(f"  Student ONNX already exists: {onnx_path}")
        return onnx_path

    student_pt = Path("logs/student/best.pt")
    if not student_pt.exists():
        print("ERROR: logs/student/best.pt not found — run step 9 distillation first.")
        sys.exit(1)

    print(f"\n  Exporting student {student_pt} → {onnx_path} ...")
    from ultralytics import YOLO
    model    = YOLO(str(student_pt))
    exported = model.export(
        format   = "onnx",
        imgsz    = img_size,
        opset    = 11,
        simplify = True,
        dynamic  = False,
        batch    = 1,
    )
    import shutil
    shutil.copy2(str(exported), str(onnx_path))
    print(f"  Student ONNX: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return onnx_path


def main():
    HAILO_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError as e:
        print(f"ERROR: Hailo DFC not importable: {e}")
        print("Run: conda activate hailo_dfc && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 compile_hailo.py")
        sys.exit(1)

    results = {}

    # ── Teacher detector ──────────────────────────────────────────────────────
    det_onnx = HAILO_DIR / "detector.onnx"
    if det_onnx.exists():
        results["detector"] = compile_yolo(det_onnx, "detector", img_size=416)
    else:
        print(f"WARNING: {det_onnx} not found — skipping teacher detector.")
        print("         Run: python3 export_hailo.py --detector-only --onnx-only")

    # ── Student detector ──────────────────────────────────────────────────────
    stu_onnx = export_student_onnx(img_size=416)
    results["student"] = compile_yolo(stu_onnx, "student", img_size=416)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Done. HEF files:")
    for name, path in results.items():
        print(f"    {name:12s}  {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"\n  Copy to Pi:")
    print(f"    scp logs/hailo/*.hef pddy@10.42.0.100:~/weapon_detection/logs/hailo/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
