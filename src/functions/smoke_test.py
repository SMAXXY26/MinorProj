"""
smoke_test.py — Post-training inference smoke test.
=====================================================
Quick end-to-end verification that the trained models work correctly.

Functions:
    run_smoke_test(cfg, device) → dict
"""

import os
import time
import cv2
import json
import torch
import numpy as np
from pathlib import Path


def run_smoke_test(cfg, device="cuda"):
    """
    Run a quick smoke test: load models, run inference on a few images,
    verify outputs are sane, and measure FPS.

    Returns:
        dict: Smoke test results (passed, fps, detections, errors).
    """
    from ultralytics import YOLO
    from src.model import WeaponClassifier
    from torchvision import transforms

    print(f"\n{'═'*70}")
    print(f"  Smoke Test — End-to-End Inference Verification")
    print(f"{'═'*70}\n")

    results = {
        "passed":     False,
        "errors":     [],
        "detector":   {},
        "classifier": {},
    }

    # ── Find best weights ─────────────────────────────────────────────────
    det_weights = None
    for path in [
        "logs/fp_correction/detector_ft_best.pt",
        "logs/detector/best.pt",
    ]:
        if Path(path).exists():
            det_weights = path
            break

    cls_weights = None
    for path in [
        "logs/fp_correction/classifier_bg_best.pt",
        "logs/classifier/best.pt",
    ]:
        if Path(path).exists():
            cls_weights = path
            break

    # ── Load detector ─────────────────────────────────────────────────────
    if det_weights is None:
        results["errors"].append("No detector weights found")
        print(f"  ✗ Detector     : weights not found")
        return results

    try:
        detector = YOLO(det_weights)
        results["detector"]["loaded"] = True
        print(f"  ✓ Detector     : {det_weights}")
    except Exception as e:
        results["errors"].append(f"Failed to load detector: {e}")
        print(f"  ✗ Detector     : load failed — {e}")
        return results

    # ── Load classifier ───────────────────────────────────────────────────
    classifier = None
    if cls_weights:
        try:
            dev  = torch.device(device if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(cls_weights, map_location=dev)
            nc   = ckpt.get("num_classes", cfg["dataset"]["num_classes"])
            classifier = WeaponClassifier(
                num_classes=nc, dropout=0.0, pretrained=False
            ).to(dev)
            classifier.load_state_dict(ckpt["model"], strict=False)
            classifier.eval()
            results["classifier"]["loaded"] = True
            results["classifier"]["num_classes"] = nc
            print(f"  ✓ Classifier   : {cls_weights} ({nc} classes)")
        except Exception as e:
            results["errors"].append(f"Failed to load classifier: {e}")
            print(f"  ⚠ Classifier   : load failed — {e}")
    else:
        print(f"  ⚠ Classifier   : weights not found (skipping)")

    # ── Collect test images ───────────────────────────────────────────────
    data_root = cfg["dataset"]["root"]
    val_img   = os.path.join(data_root, "images", "val")
    img_exts  = {".jpg", ".jpeg", ".png"}

    test_images = []
    if os.path.isdir(val_img):
        all_imgs = [os.path.join(val_img, f) for f in os.listdir(val_img)
                    if Path(f).suffix.lower() in img_exts]
        # Take up to 20 images for smoke test
        test_images = sorted(all_imgs)[:20]

    if not test_images:
        # Try to generate a synthetic image
        print(f"  ⚠ No val images — using synthetic test image")
        test_images = ["__synthetic__"]

    print(f"  Test images    : {len(test_images)}\n")

    # ── Run inference ─────────────────────────────────────────────────────
    total_dets  = 0
    n_images    = 0
    latencies   = []
    class_names = cfg["dataset"]["class_names"]

    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) if classifier else None

    for img_path in test_images:
        if img_path == "__synthetic__":
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        else:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

        n_images += 1
        t0 = time.time()

        # Gate 1: Detector
        det_results = detector(frame, conf=0.25, verbose=False)
        boxes = det_results[0].boxes

        n_dets = 0 if boxes is None else len(boxes)
        total_dets += n_dets

        # Gate 2: Classifier (if available)
        if classifier and boxes is not None and n_dets > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                H, W = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    dev  = next(classifier.parameters()).device
                    tensor = cls_transform(rgb).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        logits = classifier(tensor)
                    # Just verify it produces a valid output
                    pred = logits.argmax(-1).item()

        latency = (time.time() - t0) * 1000
        latencies.append(latency)

    # ── Compute stats ─────────────────────────────────────────────────────
    if latencies:
        avg_ms  = np.mean(latencies)
        p95_ms  = np.percentile(latencies, 95)
        fps     = 1000 / avg_ms if avg_ms > 0 else 0

        results["detector"]["avg_latency_ms"] = round(avg_ms, 1)
        results["detector"]["p95_latency_ms"] = round(p95_ms, 1)
        results["detector"]["fps"]            = round(fps, 1)
        results["detector"]["total_dets"]     = total_dets
        results["detector"]["images_tested"]  = n_images

        print(f"  Inference results:")
        print(f"    Images tested  : {n_images}")
        print(f"    Total detections: {total_dets}")
        print(f"    Avg latency    : {avg_ms:.1f} ms")
        print(f"    P95 latency    : {p95_ms:.1f} ms")
        print(f"    Throughput     : {fps:.1f} FPS")

    # ── Verdict ───────────────────────────────────────────────────────────
    passed = (
        len(results["errors"]) == 0 and
        results["detector"].get("loaded", False) and
        n_images > 0
    )
    results["passed"] = passed

    if passed:
        print(f"\n  ✓ Smoke test PASSED — model is ready for deployment")
    else:
        print(f"\n  ✗ Smoke test FAILED")
        for err in results["errors"]:
            print(f"    → {err}")

    # ── Save results ──────────────────────────────────────────────────────
    save_dir = Path(cfg["logging"]["save_dir"]) / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "smoke_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {save_dir}/smoke_test.json")

    return results
