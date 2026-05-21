"""
detection.py  —  Two-gate weapon detection on video
====================================================
Gate 1: YOLO student detector (blender)  (logs/student/best_blender.pt)
Gate 2: EfficientNet classifier (logs/fp_correction/classifier_bg_best.pt)  — backbone auto-detected from checkpoint

Saves compressed output video to outputs/ directory.

Usage:
    python detection.py --source video.mp4
    python detection.py --source video.mp4 --show          # live preview window
    python detection.py --source video.mp4 --no-classifier
    python detection.py --source video.mp4 --det-conf 0.35 --cls-conf 0.65
    python detection.py --source 0                         # webcam
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import subprocess
from pathlib import Path
from torchvision import transforms

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import WeaponClassifier
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["knife", "pistol", "rifle"]
CLASS_COLORS = {
    "knife":  (0,   255, 100),
    "pistol": (0,   100, 255),
    "rifle":  (255, 50,  50),
}

CLS_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
TEMP_DIR   = Path("outputs/temp")


# =============================================================================
#  Load models
# =============================================================================

def load_detector(weights="logs/student/best_blender.pt", device="cuda"):
    if not Path(weights).exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    model = YOLO(weights)
    print(f"[Gate 1] Detector loaded   : {weights}")
    return model


def load_classifier(weights="logs/fp_correction/classifier_bg_best.pt", device="cuda"):
    if not Path(weights).exists():
        print(f"[Gate 2] Classifier not found — running Gate 1 only")
        return None
    ckpt  = torch.load(weights, map_location=device)
    backbone = ckpt.get("backbone", "efficientnet_b0")
    model = WeaponClassifier(
        num_classes = ckpt.get("num_classes", 3),
        dropout     = 0.0,
        pretrained  = False,
        backbone    = backbone,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"[Gate 2] Classifier loaded : {weights}  (backbone={backbone})")
    return model


# =============================================================================
#  Inference helpers
# =============================================================================

@torch.no_grad()
def classify_crop(classifier, crop_bgr, device):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0, 0.0
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = CLS_TRANSFORM(rgb).unsqueeze(0).to(device)
    logits = classifier(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]
    cls    = probs.argmax().item()
    conf   = probs[cls].item()
    return cls, conf


def draw_detection(frame, x1, y1, x2, y2,
                   cls_name, det_conf, cls_conf=None):
    color = CLASS_COLORS.get(cls_name, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = (f"{cls_name} D:{det_conf:.2f} C:{cls_conf:.2f}"
             if cls_conf is not None
             else f"{cls_name} {det_conf:.2f}")

    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    cv2.rectangle(frame,
                  (x1, y1 - th - 8), (x1 + tw + 4, y1),
                  color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)


def process_frame(frame, detector, classifier, device,
                  det_conf=0.25, cls_conf=0.50,
                  use_classifier=True):
    H, W       = frame.shape[:2]
    alert      = False
    detections = []

    # ── Gate 1: Detector ─────────────────────────────────────────────────
    results = detector(frame, conf=det_conf, verbose=False, device=device)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return frame, False, []

    for box in boxes:
        x1, y1, x2, y2  = map(int, box.xyxy[0].tolist())
        x1, y1           = max(0, x1), max(0, y1)
        x2, y2           = min(W, x2), min(H, y2)
        det_cls          = int(box.cls[0].item())
        det_conf_val     = float(box.conf[0].item())
        cls_name         = CLASS_NAMES[min(det_cls, len(CLASS_NAMES) - 1)]

        # ── Gate 2: Classifier ────────────────────────────────────────────
        if use_classifier and classifier is not None:
            crop                = frame[y1:y2, x1:x2]
            pred_cls, pred_conf = classify_crop(classifier, crop, device)

            if pred_cls != det_cls or pred_conf < cls_conf:
                # Rejected — draw grey dashed outline
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (128, 128, 128), 1)
                cv2.putText(frame,
                            f"FP:{cls_name}",
                            (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (128, 128, 128), 1)
                continue

            draw_detection(frame, x1, y1, x2, y2,
                           cls_name, det_conf_val, pred_conf)
            alert = True
            detections.append({
                "class":    cls_name,
                "det_conf": det_conf_val,
                "cls_conf": pred_conf,
                "box":      (x1, y1, x2, y2),
            })
        else:
            draw_detection(frame, x1, y1, x2, y2,
                           cls_name, det_conf_val)
            alert = True
            detections.append({
                "class":    cls_name,
                "det_conf": det_conf_val,
                "box":      (x1, y1, x2, y2),
            })

    # ── Alert banner ──────────────────────────────────────────────────────
    if alert:
        cv2.rectangle(frame, (0, 0), (W, 45), (0, 0, 180), -1)
        found  = list({d["class"] for d in detections})
        banner = "⚠ WEAPON DETECTED: " + ", ".join(found).upper()
        cv2.putText(frame, banner, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)

    return frame, alert, detections


# =============================================================================
#  Video compression via ffmpeg
# =============================================================================

def compress_video(input_path: str, output_path: str, crf: int = 23):
    """
    Compress video using ffmpeg H.264 with CRF quality setting.
    CRF: 18=high quality  23=default  28=smaller file
    """
    ffmpeg_available = os.system("which ffmpeg > /dev/null 2>&1") == 0

    if not ffmpeg_available:
        print("  [Compress] ffmpeg not found — installing...")
        os.system("sudo apt-get install -y ffmpeg -q")

    cmd = [
        "ffmpeg", "-y",
        "-i",  input_path,
        "-vcodec", "libx264",
        "-crf",    str(crf),
        "-preset", "fast",
        "-movflags", "+faststart",   # web-optimised — plays before full download
        "-loglevel", "error",
        output_path,
    ]
    print(f"  [Compress] Compressing → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [Compress] ffmpeg error: {result.stderr}")
        # Fallback — just copy
        import shutil
        shutil.copy2(input_path, output_path)
    else:
        # Show size comparison
        in_size  = Path(input_path).stat().st_size  / 1024 / 1024
        out_size = Path(output_path).stat().st_size / 1024 / 1024
        ratio    = (1 - out_size / max(in_size, 0.001)) * 100
        print(f"  [Compress] {in_size:.1f}MB → {out_size:.1f}MB  "
              f"({ratio:.0f}% smaller)")


# =============================================================================
#  Main runner
# =============================================================================

def run(source, detector, classifier, device,
        det_conf=0.25, cls_conf=0.50,
        use_classifier=True, crf=23, show=False):

    # ── Setup output paths ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    src_name  = Path(source).stem if not str(source).isdigit() else "webcam"
    gate_tag  = "g1g2" if use_classifier else "g1"
    temp_path = str(TEMP_DIR  / f"{src_name}_{gate_tag}_raw.mp4")
    out_path  = str(OUTPUT_DIR / f"{src_name}_{gate_tag}_detected.mp4")

    # ── Open source ───────────────────────────────────────────────────────
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[Detection] Source     : {source}")
    print(f"[Detection] Resolution : {W}x{H}  FPS: {fps:.0f}  "
          f"Frames: {total_frames}")
    print(f"[Detection] Gate 1     : detector  conf>{det_conf}")
    print(f"[Detection] Gate 2     : "
          f"{'classifier conf>' + str(cls_conf) if use_classifier and classifier else 'DISABLED'}")
    print(f"[Detection] Output     : {out_path}\n")

    # ── Video writer (temp uncompressed) ──────────────────────────────────
    writer = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    frame_idx   = 0
    alert_count = 0
    alert_log   = []

    gate_str = "G1+G2" if (use_classifier and classifier) else "G1"
    pbar = (tqdm(total=total_frames, unit="frame", desc=f"[{gate_str}]")
            if _TQDM and total_frames > 0 else None)

    if show:
        cv2.namedWindow("WeaponDetection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame, alert, dets = process_frame(
            frame, detector, classifier, device,
            det_conf, cls_conf, use_classifier,
        )

        if alert:
            alert_count += 1
            alert_log.append({
                "frame":      frame_idx,
                "timestamp":  round(frame_idx / fps, 2),
                "detections": dets,
            })

        # Overlay: frame counter + alert count
        cv2.putText(
            frame,
            f"Frame:{frame_idx}/{total_frames}  "
            f"Alerts:{alert_count}  [{gate_str}]",
            (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1,
        )

        writer.write(frame)

        if show:
            cv2.imshow("WeaponDetection", frame)
            # press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  [Show] Interrupted by user.")
                break

        if pbar:
            pbar.update(1)
        elif frame_idx % 100 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  Progress: {frame_idx}/{total_frames} "
                  f"({pct:.0f}%)  Alerts: {alert_count}")

    if pbar:
        pbar.close()
    if show:
        cv2.destroyAllWindows()
    cap.release()
    writer.release()

    # ── Compress ──────────────────────────────────────────────────────────
    compress_video(temp_path, out_path, crf=crf)

    # ── Clean up temp file ────────────────────────────────────────────────
    Path(temp_path).unlink(missing_ok=True)

    # ── Save alert log ────────────────────────────────────────────────────
    import json
    log_path = str(OUTPUT_DIR / f"{src_name}_{gate_tag}_alerts.json")
    with open(log_path, "w") as f:
        json.dump({
            "source":       source,
            "total_frames": frame_idx,
            "alert_frames": alert_count,
            "alert_rate":   f"{alert_count/max(frame_idx,1)*100:.1f}%",
            "alerts":       alert_log,
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  Detection Complete")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Alert frames     : {alert_count}")
    print(f"  Alert rate       : {alert_count/max(frame_idx,1)*100:.1f}%")
    print(f"  Output video     : {out_path}")
    print(f"  Alert log        : {log_path}")
    print(f"{'═'*50}\n")


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-gate weapon detection"
    )
    parser.add_argument("--source",        type=str,   default="0")
    parser.add_argument("--det-weights",   type=str,
                        default="logs/student/best_blender.pt")
    parser.add_argument("--cls-weights",   type=str,
                        default="logs/fp_correction/classifier_bg_best.pt")
    parser.add_argument("--det-conf",      type=float, default=0.25)
    parser.add_argument("--cls-conf",      type=float, default=0.50)
    parser.add_argument("--crf",           type=int,   default=23,
                        help="Compression quality: 18=best 23=default 28=smallest")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Run Gate 1 (detector) only")
    parser.add_argument("--show",          action="store_true",
                        help="Display frames in a live window while processing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Detection] Device: {device}")

    detector   = load_detector(args.det_weights,  device)
    classifier = None if args.no_classifier else \
                 load_classifier(args.cls_weights, device)

    run(
        source         = args.source,
        detector       = detector,
        classifier     = classifier,
        device         = device,
        det_conf       = args.det_conf,
        cls_conf       = args.cls_conf,
        use_classifier = not args.no_classifier,
        crf            = args.crf,
        show           = args.show,
    )


if __name__ == "__main__":
    main()