"""
inference.py — Weapon Detection + Geometry Inference on Video

Usage:
    python inference.py --source path/to/john_wick.mp4 \
                        --det-ckpt runs/detector/best.pt \
                        --cls-ckpt runs/classifier/best.pt \
                        --output  output/annotated.mp4

What this script does:
  1. Reads frames from video (with scene-change aware sampling)
  2. Runs YOLOv8x-OBB detector → bounding boxes + rotation angle
  3. Crops each detection, runs EfficientNet-B5 classifier
  4. Extracts full geometry: corners, ellipse, keypoints, aspect ratio
  5. Runs BiLSTM temporal smoother over a sliding window of frames
  6. Draws rich annotations: OBB + contour + keypoints + metadata
  7. Writes annotated video

John Wick–specific adaptations:
  - Motion deblur triggered on low-sharpness frames (Laplacian variance)
  - Low-light CLAHE applied before detector on dark frames
  - Temporal smoother suppresses single-frame FP spikes (common in fast cuts)
  - Geometry shown: angle, aspect ratio, tip/grip keypoints
"""

import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from typing import List, Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model    import WeaponDetector, WeaponClassifier, TemporalSmoother
from src.geometry import (extract_geometry, draw_geometry,
                           compute_motion_magnitude, WeaponGeometry)

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette per weapon class
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["pistol", "revolver", "rifle", "shotgun",
                "smg", "knife", "blunt_weapon"]
CLASS_COLORS = [
    (0,   220,  80),   # pistol        — green
    (0,   180, 255),   # revolver      — cyan
    (255, 100,   0),   # rifle         — orange
    (255,  60, 200),   # shotgun       — magenta
    (200, 200,   0),   # smg           — yellow
    (0,   100, 255),   # knife         — blue
    (180,   0, 255),   # blunt_weapon  — purple
]


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame_bgr: np.ndarray,
                     img_size: int = 640,
                     apply_clahe: bool = False,
                     apply_deblur: bool = False) -> torch.Tensor:
    """
    BGR frame → normalised tensor (1, 3, H, W).

    Applies CLAHE for dark frames and optional Wiener-like sharpening for blur.
    """
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── Low-light: CLAHE on L channel ────────────────────────────────────────
    if apply_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── Motion deblur: unsharp mask (lightweight, no NN required) ────────────
    if apply_deblur:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        frame   = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        frame   = frame.clip(0, 255).astype(np.uint8)

    # ── Letterbox resize ──────────────────────────────────────────────────────
    h, w = frame.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = (img_size - nh) // 2
    pad_w = (img_size - nw) // 2
    frame = cv2.copyMakeBorder(frame, pad_h, img_size - nh - pad_h,
                                pad_w, img_size - nw - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # ── Normalise ─────────────────────────────────────────────────────────────
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def is_dark_frame(frame_bgr: np.ndarray, threshold: float = 60.0) -> bool:
    """True if mean luminance < threshold (triggers CLAHE)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def is_blurry_frame(frame_bgr: np.ndarray, threshold: float = 100.0) -> bool:
    """True if Laplacian variance < threshold (triggers deblur)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def scene_change(prev_gray: np.ndarray, curr_gray: np.ndarray,
                 threshold: float = 35.0) -> bool:
    """Detect hard cut / scene change via mean absolute frame difference."""
    if prev_gray is None:
        return True
    diff = cv2.absdiff(prev_gray, curr_gray).mean()
    return diff > threshold


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal feature buffer for BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBuffer:
    """
    Maintains a sliding window of per-frame feature vectors.
    Feeds the BiLSTM smoother every time the window is full.
    """
    def __init__(self, window_size: int = 8, feature_dim: int = 14):
        self.window   = window_size
        self.feat_dim = feature_dim
        self.buffer   = deque(maxlen=window_size)

    def push(self, feat_vec: np.ndarray):
        """feat_vec: (14,) numpy array."""
        self.buffer.append(feat_vec)

    def ready(self) -> bool:
        return len(self.buffer) == self.window

    def get_tensor(self) -> torch.Tensor:
        """Returns (1, T, 14) float32 tensor."""
        arr = np.stack(list(self.buffer), axis=0)   # (T, 14)
        return torch.from_numpy(arr).unsqueeze(0)   # (1, T, 14)

    def reset(self):
        self.buffer.clear()


def build_feature_vector(det, cls_probs: np.ndarray,
                          motion_mag: float) -> np.ndarray:
    """
    Build the 14-dim feature vector for one frame's top detection.
    [conf, cx_n, cy_n, w_n, h_n, θ_n, p0..p6, motion_mag]
    """
    if det is None:
        return np.zeros(14, dtype=np.float32)
    cx, cy, w, h, theta, conf, _ = det
    # Normalise spatial coords to [0, 1] (assume 640 img size)
    feat = np.array([
        conf,
        cx / 640, cy / 640, w / 640, h / 640,
        theta / (np.pi / 2),   # normalise angle to [-1, 1]
        *cls_probs,             # 7 softmax scores
        min(motion_mag, 1.0),
    ], dtype=np.float32)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
#  Annotation drawing
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray,
                   detections: list,
                   geometries: List[WeaponGeometry],
                   smoothed_confs: Optional[List[float]] = None) -> np.ndarray:
    """
    Draw all detections + geometry onto the BGR frame.
    Each detection dict: box_obb, conf, det_class, cls_class, cls_conf, geometry
    """
    vis = frame.copy()

    for i, (det, geom) in enumerate(zip(detections, geometries)):
        cls_name = det.get("cls_class", det.get("det_class", "weapon"))
        conf     = det.get("conf", 0.0)
        cls_conf = det.get("cls_conf", 0.0)

        # Use temporal-smoothed confidence if available
        if smoothed_confs and i < len(smoothed_confs):
            conf = smoothed_confs[i]

        cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
        color   = CLASS_COLORS[cls_idx]

        # ── Geometry overlay ─────────────────────────────────────────────────
        label = f"{cls_name} {conf:.2f} (cls:{cls_conf:.2f})"
        vis   = draw_geometry(vis, geom, label=label, color=color)

        # ── HUD: angle + aspect ratio info box ───────────────────────────────
        cx_i  = int(geom.cx_px)
        cy_i  = int(geom.cy_px)
        box_y = max(30, cy_i - int(geom.height_px/2) - 40)

        hud_lines = [
            f"θ = {geom.angle_deg:.1f}°",
            f"AR = {geom.aspect_ratio:.1f}x",
            f"ecc = {geom.eccentricity:.2f}",
            f"{geom.shape_hint}",
        ]
        for li, text in enumerate(hud_lines):
            cv2.putText(
                vis, text,
                (cx_i - 80, box_y + li * 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                tuple(max(0, c - 30) for c in color),
                1, cv2.LINE_AA,
            )

    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    source:        str,
    det_ckpt:      str,
    cls_ckpt:      str,
    smo_ckpt:      Optional[str] = None,
    output:        str = "output/annotated.mp4",
    conf_thresh:   float = 0.35,
    img_size:      int   = 640,
    sample_every:  int   = 2,       # process every Nth frame (speed/quality tradeoff)
    device_str:    str   = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[Loading] Detector …")
    detector = WeaponDetector(num_classes=7).to(device).eval()
    ckpt = torch.load(det_ckpt, map_location=device)
    detector.load_state_dict(ckpt.get("model", ckpt), strict=False)

    print("[Loading] Classifier …")
    classifier = WeaponClassifier(num_classes=7, pretrained=False).to(device).eval()
    ckpt = torch.load(cls_ckpt, map_location=device)
    classifier.load_state_dict(ckpt.get("model", ckpt), strict=False)

    smoother = None
    if smo_ckpt and Path(smo_ckpt).exists():
        print("[Loading] Temporal smoother …")
        smoother = TemporalSmoother().to(device).eval()
        ckpt = torch.load(smo_ckpt, map_location=device)
        smoother.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}×{height}  {fps:.1f}fps  {total} frames  →  {output}")

    # ── Video writer ──────────────────────────────────────────────────────────
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps / sample_every, (width, height))

    # ── Temporal buffer ───────────────────────────────────────────────────────
    t_buffer  = TemporalBuffer(window_size=8, feature_dim=14)
    prev_gray = None
    frame_idx = 0
    t0 = time.time()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Sample every Nth frame
            if frame_idx % sample_every != 0:
                continue

            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # ── Scene change detection ────────────────────────────────────────
            is_cut = scene_change(prev_gray, curr_gray)
            if is_cut:
                t_buffer.reset()   # clear temporal state on hard cut

            # ── Per-frame quality adaptations ─────────────────────────────────
            dark   = is_dark_frame(frame_bgr)
            blurry = is_blurry_frame(frame_bgr)

            # ── Compute optical flow for temporal feature ─────────────────────
            motion_mag = 0.0
            if prev_gray is not None and not is_cut:
                motion_mag = compute_motion_magnitude(prev_gray, curr_gray)

            prev_gray = curr_gray

            # ── Preprocess ───────────────────────────────────────────────────
            tensor = preprocess_frame(
                frame_bgr, img_size,
                apply_clahe  = dark,
                apply_deblur = blurry,
            ).to(device)

            # ── Stage 1: Detect ───────────────────────────────────────────────
            raw_preds = detector(tensor)
            dets_list = detector.decode_predictions(
                raw_preds, img_size=img_size, conf_thresh=conf_thresh
            )
            dets = dets_list[0]   # (N, 7) for first (only) batch item

            # ── Stage 2: Classify + Geometry ─────────────────────────────────
            results    = []
            geometries = []
            top_det    = None
            top_cls_probs = np.zeros(7, dtype=np.float32)

            for det_row in dets:
                cx, cy, w, h, theta, conf, cls_id = det_row.tolist()

                # Crop → classify
                x1 = max(0, int(cx - w/2 * 1.15))
                y1 = max(0, int(cy - h/2 * 1.15))
                x2 = min(img_size, int(cx + w/2 * 1.15))
                y2 = min(img_size, int(cy + h/2 * 1.15))

                # Map from letterboxed coords back to original frame for vis
                scale = max(height, width) / img_size
                cx_orig = int(cx * scale)
                cy_orig = int(cy * scale)
                w_orig  = int(w  * scale)
                h_orig  = int(h  * scale)

                # Classify the crop
                crop_tensor = tensor[:, :, y1:y2, x1:x2]
                if crop_tensor.numel() > 0:
                    crop_resized = F.interpolate(
                        crop_tensor, size=(224, 224),
                        mode="bilinear", align_corners=False
                    )
                    cls_logits  = classifier(crop_resized)[0]
                    cls_probs   = F.softmax(cls_logits, dim=-1).cpu().numpy()
                    cls_id_ref  = int(cls_probs.argmax())
                    cls_conf    = float(cls_probs.max())
                else:
                    cls_probs  = np.ones(7) / 7
                    cls_id_ref = int(cls_id)
                    cls_conf   = float(conf)

                # Geometry extraction
                geom = extract_geometry(
                    cx_orig, cy_orig, w_orig, h_orig,
                    theta, max(height, width)
                )
                geometries.append(geom)

                det_info = {
                    "box_obb":   (cx_orig, cy_orig, w_orig, h_orig, theta),
                    "conf":      float(conf),
                    "det_class": CLASS_NAMES[min(int(cls_id), 6)],
                    "cls_class": CLASS_NAMES[cls_id_ref],
                    "cls_conf":  cls_conf,
                    "geometry":  geom,
                }
                results.append(det_info)

                if top_det is None or conf > top_det[5]:
                    top_det = det_row.tolist()
                    top_cls_probs = cls_probs

            # ── Stage 3: Temporal smoothing ───────────────────────────────────
            smoothed_confs = None
            if smoother is not None:
                feat_vec = build_feature_vector(top_det, top_cls_probs, motion_mag)
                t_buffer.push(feat_vec)

                if t_buffer.ready():
                    win_tensor = t_buffer.get_tensor().to(device)
                    s_conf, _ = smoother(win_tensor)
                    # Last frame in window = current frame
                    smooth_val = float(s_conf[0, -1, 0])
                    smoothed_confs = [smooth_val] * len(results)

            # ── Annotate + write ──────────────────────────────────────────────
            annotated = annotate_frame(
                frame_bgr, results, geometries, smoothed_confs
            )

            # Frame counter + FPS overlay
            elapsed = time.time() - t0
            proc_fps = frame_idx / max(elapsed, 1e-3)
            cv2.putText(
                annotated,
                f"Frame {frame_idx}/{total}  |  proc {proc_fps:.1f}fps  "
                f"|  dets: {len(results)}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

            writer.write(annotated)

            if frame_idx % 100 == 0:
                print(f"  [{frame_idx}/{total}]  fps={proc_fps:.1f}  "
                      f"dets={len(results)}  dark={dark}  blur={blurry}")

    cap.release()
    writer.release()
    print(f"\n[Done] Output written → {output}")
    print(f"       Processed {frame_idx} frames in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weapon Detection Inference")
    parser.add_argument("--source",       required=True)
    parser.add_argument("--det-ckpt",     required=True)
    parser.add_argument("--cls-ckpt",     required=True)
    parser.add_argument("--smo-ckpt",     default=None)
    parser.add_argument("--output",       default="output/annotated.mp4")
    parser.add_argument("--conf",         type=float, default=0.35)
    parser.add_argument("--img-size",     type=int,   default=640)
    parser.add_argument("--sample-every", type=int,   default=2)
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    run_inference(
        source       = args.source,
        det_ckpt     = args.det_ckpt,
        cls_ckpt     = args.cls_ckpt,
        smo_ckpt     = args.smo_ckpt,
        output       = args.output,
        conf_thresh  = args.conf,
        img_size     = args.img_size,
        sample_every = args.sample_every,
        device_str   = args.device,
    )


if __name__ == "__main__":
    main()
