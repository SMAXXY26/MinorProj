"""
inference.py  —  Real-time weapon detection pipeline (v2, modular)
==================================================================
Pipeline stages (each lives in its own function / class):

  1. load_models()    — build ModelBundle from config + CLI args
  2. process_frame()  — one full forward pass on a single frame
  3. run()            — video capture loop, FPS tracking, I/O
  4. main()           — CLI argument parsing

Module layout
─────────────
  inference.py
  utils/
    person_tracker.py  — PersonDetector, WeaponPersonAssociator, draw_person_overlays
    tracker.py         — WeaponTracker, TrackedWeapon
    alert.py           — AlertManager
    dispatchers.py     — ConsoleDispatcher, FileLogDispatcher, WebhookDispatcher
    geo.py             — GpsReader / MockGpsReader / BboxToGeoProjector

Usage:
    # Full pipeline — Jetson + TRT
    python3 inference.py --source 0 --trt --config config/hyperparams.yaml

    # Desktop test with person tracking enabled
    python3 inference.py --source test_video.mp4 --person-track

    # Suppress window (headless / SSH)
    python3 inference.py --source 0 --no-display --output outputs/run.mp4
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES      = ["knife", "pistol", "rifle"]
CLASS_COLORS_BGR = {
    "knife":  (0xDD, 0x8A, 0x37),
    "pistol": (0x4A, 0x4B, 0xE2),
    "rifle":  (0x17, 0x75, 0xBA),
}
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
#  1. Config
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
#  2. Power-mode check (Jetson)
# =============================================================================

def check_power_mode() -> int | None:
    """Return 0 if MAXN, -1 if other mode, None if not on Jetson."""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if "NV Power Mode" in line:
                return 0 if ("MAXN" in line or "MODE_0" in line) else -1
        return -1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# =============================================================================
#  3. Camera / source helpers
# =============================================================================

def _gstreamer_pipeline(sensor_id: int, w: int = 1280, h: int = 720, fps: int = 30) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={w}, height={h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def open_source(source: str) -> cv2.VideoCapture:
    """Open a video source; tries GStreamer CSI pipeline first for integer sources."""
    if source.isdigit():
        sensor_id = int(source)
        cap = cv2.VideoCapture(_gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.warning("[Camera] GStreamer failed — falling back to VideoCapture")
            cap = cv2.VideoCapture(sensor_id)
    else:
        cap = cv2.VideoCapture(source)
    return cap


# =============================================================================
#  4. TRT Classifier Runner (Jetson INT8)
# =============================================================================

class TRTClassifierRunner:
    """
    Deserializes a TensorRT engine and runs batched inference using
    pinned host+device memory and a CUDA stream.  Batch size 1..8.
    """

    def __init__(self, engine_path: str) -> None:
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("tensorrt not installed — needs JetPack 6")

        self._trt = trt
        TRT_LOGGER        = trt.Logger(trt.Logger.WARNING)
        runtime           = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine  = runtime.deserialize_cuda_engine(f.read())
        self._context     = self._engine.create_execution_context()
        self._input_name  = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        self._max_batch   = 8

        in_shape  = (self._max_batch, 3, 224, 224)
        out_shape = (self._max_batch, len(CLASS_NAMES))
        self._in_host  = torch.zeros(in_shape,  dtype=torch.float32).pin_memory()
        self._out_host = torch.zeros(out_shape, dtype=torch.float32).pin_memory()
        self._in_dev   = torch.zeros(in_shape,  dtype=torch.float32).cuda()
        self._out_dev  = torch.zeros(out_shape, dtype=torch.float32).cuda()
        self._stream   = torch.cuda.Stream()
        logger.info(f"[TRT] Classifier engine: {engine_path}")

    def run(self, batch_np: np.ndarray) -> np.ndarray:
        """batch_np: (N,3,224,224) float32 → returns (N, num_classes) logits"""
        n = batch_np.shape[0]
        assert n <= self._max_batch
        self._in_host[:n] = torch.from_numpy(batch_np)
        with torch.cuda.stream(self._stream):
            self._in_dev[:n].copy_(self._in_host[:n], non_blocking=True)
        self._context.set_input_shape(self._input_name, (n, 3, 224, 224))
        self._context.execute_async_v2(
            [self._in_dev.data_ptr(), self._out_dev.data_ptr()],
            stream_handle=self._stream.cuda_stream,
        )
        with torch.cuda.stream(self._stream):
            self._out_host[:n].copy_(self._out_dev[:n], non_blocking=True)
        self._stream.synchronize()
        return self._out_host[:n].numpy()


# =============================================================================
#  5. PyTorch classifier helpers
# =============================================================================

def _load_pt_classifier(weights: str, device: str, num_classes: int):
    from src.model import WeaponClassifier
    from torchvision import transforms

    ckpt     = torch.load(weights, map_location=device)
    nc       = ckpt.get("num_classes", num_classes)
    backbone = ckpt.get("backbone", "efficientnet_b5")
    model    = WeaponClassifier(num_classes=nc, dropout=0.0, pretrained=False,
                                backbone=backbone)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(device)

    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    logger.info(f"[Classifier] PyTorch: {weights}")
    return model, xform


@torch.no_grad()
def _pt_classify_crop(model, xform, crop_bgr: np.ndarray, device: str) -> np.ndarray:
    """Returns softmax probabilities (num_classes,)."""
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = xform(rgb).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=-1)[0].cpu().numpy()


# =============================================================================
#  6. Temporal blending buffer (BiLSTM smoother)
# =============================================================================

class TemporalBuffer:
    """
    Per-track sliding window of 8-dim feature vectors.
    Blends LSTM output with classifier output once window is full.
    """

    def __init__(self, window_size: int = 16, num_classes: int = 3) -> None:
        self._window   = window_size
        self._nc       = num_classes
        self._buffers: dict[int, deque] = {}
        self._smoother = None

    def _load_smoother(self, device: str):
        path = Path("logs/temporal/best.pt")
        if not path.exists():
            return None
        try:
            from part3 import TemporalSmoother
            ckpt = torch.load(str(path), map_location=device)
            m    = TemporalSmoother(
                input_size  = ckpt["input_size"],
                hidden_size = ckpt["hidden_size"],
                num_layers  = ckpt["num_layers"],
                num_classes = ckpt["num_classes"],
            ).to(device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            logger.info("[Temporal] LSTM smoother loaded")
            return m
        except Exception as exc:
            logger.warning(f"[Temporal] Could not load smoother: {exc}")
            return None

    def update(
        self,
        track_id:  int,
        cls_probs: np.ndarray,
        bbox_xyxy: np.ndarray,
        img_w: int,
        img_h: int,
        age:   int,
        device: str,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        x_c = ((x1+x2)/2.0) / max(img_w, 1)
        y_c = ((y1+y2)/2.0) / max(img_h, 1)
        w_n = (x2-x1)       / max(img_w, 1)
        h_n = (y2-y1)       / max(img_h, 1)
        asp = w_n / max(h_n, 1e-3)
        feat = np.array(
            list(cls_probs[:self._nc]) + [x_c, y_c, w_n, h_n, asp],
            dtype=np.float32,
        )[:8]

        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self._window)
        self._buffers[track_id].append(feat)

        if age < self._window or len(self._buffers[track_id]) < self._window:
            return cls_probs
        if self._smoother is None:
            self._smoother = self._load_smoother(device)
        if self._smoother is None:
            return cls_probs

        window_arr = np.stack(list(self._buffers[track_id]))
        with torch.no_grad():
            x = torch.tensor(window_arr).unsqueeze(0).to(device)
            conf_pred, cls_logits = self._smoother(x)
            t_conf = float(conf_pred[0, -1])
            t_cls  = int(cls_logits[0, -1].argmax())

        temporal_probs = np.zeros(self._nc, dtype=np.float32)
        if 0 <= t_cls < self._nc:
            temporal_probs[t_cls] = t_conf

        blended = 0.6 * cls_probs + 0.4 * temporal_probs
        return blended / max(blended.sum(), 1e-6)

    def drop_track(self, tid: int) -> None:
        self._buffers.pop(tid, None)

    def cleanup(self, active_ids: set[int]) -> None:
        for tid in [t for t in self._buffers if t not in active_ids]:
            self.drop_track(tid)


# =============================================================================
#  7. Drawing helpers
# =============================================================================

def draw_weapon_track(frame: np.ndarray, track: Any) -> None:
    x1, y1, x2, y2 = map(int, track.bbox_xyxy)
    color  = CLASS_COLORS_BGR.get(track.class_name, (0, 255, 0))
    label  = f"{track.class_name.upper()} #{track.track_id} | {track.confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
    cv2.putText(frame, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    n_persons: int,
    n_armed: int,
    geo: Any,
    power_mode: int | None,
) -> None:
    H, W = frame.shape[:2]

    # Top-left: FPS + weapon tracks
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Weapons: {n_tracks}  |  Armed persons: {n_armed}/{n_persons}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 2,
    )

    # Top-right: power mode dot
    if power_mode is not None:
        dot_color = (0, 255, 0) if power_mode == 0 else (0, 165, 255)
        mode_txt  = "MAXN" if power_mode == 0 else "ECO"
        cv2.circle(frame, (W-20, 20), 8, dot_color, -1)
        cv2.putText(frame, mode_txt, (W-60, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

    # Bottom-left: GPS
    geo_str = (
        f"GPS: {geo.lat:.5f}, {geo.lon:.5f}  |  Alt: {geo.alt_m:.1f}m  |  "
        f"Hdg: {geo.heading_deg:.0f}\u00b0"
        if geo else "GPS: N/A"
    )
    cv2.putText(frame, geo_str, (10, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# =============================================================================
#  8. ModelBundle — holds every loaded model / helper for the pipeline
# =============================================================================

@dataclass
class ModelBundle:
    detector:       Any                         # YOLO weapon detector
    pt_classifier:  Any | None                  # EfficientNet PyTorch
    pt_xform:       Any | None
    trt_runner:     TRTClassifierRunner | None
    tracker:        Any                         # WeaponTracker
    temporal:       TemporalBuffer
    alert_manager:  Any                         # AlertManager
    gps_reader:     Any | None
    projector:      Any | None                  # BboxToGeoProjector
    person_detector: Any | None                 # PersonDetector (optional)
    associator:     Any | None                  # WeaponPersonAssociator (optional)
    device:         str
    num_classes:    int
    det_conf:       float


# =============================================================================
#  9. load_models — builds a ModelBundle from args + config
# =============================================================================

def load_models(args: argparse.Namespace, cfg: dict, device: str) -> ModelBundle:
    from ultralytics import YOLO
    from utils.tracker     import WeaponTracker
    from utils.alert       import AlertManager
    from utils.dispatchers import ConsoleDispatcher, FileLogDispatcher, WebhookDispatcher

    trt_cfg     = cfg.get("trt", {})
    num_classes = cfg["dataset"]["num_classes"]
    det_conf    = float(cfg["detector"].get("conf_threshold", 0.45))

    # ── Detector ─────────────────────────────────────────────────────────────
    if args.trt:
        det_engine = trt_cfg.get("detector_engine", "logs/trt/detector_int8.engine")
        if Path(det_engine).exists():
            detector = YOLO(det_engine)
            logger.info(f"[Detector] TRT engine: {det_engine}")
        else:
            logger.warning("[Detector] TRT engine not found — falling back to PyTorch")
            args.trt = False
            detector = YOLO("logs/detector/best.pt")
    else:
        det_path = args.det_ckpt or "logs/detector/best.pt"
        if not Path(det_path).exists():
            logger.error(f"Detector not found: {det_path}  (run part1.py first)")
            sys.exit(1)
        detector = YOLO(det_path)
        logger.info(f"[Detector] PyTorch: {det_path}")

    # ── Classifier ────────────────────────────────────────────────────────────
    trt_runner = pt_classifier = pt_xform = None

    if args.trt and trt_cfg.get("enabled", False):
        cls_engine = trt_cfg.get("classifier_engine", "logs/trt/classifier_int8.engine")
        if Path(cls_engine).exists():
            try:
                trt_runner = TRTClassifierRunner(cls_engine)
            except Exception as exc:
                logger.warning(f"[Classifier] TRT init failed: {exc} — PyTorch fallback")
        else:
            logger.warning("[Classifier] TRT engine not found — PyTorch fallback")

    if trt_runner is None:
        cls_path = "logs/classifier/best.pt"
        if Path(cls_path).exists():
            pt_classifier, pt_xform = _load_pt_classifier(cls_path, device, num_classes)
        else:
            logger.warning("[Classifier] Not found — detector-only mode")

    # ── Tracker + Temporal ────────────────────────────────────────────────────
    tracker  = WeaponTracker(cfg)
    temporal = TemporalBuffer(
        window_size = cfg["temporal"].get("window_size", 16),
        num_classes = num_classes,
    )

    # ── Geo ───────────────────────────────────────────────────────────────────
    geo_cfg    = cfg.get("geo", {})
    gps_reader = projector = None

    if geo_cfg.get("enabled", False):
        from utils.geo import BboxToGeoProjector, GpsReader, MockGpsReader
        if geo_cfg.get("use_mock_gps", False):
            gps_reader = MockGpsReader(geo_cfg)
            logger.info("[Geo] MockGpsReader")
        else:
            try:
                gps_reader = GpsReader(geo_cfg)
                logger.info(f"[Geo] GpsReader on {geo_cfg.get('gps_serial_port', '/dev/ttyTHS0')}")
            except Exception as exc:
                logger.warning(f"[Geo] GpsReader failed: {exc}")

        if gps_reader:
            projector = BboxToGeoProjector(
                fov_h_deg = float(geo_cfg.get("camera_fov_h_deg", 84.0)),
                fov_v_deg = float(geo_cfg.get("camera_fov_v_deg", 54.0)),
                img_w=1280, img_h=720,
            )

    # ── Alert manager ─────────────────────────────────────────────────────────
    alert_cfg   = cfg.get("alert", {})
    dispatchers = []
    if alert_cfg.get("console", {}).get("enabled", True):
        dispatchers.append(ConsoleDispatcher())
    if alert_cfg.get("file_log", {}).get("enabled", True):
        log_dir = alert_cfg.get("file_log", {}).get("log_dir", "logs/alerts")
        dispatchers.append(FileLogDispatcher(log_dir))
    if alert_cfg.get("webhook", {}).get("enabled", False):
        url     = alert_cfg["webhook"]["url"]
        timeout = int(alert_cfg["webhook"].get("timeout_sec", 2))
        dispatchers.append(WebhookDispatcher(url, timeout))

    alert_manager = AlertManager(cfg, dispatchers)

    # ── Person detector + associator (optional) ───────────────────────────────
    person_detector = associator = None
    if args.person_track:
        from utils.person_tracker import PersonDetector, WeaponPersonAssociator
        person_detector = PersonDetector(
            weights        = "yolov8n.pt",
            conf_threshold = 0.40,
            device         = device,
        )
        associator = WeaponPersonAssociator(
            iou_threshold        = 0.15,
            max_centroid_dist_px = 250.0,
        )
        logger.info("[PersonTracker] Enabled — IoU + centroid association")

    return ModelBundle(
        detector        = detector,
        pt_classifier   = pt_classifier,
        pt_xform        = pt_xform,
        trt_runner      = trt_runner,
        tracker         = tracker,
        temporal        = temporal,
        alert_manager   = alert_manager,
        gps_reader      = gps_reader,
        projector       = projector,
        person_detector = person_detector,
        associator      = associator,
        device          = device,
        num_classes     = num_classes,
        det_conf        = det_conf,
    )


# =============================================================================
#  10. FrameResult — output of one process_frame call
# =============================================================================

@dataclass
class FrameResult:
    confirmed:         list = field(default_factory=list)   # confirmed weapon tracks
    person_detections: list = field(default_factory=list)   # PersonDetection list
    carrier_ids:       set  = field(default_factory=set)    # armed person_ids
    geo:               Any  = None                          # latest GeoPoint


# =============================================================================
#  11. process_frame — one full pipeline pass, annotates frame in-place
# =============================================================================

def process_frame(
    frame:     np.ndarray,
    bundle:    ModelBundle,
    frame_idx: int,
    img_w:     int,
    img_h:     int,
) -> FrameResult:
    import supervision as sv
    from utils.person_tracker import draw_person_overlays

    result = FrameResult()

    # ── Gate 1: weapon detection ──────────────────────────────────────────────
    det_results = bundle.detector(
        frame, conf=bundle.det_conf, verbose=False, device=bundle.device
    )
    sv_dets = sv.Detections.from_ultralytics(det_results[0])

    # ── Tracking ──────────────────────────────────────────────────────────────
    tracks    = bundle.tracker.update(sv_dets, frame_idx)
    confirmed = bundle.tracker.get_confirmed_tracks()
    result.confirmed = confirmed

    bundle.temporal.cleanup({t.track_id for t in tracks})

    # ── GPS fix ───────────────────────────────────────────────────────────────
    frame_geo = bundle.gps_reader.get_latest() if bundle.gps_reader else None
    result.geo = frame_geo

    # ── Gate 2: classify + temporal blend on confirmed weapon tracks ──────────
    weapon_bboxes: list[np.ndarray] = []

    for trk in confirmed:
        x1, y1, x2, y2 = map(int, trk.bbox_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Classify
        if bundle.trt_runner is not None:
            c         = cv2.resize(crop, (224, 224))
            c         = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            c         = (c - IMG_MEAN) / IMG_STD
            raw       = bundle.trt_runner.run(c.transpose(2,0,1)[np.newaxis])[0]
            cls_probs = np.exp(raw) / np.exp(raw).sum()
        elif bundle.pt_classifier is not None:
            cls_probs = _pt_classify_crop(bundle.pt_classifier, bundle.pt_xform, crop, bundle.device)
        else:
            cls_probs = None

        if cls_probs is not None:
            cls_probs = bundle.temporal.update(
                track_id  = trk.track_id,
                cls_probs = cls_probs,
                bbox_xyxy = trk.bbox_xyxy,
                img_w=img_w, img_h=img_h,
                age=trk.age,
                device=bundle.device,
            )
            best_cls       = int(cls_probs.argmax())
            trk.confidence = float(cls_probs[best_cls])
            if 0 <= best_cls < len(CLASS_NAMES):
                trk.class_name = CLASS_NAMES[best_cls]
                trk.class_id   = best_cls

        # Geo project
        if frame_geo is not None and bundle.projector is not None:
            trk.geo = bundle.projector.project(trk.bbox_xyxy, frame_geo,
                                                altitude_agl_m=frame_geo.alt_m)
        else:
            trk.geo = frame_geo

        weapon_bboxes.append(trk.bbox_xyxy.copy())

        # Draw weapon box
        draw_weapon_track(frame, trk)

        # Fire alerts
        bundle.alert_manager.evaluate(trk, trk.geo, frame_idx)

    bundle.alert_manager.cleanup_stale({t.track_id for t in tracks})

    # ── Person detection + association ────────────────────────────────────────
    if bundle.person_detector is not None and bundle.associator is not None:
        persons     = bundle.person_detector.detect(frame)
        carrier_ids = bundle.associator.associate(weapon_bboxes, persons)

        # Pick the weapon class name for the label (most common among confirmed)
        weapon_label = ""
        if confirmed and carrier_ids:
            weapon_label = confirmed[0].class_name.upper()

        draw_person_overlays(frame, persons, carrier_ids, weapon_label)
        result.person_detections = persons
        result.carrier_ids       = carrier_ids

    return result


# =============================================================================
#  12. run — capture loop (I/O + FPS; calls process_frame each iteration)
# =============================================================================

def run(args: argparse.Namespace, cfg: dict) -> None:
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    power_mode = check_power_mode()

    if power_mode is not None and power_mode != 0:
        logger.warning(
            "[WARNING] Not in MAXN mode — run: sudo nvpmodel -m 0 && sudo jetson_clocks"
        )

    bundle = load_models(args, cfg, device)

    cap = open_source(args.source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {args.source}")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if bundle.projector is not None:
        bundle.projector.img_w = W
        bundle.projector.img_h = H

    logger.info(f"[Source] {args.source}  {W}x{H}  {src_fps:.0f} fps  device={device}")

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H)
        )
        logger.info(f"[Output] Saving to {args.output}")

    fps_buffer: deque = deque(maxlen=30)
    t_prev = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Memory monitor
        if frame_idx % 100 == 0 and torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            if mem_gb > 3.5:
                logger.warning(f"[Memory] {mem_gb:.1f} GB — near 8 GB limit")

        # ── Full pipeline pass ────────────────────────────────────────────────
        result = process_frame(frame, bundle, frame_idx, W, H)

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        fps_buffer.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev   = t_now
        disp_fps = float(np.mean(fps_buffer))

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(
            frame,
            fps        = disp_fps,
            n_tracks   = len(result.confirmed),
            n_persons  = len(result.person_detections),
            n_armed    = len(result.carrier_ids),
            geo        = result.geo,
            power_mode = power_mode,
        )

        # ── Write / Display ───────────────────────────────────────────────────
        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("WeaponDetection v2", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if bundle.gps_reader:
        bundle.gps_reader.stop()
    logger.info(f"[Done] {frame_idx} frames  avg {disp_fps:.1f} fps")


# =============================================================================
#  13. Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeaponDetection v2 — Jetson Orin Nano / desktop"
    )
    parser.add_argument("--source",       type=str, default="0",
                        help="Video source: camera index or file/RTSP path")
    parser.add_argument("--config",       type=str, default="config/hyperparams.yaml")
    parser.add_argument("--trt",          action="store_true",
                        help="Use TensorRT INT8 engines (Jetson)")
    parser.add_argument("--det-ckpt",     type=str, default=None,
                        help="Override detector weights (default: logs/detector/best.pt)")
    parser.add_argument("--output",       type=str, default=None,
                        help="Save annotated video to path (e.g. outputs/run.mp4)")
    parser.add_argument("--no-display",   action="store_true",
                        help="Headless mode — suppress cv2.imshow")
    parser.add_argument("--person-track", action="store_true",
                        help="Enable person detection and weapon-carrier association overlay")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(args, cfg)


if __name__ == "__main__":
    main()
