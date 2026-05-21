"""
inference_pi5.py — Real-time weapon detection on Raspberry Pi 5 + Hailo-8L
===========================================================================
Hardware: Raspberry Pi 5 + Hailo-8L AI HAT (26 TOPS, PCIe)

Inference architecture:
  Hailo-8L: YOLOv8s teacher detection (Gate 1) + EfficientNet classification (Gate 2)
  ARM CPU:  ByteTrack tracking, LSTM temporal blending, geo-stamping, alerts

Backends (auto-selected with --hailo, fallback to ONNX/PyTorch CPU):
  --hailo   Use HEF files on Hailo-8L (fastest: ~30fps at 416px)
  (default) ONNX Runtime on CPU (slower, no HAT required)

Usage:
    python3 inference_pi5.py --source 0 --hailo              # USB cam + Hailo
    python3 inference_pi5.py --source 0 --hailo --picam      # Pi Camera + Hailo
    python3 inference_pi5.py --source test.mp4               # Video file, CPU
    python3 inference_pi5.py --source 0 --hailo --no-display # Headless / SSH
    python3 inference_pi5.py --source rtsp://...             # RTSP stream

Prepare HEF files first (on dev machine):
    python3 export_hailo.py --onnx-only            # export ONNX
    # compile with Hailo DFC, copy logs/hailo/*.hef to Pi 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES      = ["knife", "pistol", "rifle"]
CLASS_COLORS_BGR = {
    "knife":  (0xDD, 0x8A, 0x37),
    "pistol": (0x4A, 0x4B, 0xE2),
    "rifle":  (0x17, 0x75, 0xBA),
}
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═════════════════════════════════════════════════════════════════════════════
#  Camera helpers
# ═════════════════════════════════════════════════════════════════════════════

def _open_picamera2(width: int = 1280, height: int = 720, fps: int = 30):
    """Open Pi Camera Module via libcamera (picamera2). Returns cam or None."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(
            main     = {"size": (width, height), "format": "RGB888"},
            controls = {"FrameRate": fps},
        )
        cam.configure(config)
        cam.start()
        logger.info(f"[Camera] Pi Camera via picamera2 ({width}x{height} @{fps}fps)")
        return cam
    except ImportError:
        logger.warning("[Camera] picamera2 not installed — falling back to cv2.VideoCapture")
        return None
    except Exception as exc:
        logger.warning(f"[Camera] picamera2 init failed: {exc} — falling back to cv2")
        return None


def open_source(source: str, use_picam: bool = False):
    """
    Open video source.
    Returns (cap_or_cam, is_picam): cap_or_cam is cv2.VideoCapture or Picamera2.
    """
    if use_picam and source.isdigit():
        cam = _open_picamera2()
        if cam is not None:
            return cam, True

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)
    logger.info(f"[Camera] cv2.VideoCapture({source})")
    return cap, False


def read_frame(cap_or_cam, is_picam: bool):
    """Read one frame. Returns (success, frame_bgr)."""
    if is_picam:
        frame_rgb = cap_or_cam.capture_array()
        return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return cap_or_cam.read()


def release_source(cap_or_cam, is_picam: bool) -> None:
    if is_picam:
        cap_or_cam.stop()
    else:
        cap_or_cam.release()


# ═════════════════════════════════════════════════════════════════════════════
#  Hailo classifier runner
# ═════════════════════════════════════════════════════════════════════════════

class HailoClassifierRunner:
    """
    Runs EfficientNet classifier on Hailo-8L via hailort Python bindings.

    Input  : float32 (N, 3, H, W)  ImageNet-normalized (CHW, Python batch loop)
    Output : float32 (N, num_classes)  raw logits

    Notes:
    - The HEF is compiled with static batch=1, so we iterate N crops individually.
    - InferVStreams context is re-entered per call to stay compatible across
      hailort versions (4.17–4.20+).
    - Hailo expects NHWC input; we convert CHW→HWC before sending.
    """

    def __init__(self, hef_path: str, img_size: int = 224) -> None:
        try:
            from hailo_platform import (
                HEF, ConfigureParams, HailoStreamInterface,
                InputVStreamParams, OutputVStreamParams, FormatType,
                VDevice,
            )
        except ImportError:
            raise RuntimeError(
                "hailo_platform not installed. "
                "On Pi 5: sudo apt install hailo-all  (from hailo's RPi repository)"
            )

        self._img_size = img_size
        hef = HEF(hef_path)

        self._target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_groups      = self._target.configure(hef, configure_params)
        self._ng            = network_groups[0]
        self._input_params  = InputVStreamParams.make(
            self._ng, quantized=False, format_type=FormatType.FLOAT32
        )
        self._output_params = OutputVStreamParams.make(
            self._ng, quantized=False, format_type=FormatType.FLOAT32
        )
        # Cache stream names (lazily populated on first run)
        self._input_name  = None
        self._output_name = None

        logger.info(f"[HailoClassifier] Loaded: {hef_path}")

    def run(self, batch_np: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed crops.

        Parameters
        ----------
        batch_np : float32 (N, 3, H, W)  — ImageNet-normalized

        Returns
        -------
        float32 (N, num_classes) — raw logits
        """
        from hailo_platform import InferVStreams

        results = []
        for i in range(batch_np.shape[0]):
            # Hailo expects NHWC
            single_hwc = batch_np[i].transpose(1, 2, 0)     # (H, W, 3)
            inp        = single_hwc[np.newaxis]              # (1, H, W, 3)

            with self._ng.activate():
                with InferVStreams(
                    self._ng,
                    self._input_params,
                    self._output_params,
                ) as pipeline:
                    if self._input_name is None:
                        self._input_name  = pipeline.get_input_vstream_infos()[0].name
                        self._output_name = pipeline.get_output_vstream_infos()[0].name

                    out    = pipeline.infer({self._input_name: inp})
                    logits = out[self._output_name].flatten()
                    results.append(logits.astype(np.float32))

        return np.stack(results)                             # (N, num_classes)

    def close(self) -> None:
        del self._target


# ═════════════════════════════════════════════════════════════════════════════
#  ONNX classifier fallback (ARM CPU)
# ═════════════════════════════════════════════════════════════════════════════

class ONNXClassifierRunner:
    """
    EfficientNet via ONNX Runtime on CPU — fallback when Hailo HAT is absent.
    Input/output same contract as HailoClassifierRunner.
    """

    def __init__(self, onnx_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("pip install onnxruntime")

        self._session     = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info(f"[ONNXClassifier] Loaded: {onnx_path}")

    def run(self, batch_np: np.ndarray) -> np.ndarray:
        """batch_np: float32 (N, 3, H, W) — ONNX was exported with static batch=1."""
        results = []
        for i in range(batch_np.shape[0]):
            single = batch_np[i:i+1]   # (1, 3, H, W)
            out    = self._session.run([self._output_name], {self._input_name: single})[0]
            results.append(out[0])
        return np.stack(results)


# ═════════════════════════════════════════════════════════════════════════════
#  Classifier loader (priority: hailo → onnx → pytorch CPU)
# ═════════════════════════════════════════════════════════════════════════════

def _load_classifier(cfg: dict, use_hailo: bool):
    """
    Load the best available classifier backend.
    Returns (runner, runner_type) where runner_type is 'hailo'|'onnx'|'pytorch'|None.
    """
    hailo_cfg  = cfg.get("hailo", {})
    hef_path   = hailo_cfg.get("classifier_hef", "logs/hailo/classifier.hef")
    onnx_path  = hailo_cfg.get("classifier_onnx", "logs/hailo/classifier.onnx")
    img_size   = cfg["classifier"].get("img_size", 224)

    if use_hailo and Path(hef_path).exists():
        try:
            return HailoClassifierRunner(hef_path, img_size), "hailo"
        except Exception as exc:
            logger.warning(f"[Classifier] Hailo init failed ({exc}) — trying ONNX")

    if Path(onnx_path).exists():
        try:
            return ONNXClassifierRunner(onnx_path), "onnx"
        except Exception as exc:
            logger.warning(f"[Classifier] ONNX init failed ({exc}) — trying PyTorch")

    # Last resort: PyTorch CPU
    for cls_pt in [
        "logs/fp_correction/classifier_bg_best.pt",
        "logs/classifier/best.pt",
    ]:
        if not Path(cls_pt).exists():
            continue
        try:
            import torch
            from src.model import WeaponClassifier
            from torchvision import transforms

            ckpt     = torch.load(cls_pt, map_location="cpu")
            nc       = ckpt.get("num_classes", cfg["dataset"]["num_classes"])
            backbone = ckpt.get("backbone", "efficientnet_b5")
            model    = WeaponClassifier(nc, dropout=0.0, pretrained=False,
                                        backbone=backbone)
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()
            xform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info(f"[Classifier] PyTorch CPU: {cls_pt}")
            return (model, xform), "pytorch"
        except Exception as exc:
            logger.warning(f"[Classifier] PyTorch load failed: {exc}")

    logger.warning("[Classifier] No classifier found — detector-only mode")
    return None, None


def _preprocess_crop(crop_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """BGR crop → float32 (1, 3, H, W) ImageNet-normalized."""
    c = cv2.resize(crop_bgr, (img_size, img_size))
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    c = (c - IMG_MEAN) / IMG_STD
    return c.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


def _classify_crop(
    runner,
    runner_type: str,
    crop_bgr: np.ndarray,
    img_size: int,
) -> np.ndarray:
    """Classify one BGR crop. Returns softmax probabilities (num_classes,)."""
    if runner_type in ("hailo", "onnx"):
        batch = _preprocess_crop(crop_bgr, img_size)        # (1, 3, H, W)
        raw   = runner.run(batch)[0]                         # (num_classes,)
        shifted = raw - raw.max()
        probs   = np.exp(shifted)
        return probs / probs.sum()

    # pytorch fallback
    import torch
    model, xform = runner
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = xform(rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=-1)[0]
    return probs.numpy()


# ═════════════════════════════════════════════════════════════════════════════
#  Temporal buffer (self-contained copy; LSTM runs on ARM CPU)
# ═════════════════════════════════════════════════════════════════════════════

class TemporalBuffer:
    def __init__(self, window_size: int = 16, num_classes: int = 3) -> None:
        self._window   = window_size
        self._nc       = num_classes
        self._buffers: dict[int, deque] = {}
        self._smoother = None

    def _load_smoother(self):
        path = Path("logs/temporal/best.pt")
        if not path.exists():
            return None
        try:
            import torch
            from part3 import TemporalSmoother
            ckpt = torch.load(str(path), map_location="cpu")
            m    = TemporalSmoother(
                input_size  = ckpt["input_size"],
                hidden_size = ckpt["hidden_size"],
                num_layers  = ckpt["num_layers"],
                num_classes = ckpt["num_classes"],
            )
            m.load_state_dict(ckpt["model"])
            m.eval()
            logger.info("[Temporal] LSTM smoother loaded (CPU)")
            return m
        except Exception as exc:
            logger.warning(f"[Temporal] Could not load smoother: {exc}")
            return None

    def update(
        self,
        track_id: int,
        cls_probs: np.ndarray,
        bbox_xyxy: np.ndarray,
        img_w: int,
        img_h: int,
        age: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        x_c = ((x1 + x2) / 2.0) / max(img_w, 1)
        y_c = ((y1 + y2) / 2.0) / max(img_h, 1)
        w_n = (x2 - x1) / max(img_w, 1)
        h_n = (y2 - y1) / max(img_h, 1)
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
            self._smoother = self._load_smoother()
        if self._smoother is None:
            return cls_probs

        import torch
        window_arr = np.stack(list(self._buffers[track_id]))
        with torch.no_grad():
            x = torch.tensor(window_arr).unsqueeze(0)
            conf_pred, cls_logits = self._smoother(x)
            t_conf = float(conf_pred[0, -1])
            t_cls  = int(cls_logits[0, -1].argmax())

        t_probs = np.zeros(self._nc, dtype=np.float32)
        if 0 <= t_cls < self._nc:
            t_probs[t_cls] = t_conf

        blended = 0.6 * cls_probs + 0.4 * t_probs
        return blended / max(blended.sum(), 1e-6)

    def cleanup(self, active_ids: set) -> None:
        stale = [tid for tid in list(self._buffers) if tid not in active_ids]
        for tid in stale:
            del self._buffers[tid]


# ═════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _draw_track(frame: np.ndarray, track) -> None:
    x1, y1, x2, y2 = map(int, track.bbox_xyxy)
    color    = CLASS_COLORS_BGR.get(track.class_name, (0, 255, 0))
    label    = f"{track.class_name.upper()} #{track.track_id} | {track.confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    backend: str,
    geo,
) -> None:
    H, W = frame.shape[:2]
    cv2.putText(
        frame,
        f"FPS:{fps:.1f}  Tracks:{n_tracks}  [{backend}]",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2,
    )
    if geo is not None:
        cv2.putText(
            frame,
            f"GPS:{geo.lat:.5f},{geo.lon:.5f}  Alt:{geo.alt_m:.1f}m",
            (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Main inference loop
# ═════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace, cfg: dict) -> None:
    import supervision as sv
    from ultralytics import YOLO

    hailo_cfg = cfg.get("hailo", {})

    # ── Detector ──────────────────────────────────────────────────────────
    det_hef = hailo_cfg.get("detector_hef", "logs/hailo/detector.hef")

    if args.hailo and Path(det_hef).exists():
        detector    = YOLO(det_hef)
        det_backend = "hailo"
        logger.info(f"[Detector] Hailo HEF: {det_hef}")
    else:
        if args.hailo:
            logger.warning(f"[Detector] HEF not found ({det_hef}) — falling back to PyTorch CPU")
        det_weights = None
        for p in [
            "runs/detect/logs/fp_correction/weights/best.pt",
            "logs/fp_correction/detector_ft_best.pt",
            "logs/detector/best.pt",
        ]:
            if Path(p).exists():
                det_weights = p
                break
        if det_weights is None:
            logger.error("No detector weights found. Run export_hailo.py first.")
            sys.exit(1)
        detector    = YOLO(det_weights)
        det_backend = "cpu"
        logger.info(f"[Detector] PyTorch CPU: {det_weights}")

    # ── Classifier ────────────────────────────────────────────────────────
    cls_runner, cls_type = _load_classifier(cfg, use_hailo=args.hailo)
    img_size    = cfg["classifier"].get("img_size", 224)
    backend_str = f"det:{det_backend} cls:{cls_type or 'none'}"
    logger.info(f"[Backend] {backend_str}")

    # ── Tracker ───────────────────────────────────────────────────────────
    from utils.tracker import WeaponTracker
    tracker = WeaponTracker(cfg)

    # ── Temporal buffer ───────────────────────────────────────────────────
    temporal = TemporalBuffer(
        window_size = cfg["temporal"].get("window_size", 16),
        num_classes = cfg["dataset"]["num_classes"],
    )

    # ── Geo ───────────────────────────────────────────────────────────────
    geo_cfg    = cfg.get("geo", {})
    gps_reader = None
    projector  = None

    if geo_cfg.get("enabled", False):
        from utils.geo import BboxToGeoProjector, GpsReader, MockGpsReader

        # Pi 5 default UART port (different from Jetson's /dev/ttyTHS0)
        pi_uart = geo_cfg.get("pi_gps_serial_port", "/dev/serial0")
        geo_with_pi_port = {**geo_cfg, "gps_serial_port": pi_uart}

        if geo_cfg.get("use_mock_gps", False):
            gps_reader = MockGpsReader(geo_cfg)
            logger.info("[Geo] MockGpsReader")
        else:
            try:
                gps_reader = GpsReader(geo_with_pi_port)
                logger.info(f"[Geo] GpsReader on {pi_uart}")
            except Exception as exc:
                logger.warning(f"[Geo] GpsReader failed: {exc} — geo disabled")

    # ── Alert manager ─────────────────────────────────────────────────────
    from utils.alert       import AlertManager
    from utils.dispatchers import ConsoleDispatcher, FileLogDispatcher, WebhookDispatcher

    alert_cfg   = cfg.get("alert", {})
    dispatchers = []
    if alert_cfg.get("console", {}).get("enabled", True):
        dispatchers.append(ConsoleDispatcher())
    if alert_cfg.get("file_log", {}).get("enabled", True):
        dispatchers.append(FileLogDispatcher(
            alert_cfg.get("file_log", {}).get("log_dir", "logs/alerts")
        ))
    if alert_cfg.get("webhook", {}).get("enabled", False):
        dispatchers.append(WebhookDispatcher(
            alert_cfg["webhook"]["url"],
            int(alert_cfg["webhook"].get("timeout_sec", 2)),
        ))
    alert_manager = AlertManager(cfg, dispatchers)

    det_conf = float(cfg["detector"].get("conf_threshold", 0.25))

    # ── Open source ───────────────────────────────────────────────────────
    cap_or_cam, is_picam = open_source(args.source, use_picam=args.picam)

    if is_picam:
        W, H, native_fps = 1280, 720, 30.0
    else:
        W          = int(cap_or_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        H          = int(cap_or_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = cap_or_cam.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info(f"[Source] {args.source}  {W}x{H}  {native_fps:.0f}fps")

    # Build geo projector now that we know frame dimensions
    if gps_reader is not None:
        from utils.geo import BboxToGeoProjector
        projector = BboxToGeoProjector(
            fov_h_deg = float(geo_cfg.get("camera_fov_h_deg", 84.0)),
            fov_v_deg = float(geo_cfg.get("camera_fov_v_deg", 54.0)),
            img_w     = W,
            img_h     = H,
        )

    # ── Output writer ─────────────────────────────────────────────────────
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, native_fps, (W, H))
        logger.info(f"[Output] Saving to {args.output}")

    fps_buffer: deque = deque(maxlen=30)
    frame_idx  = 0
    t_prev     = time.perf_counter()
    disp_fps   = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = read_frame(cap_or_cam, is_picam)
            if not ret:
                break
            frame_idx += 1

            frame_geo = gps_reader.get_latest() if gps_reader else None

            # Gate 1 — YOLOv8s detection via Hailo-8L (or CPU fallback)
            results  = detector(frame, conf=det_conf, verbose=False)
            sv_dets  = sv.Detections.from_ultralytics(results[0])

            # Tracking (ByteTrack on CPU)
            tracks     = tracker.update(sv_dets, frame_idx)
            confirmed  = tracker.get_confirmed_tracks()
            active_ids = {t.track_id for t in tracks}
            temporal.cleanup(active_ids)

            # Gate 2 — Classify + temporal blend confirmed tracks
            for trk in confirmed:
                x1, y1, x2, y2 = map(int, trk.bbox_xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_probs = None
                if cls_runner is not None:
                    cls_probs = _classify_crop(cls_runner, cls_type, crop, img_size)

                if cls_probs is not None:
                    cls_probs = temporal.update(
                        track_id  = trk.track_id,
                        cls_probs = cls_probs,
                        bbox_xyxy = trk.bbox_xyxy,
                        img_w     = W,
                        img_h     = H,
                        age       = trk.age,
                    )
                    best_cls       = int(cls_probs.argmax())
                    trk.confidence = float(cls_probs[best_cls])
                    if 0 <= best_cls < len(CLASS_NAMES):
                        trk.class_name = CLASS_NAMES[best_cls]
                        trk.class_id   = best_cls

                # Geo projection
                trk_geo = None
                if frame_geo is not None and projector is not None:
                    trk_geo = projector.project(
                        trk.bbox_xyxy,
                        frame_geo,
                        altitude_agl_m=frame_geo.alt_m,
                    )
                trk.geo = trk_geo or frame_geo

                _draw_track(frame, trk)
                alert_manager.evaluate(trk, trk.geo, frame_idx)

            alert_manager.cleanup_stale(active_ids)

            # FPS measurement
            t_now = time.perf_counter()
            fps_buffer.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev   = t_now
            disp_fps = float(np.mean(fps_buffer))

            _draw_hud(frame, disp_fps, len(confirmed), backend_str, frame_geo)

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("WeaponDetection v2 — Pi 5", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        release_source(cap_or_cam, is_picam)
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if gps_reader:
            gps_reader.stop()
        if cls_type == "hailo" and hasattr(cls_runner, "close"):
            cls_runner.close()

    logger.info(f"[Done] {frame_idx} frames processed  avg {disp_fps:.1f} fps")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weapon detection on Raspberry Pi 5 + Hailo-8L AI HAT"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0=USB cam, path for file/RTSP",
    )
    parser.add_argument("--config",     default="config/hyperparams.yaml")
    parser.add_argument(
        "--hailo", action="store_true",
        help="Use Hailo-8L for inference (requires logs/hailo/*.hef)",
    )
    parser.add_argument(
        "--picam", action="store_true",
        help="Use Pi Camera Module via picamera2 (instead of USB webcam)",
    )
    parser.add_argument("--output",     default=None,
                        help="Save annotated video to this path")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable cv2.imshow (headless / SSH sessions)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(args, cfg)


if __name__ == "__main__":
    main()
