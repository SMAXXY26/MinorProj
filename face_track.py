"""
face_track.py — Face detection + single-axis servo tracking
============================================================
Hardware : Raspberry Pi 5 + Hailo-8L AI HAT
Servo    : Pan only — GPIO 12 (BCM), same config as servo_track.py

Backends (auto-selected):
  --hailo   Hailo-8L HEF face detector (fastest)
  default   OpenCV DNN face detector (CPU fallback, no download needed)

Usage:
    python3 face_track.py                    # OpenCV CPU
    python3 face_track.py --hailo            # Hailo-8L (needs face.hef)
    python3 face_track.py --hailo --hef logs/hailo/scrfd_2.5g.hef
    python3 face_track.py --source 1         # alternate camera index

Hailo HEF:
    Download scrfd_2.5g.hef from Hailo Model Zoo:
    https://github.com/hailo-ai/hailo_model_zoo
    Compile with: hailo compiler scrfd_2.5g.onnx --hw-arch hailo8l
"""
from __future__ import annotations

import argparse
import collections
import threading
import time

import cv2
import numpy as np
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
FRAME_W, FRAME_H = 320, 240
INFER_EVERY      = 2

PAN_PIN          = 12
PAN_MIN, PAN_MAX = -70, 70
DEADBAND         = 8      # pixels — ignore error smaller than this (prevents jitter)

# PID gains — tune these on the Pi:
#   Kp  raise if sluggish,  lower if oscillating
#   Ki  raise to eliminate steady-state offset, lower if it winds up
#   Kd  raise to dampen overshoot, lower if noisy/twitchy
PID_KP = 3.0    # higher because err_norm is in face-widths not pixels
PID_KI = 0.05
PID_KD = 0.5

AB_ALPHA = 0.5  # position smoothing (raise if tracking lags, lower if jittery)
AB_BETA  = 0.1  # velocity adaptation

DEFAULT_HEF      = "logs/hailo/face.hef"
CONF_THRESHOLD   = 0.50

# ── Alpha-Beta filter ────────────────────────────────────────────────────────
class AlphaBetaFilter:
    """
    Smooths noisy position measurements and estimates velocity.
    Feeds a cleaner position signal into the PID, reducing jitter.

    alpha — position trust  (0-1): higher = follow measurement more closely
    beta  — velocity trust  (0-1): higher = adapt velocity faster

    Typical starting point: alpha=0.5, beta=0.1
    If tracking lags behind a moving face: raise alpha
    If position is still jittery: lower alpha
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        self.alpha = alpha
        self.beta  = beta
        self._pos  = None   # estimated position
        self._vel  = 0.0    # estimated velocity (px/s)

    def update(self, measurement: float, dt: float) -> float:
        if self._pos is None:
            self._pos = measurement
            return measurement

        # Predict
        pos_pred = self._pos + self._vel * dt

        # Residual (innovation)
        residual = measurement - pos_pred

        # Correct
        self._pos = pos_pred  + self.alpha * residual
        self._vel = self._vel + (self.beta / max(dt, 1e-6)) * residual

        return self._pos

    def reset(self):
        self._pos = None
        self._vel = 0.0


# ── PID controller ────────────────────────────────────────────────────────────
class PID:
    """
    Discrete PID with anti-windup and derivative low-pass filter.
    Call update(error, dt) each frame; returns the correction to apply.
    """
    def __init__(self, kp, ki, kd,
                 out_min=-140.0, out_max=140.0,
                 integral_limit=20.0,
                 derivative_alpha=0.2):   # lower = more filtering on D term
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min        = out_min
        self.out_max        = out_max
        self.integral_limit = integral_limit
        self.alpha          = derivative_alpha

        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        # Proportional
        p = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = max(-self.integral_limit,
                              min(self.integral_limit, self._integral))
        i = self.ki * self._integral

        # Derivative with exponential low-pass filter (reduces noise)
        raw_d            = (error - self._prev_error) / dt
        self._d_filtered = (self.alpha * raw_d
                            + (1.0 - self.alpha) * self._d_filtered)
        d = self.kd * self._d_filtered

        self._prev_error = error
        output = p + i + d
        return max(self.out_min, min(self.out_max, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0


# ── Threaded camera ───────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, index, w, h):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


# ── Hailo face detector ───────────────────────────────────────────────────────
class HailoFaceDetector:
    """
    Runs a face detection HEF on the Hailo-8L.
    Expects the HEF to have NMS post-processing embedded (scrfd, retinaface).
    Output contract: list of (x1, y1, x2, y2, conf) in pixel coords.
    """
    def __init__(self, hef_path: str, img_w: int, img_h: int) -> None:
        from hailo_platform import (
            HEF, ConfigureParams, HailoStreamInterface,
            InputVStreamParams, OutputVStreamParams, FormatType, VDevice,
        )
        self._w, self._h = img_w, img_h
        hef = HEF(hef_path)

        net_groups       = hef.get_network_group_names()
        input_info       = hef.get_input_vstream_infos(net_groups[0])
        self._in_h       = input_info[0].shape[0]
        self._in_w       = input_info[0].shape[1]

        self._target     = VDevice()
        cfg_params       = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        self._ng         = self._target.configure(hef, cfg_params)[0]
        self._in_params  = InputVStreamParams.make(
            self._ng, quantized=False, format_type=FormatType.FLOAT32
        )
        self._out_params = OutputVStreamParams.make(
            self._ng, quantized=False, format_type=FormatType.FLOAT32
        )
        print(f"[HailoFace] Loaded {hef_path}  input={self._in_w}x{self._in_h}")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        from hailo_platform import InferVStreams

        inp = cv2.resize(frame_bgr, (self._in_w, self._in_h))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = inp[np.newaxis]   # (1, H, W, 3)  NHWC

        with self._ng.activate():
            with InferVStreams(self._ng, self._in_params, self._out_params) as pipe:
                in_name  = pipe.get_input_vstream_infos()[0].name
                out_name = pipe.get_output_vstream_infos()[0].name
                raw      = pipe.infer({in_name: inp})[out_name]

        return self._parse(raw)

    def _parse(self, raw: np.ndarray) -> list[tuple]:
        """Parse NMS output → list of (x1,y1,x2,y2,conf) in original pixel space."""
        boxes = []
        flat  = raw.reshape(-1, raw.shape[-1])
        for det in flat:
            if det.shape[0] < 5:
                continue
            conf = float(det[4])
            if conf < CONF_THRESHOLD:
                continue
            # Coords normalised 0-1 in model input space
            x1 = float(det[0]) / self._in_w * self._w
            y1 = float(det[1]) / self._in_h * self._h
            x2 = float(det[2]) / self._in_w * self._w
            y2 = float(det[3]) / self._in_h * self._h
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    def close(self):
        del self._target


# ── OpenCV DNN face detector (CPU fallback) ───────────────────────────────────
class CVFaceDetector:
    """
    OpenCV DNN SSD face detector — included with OpenCV, no download needed.
    Falls back to Haar cascade if DNN proto/model files are missing.
    """
    def __init__(self) -> None:
        import os
        data_dir  = cv2.data.haarcascades
        proto     = os.path.join(data_dir, "..", "..", "dnn",
                                 "deploy.prototxt")
        model     = os.path.join(data_dir, "..", "..", "dnn",
                                 "res10_300x300_ssd_iter_140000.caffemodel")

        if os.path.exists(proto) and os.path.exists(model):
            self._net  = cv2.dnn.readNetFromCaffe(proto, model)
            self._mode = "dnn"
            print("[CVFace] OpenCV DNN SSD face detector")
        else:
            self._cc   = cv2.CascadeClassifier(
                data_dir + "haarcascade_frontalface_default.xml"
            )
            self._mode = "haar"
            print("[CVFace] Haar cascade face detector (DNN model not found)")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        h, w = frame_bgr.shape[:2]
        if self._mode == "dnn":
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_bgr, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0),
            )
            self._net.setInput(blob)
            dets  = self._net.forward()
            boxes = []
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf < CONF_THRESHOLD:
                    continue
                x1 = int(dets[0, 0, i, 3] * w)
                y1 = int(dets[0, 0, i, 4] * h)
                x2 = int(dets[0, 0, i, 5] * w)
                y2 = int(dets[0, 0, i, 6] * h)
                boxes.append((x1, y1, x2, y2, conf))
            return boxes
        else:
            gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cc.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            return [
                (x, y, x + w, y + h, 1.0)
                for (x, y, w, h) in faces
            ]


# ── Stats overlay ─────────────────────────────────────────────────────────────
def draw_stats(frame, fps, infer_fps, pan_angle, n_faces, backend, err_x=0.0):
    lines = [
        f"FPS    {fps:.1f}",
        f"Infer  {infer_fps:.1f} fps",
        f"Pan    {pan_angle:.1f}°",
        f"Err    {err_x:+.0f} px",
        f"Faces  {n_faces}",
        f"HW     {backend}",
    ]
    pad, lh = 8, 22
    box_h   = len(lines) * lh + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (210, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    hw_color = (0, 255, 80) if backend == "HAILO-8L" else (0, 180, 255)
    for i, line in enumerate(lines):
        color = hw_color if i == 4 else (220, 220, 220)
        cv2.putText(frame, line, (pad + 2, pad + 18 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hailo",  action="store_true",
                        help="Use Hailo-8L HEF for face detection")
    parser.add_argument("--hef",    default=DEFAULT_HEF,
                        help=f"Path to face detection HEF (default: {DEFAULT_HEF})")
    parser.add_argument("--source", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    # ── Servo ─────────────────────────────────────────────────────────────────
    factory = LGPIOFactory()
    pan = AngularServo(PAN_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
                       min_pulse_width=0.0005, max_pulse_width=0.0025,
                       pin_factory=factory)
    pan.angle = 0
    time.sleep(0.8)

    # ── Detector backend ──────────────────────────────────────────────────────
    import os
    if args.hailo and os.path.exists(args.hef):
        detector = HailoFaceDetector(args.hef, FRAME_W, FRAME_H)
        backend  = "HAILO-8L"
    else:
        if args.hailo:
            print(f"[Backend] HEF not found at {args.hef} — falling back to CV CPU")
        detector = CVFaceDetector()
        backend  = f"CV-{detector._mode.upper()}"

    print(f"[Backend] Active: {backend}")

    # ── Camera ────────────────────────────────────────────────────────────────
    stream    = CameraStream(args.source, FRAME_W, FRAME_H)
    cx_center = FRAME_W / 2

    fps_buf       = collections.deque(maxlen=30)
    infer_ms      = 0.0
    frame_idx     = 0
    t_prev        = time.perf_counter()
    current_angle = 0.0
    last_bbox     = None   # (x1,y1,x2,y2) of best face
    n_faces       = 0
    pid           = PID(PID_KP, PID_KI, PID_KD)
    ab_filter     = AlphaBetaFilter(AB_ALPHA, AB_BETA)

    print("Running — press Q to quit")

    try:
        while True:
            frame = stream.read()
            frame_idx += 1

            # ── Inference every N frames ──────────────────────────────────
            if frame_idx % INFER_EVERY == 0:
                t0    = time.perf_counter()
                dets  = detector.detect(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                n_faces  = len(dets)

                if dets:
                    # Track the largest face (most likely the primary subject)
                    best     = max(dets, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
                    last_bbox = best[:4]
                else:
                    last_bbox = None

            # ── Draw all detections on this frame ─────────────────────────
            if frame_idx % INFER_EVERY == 0 and dets:
                for (x1, y1, x2, y2, conf) in dets:
                    is_target = (x1, y1, x2, y2) == last_bbox
                    color = (0, 255, 0) if is_target else (180, 180, 180)
                    thick = 2 if is_target else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                    cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # ── dt for PID (compute before servo block) ───────────────────
            t_now = time.perf_counter()
            dt    = max(t_now - t_prev, 1e-6)

            # ── Servo update every frame using last known bbox ────────────
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                raw_bx  = (x1 + x2) / 2
                face_w  = max(x2 - x1, 1)

                # Alpha-beta filter smooths the raw detection centre
                bx    = ab_filter.update(raw_bx, dt)
                err_x = bx - cx_center

                # Normalise by face width — distance-adaptive gain
                err_norm = err_x / face_w

                # Draw: raw detection (grey), filtered position (red dot)
                cv2.circle(frame, (int(raw_bx), (y1 + y2) // 2), 4, (180, 180, 180), 1)
                cv2.circle(frame, (int(bx),     (y1 + y2) // 2), 5, (0, 0, 255),     -1)
                cv2.line(frame, (int(cx_center), 0),
                         (int(cx_center), FRAME_H), (0, 255, 255), 1)

                if abs(err_x) > DEADBAND:
                    correction     = pid.update(err_norm, dt)
                    current_angle -= correction
                    current_angle  = max(PAN_MIN, min(PAN_MAX, current_angle))
                    pan.angle      = current_angle
                else:
                    pid.reset()
            else:
                pid.reset()
                ab_filter.reset()  # reset filter on face loss — stale velocity causes lurch

            # ── FPS ───────────────────────────────────────────────────────
            fps_buf.append(1.0 / dt)
            t_prev = t_now
            infer_fps = 1000.0 / max(infer_ms, 1e-3)

            err_display = ((last_bbox[0] + last_bbox[2]) / 2 - cx_center) if last_bbox else 0.0
            draw_stats(frame, sum(fps_buf) / len(fps_buf),
                       infer_fps, current_angle, n_faces, backend, err_display)

            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pan.angle = 0
        time.sleep(0.4)
        stream.release()
        cv2.destroyAllWindows()
        pan.close()
        if backend == "HAILO-8L":
            detector.close()


if __name__ == "__main__":
    main()
