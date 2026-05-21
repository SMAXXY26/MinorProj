"""
servo_track.py — Single-axis pan servo tracker with scan-when-idle mode.

Behaviour:
  - Object detected  → PID centres it horizontally (pan servo)
  - No object found  → servo slowly sweeps left↔right to search
  - Press Q to quit

Usage:
  python3 servo_track.py                  # ONNX CPU, webcam 0
  python3 servo_track.py --hailo          # Hailo-8L HEF
  python3 servo_track.py --source 1       # alternate camera index
"""

import argparse
import collections
import math
import os
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory as _PinFactory
from ultralytics import YOLO

# ── Hardware ──────────────────────────────────────────────────────────────────
PAN_PIN  = 12
PAN_MIN  = -70   # degrees
PAN_MAX  =  70

# ── Model paths — searched relative to this script's directory ────────────────
_HERE = Path(__file__).parent
ONNX_WEIGHTS = str(_HERE / "logs/hailo/detector.onnx")
HEF_WEIGHTS  = str(_HERE / "logs/hailo/detector.hef")

# ── Camera ────────────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 320, 240
INFER_EVERY      = 2        # run YOLO every N frames
CONF_THRESHOLD   = 0.50

# ── PID (pan only) ────────────────────────────────────────────────────────────
# From PyImageSearch pan/tilt guide + obadakatma PID tracker (tested values).
# Oscillating → lower KP. Slow → raise KP. Steady offset → raise KI.
PAN_KP, PAN_KI, PAN_KD = 0.09, 0.08, 0.002
DEADBAND  = 8      # pixels — ignore small errors to stop micro-jitter
MAX_ITERM = 30.0   # integral anti-windup clamp (degrees)

# ── Scan (idle sweep) ─────────────────────────────────────────────────────────
SCAN_SPEED   = 30.0   # degrees per second during sweep
SCAN_PAUSE_S = 0.6    # pause at each end before reversing


# ── MJPEG stream server ───────────────────────────────────────────────────────
class _MJPEGHandler(BaseHTTPRequestHandler):
    latest_frame = None
    lock = threading.Lock()

    def log_message(self, *_):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == "/":
            # Minimal HTML page that auto-refreshes the stream
            html = (
                "<html><body style='background:#111;margin:0'>"
                "<img src='/stream' style='max-width:100%;display:block;margin:auto'>"
                "</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _MJPEGHandler.lock:
                        frame = _MJPEGHandler.latest_frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    data = jpg.tobytes()
                    self.wfile.write(
                        f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                        + data + b"\r\n"
                    )
                    time.sleep(0.04)   # ~25 fps max to the browser
            except (BrokenPipeError, ConnectionResetError):
                pass


def start_stream_server(port: int = 8080):
    class _ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = _ThreadedServer(("", port), _MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[Stream] http://10.42.0.100:{port}/  (open in browser on your laptop)")
    return server


# ── PID controller ────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint  = setpoint
        self._iterm    = 0.0
        self._last_err = None
        self._last_t   = None

    def reset(self):
        self._iterm    = 0.0
        self._last_err = None
        self._last_t   = None

    def update(self, measurement):
        now = time.perf_counter()
        err = self.setpoint - measurement
        dt  = max(now - self._last_t, 1e-4) if self._last_t else 0.033

        self._iterm = max(-MAX_ITERM, min(MAX_ITERM, self._iterm + err * dt))
        dterm       = (err - self._last_err) / dt if self._last_err is not None else 0.0

        self._last_err = err
        self._last_t   = now
        return self.kp * err + self.ki * self._iterm + self.kd * dterm


# ── Threaded camera ───────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, index, w, h):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _, self.frame = self.cap.read()
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


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(use_hailo: bool, weights_override: str = None):
    if weights_override:
        label = "HAILO-8L" if weights_override.endswith(".hef") else "CUSTOM"
        print(f"[Backend] {label}: {weights_override}")
        return YOLO(weights_override), label
    if use_hailo and os.path.exists(HEF_WEIGHTS):
        print(f"[Backend] Hailo HEF: {HEF_WEIGHTS}")
        return YOLO(HEF_WEIGHTS), "HAILO-8L"
    if use_hailo:
        print("[Backend] HEF not found — falling back to ONNX CPU")
    if not os.path.exists(ONNX_WEIGHTS):
        print(f"[Backend] WARNING: {ONNX_WEIGHTS} not found")
        print("[Backend] Run from the weapon_detection folder, or use --weights <path>")
        raise FileNotFoundError(ONNX_WEIGHTS)
    print(f"[Backend] ONNX CPU: {ONNX_WEIGHTS}")
    return YOLO(ONNX_WEIGHTS), "ONNX-CPU"


# ── Overlay ───────────────────────────────────────────────────────────────────
_CLASS_COLORS = {
    "knife":  (0x37, 0x8A, 0xDD),
    "pistol": (0xE2, 0x4B, 0x4A),
    "rifle":  (0xBA, 0x75, 0x17),
}

def draw_hud(frame, fps, infer_ms, pan_angle, mode, label, conf, backend):
    lines = [
        f"FPS    {fps:.1f}",
        f"Infer  {(1000/infer_ms if infer_ms > 0 else 0):.0f} fps",
        f"Pan    {pan_angle:+.1f}°",
        f"Mode   {mode}",
        f"Det    {label} {conf:.2f}" if label != "none" else "Det    none",
        f"HW     {backend}",
    ]
    pad, lh = 8, 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (210, len(lines) * lh + 14), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    hw_c  = (0, 255, 80) if backend == "HAILO-8L" else (0, 180, 255)
    det_c = _CLASS_COLORS.get(label, (200, 200, 200))
    mode_c = (0, 200, 255) if mode == "SCAN" else (0, 255, 80)
    for i, line in enumerate(lines):
        c = hw_c if i == 5 else (det_c if i == 4 else (mode_c if i == 3 else (220, 220, 220)))
        cv2.putText(frame, line, (pad + 2, pad + 18 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hailo",   action="store_true")
    parser.add_argument("--source",  type=int, default=0)
    parser.add_argument("--weights",     type=str, default=None,
                        help="Override model path (ONNX or HEF or .pt)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Headless mode — no cv2.imshow (use when running over SSH)")
    parser.add_argument("--stream",      action="store_true",
                        help="Serve annotated video as MJPEG on port 8080")
    parser.add_argument("--port",        type=int, default=8080)
    args = parser.parse_args()

    factory = _PinFactory()
    pan = AngularServo(PAN_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
                       min_pulse_width=0.0005, max_pulse_width=0.0025,
                       pin_factory=factory)
    pan.angle = 0
    time.sleep(0.8)

    model, backend = load_model(args.hailo, weights_override=args.weights)
    if args.stream or args.no_display:
        start_stream_server(args.port)
    stream = CameraStream(args.source, FRAME_W, FRAME_H)

    cx_center = FRAME_W / 2
    pid       = PID(PAN_KP, PAN_KI, PAN_KD, setpoint=cx_center)

    pan_angle    = 0.0
    scan_dir     = 1          # +1 right, -1 left
    scan_pause   = 0.0        # timestamp when end-of-travel reached (0 = not pausing)
    fps_buf      = collections.deque(maxlen=30)
    infer_ms     = 0.0
    detect_label = "none"
    detect_conf  = 0.0
    last_bbox    = None
    frame_idx    = 0
    t_prev       = time.perf_counter()
    mode         = "SCAN"

    print("[Tracker] Running — press Q to quit")
    print(f"[Tracker] TRACK mode when object detected, SCAN sweep otherwise")

    try:
        while True:
            frame     = stream.read()
            frame_idx += 1
            t_now     = time.perf_counter()
            dt        = max(t_now - t_prev, 1e-4)
            t_prev    = t_now
            fps_buf.append(1.0 / dt)

            # ── Detection ─────────────────────────────────────────────────────
            if frame_idx % INFER_EVERY == 0:
                t0       = time.perf_counter()
                results  = model(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=416)[0]
                infer_ms = (time.perf_counter() - t0) * 1000

                if results.boxes:
                    best         = max(results.boxes, key=lambda b: float(b.conf))
                    last_bbox    = tuple(map(int, best.xyxy[0]))
                    detect_label = results.names[int(best.cls)]
                    detect_conf  = float(best.conf)
                else:
                    last_bbox    = None
                    detect_label = "none"
                    detect_conf  = 0.0

            # ── Servo logic ───────────────────────────────────────────────────
            if last_bbox is not None:
                # TRACK mode — PID centres the object
                mode = "TRACK"
                pid.reset() if mode != "TRACK" else None
                x1, y1, x2, y2 = last_bbox
                bx = (x1 + x2) / 2

                if abs(bx - cx_center) > DEADBAND:
                    correction = pid.update(bx)
                    pan_angle  = max(PAN_MIN, min(PAN_MAX, pan_angle + correction))
                    pan.angle  = pan_angle

                color = _CLASS_COLORS.get(detect_label, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(bx), (y1 + y2) // 2), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"{detect_label} {detect_conf:.2f}",
                            (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                # SCAN mode — slow sweep left ↔ right
                mode = "SCAN"
                pid.reset()

                if scan_pause > 0.0:
                    # waiting at end of travel
                    if t_now >= scan_pause:
                        scan_dir  = -scan_dir
                        scan_pause = 0.0
                else:
                    pan_angle += scan_dir * SCAN_SPEED * dt
                    if pan_angle >= PAN_MAX:
                        pan_angle  = PAN_MAX
                        scan_pause = t_now + SCAN_PAUSE_S
                    elif pan_angle <= PAN_MIN:
                        pan_angle  = PAN_MIN
                        scan_pause = t_now + SCAN_PAUSE_S
                    pan.angle = pan_angle

            # crosshair at frame centre
            cv2.drawMarker(frame, (int(cx_center), FRAME_H // 2),
                           (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

            fps_now = sum(fps_buf) / len(fps_buf)
            draw_hud(frame, fps_now, infer_ms,
                     pan_angle, mode, detect_label, detect_conf, backend)

            # Push annotated frame to MJPEG server
            if args.stream or args.no_display:
                with _MJPEGHandler.lock:
                    _MJPEGHandler.latest_frame = frame.copy()

            if not args.no_display:
                cv2.imshow("Weapon Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"\r[{mode}] pan={pan_angle:+6.1f}°  det={detect_label:<6s} {detect_conf:.2f}  fps={fps_now:.1f}  ", end="", flush=True)

    finally:
        print()
        pan.angle = 0
        time.sleep(0.4)
        stream.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        pan.close()


if __name__ == "__main__":
    main()
