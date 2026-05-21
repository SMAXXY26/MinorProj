"""
face_track_ukf.py — Face detection + 2-axis servo tracking using Unscented Kalman Filter
========================================================================================
Hardware : Raspberry Pi 5 + Hailo-8L AI HAT
Servo    : Pan — GPIO 12 (BCM), Tilt - GPIO 13 (BCM)

Backends:
  --hailo   Hailo-8L HEF face detector (fastest)
  default   OpenCV DNN face detector (CPU fallback)

Usage:
    python3 face_track_ukf.py                    # OpenCV CPU
    python3 face_track_ukf.py --hailo            # Hailo-8L
"""
from __future__ import annotations

import argparse
import collections
import threading
import time
import os

import cv2
import numpy as np
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
INFER_EVERY      = 2

PAN_PIN          = 12
TILT_PIN         = 13
PAN_MIN, PAN_MAX = -70, 70
TILT_MIN, TILT_MAX = -70, 70
DEADBAND         = 8      # pixels — ignore error smaller than this (prevents jitter)

# PID gains
PID_KP = 0.8
PID_KI = 0.05
PID_KD = 0.3

DEFAULT_HEF      = "logs/hailo/face.hef"
CONF_THRESHOLD   = 0.50

# ── Unscented Kalman Filter (2D State: x, y, vx, vy) ──────────────────────────
class UKF2D:
    def __init__(self, dt: float):
        self.dim_x = 4
        self.dim_z = 2
        self.dt = dt
        
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 100.0  
        
        # Process noise covariance
        self.Q = np.eye(self.dim_x) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(self.dim_z) * 10.0
        
        # Weights (Julier-Uhlmann)
        self.kappa = 3.0 - self.dim_x
        self.W_m = np.zeros(2 * self.dim_x + 1)
        self.W_c = np.zeros(2 * self.dim_x + 1)
        self.W_m[0] = self.kappa / (self.dim_x + self.kappa)
        self.W_c[0] = self.W_m[0]
        for i in range(1, 2 * self.dim_x + 1):
            val = 1.0 / (2 * (self.dim_x + self.kappa))
            self.W_m[i] = val
            self.W_c[i] = val
            
        self.initialized = False
            
    def _fx(self, state, dt):
        return np.array([
            state[0] + state[2] * dt,
            state[1] + state[3] * dt,
            state[2],
            state[3]
        ])
        
    def _hx(self, state):
        return np.array([state[0], state[1]])

    def _update_Q(self, dt):
        q = 0.5 * 100  # Process noise spectral density
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q

    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
            self._update_Q(dt)
            
        if not self.initialized:
            return
            
        # Generate Sigma Points
        P_scaled = (self.dim_x + self.kappa) * self.P
        # Stability epsilon for Cholesky
        P_scaled += np.eye(self.dim_x) * 1e-8
        
        U = np.linalg.cholesky(P_scaled)
        
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = self.x
        for k in range(self.dim_x):
            sigmas[k + 1] = self.x + U[k]
            sigmas[self.dim_x + k + 1] = self.x - U[k]
            
        # Transform Sigma Points (Process)
        self.sigmas_f = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            self.sigmas_f[i] = self._fx(sigmas[i], self.dt)
            
        # Predict State
        self.x = np.dot(self.W_m, self.sigmas_f)
        
        # Predict Covariance
        self.P = self.Q.copy()
        for i in range(len(self.sigmas_f)):
            y = self.sigmas_f[i] - self.x
            self.P += self.W_c[i] * np.outer(y, y)

    def update(self, z):
        if not self.initialized:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.eye(self.dim_x) * 10.0
            self.initialized = True
            return

        # Transform Sigma Points (Measurement)
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(len(self.sigmas_f)):
            sigmas_h[i] = self._hx(self.sigmas_f[i])
            
        # Predict Measurement
        z_pred = np.dot(self.W_m, sigmas_h)
        
        # Innovation and Cross Covariance
        P_zz = self.R.copy()
        P_xz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(len(self.sigmas_f)):
            y = sigmas_h[i] - z_pred
            x_diff = self.sigmas_f[i] - self.x
            P_zz += self.W_c[i] * np.outer(y, y)
            P_xz += self.W_c[i] * np.outer(x_diff, y)
            
        # Kalman Gain
        K = np.dot(P_xz, np.linalg.inv(P_zz + np.eye(self.dim_z) * 1e-8))
        
        # State and Covariance Update
        y_residual = z - z_pred
        self.x = self.x + np.dot(K, y_residual)
        self.P = self.P - np.dot(K, np.dot(P_zz, K.T))
        
    def reset(self):
        self.initialized = False
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 100.0


# ── PID controller ────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, out_min=-140.0, out_max=140.0, integral_limit=20.0, derivative_alpha=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit
        self.alpha = derivative_alpha

        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0: return 0.0
        p = self.kp * error
        self._integral += error * dt
        self._integral  = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i = self.ki * self._integral
        raw_d = (error - self._prev_error) / dt
        self._d_filtered = (self.alpha * raw_d + (1.0 - self.alpha) * self._d_filtered)
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


# ── Detectors ─────────────────────────────────────────────────────────────────
class HailoFaceDetector:
    def __init__(self, hef_path: str) -> None:
        from hailo_platform import (HEF, ConfigureParams, HailoStreamInterface, 
                                    InputVStreamParams, OutputVStreamParams, FormatType, VDevice)
        hef = HEF(hef_path)
        net_groups = hef.get_network_group_names()
        input_info = hef.get_input_vstream_infos(net_groups[0])
        self._in_h = input_info[0].shape[0]
        self._in_w = input_info[0].shape[1]

        self._target = VDevice()
        cfg_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self._ng = self._target.configure(hef, cfg_params)[0]
        self._in_params = InputVStreamParams.make(self._ng, quantized=False, format_type=FormatType.FLOAT32)
        self._out_params = OutputVStreamParams.make(self._ng, quantized=False, format_type=FormatType.FLOAT32)
        print(f"[HailoFace] Loaded {hef_path} exact camera input matched to = {self._in_w}x{self._in_h}")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        from hailo_platform import InferVStreams
        ih, iw = frame_bgr.shape[:2]
        
        # Avoid resizing if the camera resolution exactly matches the HEF specification
        if iw != self._in_w or ih != self._in_h:
            inp = cv2.resize(frame_bgr, (self._in_w, self._in_h))
            print(f"[Warn] Resizing frame from {iw}x{ih} to {self._in_w}x{self._in_h}")
        else:
            inp = frame_bgr
            
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = inp[np.newaxis]   # (1, H, W, 3) NHWC

        with self._ng.activate():
            with InferVStreams(self._ng, self._in_params, self._out_params) as pipe:
                in_name  = pipe.get_input_vstream_infos()[0].name
                out_name = pipe.get_output_vstream_infos()[0].name
                raw      = pipe.infer({in_name: inp})[out_name]

        return self._parse(raw, iw, ih)

    def _parse(self, raw: np.ndarray, w: int, h: int) -> list[tuple]:
        boxes = []
        flat  = raw.reshape(-1, raw.shape[-1])
        for det in flat:
            if det.shape[0] < 5: continue
            conf = float(det[4])
            if conf < CONF_THRESHOLD: continue
            x1 = float(det[0]) / self._in_w * w
            y1 = float(det[1]) / self._in_h * h
            x2 = float(det[2]) / self._in_w * w
            y2 = float(det[3]) / self._in_h * h
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    def close(self):
        del self._target


class CVFaceDetector:
    def __init__(self) -> None:
        data_dir = cv2.data.haarcascades
        proto = os.path.join(data_dir, "..", "..", "dnn", "deploy.prototxt")
        model = os.path.join(data_dir, "..", "..", "dnn", "res10_300x300_ssd_iter_140000.caffemodel")

        if os.path.exists(proto) and os.path.exists(model):
            self._net  = cv2.dnn.readNetFromCaffe(proto, model)
            self._mode = "dnn"
            self._in_w = 300
            self._in_h = 300
            print("[CVFace] OpenCV DNN SSD face detector exact camera input matched to = 300x300")
        else:
            self._cc   = cv2.CascadeClassifier(data_dir + "haarcascade_frontalface_default.xml")
            self._mode = "haar"
            self._in_w = 640
            self._in_h = 480
            print("[CVFace] Haar cascade face detector matched to = 640x480")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        h, w = frame_bgr.shape[:2]
        # Same check for direct input mapping
        if self._mode == "dnn":
            if w != self._in_w or h != self._in_h:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (self._in_w, self._in_h)), 1.0, (self._in_w, self._in_h), (104.0, 177.0, 123.0))
            else:
                blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (self._in_w, self._in_h), (104.0, 177.0, 123.0))
            
            self._net.setInput(blob)
            dets = self._net.forward()
            boxes = []
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf < CONF_THRESHOLD: continue
                x1 = int(dets[0, 0, i, 3] * w); y1 = int(dets[0, 0, i, 4] * h)
                x2 = int(dets[0, 0, i, 5] * w); y2 = int(dets[0, 0, i, 6] * h)
                boxes.append((x1, y1, x2, y2, conf))
            return boxes
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cc.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]

    def close(self): pass


# ── Overlay ───────────────────────────────────────────────────────────────────
def draw_stats(frame, fps, infer_fps, pan_angle, tilt_angle, n_faces, backend):
    lines = [
        f"FPS    {fps:.1f}",
        f"Infer  {infer_fps:.1f} fps",
        f"Pan    {pan_angle:.1f}",
        f"Tilt   {tilt_angle:.1f}",
        f"Faces  {n_faces}",
        f"HW     {backend}",
    ]
    pad, lh = 8, 22
    box_h = len(lines) * lh + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (210, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    hw_color = (0, 255, 80) if backend == "HAILO-8L" else (0, 180, 255)
    for i, line in enumerate(lines):
        color = hw_color if i == 5 else (220, 220, 220)
        cv2.putText(frame, line, (pad + 2, pad + 18 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hailo",  action="store_true")
    parser.add_argument("--hef",    default=DEFAULT_HEF)
    parser.add_argument("--source", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    # 1. Setup Detector and Retrieve its Exact Required Input Dimensions
    if args.hailo and os.path.exists(args.hef):
        detector = HailoFaceDetector(args.hef)
        backend = "HAILO-8L"
        FRAME_W, FRAME_H = detector._in_w, detector._in_h
    else:
        detector = CVFaceDetector()
        backend = f"CV-{detector._mode.upper()}"
        FRAME_W, FRAME_H = detector._in_w, detector._in_h

    # 2. Setup Camera matching Native Model Dimensions exactly
    stream = CameraStream(args.source, FRAME_W, FRAME_H)
    cx_center = FRAME_W / 2
    cy_center = FRAME_H / 2

    # 3. Servos
    factory = LGPIOFactory()
    pan = AngularServo(PAN_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
                       min_pulse_width=0.0005, max_pulse_width=0.0025,
                       pin_factory=factory)
    tilt = AngularServo(TILT_PIN, min_angle=TILT_MIN, max_angle=TILT_MAX,
                        min_pulse_width=0.0005, max_pulse_width=0.0025,
                        pin_factory=factory)
    pan.angle = 0
    tilt.angle = 0
    time.sleep(0.8)

    fps_buf = collections.deque(maxlen=30)
    infer_ms = 0.0
    frame_idx = 0
    t_prev = time.perf_counter()
    
    current_pan = 0.0
    current_tilt = 0.0
    last_bbox = None
    
    pid_pan = PID(PID_KP, PID_KI, PID_KD)
    pid_tilt = PID(PID_KP, PID_KI, PID_KD)
    
    ukf = UKF2D(dt=0.033)

    print(f"Running {FRAME_W}x{FRAME_H} without downsampling — press Q to quit")

    try:
        while True:
            frame = stream.read()
            if frame is None or frame.size == 0:
                continue
                
            frame_idx += 1

            # Inference
            if frame_idx % INFER_EVERY == 0:
                t0 = time.perf_counter()
                dets = detector.detect(frame)
                infer_ms = (time.perf_counter() - t0) * 1000

                if dets:
                    best = max(dets, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
                    last_bbox = best[:4]
                else:
                    last_bbox = None

            # Draw Dets
            if dets and frame_idx % INFER_EVERY == 0:
                for (x1, y1, x2, y2, conf) in dets:
                    is_tgt = (x1, y1, x2, y2) == last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if is_tgt else (180, 180, 180), 2 if is_tgt else 1)

            t_now = time.perf_counter()
            dt = max(t_now - t_prev, 1e-6)

            ukf.predict(dt)

            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                raw_bx = (x1 + x2) / 2
                raw_by = (y1 + y2) / 2
                face_w = max(x2 - x1, 1)
                face_h = max(y2 - y1, 1)

                ukf.update(np.array([raw_bx, raw_by]))
                bx, by = ukf.x[0], ukf.x[1]

                cv2.circle(frame, (int(raw_bx), int(raw_by)), 4, (180, 180, 180), 1)
                cv2.circle(frame, (int(bx), int(by)), 5, (0, 0, 255), -1)

                err_x = bx - cx_center
                err_y = by - cy_center
                err_norm_x = err_x / face_w
                err_norm_y = err_y / face_h

                if abs(err_x) > DEADBAND:
                    corr_pan = pid_pan.update(err_norm_x, dt)
                    current_pan = max(PAN_MIN, min(PAN_MAX, current_pan - corr_pan))
                    pan.angle = current_pan
                else:
                    pid_pan.reset()
                    
                if abs(err_y) > DEADBAND:
                    # Note: You may need to invert this sign (+/- corr_tilt) depending on servo mount orientation
                    corr_tilt = pid_tilt.update(err_norm_y, dt)
                    current_tilt = max(TILT_MIN, min(TILT_MAX, current_tilt - corr_tilt))
                    tilt.angle = current_tilt
                else:
                    pid_tilt.reset()
                    
            else:
                pid_pan.reset()
                pid_tilt.reset()
                ukf.reset()

            fps_buf.append(1.0 / dt)
            t_prev = t_now
            infer_fps = 1000.0 / max(infer_ms, 1e-3)

            draw_stats(frame, sum(fps_buf) / len(fps_buf), infer_fps, current_pan, current_tilt, len(dets) if 'dets' in locals() else 0, backend)

            cv2.imshow("UKF 2-Axis Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pan.angle = 0
        tilt.angle = 0
        time.sleep(0.4)
        stream.release()
        cv2.destroyAllWindows()
        pan.close()
        tilt.close()
        detector.close()

if __name__ == "__main__":
    main()
