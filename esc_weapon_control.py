"""
ESC Weapon Detection Controller
================================
Runs the weapon detector on a live camera / video and drives a BLDC motor
via ESC based on detection state.

Motor behaviour
---------------
  IDLE        No weapon visible for > COOLDOWN_S seconds → throttle = IDLE_THROTTLE
  ALERT       Weapon detected → ramp to ALERT_THROTTLE instantly
  COOLDOWN    Weapon lost → hold ALERT_THROTTLE for COOLDOWN_S, then ramp back down

Typical wiring (Jetson Orin Nano)
----------------------------------
  ESC signal wire  → GPIO pin 32  (PWM0, BOARD numbering)
  ESC +5V BEC     → Jetson 5V pin  (or separate BEC)
  ESC GND         → common GND

Run
---
  python3 esc_weapon_control.py                  # webcam, default pin 32
  python3 esc_weapon_control.py --source 0 --pin 32
  python3 esc_weapon_control.py --source video.mp4 --sim   # force simulation
  python3 esc_weapon_control.py --no-display               # headless on Jetson
"""

import argparse
import logging
import time
import cv2
import numpy as np
from ultralytics import YOLO

from utils.esc_pwm import ESCController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
DETECTOR_WEIGHTS = "logs/detector/best.pt"   # weapon-trained weights
FALLBACK_WEIGHTS = "yolov8n.pt"              # COCO fallback (person class)
WEAPON_CLASSES   = [0, 1, 2]                 # knife, pistol, rifle
CONF_THRESH      = 0.50

IDLE_THROTTLE    = 0.00   # motor off when no weapon
ALERT_THROTTLE   = 0.60   # throttle when weapon detected  (0–1)
RAMP_UP_S        = 0.3    # seconds to ramp up to alert throttle
RAMP_DOWN_S      = 2.0    # seconds to ramp back to idle
COOLDOWN_S       = 3.0    # hold alert for this long after weapon disappears

CLASS_NAMES = {0: "knife", 1: "pistol", 2: "rifle"}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="ESC Weapon Detection Controller")
    p.add_argument("--source", default="0",
                   help="Webcam index or video file path (default: 0)")
    p.add_argument("--pin", type=int, default=32,
                   help="GPIO BOARD pin for ESC signal (default: 32)")
    p.add_argument("--sim", action="store_true",
                   help="Force simulation mode (no real GPIO output)")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless — no OpenCV window (Jetson inference mode)")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help=f"Detection confidence threshold (default: {CONF_THRESH})")
    p.add_argument("--alert-throttle", type=float, default=ALERT_THROTTLE,
                   help=f"Motor throttle on weapon detection 0–1 (default: {ALERT_THROTTLE})")
    p.add_argument("--idle-throttle", type=float, default=IDLE_THROTTLE,
                   help=f"Motor throttle when idle 0–1 (default: {IDLE_THROTTLE})")
    return p.parse_args()


def load_model(sim: bool) -> tuple[YOLO, list[int], dict]:
    """Load weapon model if weights exist, else fall back to COCO yolov8n."""
    import os
    if not sim and os.path.exists(DETECTOR_WEIGHTS):
        log.info(f"[model] loading weapon detector: {DETECTOR_WEIGHTS}")
        model = YOLO(DETECTOR_WEIGHTS)
        classes = WEAPON_CLASSES
        names = CLASS_NAMES
    else:
        log.warning(f"[model] weapon weights not found — using {FALLBACK_WEIGHTS} (person detection)")
        model = YOLO(FALLBACK_WEIGHTS)
        classes = [0]   # person class in COCO
        names = {0: "person"}
    return model, classes, names


def draw_hud(frame: np.ndarray, detections: list, state: str,
             throttle: float, cooldown_left: float, names: dict):
    fh, fw = frame.shape[:2]

    # detection boxes
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # state banner
    if state == "ALERT":
        cv2.rectangle(frame, (0, 0), (fw, 52), (0, 0, 160), -1)
        cv2.putText(frame, "!  WEAPON DETECTED — MOTOR ACTIVE  !",
                    (12, 37), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)
    elif state == "COOLDOWN":
        cv2.rectangle(frame, (0, 0), (fw, 52), (0, 80, 140), -1)
        cv2.putText(frame, f"COOLDOWN — holding {cooldown_left:.1f}s",
                    (12, 37), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 230, 100), 2)

    # throttle bar (bottom strip)
    bar_max = fw - 160
    bar_fill = int(bar_max * throttle)
    cv2.rectangle(frame, (8, fh - 28), (8 + bar_max, fh - 10), (50, 50, 50), -1)
    col = (0, 200, 80) if state == "IDLE" else (0, 80, 255)
    if bar_fill > 0:
        cv2.rectangle(frame, (8, fh - 28), (8 + bar_fill, fh - 10), col, -1)
    cv2.putText(frame, f"ESC throttle: {throttle:.0%}  [{state}]",
                (8, fh - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model, det_classes, names = load_model(args.sim)

    # ESC setup — force simulation if --sim flag set
    if args.sim:
        import utils.esc_pwm as _m
        _m._make_backend = lambda pin: _m._SimulatedPWM(pin)

    esc = ESCController(pin=args.pin)
    esc.arm()
    esc.set_throttle(args.idle_throttle)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    if not args.no_display:
        cv2.namedWindow("ESC Weapon Control", cv2.WINDOW_NORMAL)

    # state machine
    state = "IDLE"          # IDLE | ALERT | COOLDOWN
    last_detection_t = 0.0
    frame_idx = 0

    log.info("[run] starting — press Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            now = time.time()

            # ── detect ────────────────────────────────────────────────────
            results = model(frame, classes=det_classes, conf=args.conf,
                            verbose=False)[0]

            detections = []
            weapon_found = False
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls_id))
                weapon_found = True

            # ── state machine → ESC ───────────────────────────────────────
            if weapon_found:
                last_detection_t = now
                if state != "ALERT":
                    log.info(f"[{now:.1f}] WEAPON DETECTED → ramping motor to "
                             f"{args.alert_throttle:.0%}")
                    state = "ALERT"
                    esc.ramp_to(args.alert_throttle, duration=RAMP_UP_S)

            elif state == "ALERT":
                state = "COOLDOWN"
                log.info(f"[{now:.1f}] weapon lost → COOLDOWN {COOLDOWN_S}s")

            elif state == "COOLDOWN":
                elapsed = now - last_detection_t
                if elapsed >= COOLDOWN_S:
                    state = "IDLE"
                    log.info(f"[{now:.1f}] cooldown done → ramping motor to idle")
                    esc.ramp_to(args.idle_throttle, duration=RAMP_DOWN_S)

            cooldown_left = max(0.0, COOLDOWN_S - (now - last_detection_t))

            # ── display ───────────────────────────────────────────────────
            if not args.no_display:
                draw_hud(frame, detections, state, esc.throttle, cooldown_left, names)
                cv2.imshow("ESC Weapon Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                # headless: print status every 30 frames
                if frame_idx % 30 == 0:
                    log.info(f"frame={frame_idx}  state={state}  "
                             f"throttle={esc.throttle:.0%}  "
                             f"detections={len(detections)}")

    finally:
        log.info("[shutdown] stopping motor …")
        esc.ramp_to(0.0, duration=1.0)
        esc.stop()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log.info("[shutdown] done")


if __name__ == "__main__":
    main()
