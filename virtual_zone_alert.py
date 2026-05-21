"""
3D Virtual Zone Monitor — Depth Anything V2 Edition
====================================================
Combines YOLOv8n person detection with monocular depth estimation
(Depth Anything V2 Small) to build a real depth-aware 3D zone.

How it works
------------
1. Depth Anything V2 estimates a per-pixel depth map every few frames.
2. You click 4 ground-plane corners — the zone's depth band is sampled
   from those exact pixels in the depth map.
3. Each detected person's foot-point is checked against the 2D polygon
   AND against the zone's depth band. Both must match for a breach.
4. A Bird's Eye View (BEV) panel shows all persons and the depth band
   from above so you can see who is in range.

Controls
--------
  Click ×4    — place ground corners in order (TL → TR → BR → BL)
  D           — toggle depth colormap overlay
  + / -       — widen / narrow depth band
  R           — reset zone
  Q / Esc     — quit
"""

import argparse
import time
import threading
import queue
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image as PILImage
    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False

# ── config ────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"
DEPTH_MODEL_ID  = "depth-anything/Depth-Anything-V2-Small-hf"
PERSON_CLASS    = 0
CONF_THRESH     = 0.45
ALERT_COOLDOWN  = 2.0            # seconds between console warnings
DEPTH_SKIP      = 4              # run depth every N frames
DEPTH_HALF_BAND = 0.12           # initial depth band half-width (0–1 scale)
BEV_SIZE        = 260            # bird's eye view panel size (pixels)
ZONE_HEIGHT_PX  = 140            # visual 3D extrusion height for rendering
# ─────────────────────────────────────────────────────────────────────────────


# ── depth estimator ───────────────────────────────────────────────────────────

class DepthEstimator:
    def __init__(self, device: str):
        print(f"[depth] loading {DEPTH_MODEL_ID} on {device} …")
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        self.model.to(device).eval()
        self.device = device
        print("[depth] ready\n")

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return HxW float32 depth map normalized to [0, 1]. 1 = far, 0 = close."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        d = out.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        lo, hi = d.min(), d.max()
        return (d - lo) / (hi - lo + 1e-8)


# ── 3D zone ───────────────────────────────────────────────────────────────────

class Zone3D:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ground_pts: list[tuple[int, int]] = []
        self.d_center: float | None = None
        self.d_half: float = DEPTH_HALF_BAND

    @property
    def ready(self) -> bool:
        return len(self.ground_pts) == 4 and self.d_center is not None

    def add_point(self, pt: tuple[int, int]):
        if len(self.ground_pts) < 4:
            self.ground_pts.append(pt)

    def sample_depth(self, depth_map: np.ndarray):
        h, w = depth_map.shape
        vals = []
        for (x, y) in self.ground_pts:
            xi = min(max(x, 0), w - 1)
            yi = min(max(y, 0), h - 1)
            vals.append(float(depth_map[yi, xi]))
        if vals:
            self.d_center = float(np.mean(vals))

    def contains(self, foot: tuple[int, int], foot_depth: float) -> bool:
        if not self.ready:
            return False
        poly = np.array(self.ground_pts, np.float32).reshape(-1, 1, 2)
        in_poly = cv2.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0
        in_depth = abs(foot_depth - self.d_center) <= self.d_half
        return in_poly and in_depth

    def adjust_band(self, delta: float):
        self.d_half = float(np.clip(self.d_half + delta, 0.02, 0.5))


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_3d_zone(frame: np.ndarray, zone: Zone3D, breached: bool):
    g = zone.ground_pts
    n = len(g)
    if n == 0:
        return

    h_px = ZONE_HEIGHT_PX
    c = [(x, y - h_px) for (x, y) in g]

    ec = (0, 0, 255) if breached else (30, 210, 255)

    overlay = frame.copy()

    def fill(pts, col):
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], col)

    if n == 4:
        dk = (0, 0, 60) if breached else (0, 40, 50)
        fill(c,                       dk)
        fill([g[3], g[2], c[2], c[3]], (0, 10, 80) if breached else (0, 35, 55))
        fill([g[2], g[1], c[1], c[2]], (0,  5, 50) if breached else (0, 20, 35))
        fill([g[0], g[3], c[3], c[0]], (0,  5, 50) if breached else (0, 20, 35))
        fill([g[0], g[1], c[1], c[0]], (0, 10, 80) if breached else (0, 35, 55))
        fill(g,                        (0,  5, 40) if breached else (0, 15, 30))

    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    for i in range(n):
        gi, gn = tuple(map(int, g[i])), tuple(map(int, g[(i+1) % n]))
        ci, cn = tuple(map(int, c[i])), tuple(map(int, c[(i+1) % n]))
        cv2.line(frame, gi, gn, ec, 2)    # ground edge
        cv2.line(frame, ci, cn, ec, 2)    # ceiling edge
        cv2.line(frame, gi, ci, ec, 2)    # vertical pillar

    # zone label above ceiling centroid
    cx = int(np.mean([p[0] for p in g]))
    cy = int(np.mean([p[1] for p in g])) - h_px - 10
    if zone.d_center is not None:
        label = (f"ZONE [BREACHED]  d={zone.d_center:.2f}±{zone.d_half:.2f}"
                 if breached else
                 f"ZONE 3D  d={zone.d_center:.2f}±{zone.d_half:.2f}")
    else:
        label = "ZONE 3D  (sampling depth…)"
    cv2.putText(frame, label, (cx - 110, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ec, 2)


def draw_placement(frame: np.ndarray, zone: Zone3D, preview: tuple[int, int] | None):
    """Draw corner dots and guide lines while the user places corners."""
    for i, p in enumerate(zone.ground_pts):
        cv2.circle(frame, p, 7, (0, 255, 255), -1)
        cv2.putText(frame, str(i + 1), (p[0] + 9, p[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if preview:
        cv2.circle(frame, preview, 5, (200, 200, 0), -1)
        if zone.ground_pts:
            cv2.line(frame, zone.ground_pts[-1], preview, (80, 80, 80), 1)


def draw_bev(zone: Zone3D, persons: list[tuple[float, float, bool]]) -> np.ndarray:
    """
    Bird's Eye View panel.
    X-axis = horizontal screen position (left→right).
    Y-axis = depth (far = top, near = bottom).
    """
    S = BEV_SIZE
    bev = np.full((S, S, 3), 18, dtype=np.uint8)

    # grid lines
    for i in range(1, 4):
        y = int(i * S / 4)
        cv2.line(bev, (0, y), (S, y), (35, 35, 35), 1)
        cv2.line(bev, (y, 0), (y, S), (35, 35, 35), 1)

    cv2.putText(bev, "BEV (top-down)", (S // 2 - 55, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
    cv2.putText(bev, "far",  (4, 22),   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)
    cv2.putText(bev, "near", (4, S - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

    margin = 14

    def to_bev(x_norm: float, depth: float) -> tuple[int, int]:
        bx = int(x_norm * (S - 2 * margin) + margin)
        bd = int((1.0 - depth) * (S - 2 * margin) + margin)  # far at top
        return (bx, bd)

    # zone depth band
    if zone.d_center is not None:
        d_lo = max(zone.d_center - zone.d_half, 0.0)
        d_hi = min(zone.d_center + zone.d_half, 1.0)
        by_hi = int((1.0 - d_hi) * (S - 2 * margin) + margin)
        by_lo = int((1.0 - d_lo) * (S - 2 * margin) + margin)
        cv2.rectangle(bev, (margin, by_hi), (S - margin, by_lo), (20, 70, 50), -1)
        cv2.rectangle(bev, (margin, by_hi), (S - margin, by_lo), (30, 160, 110), 1)
        cv2.putText(bev, f"zone band", (margin + 2, by_hi - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (30, 160, 110), 1)

    for (x_norm, depth, in_zone) in persons:
        bpt = to_bev(x_norm, depth)
        col = (0, 0, 255) if in_zone else (0, 210, 80)
        cv2.circle(bev, bpt, 8, col, -1)
        cv2.circle(bev, bpt, 8, (255, 255, 255), 1)
        cv2.putText(bev, f"{depth:.2f}", (bpt[0] + 9, bpt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    return bev


# ── mouse state ───────────────────────────────────────────────────────────────

zone = Zone3D()
preview_pt: tuple[int, int] | None = None


def on_mouse(event, x, y, flags, param):
    global preview_pt
    preview_pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        zone.add_point((x, y))


# ── zone-drawing freeze (video mode) ─────────────────────────────────────────

def zone_setup_screen(frame: np.ndarray, depth_map: np.ndarray | None) -> bool:
    """
    Show a frozen frame so the user can click 4 zone corners before playback.
    Returns True when ready (4 pts placed + Space/Enter pressed), False to quit.
    """
    fh, fw = frame.shape[:2]
    while True:
        disp = frame.copy()

        # depth overlay hint
        if depth_map is not None:
            cmap = cv2.applyColorMap((depth_map * 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)
            disp = cv2.addWeighted(disp, 0.55, cmap, 0.45, 0)

        draw_placement(disp, zone, preview_pt)

        if len(zone.ground_pts) == 4:
            draw_3d_zone(disp, zone, False)
            msg = "Zone set!  Press SPACE to start  |  R to redo  |  Q to quit"
        else:
            msg = f"Click corner {len(zone.ground_pts)+1}/4 on the ground plane"

        cv2.rectangle(disp, (0, fh - 30), (fw, fh), (0, 0, 0), -1)
        cv2.putText(disp, msg, (8, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)
        cv2.imshow("3D Zone Monitor", disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            return False
        elif key == ord('r'):
            zone.reset()
        elif key in (ord(' '), 13) and len(zone.ground_pts) == 4:
            return True


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="3D Virtual Zone Monitor")
    p.add_argument("--source", default="0",
                   help="Video file path or webcam index (default: 0)")
    p.add_argument("--output", default="",
                   help="Save annotated output to this .mp4 path (optional)")
    p.add_argument("--conf", type=float, default=CONF_THRESH,
                   help=f"Person detection confidence (default: {CONF_THRESH})")
    return p.parse_args()


def main():
    global zone
    args = parse_args()

    TARGET_FPS    = 30
    FRAME_BUDGET  = 1.0 / TARGET_FPS   # seconds per frame

    source = int(args.source) if args.source.isdigit() else args.source
    is_video_file = isinstance(source, str)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  source={source}  target={TARGET_FPS}fps")

    yolo = YOLO(YOLO_MODEL)

    depth_est: DepthEstimator | None = None
    if HAS_DEPTH:
        depth_est = DepthEstimator(device)
    else:
        print("[WARN] transformers not installed — no depth estimation.")
        print("       pip install transformers pillow\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # tell OpenCV to buffer only 1 frame so reads are always fresh
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {fw}x{fh} @ {src_fps:.1f}fps  frames={total_frames}")

    writer = None

    cv2.namedWindow("3D Zone Monitor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("3D Zone Monitor", on_mouse)

    # ── depth background thread ────────────────────────────────────────────
    # The thread consumes frames from depth_in_q and posts depth maps to
    # depth_out_q. Main loop reads the latest result without blocking.
    depth_in_q:  queue.Queue = queue.Queue(maxsize=1)
    depth_out_q: queue.Queue = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    def depth_worker():
        while not stop_evt.is_set():
            try:
                frame_bgr = depth_in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            raw = depth_est.estimate(frame_bgr)
            resized = cv2.resize(raw, (fw, fh))
            # keep only the freshest result
            if not depth_out_q.empty():
                try:
                    depth_out_q.get_nowait()
                except queue.Empty:
                    pass
            depth_out_q.put(resized)

    if depth_est:
        t = threading.Thread(target=depth_worker, daemon=True)
        t.start()

    depth_map: np.ndarray | None = None
    frame_idx   = 0
    last_alert  = 0.0
    show_depth  = False
    alert_count = 0

    # FPS tracking
    fps_t0      = time.perf_counter()
    fps_count   = 0
    display_fps = 0.0

    print("Click 4 ground-plane corners on the running video to define the 3D zone.")
    print("D=depth overlay | +/-=depth band | R=reset | Q=quit\n")

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── feed depth thread (non-blocking, drop if busy) ────────────────
        if depth_est and frame_idx % DEPTH_SKIP == 0:
            if depth_in_q.empty():
                depth_in_q.put_nowait(frame.copy())

        # ── collect latest depth result (non-blocking) ────────────────────
        try:
            depth_map = depth_out_q.get_nowait()
            if len(zone.ground_pts) == 4 and zone.d_center is None:
                zone.sample_depth(depth_map)
        except queue.Empty:
            pass

        # ── depth overlay ──────────────────────────────────────────────────
        if show_depth and depth_map is not None:
            cmap = cv2.applyColorMap((depth_map * 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)
            frame = cv2.addWeighted(frame, 0.50, cmap, 0.50, 0)

        # ── YOLO detection ─────────────────────────────────────────────────
        results = yolo(frame, classes=[PERSON_CLASS], conf=args.conf,
                       verbose=False)[0]

        persons_bev: list[tuple[float, float, bool]] = []
        intrusion = False

        for box in results.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            foot = ((px1 + px2) // 2, py2)
            foot_depth = 0.5
            if depth_map is not None:
                xi = min(max(foot[0], 0), fw - 1)
                yi = min(max(foot[1], 0), fh - 1)
                foot_depth = float(depth_map[yi, xi])

            in_zone = zone.contains(foot, foot_depth)
            if in_zone:
                intrusion = True

            col = (0, 0, 255) if in_zone else (0, 220, 70)
            cv2.rectangle(frame, (px1, py1), (px2, py2), col, 2)
            cv2.circle(frame, foot, 5, col, -1)
            depth_str = f" d={foot_depth:.2f}" if depth_map is not None else ""
            cv2.putText(frame, f"person {conf:.2f}{depth_str}",
                        (px1, py1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            persons_bev.append((foot[0] / fw, foot_depth, in_zone))

        # ── draw 3D zone ───────────────────────────────────────────────────
        if len(zone.ground_pts) > 0:
            draw_3d_zone(frame, zone, intrusion)
        if not zone.ready:
            draw_placement(frame, zone, preview_pt)

        # ── BEV panel ──────────────────────────────────────────────────────
        bev = draw_bev(zone, persons_bev)
        bx, by = fw - BEV_SIZE - 8, fh - BEV_SIZE - 8
        if bx > 0 and by > 0:
            frame[by:by + BEV_SIZE, bx:bx + BEV_SIZE] = bev
            cv2.rectangle(frame, (bx - 1, by - 1),
                          (bx + BEV_SIZE, by + BEV_SIZE), (70, 70, 70), 1)

        # ── progress bar ──────────────────────────────────────────────────
        if is_video_file and total_frames > 0:
            bar_w = int(fw * frame_idx / total_frames)
            cv2.rectangle(frame, (0, fh - 4), (bar_w, fh), (0, 200, 120), -1)

        # ── alert banner ───────────────────────────────────────────────────
        now = time.time()
        if intrusion:
            alert_count += 1
            cv2.rectangle(frame, (0, 0), (fw, 52), (0, 0, 160), -1)
            cv2.putText(frame, f"!  INTRUSION DETECTED  (#{alert_count})  !",
                        (14, 37), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2)
            if now - last_alert > ALERT_COOLDOWN:
                depths = [d for (_, d, iz) in persons_bev if iz]
                ts = time.strftime('%H:%M:%S')
                d_str = f"depth={depths[0]:.2f}" if depths else ""
                print(f"[{ts}] WARNING: Person in 3D zone! frame={frame_idx} {d_str}")
                last_alert = now

        # ── FPS counter ───────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.perf_counter() - fps_t0
        if elapsed >= 1.0:
            display_fps = fps_count / elapsed
            fps_count   = 0
            fps_t0      = time.perf_counter()

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"{display_fps:.1f} fps", (fw - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
        if zone.d_center is not None:
            hud = (f"frame {frame_idx}/{total_frames}  "
                   f"band ±{zone.d_half:.2f}  "
                   f"alerts={alert_count}  D=depth +/-=band R=reset Q=quit")
        else:
            hud = ("Click 4 corners to set zone" if len(zone.ground_pts) < 4
                   else "Waiting for depth sample…")
        cv2.putText(frame, hud, (8, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        if writer:
            writer.write(frame)

        cv2.imshow("3D Zone Monitor", frame)

        # ── frame pacing: sleep off remaining budget ───────────────────────
        elapsed_loop = time.perf_counter() - loop_start
        wait_ms = max(1, int((FRAME_BUDGET - elapsed_loop) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            zone.reset()
            print("Zone reset.")
        elif key == ord('d'):
            show_depth = not show_depth
        elif key in (ord('+'), ord('=')):
            zone.adjust_band(+0.02)
        elif key == ord('-'):
            zone.adjust_band(-0.02)

    stop_evt.set()
    cap.release()
    if writer:
        writer.release()
    print(f"[done] {frame_idx} frames  {display_fps:.1f}fps  {alert_count} alerts")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
