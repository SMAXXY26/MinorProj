"""
geometry.py — Weapon Geometry Extraction

Given an OBB detection (cx, cy, w, h, θ) this module computes:
  1. Rotated bounding box corners  (4×2 pixel coords)
  2. Minimum enclosing ellipse     (axes a, b, tilt angle)
  3. Oriented bounding rectangle   (from contour — most accurate angle)
  4. Aspect ratio                  (length / width — barrel detection)
  5. Estimated weapon length       (pixels → metric via calibration)
  6. Contour polygon               (Douglas-Peucker simplified)
  7. Geometric class heuristics    (long+thin = rifle/blade, square = pistol)
  8. Keypoint locations            (tip/grip/barrel for downstream analysis)

All functions operate on numpy arrays and are called during inference.
No gradients needed — pure OpenCV + numpy.
"""

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeaponGeometry:
    # OBB parameters
    cx_px:          float = 0.0
    cy_px:          float = 0.0
    width_px:       float = 0.0
    height_px:      float = 0.0
    angle_deg:      float = 0.0          # rotation angle in degrees

    # Derived geometric properties
    corners:        np.ndarray = field(default_factory=lambda: np.zeros((4,2)))
    aspect_ratio:   float = 0.0          # length / width (long side / short side)
    area_px2:       float = 0.0
    perimeter_px:   float = 0.0

    # Ellipse fit  (on the mask / contour)
    ellipse_major:  float = 0.0          # semi-major axis (px)
    ellipse_minor:  float = 0.0          # semi-minor axis (px)
    ellipse_angle:  float = 0.0          # tilt angle (degrees)
    eccentricity:   float = 0.0          # 0 = circle, 1 = line segment

    # Contour polygon (simplified)
    polygon:        List[Tuple[int,int]] = field(default_factory=list)
    poly_vertices:  int = 0

    # Keypoints (tip, grip)
    keypoints:      dict = field(default_factory=dict)

    # Geometric class hint
    shape_hint:     str = "unknown"      # "long_weapon", "handgun", "blade", "blunt"

    # Orientation metadata
    is_horizontal:  bool = False
    is_vertical:    bool = False
    orientation_label: str = ""          # "pointing_left", "pointing_right", etc.

    def to_dict(self) -> dict:
        return {
            "center":       (round(self.cx_px, 1), round(self.cy_px, 1)),
            "size_px":      (round(self.width_px, 1), round(self.height_px, 1)),
            "angle_deg":    round(self.angle_deg, 2),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "eccentricity": round(self.eccentricity, 3),
            "ellipse":      {
                "major":    round(self.ellipse_major, 1),
                "minor":    round(self.ellipse_minor, 1),
                "angle":    round(self.ellipse_angle, 2),
            },
            "poly_vertices":self.poly_vertices,
            "keypoints":    self.keypoints,
            "shape_hint":   self.shape_hint,
            "orientation":  self.orientation_label,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_geometry(cx: float, cy: float,
                     w: float, h: float,
                     theta: float,
                     img_size: int = 640,
                     mask: Optional[np.ndarray] = None,
                     epsilon_ratio: float = 0.02) -> WeaponGeometry:
    """
    Compute full geometric properties for a weapon detection.

    Parameters
    ----------
    cx, cy  : centre of OBB in *pixels*
    w, h    : width/height of OBB in pixels
    theta   : rotation angle in radians (from OBB head, ∈ [-π/2, π/2])
    img_size: canvas size (used for orientation labelling)
    mask    : optional binary mask (H×W uint8).
              If provided, contour + ellipse fit on mask.
              If None, synthetic contour from OBB corners is used.
    epsilon_ratio: DP simplification ε = ratio × arc_length
    """
    geom = WeaponGeometry()
    theta_deg = math.degrees(theta)

    # ── 1. OBB corners ──────────────────────────────────────────────────────
    geom.cx_px      = cx
    geom.cy_px      = cy
    geom.width_px   = max(w, h)     # always make "width" the long side
    geom.height_px  = min(w, h)
    geom.angle_deg  = theta_deg
    geom.area_px2   = w * h
    geom.corners    = compute_obb_corners(cx, cy, w, h, theta)
    geom.aspect_ratio = (max(w, h) / (min(w, h) + 1e-6))
    geom.perimeter_px = 2 * (w + h)

    # ── 2. Contour ──────────────────────────────────────────────────────────
    if mask is not None and mask.sum() > 50:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cnt = max(contours, key=cv2.contourArea)
        else:
            cnt = corners_to_contour(geom.corners)
    else:
        cnt = corners_to_contour(geom.corners)

    # ── 3. Minimum enclosing ellipse ────────────────────────────────────────
    if cnt is not None and len(cnt) >= 5:
        try:
            (ex, ey), (ea, eb), e_angle = cv2.fitEllipse(cnt)
            geom.ellipse_major  = max(ea, eb) / 2
            geom.ellipse_minor  = min(ea, eb) / 2
            geom.ellipse_angle  = e_angle
            if geom.ellipse_major > 1e-3:
                # eccentricity = sqrt(1 - (b/a)^2)
                ratio = geom.ellipse_minor / geom.ellipse_major
                geom.eccentricity = math.sqrt(max(0, 1 - ratio ** 2))
        except cv2.error:
            pass

    # ── 4. Simplified polygon (Douglas-Peucker) ─────────────────────────────
    if cnt is not None and len(cnt) > 2:
        arc     = cv2.arcLength(cnt, closed=True)
        epsilon = epsilon_ratio * arc
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=True)
        geom.polygon      = [(int(p[0][0]), int(p[0][1])) for p in approx]
        geom.poly_vertices = len(approx)

    # ── 5. Minimum area rotated rect from contour ───────────────────────────
    if cnt is not None and len(cnt) >= 4:
        rect = cv2.minAreaRect(cnt)
        # rect = ((cx,cy), (w,h), angle)
        # Override angle with more accurate contour-derived value
        geom.angle_deg = rect[2]

    # ── 6. Keypoints ────────────────────────────────────────────────────────
    geom.keypoints = compute_keypoints(geom.corners, theta)

    # ── 7. Shape heuristic classification ───────────────────────────────────
    geom.shape_hint = classify_shape(geom.aspect_ratio, geom.eccentricity)

    # ── 8. Orientation label ─────────────────────────────────────────────────
    geom.is_horizontal = abs(theta_deg) < 30 or abs(theta_deg) > 150
    geom.is_vertical   = 60 < abs(theta_deg) < 120
    geom.orientation_label = compute_orientation_label(theta_deg, cx, img_size)

    return geom


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry sub-routines
# ─────────────────────────────────────────────────────────────────────────────

def compute_obb_corners(cx: float, cy: float,
                        w: float, h: float,
                        theta: float) -> np.ndarray:
    """
    Return the 4 corners of an oriented bounding box in pixel space.
    Order: top-left, top-right, bottom-right, bottom-left (relative to
    the un-rotated box; rotation applied after).
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    hw, hh = w / 2, h / 2

    local = np.array([
        [-hw, -hh],  # TL
        [ hw, -hh],  # TR
        [ hw,  hh],  # BR
        [-hw,  hh],  # BL
    ], dtype=np.float32)

    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    rotated = local @ R.T + np.array([[cx, cy]])
    return rotated.astype(np.float32)


def corners_to_contour(corners: np.ndarray) -> np.ndarray:
    """Convert 4 corners (4×2) to OpenCV contour format (4,1,2 int32)."""
    return corners.reshape(4, 1, 2).astype(np.int32)


def compute_keypoints(corners: np.ndarray, theta: float) -> dict:
    """
    Estimate weapon keypoints from OBB corners.

    Convention: for a weapon in its canonical orientation (horizontal,
    barrel pointing right), the tip is the right-most point and the
    grip is the left-most point.  We rotate this interpretation by θ.

    Returns dict with 'tip', 'grip', 'centre', 'midline_vector'.
    """
    # Corners are in pixel space. Find the "long axis" endpoints.
    # The long axis connects the midpoints of the two short edges.
    mid_top    = (corners[0] + corners[1]) / 2   # midpoint of top edge
    mid_bottom = (corners[3] + corners[2]) / 2   # midpoint of bottom edge
    mid_left   = (corners[0] + corners[3]) / 2   # midpoint of left edge
    mid_right  = (corners[1] + corners[2]) / 2   # midpoint of right edge

    # Long axis: connects the two midpoints that are further apart
    d_lr = np.linalg.norm(mid_right - mid_left)
    d_tb = np.linalg.norm(mid_bottom - mid_top)

    if d_lr >= d_tb:
        tip  = mid_right
        grip = mid_left
    else:
        tip  = mid_top
        grip = mid_bottom

    midline_vec = tip - grip
    midline_norm = midline_vec / (np.linalg.norm(midline_vec) + 1e-6)

    centre = (corners.mean(axis=0))

    return {
        "tip":             (int(tip[0]),    int(tip[1])),
        "grip":            (int(grip[0]),   int(grip[1])),
        "centre":          (int(centre[0]), int(centre[1])),
        "midline_vector":  (float(midline_norm[0]), float(midline_norm[1])),
        "pointing_angle":  float(math.degrees(
            math.atan2(midline_norm[1], midline_norm[0])
        )),
    }


def classify_shape(aspect_ratio: float, eccentricity: float) -> str:
    """
    Coarse shape heuristic based purely on geometry.
    Complements the neural classifier — useful as a sanity check.

    Rifle/SMG:    very elongated (AR > 3.5), high eccentricity
    Pistol:       moderately elongated (AR 1.5–3.5)
    Knife/blade:  very elongated + thin
    Blunt weapon: AR ~1–2, lower eccentricity
    """
    if aspect_ratio > 4.5 and eccentricity > 0.92:
        return "long_weapon"         # rifle, SMG, shotgun
    elif aspect_ratio > 3.0 and eccentricity > 0.85:
        return "blade_or_rifle"
    elif 1.5 < aspect_ratio <= 3.0:
        return "handgun"
    elif aspect_ratio <= 1.5 and eccentricity < 0.7:
        return "blunt"
    else:
        return "unknown"


def compute_orientation_label(angle_deg: float,
                               cx: float,
                               img_size: int) -> str:
    """
    Human-readable orientation string.
    Also records which half of the frame the weapon is in
    (useful for scene understanding: "left side, pointing right" = threat vector).
    """
    # Normalise angle to [-180, 180)
    a = ((angle_deg + 90) % 180) - 90

    if   -22.5 <= a <  22.5:   direction = "horizontal_right"
    elif  22.5 <= a <  67.5:   direction = "diagonal_down_right"
    elif  67.5 <= a <= 90.0:   direction = "vertical"
    elif -67.5 <= a < -22.5:   direction = "diagonal_up_right"
    else:                       direction = "horizontal_left"

    side = "left_frame" if cx < img_size / 2 else "right_frame"
    return f"{direction}_{side}"


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def draw_geometry(img: np.ndarray, geom: WeaponGeometry,
                  label: str = "", color: Tuple = (0, 255, 80),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw all geometry annotations onto an image (for debugging / demo).
    Returns annotated copy.
    """
    vis = img.copy()

    # ── OBB rotated rectangle ───────────────────────────────────────────────
    pts = geom.corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

    # ── Contour polygon ──────────────────────────────────────────────────────
    if geom.polygon:
        poly_pts = np.array(geom.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_pts], isClosed=True,
                      color=(255, 200, 0), thickness=1)

    # ── Ellipse ───────────────────────────────────────────────────────────────
    if geom.ellipse_major > 1:
        cx_i = int(geom.cx_px)
        cy_i = int(geom.cy_px)
        try:
            cv2.ellipse(vis,
                        center = (cx_i, cy_i),
                        axes   = (int(geom.ellipse_major), int(geom.ellipse_minor)),
                        angle  = geom.ellipse_angle,
                        startAngle=0, endAngle=360,
                        color=(200, 100, 255), thickness=1)
        except cv2.error:
            pass

    # ── Keypoints ────────────────────────────────────────────────────────────
    if "tip" in geom.keypoints:
        tip  = geom.keypoints["tip"]
        grip = geom.keypoints["grip"]
        cv2.circle(vis, tip,  5, (0,   0, 255), -1)   # tip  = red
        cv2.circle(vis, grip, 5, (255, 0,   0), -1)   # grip = blue
        # Midline
        cv2.arrowedLine(vis, grip, tip, (0, 255, 200), 1, tipLength=0.15)

    # ── Angle text ────────────────────────────────────────────────────────────
    cx_i = int(geom.cx_px)
    cy_i = int(geom.cy_px)
    info = (f"{label}  AR:{geom.aspect_ratio:.1f}  "
            f"θ:{geom.angle_deg:.1f}°  {geom.shape_hint}")
    cv2.putText(vis, info,
                (max(0, cx_i - 80), max(20, cy_i - int(geom.height_px/2) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Batch geometry extraction (for multiple detections in one frame)
# ─────────────────────────────────────────────────────────────────────────────

def extract_batch_geometry(detections: list,
                            img_size: int = 640,
                            masks: Optional[List[np.ndarray]] = None
                           ) -> List[WeaponGeometry]:
    """
    Run geometry extraction for all detections in a single frame.

    detections : list of (cx, cy, w, h, θ) tuples (pixel coords)
    masks      : optional list of binary masks, one per detection
    """
    results = []
    for i, det in enumerate(detections):
        cx, cy, w, h, theta = det
        mask = masks[i] if (masks and i < len(masks)) else None
        geom = extract_geometry(cx, cy, w, h, theta, img_size, mask)
        results.append(geom)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Motion geometry (optical flow magnitude — used as BiLSTM input feature)
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_magnitude(prev_gray: np.ndarray,
                              curr_gray: np.ndarray,
                              roi: Optional[Tuple[int,int,int,int]] = None
                             ) -> float:
    """
    Estimate motion magnitude in a region using dense optical flow.
    Used as the 14th feature in the BiLSTM input vector.

    prev_gray, curr_gray : grayscale frames (H, W) uint8
    roi                  : (x1, y1, x2, y2) in pixels — restrict to weapon region
    Returns              : mean flow magnitude (normalised 0–1)
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        prev_gray = prev_gray[y1:y2, x1:x2]
        curr_gray = curr_gray[y1:y2, x1:x2]

    if prev_gray.size == 0 or curr_gray.size == 0:
        return 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow       = None,
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 15,
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Normalise: typical fast-action max flow ~50px/frame
    return float(np.mean(magnitude) / 50.0)
