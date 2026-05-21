"""
utils/person_tracker.py  —  Person detection and weapon–person association
=========================================================================
Detects people in a frame (YOLOv8n COCO pretrained), associates each weapon
bounding box with the person most likely holding it, and draws a distinct
visual overlay on armed individuals.

Association strategy (in priority order):
  1. IoU overlap  — person bbox that overlaps weapon bbox with IoU > threshold
  2. Centroid distance fallback — nearest person centroid within max_dist_px

Usage in inference.py:
    from utils.person_tracker import (
        PersonDetector, WeaponPersonAssociator, draw_person_overlays
    )
    person_det  = PersonDetector(device=device)
    associator  = WeaponPersonAssociator()

    # Per frame:
    persons      = person_det.detect(frame)
    carrier_ids  = associator.associate(weapon_bboxes, persons)
    draw_person_overlays(frame, persons, carrier_ids)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Visual constants ──────────────────────────────────────────────────────────
_COLOR_ARMED      = (0,   0,   220)   # red   — weapon carrier
_COLOR_NEUTRAL    = (100, 100, 100)   # grey  — other persons
_COLOR_LABEL_BG   = (0,   0,   180)   # dark red label background
_THICK_ARMED      = 3
_THICK_NEUTRAL    = 1
_COCO_PERSON_CLS  = 0


# =============================================================================
#  PersonDetection dataclass
# =============================================================================

@dataclass
class PersonDetection:
    person_id:  int             # index in the per-frame list
    bbox_xyxy:  np.ndarray      # [x1, y1, x2, y2] float32
    confidence: float


# =============================================================================
#  PersonDetector
# =============================================================================

class PersonDetector:
    """
    Detects people using YOLOv8n pretrained on COCO (class 0 = person).
    No weapon training needed — runs alongside the weapon detector.
    """

    def __init__(
        self,
        weights:        str   = "yolov8n.pt",
        conf_threshold: float = 0.40,
        device:         str   = "cpu",
    ) -> None:
        from ultralytics import YOLO
        self._model  = YOLO(weights)
        self._conf   = conf_threshold
        self._device = device
        logger.info(f"[PersonDetector] {weights}  conf>{conf_threshold}")

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Return all PersonDetection objects visible in frame."""
        results = self._model(
            frame,
            conf    = self._conf,
            classes = [_COCO_PERSON_CLS],
            verbose = False,
            device  = self._device,
        )
        out: list[PersonDetection] = []
        if results and len(results[0].boxes):
            boxes = results[0].boxes
            for i, (xyxy, conf) in enumerate(
                zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())
            ):
                out.append(PersonDetection(
                    person_id  = i,
                    bbox_xyxy  = xyxy.astype(np.float32),
                    confidence = float(conf),
                ))
        return out


# =============================================================================
#  WeaponPersonAssociator
# =============================================================================

class WeaponPersonAssociator:
    """
    Associates weapon bounding boxes to persons using IoU + centroid fallback.
    Returns the set of person_id values that are carrying at least one weapon.
    """

    def __init__(
        self,
        iou_threshold:        float = 0.15,
        max_centroid_dist_px: float = 250.0,
    ) -> None:
        self._iou_thresh = iou_threshold
        self._max_dist   = max_centroid_dist_px

    def associate(
        self,
        weapon_bboxes:     list[np.ndarray],      # each [x1,y1,x2,y2]
        person_detections: list[PersonDetection],
    ) -> set[int]:
        """Return set of person_id values that are armed."""
        if not person_detections or not weapon_bboxes:
            return set()

        carriers: set[int] = set()
        p_boxes = np.array([p.bbox_xyxy for p in person_detections])

        for w_box in weapon_bboxes:
            best_iou  = 0.0
            best_pid: Optional[int] = None

            # Pass 1 — IoU
            for pid, p_box in enumerate(p_boxes):
                iou = _iou(w_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid

            if best_iou >= self._iou_thresh and best_pid is not None:
                carriers.add(best_pid)
                continue

            # Pass 2 — centroid distance fallback
            w_cx = (w_box[0] + w_box[2]) / 2.0
            w_cy = (w_box[1] + w_box[3]) / 2.0
            best_dist = self._max_dist

            for pid, p_box in enumerate(p_boxes):
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0
                dist = float(np.hypot(w_cx - p_cx, w_cy - p_cy))
                if dist < best_dist:
                    best_dist = dist
                    best_pid  = pid

            if best_pid is not None:
                carriers.add(best_pid)

        return carriers


# =============================================================================
#  Drawing
# =============================================================================

def draw_person_overlays(
    frame:             np.ndarray,
    person_detections: list[PersonDetection],
    carrier_ids:       set[int],
    weapon_label:      str = "",          # e.g. "RIFLE" — appended to the ARMED tag
) -> None:
    """
    Draw overlays on all detected persons.
      Armed carriers  → thick red box + corner brackets + "ARMED" label
      Other persons   → thin grey box (situational awareness only)
    """
    for p in person_detections:
        x1, y1, x2, y2 = map(int, p.bbox_xyxy)
        armed = p.person_id in carrier_ids

        color = _COLOR_ARMED   if armed else _COLOR_NEUTRAL
        thick = _THICK_ARMED   if armed else _THICK_NEUTRAL

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        if armed:
            tag = f"ARMED  {weapon_label}".strip() if weapon_label else "ARMED"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10), (x1 + tw + 6, y1),
                _COLOR_LABEL_BG, -1,
            )
            cv2.putText(
                frame, tag, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            _draw_corner_brackets(frame, x1, y1, x2, y2, color)


# =============================================================================
#  Internal helpers
# =============================================================================

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


def _draw_corner_brackets(
    frame:     np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color:     tuple,
    length:    int = 18,
    thickness: int = 3,
) -> None:
    """L-shaped corner brackets drawn inside the bounding box for emphasis."""
    corners = [
        ((x1, y1 + length), (x1, y1),  (x1 + length, y1)),   # top-left
        ((x2 - length, y1), (x2, y1),  (x2, y1 + length)),   # top-right
        ((x1, y2 - length), (x1, y2),  (x1 + length, y2)),   # bottom-left
        ((x2 - length, y2), (x2, y2),  (x2, y2 - length)),   # bottom-right
    ]
    for p0, corner, p1 in corners:
        cv2.line(frame, p0, corner, color, thickness)
        cv2.line(frame, corner, p1,  color, thickness)
