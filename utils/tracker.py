"""
utils/tracker.py  —  ByteTrack multi-object tracker wrapper
=============================================================
Wraps supervision's ByteTracker and provides TrackedWeapon dataclass.

Requires: pip install supervision>=0.21.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import supervision as sv


@dataclass
class TrackedWeapon:
    track_id: int
    bbox_xyxy: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    age: int                              # frames this track has been alive
    is_confirmed: bool                    # True when age >= min_frames_to_confirm
    last_seen_frame: int
    velocity_px_per_frame: tuple[float, float]   # (vx, vy)
    geo: object | None = None            # GeoPoint | None — filled by inference.py


class WeaponTracker:
    """
    Multi-object tracker backed by supervision's ByteTracker.

    Example config keys consumed:
        tracking.track_activation_threshold  (float, default 0.45)
        tracking.lost_track_buffer           (int,   default 30)
        tracking.minimum_matching_threshold  (float, default 0.8)
        tracking.min_frames_to_confirm       (int,   default 3)
        tracking.frame_rate                  (int,   default 30)
    """

    CLASS_NAMES: list[str] = ["knife", "pistol", "rifle"]

    def __init__(self, cfg: dict) -> None:
        trk = cfg.get("tracking", {})

        self._tracker = sv.ByteTrack(
            track_activation_threshold = trk.get("track_activation_threshold", 0.45),
            lost_track_buffer          = trk.get("lost_track_buffer", 30),
            minimum_matching_threshold = trk.get("minimum_matching_threshold", 0.8),
            frame_rate                 = trk.get("frame_rate", 30),
        )
        self.min_frames_to_confirm: int = trk.get("min_frames_to_confirm", 3)

        # Per-track state
        self._history: dict[int, list[np.ndarray]] = {}   # last 5 bbox centres
        self._age:     dict[int, int]               = {}
        self._last_seen: dict[int, int]             = {}

        # Cache of most recent TrackedWeapon objects so callers can query them
        self._current_tracks: list[TrackedWeapon] = []

    # ------------------------------------------------------------------
    def update(
        self,
        detections: sv.Detections,
        frame_id: int,
    ) -> list[TrackedWeapon]:
        """
        Run ByteTrack on the current-frame detections.

        Parameters
        ----------
        detections : sv.Detections  — output of sv.Detections.from_ultralytics(...)
        frame_id   : int            — monotonically increasing frame counter

        Returns
        -------
        list[TrackedWeapon]  — all active tracks (confirmed + unconfirmed)
        """
        if len(detections) == 0:
            self._current_tracks = []
            return []

        tracked = self._tracker.update_with_detections(detections)

        results: list[TrackedWeapon] = []

        for i in range(len(tracked)):
            tid  = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i]            # np.ndarray [x1,y1,x2,y2]
            cls  = int(tracked.class_id[i])
            conf = float(tracked.confidence[i])

            # --- age ---
            self._age[tid] = self._age.get(tid, 0) + 1
            self._last_seen[tid] = frame_id

            # --- bbox centre history (keep last 5) ---
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            centre = np.array([cx, cy], dtype=np.float32)
            if tid not in self._history:
                self._history[tid] = []
            self._history[tid].append(centre)
            if len(self._history[tid]) > 5:
                self._history[tid].pop(0)

            # --- velocity (pixel/frame between last 2 positions) ---
            hist = self._history[tid]
            if len(hist) >= 2:
                delta = hist[-1] - hist[-2]
                vx, vy = float(delta[0]), float(delta[1])
            else:
                vx, vy = 0.0, 0.0

            age       = self._age[tid]
            confirmed = age >= self.min_frames_to_confirm
            cls_name  = (self.CLASS_NAMES[cls]
                         if 0 <= cls < len(self.CLASS_NAMES)
                         else f"cls{cls}")

            results.append(TrackedWeapon(
                track_id              = tid,
                bbox_xyxy             = bbox,
                class_id              = cls,
                class_name            = cls_name,
                confidence            = conf,
                age                   = age,
                is_confirmed          = confirmed,
                last_seen_frame       = frame_id,
                velocity_px_per_frame = (vx, vy),
            ))

        self._current_tracks = results
        return results

    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> list[TrackedWeapon]:
        """Return only tracks where is_confirmed=True."""
        return [t for t in self._current_tracks if t.is_confirmed]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all internal state."""
        self._history.clear()
        self._age.clear()
        self._last_seen.clear()
        self._current_tracks = []
