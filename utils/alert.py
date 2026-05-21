"""
utils/alert.py  —  Alert state management and dispatch pipeline
================================================================
Classes
-------
  AlertPayload   — dataclass passed to all dispatchers
  AlertState     — per-track state tracker
  AlertManager   — evaluates tracks and fires dispatchers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  AlertPayload
# ---------------------------------------------------------------------------

@dataclass
class AlertPayload:
    timestamp_iso: str
    track_id:      int
    class_name:    str
    confidence:    float
    geo:           object | None   # GeoPoint | None
    frame_id:      int
    drone_id:      str
    threat_score:  float


# ---------------------------------------------------------------------------
#  AlertState  (per track)
# ---------------------------------------------------------------------------

@dataclass
class AlertState:
    track_id:         int
    last_alert_frame: int = -9999   # initialise to a safely far-past frame


# ---------------------------------------------------------------------------
#  AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Evaluates tracks against alert criteria and calls dispatchers.

    Config keys (from cfg['alert']):
        confidence_threshold  float  default 0.70
        cooldown_frames       int    default 90
        drone_id              str    default 'DRONE_01'
    """

    def __init__(self, cfg: dict, dispatchers: list) -> None:
        alert_cfg = cfg.get("alert", {})
        self._conf_thresh    = float(alert_cfg.get("confidence_threshold", 0.70))
        self._cooldown       = int(alert_cfg.get("cooldown_frames", 90))
        self._drone_id       = str(alert_cfg.get("drone_id", "DRONE_01"))
        self._dispatchers    = dispatchers
        self._states: dict[int, AlertState] = {}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        track:    "TrackedWeapon",  # type: ignore[name-defined]
        geo:      object | None,    # GeoPoint | None
        frame_id: int,
    ) -> bool:
        """
        Fire alert when ALL three conditions are met:
          1. track.is_confirmed
          2. track.confidence >= confidence_threshold
          3. frame_id - last_alert_frame >= cooldown_frames

        Returns True if an alert was fired.
        """
        # Condition 1 — confirmed track
        if not track.is_confirmed:
            return False

        # Condition 2 — confidence gate
        if track.confidence < self._conf_thresh:
            return False

        # Condition 3 — cooldown
        state = self._states.setdefault(track.track_id, AlertState(track.track_id))
        if frame_id - state.last_alert_frame < self._cooldown:
            return False

        # --- All conditions met — fire ---
        state.last_alert_frame = frame_id

        from utils.threat_scorer import score_threat
        threat = score_threat(track, {})

        payload = AlertPayload(
            timestamp_iso = datetime.now(timezone.utc).isoformat(),
            track_id      = track.track_id,
            class_name    = track.class_name,
            confidence    = track.confidence,
            geo           = geo,
            frame_id      = frame_id,
            drone_id      = self._drone_id,
            threat_score  = threat,
        )

        for dispatcher in self._dispatchers:
            try:
                dispatcher.dispatch(payload)
            except Exception as exc:
                logger.warning(f"[AlertManager] Dispatcher error: {exc}")

        logger.info(
            f"[ALERT] track={track.track_id} cls={track.class_name} "
            f"conf={track.confidence:.2f} score={threat:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    def cleanup_stale(self, active_track_ids: set[int]) -> None:
        """Remove state for tracks no longer in the tracker."""
        stale = [tid for tid in self._states if tid not in active_track_ids]
        for tid in stale:
            del self._states[tid]
