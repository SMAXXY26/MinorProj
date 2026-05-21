"""
utils/threat_scorer.py  —  Threat score computation
=====================================================
Computes a 0–1 threat score from a TrackedWeapon and optional context.

Weights
-------
  Class severity    0.35   rifle=1.0, pistol=0.8, knife=0.5
  Detection conf    0.25   raw value
  Track persistence 0.20   min(age/30, 1.0)
  Velocity mag      0.10   min(speed/20, 1.0)
  Time of day       0.10   22-06h=1.0, 18-22/06-08h=0.7, else=0.4
"""
from __future__ import annotations

import math
from datetime import datetime, timezone


# Class severity mapping
_SEVERITY: dict[str, float] = {
    "rifle":  1.0,
    "pistol": 0.8,
    "knife":  0.5,
}


def score_threat(track: "TrackedWeapon", context: dict) -> float:  # type: ignore[name-defined]
    """
    Compute a 0–1 threat score.

    Parameters
    ----------
    track   : TrackedWeapon instance
    context : dict — may contain 'local_hour' (int 0-23).
                     If absent, uses current UTC hour.

    Returns
    -------
    float in [0, 1]
    """
    # --- Class severity (0.35) ---
    severity = _SEVERITY.get(track.class_name.lower(), 0.5)

    # --- Detection confidence (0.25) ---
    conf = float(track.confidence)

    # --- Track persistence (0.20) ---
    persistence = min(track.age / 30.0, 1.0)

    # --- Velocity magnitude (0.10) ---
    vx, vy = track.velocity_px_per_frame
    speed  = math.sqrt(vx * vx + vy * vy)
    vel_score = min(speed / 20.0, 1.0)

    # --- Time of day (0.10) ---
    hour = context.get("local_hour", datetime.now(timezone.utc).hour)
    if 22 <= hour or hour < 6:
        tod_score = 1.0
    elif (18 <= hour < 22) or (6 <= hour < 8):
        tod_score = 0.7
    else:
        tod_score = 0.4

    # --- Weighted sum ---
    score = (
        0.35 * severity
        + 0.25 * conf
        + 0.20 * persistence
        + 0.10 * vel_score
        + 0.10 * tod_score
    )
    return float(min(max(score, 0.0), 1.0))
