"""
utils/dispatchers.py  —  Alert dispatcher implementations
==========================================================
Three dispatchers that implement the dispatch(payload) interface:

  WebhookDispatcher   — HTTP POST with retry
  FileLogDispatcher   — appends to alerts.jsonl + alerts.log
  ConsoleDispatcher   — ANSI-coloured terminal output
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ── ANSI colours ─────────────────────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


# ---------------------------------------------------------------------------
#  WebhookDispatcher
# ---------------------------------------------------------------------------

class WebhookDispatcher:
    """HTTP POST the alert payload as JSON. Never raises."""

    def __init__(self, url: str, timeout: int = 2) -> None:
        self._url     = url
        self._timeout = timeout

    def dispatch(self, payload: "AlertPayload") -> None:  # type: ignore[name-defined]
        try:
            import requests
        except ImportError:
            logger.warning("[Webhook] requests not installed — skipping")
            return

        data = asdict(payload)
        # GeoPoint may be nested dataclass — convert sub-dict safely
        if data.get("geo") and hasattr(data["geo"], "__dict__"):
            data["geo"] = asdict(data["geo"])

        try:
            resp = requests.post(self._url, json=data, timeout=self._timeout)
            resp.raise_for_status()
        except requests.ConnectionError:
            # Single retry
            try:
                resp = requests.post(self._url, json=data, timeout=self._timeout)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning(f"[Webhook] Retry failed: {exc}")
        except Exception as exc:
            logger.warning(f"[Webhook] POST failed: {exc}")


# ---------------------------------------------------------------------------
#  FileLogDispatcher
# ---------------------------------------------------------------------------

class FileLogDispatcher:
    """Appends alerts to JSONL and human-readable log files."""

    def __init__(self, log_dir: str = "logs/alerts") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._dir / "alerts.jsonl"
        self._log   = self._dir / "alerts.log"

    def dispatch(self, payload: "AlertPayload") -> None:  # type: ignore[name-defined]
        from dataclasses import asdict
        data = asdict(payload)

        # JSONL — one JSON object per line
        with open(self._jsonl, "a") as f:
            f.write(json.dumps(data) + "\n")

        # Human-readable log
        geo = payload.geo
        geo_str = (f"{geo.lat:.5f},{geo.lon:.5f}" if geo else "N/A")
        line = (
            f"[{payload.timestamp_iso}] "
            f"ALERT | {payload.class_name.upper()} "
            f"| conf={payload.confidence:.2f} "
            f"| track={payload.track_id} "
            f"| geo={geo_str} "
            f"| score={payload.threat_score:.2f} "
            f"| drone={payload.drone_id}\n"
        )
        with open(self._log, "a") as f:
            f.write(line)


# ---------------------------------------------------------------------------
#  ConsoleDispatcher
# ---------------------------------------------------------------------------

class ConsoleDispatcher:
    """ANSI-coloured terminal output for alerts."""

    def dispatch(self, payload: "AlertPayload") -> None:  # type: ignore[name-defined]
        cls = payload.class_name.lower()
        colour = _RED if cls in ("pistol", "rifle") else _YELLOW

        geo = payload.geo
        geo_str = (f"{geo.lat:.5f},{geo.lon:.5f}" if geo else "N/A,N/A")

        msg = (
            f"{colour}[ALERT] {payload.timestamp_iso} "
            f"| {payload.class_name.upper()} "
            f"| conf={payload.confidence:.2f} "
            f"| track={payload.track_id} "
            f"| geo={geo_str} "
            f"| score={payload.threat_score:.2f}{_RESET}"
        )
        print(msg)
