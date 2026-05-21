"""
utils/geo.py  —  GPS / geo-stamping utilities
===============================================
Provides:
  GeoPoint            — dataclass for a GPS fix
  GpsReader           — serial NMEA reader (background thread)
  MockGpsReader       — fixed-position stub for desktop/CI testing
  BboxToGeoProjector  — maps bbox pixel coords → ground GeoPoint
  MavlinkTelemetry    — optional MAVLink altitude/heading reader
"""
from __future__ import annotations

import math
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  GeoPoint
# ---------------------------------------------------------------------------

@dataclass
class GeoPoint:
    lat: float
    lon: float
    alt_m: float
    heading_deg: float
    timestamp_utc: str


# ---------------------------------------------------------------------------
#  GpsReader  (real hardware — pyserial + pynmea2)
# ---------------------------------------------------------------------------

class GpsReader:
    """
    Reads NMEA sentences from a serial UART in a background daemon thread.
    Parses GPGGA (lat/lon/alt) and GPRMC (heading/speed).

    Usage::
        reader = GpsReader(cfg)
        # ... later ...
        fix = reader.get_latest()   # GeoPoint | None
    """

    def __init__(self, cfg: dict) -> None:
        self._port  = cfg.get("gps_serial_port", "/dev/ttyTHS0")
        self._baud  = int(cfg.get("gps_baud", 9600))
        self._fix:  GeoPoint | None = None
        self._lock  = threading.Lock()
        self._last_fix_time: float = 0.0
        self._running = True

        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def _read_loop(self) -> None:
        try:
            import serial
            import pynmea2
        except ImportError:
            logger.error("[GPS] pyserial or pynmea2 not installed — GPS disabled")
            return

        while self._running:
            try:
                with serial.Serial(self._port, self._baud, timeout=1) as s:
                    lat = lon = alt = heading = 0.0
                    while self._running:
                        raw = s.readline().decode("ascii", errors="replace").strip()
                        try:
                            msg = pynmea2.parse(raw)
                        except pynmea2.ParseError:
                            continue

                        if isinstance(msg, pynmea2.types.talker.GGA):
                            if msg.gps_qual and int(msg.gps_qual) > 0:
                                lat = float(msg.latitude)
                                lon = float(msg.longitude)
                                alt = float(msg.altitude or 0)
                                fix = GeoPoint(
                                    lat           = lat,
                                    lon           = lon,
                                    alt_m         = alt,
                                    heading_deg   = heading,
                                    timestamp_utc = datetime.now(timezone.utc).isoformat(),
                                )
                                with self._lock:
                                    self._fix = fix
                                    self._last_fix_time = time.monotonic()

                        elif isinstance(msg, pynmea2.types.talker.RMC):
                            if hasattr(msg, "true_course") and msg.true_course:
                                heading = float(msg.true_course)

            except Exception as exc:
                logger.warning(f"[GPS] Serial error: {exc}  — retrying in 5 s")
                time.sleep(5)

    def get_latest(self) -> GeoPoint | None:
        """Return latest GPS fix or None if no fix / stale."""
        with self._lock:
            fix = self._fix
        if fix is not None:
            stale = time.monotonic() - self._last_fix_time
            if stale > 5.0:
                logger.warning(f"[GPS] No fix for >{stale:.0f} s")
        return fix

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
#  MockGpsReader  (desktop / CI testing)
# ---------------------------------------------------------------------------

class MockGpsReader:
    """
    Returns a fixed GeoPoint from config.
    Drop-in replacement for GpsReader — identical interface.
    """

    def __init__(self, cfg: dict) -> None:
        self._point = GeoPoint(
            lat           = float(cfg.get("mock_gps_lat",   28.6139)),
            lon           = float(cfg.get("mock_gps_lon",   77.2090)),
            alt_m         = float(cfg.get("mock_gps_alt_m", 50.0)),
            heading_deg   = 0.0,
            timestamp_utc = datetime.now(timezone.utc).isoformat(),
        )

    def get_latest(self) -> GeoPoint | None:
        # Update timestamp each call so it looks "live"
        return GeoPoint(
            lat           = self._point.lat,
            lon           = self._point.lon,
            alt_m         = self._point.alt_m,
            heading_deg   = self._point.heading_deg,
            timestamp_utc = datetime.now(timezone.utc).isoformat(),
        )

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
#  BboxToGeoProjector
# ---------------------------------------------------------------------------

class BboxToGeoProjector:
    """
    Projects a bounding box centre (pixel coords) to a ground-level GeoPoint
    using a flat-earth approximation (valid below ~500 m altitude).

    FOV angles refer to the full horizontal / vertical field of view.
    """

    def __init__(
        self,
        fov_h_deg: float,
        fov_v_deg: float,
        img_w: int,
        img_h: int,
    ) -> None:
        self.fov_h_rad = math.radians(fov_h_deg)
        self.fov_v_rad = math.radians(fov_v_deg)
        self.img_w     = img_w
        self.img_h     = img_h

    def project(
        self,
        bbox_xyxy: "np.ndarray",          # [x1, y1, x2, y2] in pixels
        drone_geo: GeoPoint,
        altitude_agl_m: float | None = None,
    ) -> GeoPoint:
        """
        Returns the estimated ground GeoPoint of the detected object.

        Parameters
        ----------
        bbox_xyxy      : detection bounding box in pixel coords
        drone_geo      : current drone GPS fix
        altitude_agl_m : AGL altitude in metres; falls back to drone_geo.alt_m
        """
        import math

        alt = altitude_agl_m if altitude_agl_m is not None else drone_geo.alt_m

        # Bbox centre pixel coords
        cx_px = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy_px = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0

        # Offset from image centre (normalised -1..+1)
        dx_norm = (cx_px - self.img_w / 2.0) / (self.img_w / 2.0)
        dy_norm = (cy_px - self.img_h / 2.0) / (self.img_h / 2.0)

        # Ground footprint from altitude + FOV
        footprint_x_m = 2.0 * alt * math.tan(self.fov_h_rad / 2.0)
        footprint_y_m = 2.0 * alt * math.tan(self.fov_v_rad / 2.0)

        # Raw ground offsets in metres (image frame)
        raw_dx_m = dx_norm * footprint_x_m / 2.0
        raw_dy_m = dy_norm * footprint_y_m / 2.0

        # Apply drone heading rotation (image X,Y → North/East)
        hdg_rad = math.radians(drone_geo.heading_deg)
        north_m =  raw_dx_m * math.sin(hdg_rad) + raw_dy_m * math.cos(hdg_rad)
        east_m  =  raw_dx_m * math.cos(hdg_rad) - raw_dy_m * math.sin(hdg_rad)

        # Flat-earth lat/lon conversion (valid < 500 m)
        lat_rad = math.radians(drone_geo.lat)
        d_lat   = north_m / 111_320.0
        d_lon   = east_m  / (111_320.0 * math.cos(lat_rad))

        return GeoPoint(
            lat           = drone_geo.lat + d_lat,
            lon           = drone_geo.lon + d_lon,
            alt_m         = 0.0,   # ground level
            heading_deg   = drone_geo.heading_deg,
            timestamp_utc = drone_geo.timestamp_utc,
        )


# ---------------------------------------------------------------------------
#  MavlinkTelemetry  (optional)
# ---------------------------------------------------------------------------

class MavlinkTelemetry:
    """
    Reads altitude AGL and heading from a MAVLink-capable flight controller.

    Falls back gracefully if pymavlink is not installed or port is None.
    """

    def __init__(self, cfg: dict) -> None:
        self._port    = cfg.get("mavlink_serial_port", None)
        self._running = False
        self._telem: dict = {
            "alt_agl_m":  None,
            "heading_deg": 0.0,
            "roll_deg":    0.0,
            "pitch_deg":   0.0,
        }
        self._lock = threading.Lock()

        if self._port is None:
            logger.info("[MAVLink] No port configured — skipping MAVLink")
            return

        try:
            import pymavlink.mavutil as mavutil  # type: ignore
            self._mav = mavutil.mavlink_connection(self._port, baud=57600)
            self._running = True
            t = threading.Thread(target=self._read_loop, daemon=True)
            t.start()
            logger.info(f"[MAVLink] Connected to {self._port}")
        except Exception as exc:
            logger.warning(f"[MAVLink] Could not connect: {exc}")

    def _read_loop(self) -> None:
        try:
            import pymavlink.mavutil as mavutil  # type: ignore
            while self._running:
                msg = self._mav.recv_match(
                    type=["GLOBAL_POSITION_INT", "ATTITUDE"],
                    blocking=True, timeout=1,
                )
                if msg is None:
                    continue
                mtype = msg.get_type()
                if mtype == "GLOBAL_POSITION_INT":
                    with self._lock:
                        # relative_alt in mm → m
                        self._telem["alt_agl_m"] = msg.relative_alt / 1000.0
                        self._telem["heading_deg"] = msg.hdg / 100.0
                elif mtype == "ATTITUDE":
                    with self._lock:
                        self._telem["roll_deg"]  = math.degrees(msg.roll)
                        self._telem["pitch_deg"] = math.degrees(msg.pitch)
        except Exception as exc:
            logger.warning(f"[MAVLink] Read error: {exc}")

    def get_telemetry(self) -> dict:
        with self._lock:
            return dict(self._telem)

    def stop(self) -> None:
        self._running = False
