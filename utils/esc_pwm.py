"""
ESC PWM controller for BLDC motors.

Standard servo/ESC protocol:
  50 Hz signal (20 ms period)
  1000 µs pulse → minimum / disarmed
  1500 µs pulse → mid-point
  2000 µs pulse → maximum

Duty cycle = pulse_µs / 20000 * 100
  min  = 5.0 %
  mid  = 7.5 %
  max  = 10.0 %

Backends (tried in order):
  1. Jetson.GPIO  — hardware PWM on Jetson Orin Nano (primary target)
  2. RPi.GPIO     — Raspberry Pi
  3. SimulatedPWM — desktop / dev machine (prints duty cycle, no real output)
"""

import time
import threading
import logging

log = logging.getLogger(__name__)

PWM_FREQ_HZ  = 50          # standard ESC signal frequency
PULSE_MIN_US = 1000        # disarmed / stopped
PULSE_MAX_US = 2000        # full throttle
PULSE_MID_US = 1500        # mid / neutral
ARM_HOLD_S   = 2.0         # seconds to hold min pulse during arm sequence


def _us_to_duty(pulse_us: float) -> float:
    """Convert pulse width in µs to duty-cycle percentage (0–100)."""
    period_us = 1_000_000 / PWM_FREQ_HZ
    return pulse_us / period_us * 100.0


# ── backends ──────────────────────────────────────────────────────────────────

class _JetsonPWM:
    def __init__(self, pin: int):
        import Jetson.GPIO as GPIO
        self._GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        self._pwm = GPIO.PWM(pin, PWM_FREQ_HZ)
        self._pwm.start(_us_to_duty(PULSE_MIN_US))
        self._pin = pin

    def set_duty(self, duty: float):
        self._pwm.ChangeDutyCycle(duty)

    def stop(self):
        self._pwm.stop()
        self._GPIO.cleanup(self._pin)


class _RPiPWM:
    def __init__(self, pin: int):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        self._pwm = GPIO.PWM(pin, PWM_FREQ_HZ)
        self._pwm.start(_us_to_duty(PULSE_MIN_US))
        self._pin = pin

    def set_duty(self, duty: float):
        self._pwm.ChangeDutyCycle(duty)

    def stop(self):
        self._pwm.stop()
        self._GPIO.cleanup(self._pin)


class _SimulatedPWM:
    """Prints duty cycle changes — no real GPIO output."""
    def __init__(self, pin: int):
        log.warning("[ESC] No GPIO library found — running in simulation mode.")
        self._pin = pin
        self._duty = _us_to_duty(PULSE_MIN_US)

    def set_duty(self, duty: float):
        if abs(duty - self._duty) > 0.05:
            pulse_us = duty / 100.0 * (1_000_000 / PWM_FREQ_HZ)
            throttle = (pulse_us - PULSE_MIN_US) / (PULSE_MAX_US - PULSE_MIN_US)
            log.info(f"[ESC-SIM] pin={self._pin}  duty={duty:.2f}%  "
                     f"pulse={pulse_us:.0f}µs  throttle={throttle:.2%}")
        self._duty = duty

    def stop(self):
        log.info(f"[ESC-SIM] pin={self._pin} stopped")


def _make_backend(pin: int):
    for cls, lib in [(_JetsonPWM, "Jetson.GPIO"), (_RPiPWM, "RPi.GPIO")]:
        try:
            __import__(lib.replace(".", "_") if "." in lib else lib)
            return cls(pin)
        except (ImportError, RuntimeError):
            pass
    return _SimulatedPWM(pin)


# ── public ESC controller ─────────────────────────────────────────────────────

class ESCController:
    """
    Thread-safe ESC controller.

    Usage
    -----
    esc = ESCController(pin=32)
    esc.arm()                    # send min pulse for ARM_HOLD_S seconds
    esc.set_throttle(0.0)        # 0.0 = min, 1.0 = max
    esc.ramp_to(0.6, duration=1.5)  # smooth ramp
    esc.stop()                   # cut output and cleanup GPIO
    """

    def __init__(self, pin: int = 32):
        """
        Parameters
        ----------
        pin : int
            BOARD-mode GPIO pin connected to the ESC signal wire.
            Jetson Orin Nano hardware PWM pins: 32 (PWM0) or 33 (PWM2).
        """
        self._backend = _make_backend(pin)
        self._lock = threading.Lock()
        self._throttle = 0.0
        self._armed = False
        log.info(f"[ESC] init on pin {pin}")

    # ── internal ──────────────────────────────────────────────────────────

    def _set_pulse(self, pulse_us: float):
        pulse_us = max(PULSE_MIN_US, min(PULSE_MAX_US, pulse_us))
        with self._lock:
            self._backend.set_duty(_us_to_duty(pulse_us))

    # ── public API ────────────────────────────────────────────────────────

    def arm(self):
        """
        Send minimum pulse for ARM_HOLD_S seconds — required by most ESCs
        before they accept throttle commands.
        """
        log.info("[ESC] arming …")
        self._set_pulse(PULSE_MIN_US)
        time.sleep(ARM_HOLD_S)
        self._armed = True
        log.info("[ESC] armed")

    def set_throttle(self, throttle: float):
        """Set throttle 0.0 (stopped) → 1.0 (full)."""
        throttle = max(0.0, min(1.0, throttle))
        pulse_us = PULSE_MIN_US + throttle * (PULSE_MAX_US - PULSE_MIN_US)
        self._set_pulse(pulse_us)
        self._throttle = throttle

    def ramp_to(self, target: float, duration: float = 1.0, steps: int = 50):
        """Smoothly ramp throttle to target over duration seconds."""
        start = self._throttle
        target = max(0.0, min(1.0, target))
        for i in range(steps + 1):
            t = i / steps
            self.set_throttle(start + (target - start) * t)
            time.sleep(duration / steps)

    def stop(self):
        """Cut throttle to zero and release GPIO."""
        self.set_throttle(0.0)
        time.sleep(0.1)
        with self._lock:
            self._backend.stop()
        self._armed = False
        log.info("[ESC] stopped")

    @property
    def throttle(self) -> float:
        return self._throttle
