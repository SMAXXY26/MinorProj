"""
servo_check.py — Test servo movement before running the full tracker.
Sweeps the servo left, center, right, then lets you manually set angles.
"""
import time
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

# --- Config ---
PAN_PIN  = 12
PAN_MIN  = -70
PAN_MAX  =  70

factory = LGPIOFactory()
pan = AngularServo(PAN_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
                   min_pulse_width=0.0005, max_pulse_width=0.0025,
                   pin_factory=factory)

def move(angle, label):
    print(f"  {label:10s} → {angle}°")
    pan.angle = angle
    time.sleep(1)

try:
    print("\n--- Servo Sweep Test ---")
    move(0,       "center")
    move(PAN_MAX, "right")
    move(0,       "center")
    move(PAN_MIN, "left")
    move(0,       "center")

    print("\n--- Manual Control ---")
    print("Enter angle (-70 to 70), or 'q' to quit\n")

    while True:
        val = input("Angle: ").strip()
        if val.lower() == 'q':
            break
        try:
            angle = float(val)
            angle = max(PAN_MIN, min(PAN_MAX, angle))
            pan.angle = angle
            print(f"  Moved to {angle}°")
        except ValueError:
            print("  Invalid — enter a number")

finally:
    print("\nCentering and closing...")
    pan.angle = 0
    time.sleep(0.5)
    pan.close()
    print("Done.")
