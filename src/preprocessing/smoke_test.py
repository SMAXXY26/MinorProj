"""
Quick smoke test for the preprocessing module.
Run from the project root:
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate ml_env
    python -m src.preprocessing.smoke_test
"""

import sys
import numpy as np

# --- bilateral filter ---
from src.preprocessing.bilateral import apply_bilateral_filter
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.tracker import PreprocessingTracker
from src.preprocessing.pipeline import PreprocessingPipeline


def _make_noisy_bgr(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = np.full((h, w, 3), 80, dtype=np.uint8)
    noise = rng.integers(-30, 30, size=(h, w, 3), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_dark_bgr(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 40, size=(h, w, 3), dtype=np.uint8)


def test_bilateral():
    img = _make_noisy_bgr()
    tracker = PreprocessingTracker(run_id="test_bilateral")
    out = apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75,
                                  tracker=tracker, image_id="noisy_frame")
    assert out.shape == img.shape, "shape mismatch"
    assert out.dtype == img.dtype, "dtype mismatch"
    assert len(tracker) == 1
    print("[PASS] bilateral filter")
    print(tracker.summary())


def test_clahe():
    img = _make_dark_bgr()
    tracker = PreprocessingTracker(run_id="test_clahe")
    out = apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8),
                      tracker=tracker, image_id="dark_frame")
    assert out.shape == img.shape, "shape mismatch"
    assert out.dtype == img.dtype, "dtype mismatch"
    assert len(tracker) == 1
    # CLAHE should lift contrast
    def rms(x):
        import cv2
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return float(np.std(g))
    assert rms(out) >= rms(img), "CLAHE did not increase contrast"
    print("[PASS] CLAHE")
    print(tracker.summary())


def test_pipeline():
    img = _make_dark_bgr()
    pipe = PreprocessingPipeline(bilateral=True, clahe=True, log_dir="/tmp/test_preproc")
    out = pipe(img, image_id="combined_frame")
    assert out.shape == img.shape
    assert len(pipe.tracker) == 2
    saved = pipe.save_log()
    assert saved.exists(), f"log not written to {saved}"
    print(f"[PASS] pipeline  — log saved to {saved}")
    print(pipe.tracker.summary())


if __name__ == "__main__":
    test_bilateral()
    print()
    test_clahe()
    print()
    test_pipeline()
    print("\nAll smoke tests passed.")
    sys.exit(0)
