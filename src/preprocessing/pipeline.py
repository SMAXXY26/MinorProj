"""
PreprocessingPipeline — chains bilateral filtering and CLAHE in one call.

Typical usage
-------------
    from src.preprocessing.pipeline import PreprocessingPipeline

    pipe = PreprocessingPipeline(
        bilateral=True, bilateral_d=9, bilateral_sigma=75,
        clahe=True, clahe_clip=2.0, clahe_tile=(8, 8),
        log_dir="logs/preprocessing",
    )

    frame_out = pipe(frame, image_id="frame_0042")

    # after a run:
    pipe.tracker.summary()
    pipe.save_log()          # writes logs/preprocessing/<run_id>.json
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .bilateral import apply_bilateral_filter
from .clahe import apply_clahe
from .tracker import PreprocessingTracker


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline: bilateral filter → CLAHE.
    Either stage can be disabled independently.
    """

    def __init__(
        self,
        # bilateral filter knobs
        bilateral: bool = True,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        # CLAHE knobs
        clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_tile: Tuple[int, int] = (8, 8),
        # tracker
        log_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.bilateral_enabled = bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        self.clahe_enabled = clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

        self.log_dir = Path(log_dir) if log_dir else Path("logs/preprocessing")
        self.tracker = PreprocessingTracker(run_id=run_id)

    # ------------------------------------------------------------------

    def __call__(
        self,
        image: np.ndarray,
        image_id: Optional[str] = None,
    ) -> np.ndarray:
        """Process one image through the enabled stages."""
        out = image

        if self.bilateral_enabled:
            out = apply_bilateral_filter(
                out,
                d=self.bilateral_d,
                sigma_color=self.bilateral_sigma_color,
                sigma_space=self.bilateral_sigma_space,
                tracker=self.tracker,
                image_id=image_id,
            )

        if self.clahe_enabled:
            out = apply_clahe(
                out,
                clip_limit=self.clahe_clip,
                tile_grid_size=self.clahe_tile,
                tracker=self.tracker,
                image_id=image_id,
            )

        return out

    def save_log(self, filename: Optional[str] = None) -> Path:
        """Write the tracker log to disk. Returns the saved path."""
        fname = filename or f"{self.tracker.run_id}.json"
        dest = self.log_dir / fname
        return self.tracker.save(dest)

    def reset(self) -> None:
        """Clear tracker state between runs (keeps pipeline config)."""
        self.tracker.reset()

    def __repr__(self) -> str:
        stages = []
        if self.bilateral_enabled:
            stages.append(
                f"Bilateral(d={self.bilateral_d}, σ_c={self.bilateral_sigma_color}, σ_s={self.bilateral_sigma_space})"
            )
        if self.clahe_enabled:
            stages.append(f"CLAHE(clip={self.clahe_clip}, tile={self.clahe_tile})")
        return f"PreprocessingPipeline([{' → '.join(stages) or 'disabled'}])"
