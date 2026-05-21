"""
PreprocessingTracker — lightweight change-log for preprocessing operations.

Every call to apply_bilateral_filter / apply_clahe (when a tracker is passed)
appends one entry here. The log can be:
  - inspected in memory  : tracker.entries
  - saved to JSON        : tracker.save("run_log.json")
  - printed as a table   : tracker.summary()
  - reset between runs   : tracker.reset()
"""

import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PreprocessingTracker:
    """Records preprocessing operations applied during a run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id: str = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict[str, Any]] = []
        self._op_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        operation: str,
        image_id: Optional[str],
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Append one preprocessing event to the log."""
        self._op_counts[operation] = self._op_counts.get(operation, 0) + 1
        entry: Dict[str, Any] = {
            "seq": len(self.entries) + 1,
            "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "operation": operation,
            "image_id": image_id,
            "params": params,
            "metrics": metrics,
        }
        self.entries.append(entry)

    def reset(self) -> None:
        """Clear all recorded entries (keeps run_id)."""
        self.entries.clear()
        self._op_counts.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary table."""
        if not self.entries:
            return "PreprocessingTracker: no operations recorded."

        lines = [
            f"Run ID : {self.run_id}",
            f"Total  : {len(self.entries)} operations",
            "",
            f"{'#':<5} {'Operation':<22} {'Image ID':<30} {'Key metric'}",
            "-" * 80,
        ]
        for e in self.entries:
            op = e["operation"]
            img = str(e["image_id"] or "—")[:29]
            # pick the most informative scalar metric per operation type
            m = e["metrics"]
            if op == "bilateral_filter":
                metric_str = f"mean Δ = {m.get('mean_delta', 'n/a')}"
            elif op == "clahe":
                metric_str = f"contrast gain = {m.get('contrast_gain', 'n/a')}"
            else:
                metric_str = str(m)
            lines.append(f"{e['seq']:<5} {op:<22} {img:<30} {metric_str}")

        lines += ["", "Op counts: " + ", ".join(f"{k}={v}" for k, v in self._op_counts.items())]
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Serialise the log to a JSON file. Returns the resolved path."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "total_ops": len(self.entries),
            "op_counts": self._op_counts,
            "entries": self.entries,
        }
        dest.write_text(json.dumps(payload, indent=2))
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingTracker":
        """Reconstruct a tracker from a previously saved JSON file."""
        data = json.loads(Path(path).read_text())
        tracker = cls(run_id=data.get("run_id"))
        tracker.entries = data.get("entries", [])
        tracker._op_counts = data.get("op_counts", {})
        return tracker

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self._op_counts.items()) or "empty"
        return f"PreprocessingTracker(run_id={self.run_id!r}, ops=[{counts}])"
