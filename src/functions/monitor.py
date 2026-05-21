"""
monitor.py — Real-time training monitor.
=========================================
Logs GPU utilization, VRAM, system RAM, ETA, and per-epoch
metrics to both console and a JSONL file during training.

Usage:
    mon = TrainingMonitor(log_dir="logs/runs/run_XYZ", step_name="detector")
    mon.start_epoch(epoch, total_epochs)
    # ... training ...
    mon.end_epoch(metrics_dict)
    mon.close()
"""

import time
import json
import threading
from pathlib import Path


def _gpu_stats():
    """Return (gpu_util_pct, vram_used_mb, vram_total_mb) or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        used  = torch.cuda.memory_allocated(0)  / 1e6
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        # utilization requires pynvml or subprocess
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            ).decode().strip()
            util = int(out.split("\n")[0])
        except Exception:
            util = -1
        return util, round(used), round(total)
    except Exception:
        return None


def _ram_stats():
    """Return (used_mb, total_mb) or None."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.used / 1e6), round(vm.total / 1e6)
    except Exception:
        return None


class TrainingMonitor:
    """
    Lightweight per-epoch training monitor.

    Writes each epoch's metrics to <log_dir>/<step_name>_monitor.jsonl
    and prints a one-liner summary to stdout.
    """

    def __init__(self, log_dir: str, step_name: str):
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_name = step_name
        self.log_path  = self.log_dir / f"{step_name}_monitor.jsonl"
        self._fh       = open(self.log_path, "a")
        self._epoch_start: float = 0.0
        self._run_start:   float = time.time()

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._epoch       = epoch
        self._total       = total_epochs
        self._epoch_start = time.time()

    def end_epoch(self, metrics: dict) -> None:
        elapsed   = time.time() - self._epoch_start
        run_total = time.time() - self._run_start
        remaining = (elapsed * (self._total - self._epoch - 1))

        gpu   = _gpu_stats()
        ram   = _ram_stats()

        record = {
            "step":    self.step_name,
            "epoch":   self._epoch + 1,
            "total":   self._total,
            "elapsed_s": round(elapsed, 1),
            "eta_s":   round(remaining),
            **metrics,
        }
        if gpu:
            record["gpu_util_pct"]  = gpu[0]
            record["vram_used_mb"]  = gpu[1]
            record["vram_total_mb"] = gpu[2]
        if ram:
            record["ram_used_mb"]  = ram[0]
            record["ram_total_mb"] = ram[1]

        # Write JSONL
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

        # Console one-liner
        gpu_str = (f"  GPU {gpu[0]}% {gpu[1]}/{gpu[2]}MB" if gpu and gpu[0] >= 0
                   else f"  VRAM {gpu[1]}MB" if gpu else "")
        eta_str = f"  ETA {remaining//60:.0f}m{remaining%60:.0f}s"
        metrics_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(
            f"  [{self.step_name}] "
            f"Epoch {self._epoch+1}/{self._total}  "
            f"{metrics_str}{gpu_str}{eta_str}  ({elapsed:.0f}s/ep)"
        )

    def log_event(self, msg: str) -> None:
        record = {"step": self.step_name, "event": msg,
                  "t": round(time.time() - self._run_start, 1)}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        print(f"  [{self.step_name}] {msg}")

    def close(self) -> None:
        self._fh.close()
