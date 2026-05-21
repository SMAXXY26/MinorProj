"""
train_all.py — Industry-Ready End-to-End Training Pipeline
============================================================
Full weapon detection pipeline with production-grade features:

  Step 0: Pre-flight checks         (environment, GPU, disk, deps)
  Step 1: Data validation            (integrity, balance, corruption)
  Step 2: Train YOLOv8s detector     (Ultralytics, 640px, aerial aug)
  Step 3: Generate crops + Classifier (EfficientNet-B5)
  Step 4: Temporal smoother          (BiLSTM — optional, needs videos)
  Step 5: False positive correction  (hard neg mining + fine-tune)
  Step 6: Model evaluation           (full val metrics, quality gates)
  Step 7: Model export               (ONNX for deployment)
  Step 8: Smoke test                 (end-to-end inference verify)
  Step 9: Student distillation       (YOLOv8n 320px for Pi 5 AI HAT)

All model code, math, training loops, and utilities live in src/functions/.
This file contains ONLY imports and orchestration.

Usage:
    python train_all.py                              # Steps 0-3, 5-8
    python train_all.py --all                        # All steps 0-9
    python train_all.py --sequences data/sequences/  # Include temporal
    python train_all.py --skip 2 3                   # Skip detector+classifier
    python train_all.py --only 6 7 8                 # Only eval+export+test
    python train_all.py --only 9                     # Only Pi 5 distillation
    python train_all.py --no-export                  # Skip ONNX export
    python train_all.py --no-smoke                   # Skip smoke test
    python train_all.py --fail-fast                  # Abort on any failure

Output:
    logs/detector/best.pt                    — YOLOv8s detector (640px, aerial)
    logs/classifier/best.pt                  — EfficientNet-B5 classifier
    logs/temporal/best.pt                    — BiLSTM temporal smoother
    logs/fp_correction/detector_ft_best.pt   — Fine-tuned detector
    logs/fp_correction/classifier_bg_best.pt — Fine-tuned classifier (bg class)
    logs/student/best.pt                     — YOLOv8n 320px for Pi 5 AI HAT
    logs/export/student_detector.onnx        — ONNX for Hailo HEF pipeline
    logs/export/student_detector_ncnn/       — NCNN for Pi 5 CPU fallback
    logs/evaluation/metrics.json             — Full evaluation metrics
    logs/evaluation/smoke_test.json          — Smoke test results
    logs/export/detector.onnx                — ONNX teacher detector
    logs/export/classifier.onnx              — ONNX classifier
    logs/runs/<run_id>/config_snapshot.yaml  — Config at time of training
    logs/runs/<run_id>/run_report.json       — Full run report
    config/hyperparams.yaml                  — Updated conf threshold
"""

import os
import sys
import time
import json
import argparse
import hashlib
import platform
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  Ensure project root is on path
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════
#  Import ALL logic from src/functions/
# ═══════════════════════════════════════════════════════════════════════
from src.functions.common import (
    load_config,
    set_seed,
    get_device,
    fmt_time,
    print_banner,
    print_summary,
)
from src.functions.validation import (
    preflight_checks,
    validate_dataset,
    evaluate_model,
)
from src.functions.detector import (
    prepare_dataset_yaml,
    train_detector,
)
from src.functions.classifier import (
    generate_crops,
    train_classifier,
)
from src.functions.temporal import (
    extract_features_from_folder,
    train_temporal,
)
from src.functions.fp_correction import (
    run_fp_correction,
)
from src.functions.export import (
    export_models,
)
from src.functions.smoke_test import (
    run_smoke_test,
)
from src.functions.distill import (
    train_student,
    export_student,
)


# ═══════════════════════════════════════════════════════════════════════
#  Run metadata — experiment tracking
# ═══════════════════════════════════════════════════════════════════════

def generate_run_id():
    """Generate a unique run ID based on timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


def save_run_metadata(run_id, cfg, config_path):
    """Save config snapshot and environment info at start of run."""
    import yaml

    run_dir = Path(cfg["logging"]["save_dir"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Environment info
    env_info = {
        "run_id":       run_id,
        "timestamp":    datetime.now().isoformat(),
        "python":       platform.python_version(),
        "platform":     platform.platform(),
        "hostname":     platform.node(),
        "config_path":  config_path,
    }

    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["gpu"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
    except Exception:
        pass

    try:
        import ultralytics
        env_info["ultralytics_version"] = ultralytics.__version__
    except Exception:
        pass

    # Git hash
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env_info["git_hash"] = git_hash[:12]
    except Exception:
        env_info["git_hash"] = "unknown"

    with open(run_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    return run_dir


def save_run_report(run_dir, timings, step_names, step_results, cfg):
    """Save final run report with all results."""
    report = {
        "timestamp":    datetime.now().isoformat(),
        "timings":      {
            step_names[s]: {
                "seconds": round(t, 1),
                "display": fmt_time(t),
            } for s, t in timings.items()
        },
        "total_time":   fmt_time(sum(timings.values())),
        "steps_run":    list(timings.keys()),
        "results":      step_results,
        "output_files": {},
    }

    # Check all output files
    outputs = [
        ("logs/detector/best.pt",                    "detector"),
        ("logs/classifier/best.pt",                  "classifier"),
        ("logs/temporal/best.pt",                    "temporal_smoother"),
        ("logs/fp_correction/detector_ft_best.pt",   "ft_detector"),
        ("logs/fp_correction/classifier_bg_best.pt", "ft_classifier"),
        ("logs/export/detector.onnx",                "detector_onnx"),
        ("logs/export/classifier.onnx",              "classifier_onnx"),
        ("logs/evaluation/metrics.json",             "evaluation"),
        ("logs/evaluation/smoke_test.json",          "smoke_test"),
    ]
    for path, name in outputs:
        report["output_files"][name] = {
            "path":   path,
            "exists": Path(path).exists(),
            "size_mb": round(Path(path).stat().st_size / 1e6, 2)
                       if Path(path).exists() else None,
        }

    with open(run_dir / "run_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Run report → {run_dir / 'run_report.json'}")
    return report


# ═══════════════════════════════════════════════════════════════════════
#  Step 0 — Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════════

def run_step0(cfg, config_path, fail_fast=False):
    print_banner(0, "Pre-flight Checks")
    t0 = time.time()

    passed = preflight_checks(cfg)

    elapsed = time.time() - t0
    status = "PASSED" if passed else "FAILED"
    print_summary(0, "Pre-flight Checks", elapsed, status)

    if not passed and fail_fast:
        print("  ✗ ABORTING: Pre-flight checks failed (--fail-fast)")
        sys.exit(1)

    return elapsed, {"passed": passed}


# ═══════════════════════════════════════════════════════════════════════
#  Step 1 — Data Validation
# ═══════════════════════════════════════════════════════════════════════

def run_step1(cfg, config_path, fail_fast=False):
    print_banner(1, "Data Validation")
    t0 = time.time()

    report = validate_dataset(cfg)

    elapsed = time.time() - t0
    status = "PASSED" if report["valid"] else "ISSUES FOUND"
    print_summary(1, "Data Validation", elapsed, status)

    if not report["valid"] and fail_fast:
        print("  ✗ ABORTING: Dataset validation failed (--fail-fast)")
        sys.exit(1)

    return elapsed, report


# ═══════════════════════════════════════════════════════════════════════
#  Step 2 — Train YOLOv8n Detector
# ═══════════════════════════════════════════════════════════════════════

def run_step2(cfg, config_path):
    print_banner(2, "Train YOLOv8n Detector")
    t0 = time.time()

    train_detector(cfg, resume=False)

    elapsed = time.time() - t0
    print_summary(2, "Detector Training", elapsed)

    det_path = Path("logs/detector/best.pt")
    if det_path.exists():
        size = det_path.stat().st_size / 1e6
        print(f"  ✓ Detector weights: {det_path} ({size:.1f} MB)")
    else:
        print(f"  ✗ WARNING: Detector weights not found at {det_path}")

    return elapsed, {"weights": str(det_path), "exists": det_path.exists()}


# ═══════════════════════════════════════════════════════════════════════
#  Step 3 — Generate Crops + Train Classifier
# ═══════════════════════════════════════════════════════════════════════

def run_step3(cfg, config_path):
    print_banner(3, "Crop Generation + Classifier Training")
    t0 = time.time()

    set_seed(cfg.get("project", {}).get("seed", 42))
    device = get_device(cfg)

    # Step 3A: Generate crops
    det_ckpt = "logs/detector/best.pt"
    if not Path(det_ckpt).exists():
        print("  [ERROR] Detector weights not found — run step 2 first")
        return 0, {"error": "no_detector_weights"}

    generate_crops(cfg, det_ckpt)

    # Step 3B: Train classifier
    train_classifier(cfg, device)

    elapsed = time.time() - t0
    print_summary(3, "Classifier Training", elapsed)

    cls_path = Path("logs/classifier/best.pt")
    if cls_path.exists():
        size = cls_path.stat().st_size / 1e6
        print(f"  ✓ Classifier weights: {cls_path} ({size:.1f} MB)")
    else:
        print(f"  ✗ WARNING: Classifier weights not found at {cls_path}")

    return elapsed, {"weights": str(cls_path), "exists": cls_path.exists()}


# ═══════════════════════════════════════════════════════════════════════
#  Step 4 — Temporal Smoother Training (Optional)
# ═══════════════════════════════════════════════════════════════════════

def run_step4(cfg, config_path, sequences_dir):
    print_banner(4, "Temporal Smoother Training")
    t0 = time.time()

    if not sequences_dir or not Path(sequences_dir).exists():
        print("  [SKIP] No sequences directory provided or found.")
        print("  To train the temporal smoother, run:")
        print("    python scripts/make_sequences.py   # generate sequences")
        print("    python train_all.py --sequences data/sequences/")
        print_summary(4, "Temporal Smoother", 0, "SKIPPED")
        return 0, {"skipped": True}

    from ultralytics import YOLO

    set_seed(cfg.get("project", {}).get("seed", 42))
    device = get_device(cfg)

    det_ckpt = "logs/detector/best.pt"
    if not Path(det_ckpt).exists():
        print("  [ERROR] Detector weights not found — run step 2 first")
        return 0, {"error": "no_detector_weights"}

    detector = YOLO(det_ckpt)

    # Extract features
    print(f"\n  Extracting features from: {sequences_dir}")
    sequences = extract_features_from_folder(
        sequences_dir, detector,
        cfg["dataset"]["class_names"], cfg,
    )

    if not sequences:
        print(f"  [WARN] No sequences found in {sequences_dir}")
        print_summary(4, "Temporal Smoother", 0, "NO DATA")
        return 0, {"skipped": True, "reason": "no_sequences"}

    train_temporal(cfg, sequences, device, resume=False)

    elapsed = time.time() - t0
    print_summary(4, "Temporal Smoother", elapsed)

    tmp_path = Path("logs/temporal/best.pt")
    if tmp_path.exists():
        print(f"  ✓ Temporal weights: {tmp_path}")

    return elapsed, {"weights": str(tmp_path), "exists": tmp_path.exists()}


# ═══════════════════════════════════════════════════════════════════════
#  Step 5 — False Positive Correction
# ═══════════════════════════════════════════════════════════════════════

def run_step5(cfg, config_path):
    print_banner(5, "False Positive Correction")
    t0 = time.time()

    set_seed(cfg.get("project", {}).get("seed", 42))

    # Check prerequisites
    for path, name in [
        ("logs/detector/best.pt",   "Detector (step 2)"),
        ("logs/classifier/best.pt", "Classifier (step 3)"),
    ]:
        if not Path(path).exists():
            print(f"  [ERROR] {name} not found at {path}")
            return 0, {"error": f"missing_{name}"}

    # Check negatives exist
    neg_dir = cfg.get("fp_correction", {}).get(
        "negative_image_dir", "data/negatives/images"
    )
    if not Path(neg_dir).exists() or len(list(Path(neg_dir).iterdir())) == 0:
        print(f"  [WARN] No negative images found in {neg_dir}")
        print(f"  FP correction needs negative images to mine hard negatives.")
        print(f"  Run: python download_negatives.py")
        print_summary(5, "FP Correction", 0, "NO NEGATIVES")
        return 0, {"skipped": True, "reason": "no_negatives"}

    run_fp_correction(cfg, config_path, skip_mining=False)

    elapsed = time.time() - t0
    print_summary(5, "FP Correction", elapsed)
    return elapsed, {"completed": True}


# ═══════════════════════════════════════════════════════════════════════
#  Step 6 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_step6(cfg, config_path):
    print_banner(6, "Model Evaluation")
    t0 = time.time()

    metrics = evaluate_model(cfg)

    elapsed = time.time() - t0
    passed = metrics.get("quality_gates_passed", False)
    status = "PASSED" if passed else "GATES FAILED"
    print_summary(6, "Model Evaluation", elapsed, status)
    return elapsed, metrics


# ═══════════════════════════════════════════════════════════════════════
#  Step 7 — Model Export (ONNX)
# ═══════════════════════════════════════════════════════════════════════

def run_step7(cfg, config_path):
    print_banner(7, "Model Export (ONNX)")
    t0 = time.time()

    exports = export_models(cfg)

    elapsed = time.time() - t0
    n_exported = len(exports)
    print_summary(7, "Model Export", elapsed, f"{n_exported} models")
    return elapsed, exports


# ═══════════════════════════════════════════════════════════════════════
#  Step 8 — Smoke Test
# ═══════════════════════════════════════════════════════════════════════

def run_step8(cfg, config_path):
    print_banner(8, "Smoke Test")
    t0 = time.time()

    device = "cuda" if cfg["hardware"]["device"] == "cuda" else "cpu"
    results = run_smoke_test(cfg, device)

    elapsed = time.time() - t0
    status = "PASSED" if results["passed"] else "FAILED"
    print_summary(8, "Smoke Test", elapsed, status)
    return elapsed, results


# ═══════════════════════════════════════════════════════════════════════
#  Step 9 — Student Distillation (Pi 5 AI HAT — Hailo-8L)
# ═══════════════════════════════════════════════════════════════════════

def run_step9(cfg, config_path):
    st_cfg   = cfg.get("student", {})
    img_size = st_cfg.get("img_size", 416)
    print_banner(9, f"Student Distillation — YOLOv8n {img_size}px (multi-teacher KD)")
    t0 = time.time()

    # Config override: `student.teachers: [path1, path2, ...]`
    # Else: auto-discover all existing teacher checkpoints down the chain
    cfg_teachers = st_cfg.get("teachers", None)
    if cfg_teachers:
        teacher_candidates = list(cfg_teachers)
        print(f"  Using {len(teacher_candidates)} teacher(s) from config:")
    else:
        teacher_candidates = [
            "runs/detect/logs/fp_correction/weights/best.pt",
            "logs/fp_correction/detector_ft_best.pt",
            "logs/detector/best.pt",
        ]
        print(f"  Auto-discovering teachers from candidate list...")

    existing_teachers = [p for p in teacher_candidates if Path(p).exists()]
    missing_teachers  = [p for p in teacher_candidates if not Path(p).exists()]

    if not existing_teachers:
        print("  [ERROR] No teacher weights found — run steps 2 or 5 first")
        return 0, {"error": "no_teacher_weights"}

    print(f"  Teachers found ({len(existing_teachers)}):")
    for i, tp in enumerate(existing_teachers):
        size = Path(tp).stat().st_size / 1e6
        print(f"    [{i}] {tp}  ({size:.1f} MB)")
    for tp in missing_teachers:
        print(f"    [skip] {tp}  (not found)")

    # Train student with the teacher ensemble
    student_ckpt = train_student(cfg, existing_teachers)

    # Export to NCNN + ONNX
    if student_ckpt and Path(student_ckpt).exists():
        exports = export_student(cfg, student_ckpt)
    else:
        exports = {}

    elapsed = time.time() - t0
    n = len(exports)
    print_summary(9, "Student Distillation", elapsed, f"{n} formats exported")

    student_path = Path("logs/student/best.pt")
    if student_path.exists():
        size = student_path.stat().st_size / 1e6
        print(f"  ✓ Student weights : {student_path} ({size:.1f} MB)")
    for name, path in exports.items():
        print(f"  ✓ {name:15s} : {path}")

    return elapsed, {
        "weights":  str(student_path),
        "exports":  exports,
        "teachers": existing_teachers,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Industry-ready weapon detection training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py                              # Full pipeline (0-3, 5-8)
  python train_all.py --all                        # All 9 steps
  python train_all.py --sequences data/sequences/  # Include temporal
  python train_all.py --skip 2 3                   # Skip training (reuse)
  python train_all.py --only 6 7 8                 # Only eval+export+test
  python train_all.py --fail-fast                  # Abort on any failure
  python train_all.py --no-export --no-smoke       # Skip post-training steps
        """,
    )
    parser.add_argument(
        "--config", default="config/hyperparams.yaml",
        help="Path to hyperparams config",
    )
    parser.add_argument(
        "--sequences", default=None,
        help="Path to video sequences dir for temporal training (step 4)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 9 steps (requires --sequences for step 4)",
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, default=[],
        help="Steps to skip, e.g. --skip 2 3",
    )
    parser.add_argument(
        "--only", nargs="+", type=int, default=[],
        help="Run ONLY these steps, e.g. --only 6 7 8",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Abort pipeline if any validation step fails",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip ONNX model export (step 7)",
    )
    parser.add_argument(
        "--no-smoke", action="store_true",
        help="Skip smoke test (step 8)",
    )

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # ── Generate run ID ───────────────────────────────────────────────
    run_id = generate_run_id()

    # ── Determine which steps to run ──────────────────────────────────
    # Default: 0,1,2,3,5,6,7,8,9 (skip temporal=4)
    all_steps = list(range(10)) if args.all else [0, 1, 2, 3, 5, 6, 7, 8, 9]
    if args.sequences:
        all_steps = list(range(10))

    if args.no_export:
        all_steps = [s for s in all_steps if s != 7]
    if args.no_smoke:
        all_steps = [s for s in all_steps if s != 8]

    if args.only:
        steps_to_run = sorted(args.only)
    else:
        steps_to_run = [s for s in all_steps if s not in args.skip]

    # ── Step names ────────────────────────────────────────────────────
    step_names = {
        0: "Pre-flight Checks",
        1: "Data Validation",
        2: "Train YOLOv8s Detector (640px)",
        3: "Crop + Classifier Training",
        4: "Temporal Smoother (BiLSTM)",
        5: "False Positive Correction",
        6: "Model Evaluation",
        7: "Model Export (ONNX)",
        8: "Smoke Test",
        9: "Student Distil — Pi 5 AI HAT (320px)",
    }

    # ── Print plan ────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  🔫 Weapon Detection — Industry-Ready Training Pipeline")
    print(f"{'═'*70}")
    print(f"\n  Run ID   : {run_id}")
    print(f"  Config   : {args.config}")
    print(f"  Steps    : {steps_to_run}")
    print(f"  Fail-fast: {'ON' if args.fail_fast else 'OFF'}")
    if args.sequences:
        print(f"  Sequences: {args.sequences}")
    print(f"\n  Pipeline:")
    for s in range(10):
        if s in steps_to_run:
            marker = "→"
        else:
            marker = "  skip"
        phase = ""
        if s <= 1:
            phase = "[pre-train]"
        elif s <= 5:
            phase = "[training] "
        else:
            phase = "[post-train]"
        print(f"    {marker} Step {s}: {step_names[s]:35s} {phase}")
    print(f"\n{'═'*70}")

    # ── Save run metadata ─────────────────────────────────────────────
    run_dir = save_run_metadata(run_id, cfg, args.config)
    print(f"\n  Run directory → {run_dir}")

    # ── Dataset stats ─────────────────────────────────────────────────
    train_lbl = Path(cfg["dataset"]["root"]) / "labels" / "train"
    if train_lbl.exists():
        class_counts = {}
        for name in cfg["dataset"]["class_names"]:
            class_counts[name] = 0
        for f in train_lbl.iterdir():
            if f.suffix != ".txt":
                continue
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        ci = int(parts[0])
                        if ci < len(cfg["dataset"]["class_names"]):
                            class_counts[cfg["dataset"]["class_names"][ci]] += 1
        print(f"\n  Dataset statistics:")
        for name, count in class_counts.items():
            print(f"    {name:10s}: {count:,} annotations")
        try:
            total_imgs = len(list(
                (Path(cfg["dataset"]["root"]) / "images" / "train").iterdir()
            ))
            print(f"    {'total':10s}: {total_imgs:,} images")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  Execute pipeline
    # ══════════════════════════════════════════════════════════════════
    timings      = {}
    step_results = {}
    pipeline_start = time.time()

    # ── Pre-training steps ────────────────────────────────────────────
    if 0 in steps_to_run:
        t, r = run_step0(cfg, args.config, args.fail_fast)
        timings[0] = t
        step_results[0] = r

    if 1 in steps_to_run:
        t, r = run_step1(cfg, args.config, args.fail_fast)
        timings[1] = t
        step_results[1] = r

    # ── Training steps ────────────────────────────────────────────────
    if 2 in steps_to_run:
        t, r = run_step2(cfg, args.config)
        timings[2] = t
        step_results[2] = r

    if 3 in steps_to_run:
        t, r = run_step3(cfg, args.config)
        timings[3] = t
        step_results[3] = r

    if 4 in steps_to_run:
        seq_dir = args.sequences or "data/sequences"
        t, r = run_step4(cfg, args.config, seq_dir)
        timings[4] = t
        step_results[4] = r

    if 5 in steps_to_run:
        # Reload config in case previous steps modified it
        cfg = load_config(args.config)
        t, r = run_step5(cfg, args.config)
        timings[5] = t
        step_results[5] = r

    # ── Post-training steps ───────────────────────────────────────────
    if 6 in steps_to_run:
        cfg = load_config(args.config)
        t, r = run_step6(cfg, args.config)
        timings[6] = t
        step_results[6] = r

    if 7 in steps_to_run:
        t, r = run_step7(cfg, args.config)
        timings[7] = t
        step_results[7] = r

    if 8 in steps_to_run:
        t, r = run_step8(cfg, args.config)
        timings[8] = t
        step_results[8] = r

    if 9 in steps_to_run:
        cfg = load_config(args.config)
        t, r = run_step9(cfg, args.config)
        timings[9] = t
        step_results[9] = r

    total_time = time.time() - pipeline_start

    # ══════════════════════════════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  🏁 Pipeline Complete!")
    print(f"{'═'*70}")

    # Timing table
    print(f"\n  Timing:")
    for s in range(10):
        if s in timings:
            t = timings[s]
            status = "✓" if t > 0 else "skipped"
            print(f"    Step {s} ({step_names[s]:35s}): "
                  f"{fmt_time(t):>8s}  {status}")
    print(f"    {'─'*55}")
    print(f"    {'Total':41s}: {fmt_time(total_time):>8s}")

    # Output files
    print(f"\n  Output files:")
    outputs = [
        ("logs/detector/best.pt",                    "Detector"),
        ("logs/classifier/best.pt",                  "Classifier"),
        ("logs/temporal/best.pt",                    "Temporal Smoother"),
        ("logs/fp_correction/detector_ft_best.pt",   "FT Detector"),
        ("logs/fp_correction/classifier_bg_best.pt", "FT Classifier"),
        ("logs/export/detector.onnx",                "Detector ONNX"),
        ("logs/export/classifier.onnx",              "Classifier ONNX"),
        ("logs/evaluation/metrics.json",             "Eval Metrics"),
        ("logs/evaluation/smoke_test.json",          "Smoke Test"),
    ]
    for path, name in outputs:
        p = Path(path)
        if p.exists():
            size = p.stat().st_size / 1e6
            print(f"    ✓ {name:20s}: {path} ({size:.1f} MB)")
        else:
            print(f"    ✗ {name:20s}: {path}")

    # Quality summary
    eval_metrics = step_results.get(6, {})
    if eval_metrics and "mAP50" in eval_metrics:
        print(f"\n  Model quality:")
        print(f"    mAP@50     : {eval_metrics['mAP50']:.4f}")
        print(f"    mAP@50-95  : {eval_metrics.get('mAP50_95', 0):.4f}")
        print(f"    Precision  : {eval_metrics.get('precision', 0):.4f}")
        print(f"    Recall     : {eval_metrics.get('recall', 0):.4f}")
        gates = "✓ PASSED" if eval_metrics.get("quality_gates_passed") else "✗ FAILED"
        print(f"    Gates      : {gates}")

    smoke = step_results.get(8, {})
    if smoke:
        det_info = smoke.get("detector", {})
        if "fps" in det_info:
            print(f"\n  Inference speed:")
            print(f"    FPS        : {det_info['fps']:.1f}")
            print(f"    Latency    : {det_info.get('avg_latency_ms', 0):.1f} ms avg")

    # Save run report
    save_run_report(run_dir, timings, step_names, step_results, cfg)

    # Reload latest config
    cfg = load_config(args.config)
    det_weights = "logs/fp_correction/detector_ft_best.pt"
    if not Path(det_weights).exists():
        det_weights = "logs/detector/best.pt"
    conf = cfg["detector"].get("conf_threshold", 0.15)

    print(f"\n  Run ID   : {run_id}")
    print(f"\n  Test with:")
    print(f"    python detection.py --source <video>")
    print(f"    python check_video.py --source <video> --gates 123")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
