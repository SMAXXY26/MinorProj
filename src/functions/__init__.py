"""
src/functions — All training logic for the weapon detection pipeline.
=====================================================================
Import everything from here for a clean API:

    from src.functions import train_detector, train_classifier, ...
"""

# ── Monitor ──────────────────────────────────────────────────────────────────
from src.functions.monitor import TrainingMonitor

# ── Common utilities ─────────────────────────────────────────────────────────
from src.functions.common import (
    load_config,
    save_config,
    set_seed,
    get_device,
    save_checkpoint,
    ModelEMA,
    mixup_data,
    fmt_time,
    print_banner,
    print_summary,
)

# ── Validation & Pre-flight ──────────────────────────────────────────────────
from src.functions.validation import (
    preflight_checks,
    validate_dataset,
    evaluate_model,
)

# ── Step 1: Detector ─────────────────────────────────────────────────────────
from src.functions.detector import (
    prepare_dataset_yaml,
    train_detector,
)

# ── Step 2: Classifier ───────────────────────────────────────────────────────
from src.functions.classifier import (
    generate_crops,
    evaluate_classifier,
    train_classifier,
)

# ── Step 3: Temporal Smoother ────────────────────────────────────────────────
from src.functions.temporal import (
    TemporalSmoother,
    TemporalLoss,
    TemporalDataset,
    extract_features_from_video,
    extract_features_from_folder,
    evaluate_temporal,
    train_temporal,
    demo_smoother,
)

# ── Step 4: False Positive Correction ────────────────────────────────────────
from src.functions.fp_correction import (
    mine_hard_negatives,
    find_optimal_threshold,
    finetune_detector,
    HardNegativeDataset,
    finetune_classifier_with_background,
    run_fp_correction,
)

# ── Export ────────────────────────────────────────────────────────────────────
from src.functions.export import (
    export_detector_onnx,
    export_classifier_onnx,
    export_models,
)

# ── Smoke Test ───────────────────────────────────────────────────────────────
from src.functions.smoke_test import (
    run_smoke_test,
)

# ── Student Distillation (Pi 5 AI HAT) ───────────────────────────────────────
from src.functions.distill import (
    train_student,
    export_student,
)
