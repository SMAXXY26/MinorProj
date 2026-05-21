#!/bin/bash
# =============================================================================
#  run_pipeline.sh  —  WeaponDetection v2  Full Pipeline Runner
# =============================================================================
#
#  USAGE
#  ─────
#  Full training pipeline (Steps 1-4):
#      bash run_pipeline.sh
#
#  Include temporal smoother (Step 3 — needs video clips in data/sequences/):
#      bash run_pipeline.sh --temporal
#
#  Skip already-done steps:
#      bash run_pipeline.sh --skip-step1       # reuse existing detector
#      bash run_pipeline.sh --skip-step1 --skip-step2
#
#  Only run inference (models already trained):
#      bash run_pipeline.sh --infer-only --source test_video.mp4
#      bash run_pipeline.sh --infer-only --source 0          # webcam/CSI
#      bash run_pipeline.sh --infer-only --source 0 --trt    # Jetson TRT mode
#
#  Export to TensorRT INT8 (Jetson only):
#      bash run_pipeline.sh --export-trt
#
#  Full Jetson workflow (train → trt export → deploy):
#      bash run_pipeline.sh --temporal --export-trt --deploy --source 0
#
# =============================================================================

set -e   # abort on any error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner()  { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════════════${NC}"; \
            echo -e "${CYAN}${BOLD}  $1${NC}"; \
            echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════${NC}\n"; }
ok()      { echo -e "  ${GREEN}✓${NC}  $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC}  $1"; }
err()     { echo -e "  ${RED}✗${NC}  $1"; }
info()    { echo -e "  ${CYAN}→${NC}  $1"; }
step()    { echo -e "\n${BOLD}[ STEP $1 ]${NC}  $2\n"; }

# ── Defaults ─────────────────────────────────────────────────────────────────
CONFIG="config/hyperparams.yaml"
PYTHON="python3"

DO_STEP1=true
DO_STEP2=true
DO_STEP3=false
DO_STEP4=true
DO_TRT=false
DO_INFER=false
DO_DEPLOY=false

SOURCE="0"
TRT_FLAG=""
SEQUENCES_DIR="data/sequences"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --temporal)       DO_STEP3=true; shift ;;
        --skip-step1)     DO_STEP1=false; shift ;;
        --skip-step2)     DO_STEP2=false; shift ;;
        --skip-step3)     DO_STEP3=false; shift ;;
        --skip-step4)     DO_STEP4=false; shift ;;
        --export-trt)     DO_TRT=true; shift ;;
        --infer-only)     DO_STEP1=false; DO_STEP2=false; DO_STEP3=false
                          DO_STEP4=false; DO_INFER=true; shift ;;
        --deploy)         DO_INFER=true; shift ;;
        --source)         SOURCE="$2"; shift 2 ;;
        --trt)            TRT_FLAG="--trt"; shift ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --sequences)      SEQUENCES_DIR="$2"; DO_STEP3=true; shift 2 ;;
        -h|--help)
            head -40 "$0" | grep "^#  " | sed 's/^#  //'
            exit 0 ;;
        *)  echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Sanity: check Python ──────────────────────────────────────────────────────
banner "WeaponDetection v2 — Pipeline Runner"
info "Python : $($PYTHON --version 2>&1)"
info "Config : $CONFIG"
info "WorkDir: $(pwd)"

# ── Check dataset exists ──────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    err "Config not found: $CONFIG"; exit 1
fi

TRAIN_IMGS="data/images/train"
if [[ ! -d "$TRAIN_IMGS" ]]; then
    err "Training data not found at $TRAIN_IMGS"
    err "Add images to data/images/train/ and labels to data/labels/train/"
    exit 1
fi
IMG_COUNT=$(ls "$TRAIN_IMGS" 2>/dev/null | wc -l)
ok "Training images: $IMG_COUNT"

# ── Check for negatives (needed for step 4) ───────────────────────────────────
NEG_DIR="data/negatives/images"
NEG_COUNT=0
if [[ -d "$NEG_DIR" ]]; then
    NEG_COUNT=$(ls "$NEG_DIR" 2>/dev/null | wc -l)
fi
if [[ $NEG_COUNT -eq 0 ]] && $DO_STEP4; then
    warn "No negative images found in $NEG_DIR"
    warn "Step 4 (FP correction) will be skipped."
    warn "To get negatives: python3 download_negatives.py"
    DO_STEP4=false
fi

# ── Print plan ────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Run plan:${NC}"
$DO_STEP1 && echo -e "  ${GREEN}→${NC} Step 1: Train YOLOv8s Detector (640px)" \
           || echo -e "  ${YELLOW}  skip${NC}  Step 1 (using existing detector)"
$DO_STEP2 && echo -e "  ${GREEN}→${NC} Step 2: Generate Crops + Train EfficientNet-B5 Classifier" \
           || echo -e "  ${YELLOW}  skip${NC}  Step 2"
$DO_STEP3 && echo -e "  ${GREEN}→${NC} Step 3: Train Temporal LSTM Smoother" \
           || echo -e "  ${YELLOW}  skip${NC}  Step 3 (add --temporal to enable)"
$DO_STEP4 && echo -e "  ${GREEN}→${NC} Step 4: False Positive Correction (hard-negative mining)" \
           || echo -e "  ${YELLOW}  skip${NC}  Step 4"
$DO_TRT   && echo -e "  ${GREEN}→${NC} TRT  : Export models to TensorRT INT8"
$DO_INFER && echo -e "  ${GREEN}→${NC} Infer: Real-time inference on source='$SOURCE' $TRT_FLAG"
echo ""

# ── Timing helper ─────────────────────────────────────────────────────────────
declare -A TIMINGS

start_timer() { T_START=$(date +%s); }
stop_timer()  {
    local elapsed=$(( $(date +%s) - T_START ))
    local m=$(( elapsed / 60 )); local s=$(( elapsed % 60 ))
    TIMINGS["$1"]="${m}m ${s}s"
}

# =============================================================================
#  STEP 1 — Train YOLOv8s Detector
# =============================================================================
if $DO_STEP1; then
    step 1 "Train YOLOv8s Detector (img_size=640)"
    start_timer

    $PYTHON part1.py --config "$CONFIG"

    if [[ -f "logs/detector/best.pt" ]]; then
        ok "Detector weights saved: logs/detector/best.pt"
    else
        # Search for where Ultralytics actually saved them
        FOUND=$(find runs/detect -name "best.pt" 2>/dev/null | head -1)
        if [[ -n "$FOUND" ]]; then
            mkdir -p logs/detector
            cp "$FOUND" logs/detector/best.pt
            ok "Detector weights copied from $FOUND → logs/detector/best.pt"
        else
            err "Detector weights not found — check training output above"
            exit 1
        fi
    fi
    stop_timer "step1"
    ok "Step 1 complete  [${TIMINGS[step1]}]"
else
    if [[ ! -f "logs/detector/best.pt" ]]; then
        err "Detector weights not found at logs/detector/best.pt"
        err "Remove --skip-step1 or copy weights manually."
        exit 1
    fi
    ok "Step 1 skipped — using existing logs/detector/best.pt"
fi

# =============================================================================
#  STEP 2 — Generate Crops + Train Classifier
# =============================================================================
if $DO_STEP2; then
    step 2 "Generate Crops + Train EfficientNet-B5 Classifier"
    start_timer

    $PYTHON part2.py --config "$CONFIG" --det-ckpt logs/detector/best.pt

    if [[ -f "logs/classifier/best.pt" ]]; then
        ok "Classifier weights saved: logs/classifier/best.pt"
    else
        err "Classifier weights not found."
        exit 1
    fi
    stop_timer "step2"
    ok "Step 2 complete  [${TIMINGS[step2]}]"
else
    [[ -f "logs/classifier/best.pt" ]] \
        && ok "Step 2 skipped — using existing logs/classifier/best.pt" \
        || warn "Step 2 skipped and no classifier found — Gate 2 won't run"
fi

# =============================================================================
#  STEP 3 — Temporal LSTM Smoother  (optional)
# =============================================================================
if $DO_STEP3; then
    step 3 "Train Temporal BiLSTM Smoother"

    if [[ ! -d "$SEQUENCES_DIR" ]] || [[ -z "$(ls "$SEQUENCES_DIR" 2>/dev/null)" ]]; then
        warn "No video sequences found in $SEQUENCES_DIR"
        warn "Add .mp4 / .avi clips and re-run with --temporal"
        warn "Skipping temporal training."
    else
        start_timer
        VID_COUNT=$(find "$SEQUENCES_DIR" -name "*.mp4" -o -name "*.avi" -o \
                         -name "*.mov" -o -name "*.mkv" 2>/dev/null | wc -l)
        info "Found $VID_COUNT video files in $SEQUENCES_DIR"

        $PYTHON part3.py --config "$CONFIG" --sequences "$SEQUENCES_DIR"

        if [[ -f "logs/temporal/best.pt" ]]; then
            ok "Temporal weights saved: logs/temporal/best.pt"
        else
            warn "Temporal weights not found — temporal blending will be skipped in inference"
        fi
        stop_timer "step3"
        ok "Step 3 complete  [${TIMINGS[step3]}]"
    fi
else
    info "Step 3 skipped  (pass --temporal to enable)"
fi

# =============================================================================
#  STEP 4 — False Positive Correction
# =============================================================================
if $DO_STEP4; then
    step 4 "False Positive Correction (hard-negative mining)"
    start_timer

    $PYTHON part4.py --config "$CONFIG"

    if [[ -f "logs/fp_correction/detector_ft_best.pt" ]]; then
        ok "FT Detector:   logs/fp_correction/detector_ft_best.pt"
    fi
    if [[ -f "logs/fp_correction/classifier_bg_best.pt" ]]; then
        ok "FT Classifier: logs/fp_correction/classifier_bg_best.pt"
    fi
    stop_timer "step4"
    ok "Step 4 complete  [${TIMINGS[step4]}]"
fi

# =============================================================================
#  TRT EXPORT  (Jetson only)
# =============================================================================
if $DO_TRT; then
    step "TRT" "Export to TensorRT INT8 (Jetson Orin Nano)"

    # Warn if not on Jetson
    if ! command -v nvpmodel &>/dev/null; then
        warn "nvpmodel not found — probably not on Jetson."
        warn "TRT export may fail without CUDA TensorRT installed."
    else
        sudo nvpmodel -m 0 2>/dev/null && ok "MAXN power mode set" || true
        sudo jetson_clocks 2>/dev/null && ok "jetson_clocks applied"  || true
    fi

    start_timer
    $PYTHON export_trt.py --config "$CONFIG"

    [[ -f "logs/trt/detector_int8.engine" ]]    && ok "logs/trt/detector_int8.engine"
    [[ -f "logs/trt/classifier_int8.engine" ]]  && ok "logs/trt/classifier_int8.engine"
    stop_timer "trt"
    ok "TRT export complete  [${TIMINGS[trt]}]"
fi

# =============================================================================
#  INFERENCE  (real-time pipeline)
# =============================================================================
if $DO_INFER; then
    banner "Real-Time Inference"

    # Pick the best available weights
    DET_WEIGHTS="logs/detector/best.pt"
    if [[ -f "logs/fp_correction/detector_ft_best.pt" ]]; then
        DET_WEIGHTS="logs/fp_correction/detector_ft_best.pt"
        info "Using fine-tuned detector: $DET_WEIGHTS"
    else
        info "Using base detector: $DET_WEIGHTS"
    fi

    # TRT mode check
    if [[ -n "$TRT_FLAG" ]]; then
        if [[ ! -f "logs/trt/detector_int8.engine" ]]; then
            warn "TRT engine not found — run with --export-trt first."
            warn "Falling back to PyTorch."
            TRT_FLAG=""
        else
            ok "TRT engines found — running in INT8 mode"
        fi
    fi

    info "Source: $SOURCE"
    $DO_STEP3 && [[ -f "logs/temporal/best.pt" ]] \
        && info "Temporal smoother: ENABLED" \
        || info "Temporal smoother: disabled (no weights)"

    echo ""
    info "Starting inference... (press Q to quit)"
    echo ""

    $PYTHON inference.py \
        --source  "$SOURCE" \
        --config  "$CONFIG" \
        $TRT_FLAG
fi

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
if ! $DO_INFER; then
    banner "Pipeline Complete"

    echo -e "  ${BOLD}Timing summary:${NC}"
    for key in step1 step2 step3 step4 trt; do
        [[ -n "${TIMINGS[$key]}" ]] && echo "    $key : ${TIMINGS[$key]}"
    done

    echo ""
    echo -e "  ${BOLD}Output weights:${NC}"
    weights=(
        "logs/detector/best.pt                   | Base Detector"
        "logs/classifier/best.pt                 | Base Classifier"
        "logs/temporal/best.pt                   | Temporal Smoother"
        "logs/fp_correction/detector_ft_best.pt  | Fine-tuned Detector"
        "logs/fp_correction/classifier_bg_best.pt| Fine-tuned Classifier (+bg)"
        "logs/trt/detector_int8.engine           | TRT INT8 Detector"
        "logs/trt/classifier_int8.engine         | TRT INT8 Classifier"
    )
    for entry in "${weights[@]}"; do
        path=$(echo "$entry" | cut -d'|' -f1 | xargs)
        name=$(echo "$entry" | cut -d'|' -f2 | xargs)
        if [[ -f "$path" ]]; then
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            echo -e "    ${GREEN}✓${NC}  [$size]  $name  →  $path"
        else
            echo -e "    ${YELLOW}–${NC}         $name  (not generated)"
        fi
    done

    echo ""
    echo -e "  ${BOLD}Next commands:${NC}"
    echo "    # Real-time on webcam (PyTorch)"
    echo "    bash run_pipeline.sh --infer-only --source 0"
    echo ""
    echo "    # Real-time on video file"
    echo "    bash run_pipeline.sh --infer-only --source /path/to/video.mp4"
    echo ""
    echo "    # Jetson: export TRT then run"
    echo "    bash run_pipeline.sh --export-trt"
    echo "    bash run_pipeline.sh --infer-only --source 0 --trt"
    echo ""
    echo "    # Check alert log"
    echo "    cat logs/alerts/alerts.jsonl | python3 -m json.tool | head -20"
    echo ""
    echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════${NC}"
fi
