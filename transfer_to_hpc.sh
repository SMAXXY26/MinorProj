#!/usr/bin/env bash
# transfer_to_hpc.sh — Transfer everything needed for HPC training.
#
# Usage:
#   ./transfer_to_hpc.sh                     # step 9 only (distillation)
#   ./transfer_to_hpc.sh --full              # full pipeline (steps 1-5, 9)
#   ./transfer_to_hpc.sh --fp-correction     # FP correction (step 5)
#   ./transfer_to_hpc.sh --classifier        # classifier training (step 3)
#   ./transfer_to_hpc.sh --all               # FP corr + classifier + distillation
#   ./transfer_to_hpc.sh --dry-run           # preview only, nothing transferred
#   ./transfer_to_hpc.sh --code-only         # skip images + labels (already on HPC)
#   ./transfer_to_hpc.sh --new-data          # only transfer new drone_*/visdrone_* files
#
# Modes can be combined:
#   ./transfer_to_hpc.sh --full --dry-run
#   ./transfer_to_hpc.sh --full --code-only  # re-transfer code after edits, skip data
#
# Safe to re-run: rsync is idempotent and skips unchanged files.
# Safe to interrupt: rsync --partial resumes where it left off.

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit before first run
# ══════════════════════════════════════════════════════════════════════════════
HPC_USER="btech10170.23"
HPC_HOST="172.16.220.100"
HPC_DIR="~/Weapon_detection"
SSH_PORT="22"

# ══════════════════════════════════════════════════════════════════════════════
#  Flags
# ══════════════════════════════════════════════════════════════════════════════
DRY_RUN=""
CODE_ONLY=0
NEW_DATA=0
FULL=0
CLASSIFIER=0
FP_CORRECTION=0
TEMPORAL=0
DETECTOR=0
SYNTHETIC=0

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN="--dry-run"
                        echo "  DRY-RUN MODE — nothing will actually transfer" ;;
        --code-only)    CODE_ONLY=1
                        echo "  CODE-ONLY — skipping images + labels" ;;
        --new-data)     NEW_DATA=1
                        echo "  NEW-DATA — only drone_* and visdrone_* files (~1.1 GB)" ;;
        --full)         FULL=1; DETECTOR=1; CLASSIFIER=1; TEMPORAL=1; FP_CORRECTION=1
                        echo "  FULL MODE — complete pipeline transfer (steps 1-5, 9)" ;;
        --classifier)   CLASSIFIER=1
                        echo "  CLASSIFIER — crops + part2 scripts" ;;
        --fp-correction)FP_CORRECTION=1
                        echo "  FP-CORRECTION — drone negatives + part4 scripts" ;;
        --temporal)     TEMPORAL=1
                        echo "  TEMPORAL — video sequences + temporal scripts" ;;
        --detector)     DETECTOR=1
                        echo "  DETECTOR — part1 scripts + yolov8s pretrain" ;;
        --synthetic)    SYNTHETIC=1
                        echo "  SYNTHETIC — rendered aerial images (~304 MB)" ;;
        --all)          CLASSIFIER=1; FP_CORRECTION=1
                        echo "  ALL MODE — FP correction + classifier + distillation" ;;
        *)  echo "  unknown arg: $arg"; exit 1 ;;
    esac
done

if [ "$CODE_ONLY" -eq 1 ] && [ "$NEW_DATA" -eq 1 ]; then
    echo "  ERROR: --code-only and --new-data are mutually exclusive"; exit 1
fi

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_ROOT"

# ── SSH multiplexing — one password prompt for the entire run ─────────────────
SSH_CTRL_DIR="${TMPDIR:-/tmp}/ssh_ctrl_$$"
mkdir -p "$SSH_CTRL_DIR" && chmod 700 "$SSH_CTRL_DIR"
SSH_CTRL_PATH="$SSH_CTRL_DIR/%r@%h:%p"
SSH_MUX=(-o "ControlMaster=auto" -o "ControlPath=$SSH_CTRL_PATH" -o "ControlPersist=30m")
SSH_CMD="ssh -p $SSH_PORT ${SSH_MUX[*]}"

cleanup_ssh() {
    ssh "${SSH_MUX[@]}" -O exit -p "$SSH_PORT" "$HPC_USER@$HPC_HOST" 2>/dev/null || true
    rm -rf "$SSH_CTRL_DIR" 2>/dev/null || true
}
trap cleanup_ssh EXIT

RSYNC=(-avzh --progress --partial ${DRY_RUN:-} -e "$SSH_CMD")
REMOTE="$HPC_USER@$HPC_HOST"

# ══════════════════════════════════════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════"
echo "  HPC Transfer — Weapon Detection Pipeline"
echo "══════════════════════════════════════════════════════════════"
echo "  Local  : $LOCAL_ROOT"
echo "  Remote : $REMOTE:$HPC_DIR"
echo "  Mode   : ${DRY_RUN:-LIVE}"
echo ""
echo "  Training targets:"
[ "$DETECTOR"    -eq 1 ] && echo "    ✓ Detector (step 2)"
[ "$CLASSIFIER"  -eq 1 ] && echo "    ✓ Classifier (step 3)"
[ "$TEMPORAL"    -eq 1 ] && echo "    ✓ Temporal smoother (step 4)"
[ "$FP_CORRECTION" -eq 1 ] && echo "    ✓ FP correction (step 5)"
[ "$SYNTHETIC"   -eq 1 ] && echo "    ✓ Synthetic aerial data"
echo "    ✓ Student distillation (step 9)  [always]"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  Pre-flight: verify required files exist locally
# ══════════════════════════════════════════════════════════════════════════════
echo "── Pre-flight checks ─────────────────────────────────────────"

required=(
    train_all.py
    requirements_hpc.txt
    yolo11n.pt
    run_step9.pbs
    run_all.pbs
    config/hyperparams.yaml
    config/yolo11n_p2.yaml
    src/__init__.py
    src/model.py
    src/dataset.py
    src/losses.py
    src/augmentations.py
    src/functions/__init__.py
    src/functions/distill.py
    src/functions/small_object_aug.py
    src/functions/monitor.py
    src/functions/common.py
    data/images/train
    data/images/val
    data/labels/train
    data/labels/val
    runs/detect/logs/fp_correction/weights/best.pt
)

[ "$DETECTOR"    -eq 1 ] && required+=(part1.py run_detector.pbs src/functions/detector.py)
[ "$CLASSIFIER"  -eq 1 ] && required+=(part2.py run_part2.pbs src/functions/classifier.py
                                        data/classifier_crops/knife
                                        data/classifier_crops/pistol
                                        data/classifier_crops/rifle)
[ "$TEMPORAL"    -eq 1 ] && required+=(part3.py run_temporal.pbs src/functions/temporal.py
                                        data/sequences)
[ "$FP_CORRECTION" -eq 1 ] && required+=(part4.py run_part4.pbs src/functions/fp_correction.py
                                          logs/detector/best.pt logs/classifier/best.pt)
[ "$SYNTHETIC"   -eq 1 ] && required+=(run_synthetic.pbs data/synthetic/images data/synthetic/labels)

MISSING=0
for f in "${required[@]}"; do
    if [ -e "$LOCAL_ROOT/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  ERROR: $MISSING required file(s) missing. Aborting."
    [ ! -f "$LOCAL_ROOT/yolo11n.pt" ] && \
        echo "  Tip: python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
    exit 1
fi

# ── SSH reachability ──────────────────────────────────────────────────────────
echo ""
echo "  Testing SSH to $REMOTE ..."
echo "  (Enter password once — all subsequent stages reuse this connection.)"
if ssh "${SSH_MUX[@]}" -p "$SSH_PORT" -o ConnectTimeout=30 "$REMOTE" "test -d $HPC_DIR"; then
    echo "  ✓ SSH OK, $HPC_DIR exists"
else
    echo "  ✗ Cannot reach $REMOTE or $HPC_DIR does not exist."
    echo "    Create it: ssh $REMOTE 'mkdir -p $HPC_DIR'"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Remote directory tree
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Stage 1: Remote directory tree ───────────────────────────"
REMOTE_DIRS=(
    "$HPC_DIR/config"
    "$HPC_DIR/src"
    "$HPC_DIR/src/functions"
    "$HPC_DIR/utils"
    "$HPC_DIR/data/images/train"
    "$HPC_DIR/data/images/val"
    "$HPC_DIR/data/labels/train"
    "$HPC_DIR/data/labels/val"
    "$HPC_DIR/runs/detect/logs/fp_correction/weights"
    "$HPC_DIR/runs/detect/logs/student/weights"
    "$HPC_DIR/logs/student"
    "$HPC_DIR/logs/detector"
    "$HPC_DIR/logs/classifier"
    "$HPC_DIR/logs/temporal"
    "$HPC_DIR/logs/fp_correction"
    "$HPC_DIR/.cache/torch/hub/checkpoints"
)
[ "$TEMPORAL"    -eq 1 ] && REMOTE_DIRS+=("$HPC_DIR/data/sequences")
[ "$CLASSIFIER"  -eq 1 ] && REMOTE_DIRS+=(
    "$HPC_DIR/data/classifier_crops/knife"
    "$HPC_DIR/data/classifier_crops/pistol"
    "$HPC_DIR/data/classifier_crops/rifle"
)
[ "$FP_CORRECTION" -eq 1 ] && REMOTE_DIRS+=(
    "$HPC_DIR/data/negatives/visdrone"
    "$HPC_DIR/logs/fp_correction/hard_negatives"
)
[ "$SYNTHETIC"   -eq 1 ] && REMOTE_DIRS+=(
    "$HPC_DIR/data/synthetic/images"
    "$HPC_DIR/data/synthetic/labels"
)

ssh "${SSH_MUX[@]}" -p "$SSH_PORT" "$REMOTE" "mkdir -p ${REMOTE_DIRS[*]}"
echo "  ✓ Remote dirs ready"

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — HPC dataset.yaml (relative paths — works from any workdir)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Stage 2: HPC-compatible dataset.yaml ─────────────────────"
TMP_YAML="$(mktemp)"
cat > "$TMP_YAML" <<'EOF'
names:
- knife
- pistol
- rifle
nc: 3
path: data
train: images/train
val: images/val
EOF
rsync "${RSYNC[@]}" "$TMP_YAML" "$REMOTE:$HPC_DIR/data/dataset.yaml"
rm -f "$TMP_YAML"
echo "  ✓ Patched dataset.yaml (path: data — relative to PBS workdir)"

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Code, configs, PBS scripts (~7 MB)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Stage 3: Code + configs + PBS scripts ────────────────────"

# Core files always transferred
CORE_FILES=(
    train_all.py
    requirements_hpc.txt
    yolo11n.pt
    run_step9.pbs
    run_all.pbs
    CLAUDE.md
    CITATIONS.md
)
[ "$DETECTOR"      -eq 1 ] && CORE_FILES+=(part1.py run_detector.pbs)
[ "$CLASSIFIER"    -eq 1 ] && CORE_FILES+=(part2.py run_part2.pbs)
[ "$TEMPORAL"      -eq 1 ] && CORE_FILES+=(part3.py run_temporal.pbs)
[ "$FP_CORRECTION" -eq 1 ] && CORE_FILES+=(part4.py run_part4.pbs)
[ "$SYNTHETIC"     -eq 1 ] && CORE_FILES+=(run_synthetic.pbs)

rsync "${RSYNC[@]}" "${CORE_FILES[@]}" "$REMOTE:$HPC_DIR/"

# Full config/ directory (includes hyperparams.yaml AND yolo11n_p2.yaml)
rsync "${RSYNC[@]}" config/ "$REMOTE:$HPC_DIR/config/"

# Full src/ tree (includes small_object_aug.py and everything it imports)
rsync "${RSYNC[@]}" \
    --exclude '__pycache__' --exclude '*.pyc' \
    src/ "$REMOTE:$HPC_DIR/src/"

# utils/ (tracker, alert, geo — imported at inference + some training steps)
if [ -d utils ]; then
    rsync "${RSYNC[@]}" \
        --exclude '__pycache__' --exclude '*.pyc' \
        utils/ "$REMOTE:$HPC_DIR/utils/"
fi

echo "  ✓ Code + configs transferred"

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4 — Model weights
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Stage 4: Model weights ────────────────────────────────────"

# T1 teacher — FP-corrected YOLOv8s (always needed for distillation)
rsync "${RSYNC[@]}" \
    runs/detect/logs/fp_correction/weights/best.pt \
    "$REMOTE:$HPC_DIR/runs/detect/logs/fp_correction/weights/"
echo "  ✓ T1 teacher  runs/detect/logs/fp_correction/weights/best.pt"

# T2 teacher — FP-correction fine-tuned detector
if [ -f logs/fp_correction/detector_ft_best.pt ]; then
    rsync "${RSYNC[@]}" \
        logs/fp_correction/detector_ft_best.pt \
        "$REMOTE:$HPC_DIR/logs/fp_correction/"
    echo "  ✓ T2 teacher  logs/fp_correction/detector_ft_best.pt"
fi

# T3 teacher — base detector (pre FP correction)
if [ -f logs/detector/best.pt ]; then
    rsync "${RSYNC[@]}" \
        logs/detector/best.pt \
        "$REMOTE:$HPC_DIR/logs/detector/"
    echo "  ✓ T3 teacher  logs/detector/best.pt"
fi

# Classifier weights
if [ -f logs/classifier/best.pt ]; then
    rsync "${RSYNC[@]}" \
        logs/classifier/best.pt \
        "$REMOTE:$HPC_DIR/logs/classifier/"
    echo "  ✓ Classifier  logs/classifier/best.pt"
fi

# Temporal weights
if [ -f logs/temporal/best.pt ]; then
    rsync "${RSYNC[@]}" \
        logs/temporal/best.pt \
        "$REMOTE:$HPC_DIR/logs/temporal/"
    echo "  ✓ Temporal    logs/temporal/best.pt"
fi

# Round 1 student — for iterative KD warm-start in step 9
if [ -f logs/student/best.pt ]; then
    rsync "${RSYNC[@]}" \
        logs/student/best.pt \
        "$REMOTE:$HPC_DIR/logs/student/best.pt"
    echo "  ✓ R1 student  logs/student/best.pt  (iterative KD warm-start)"
else
    echo "  ⚠ No R1 student found — step 9 will start fresh from yolo11n.pt"
fi

# EfficientNet-B5 ImageNet pretrain cache (avoids download from compute node)
EFFB5="$HOME/.cache/torch/hub/checkpoints/efficientnet_b5_lukemelas-b6417697.pth"
if [ -f "$EFFB5" ]; then
    rsync "${RSYNC[@]}" "$EFFB5" \
        "$REMOTE:$HPC_DIR/.cache/torch/hub/checkpoints/"
    echo "  ✓ EfficientNet-B5 pretrain weights cached"
else
    echo "  ⚠ EfficientNet-B5 cache not found at $EFFB5"
    echo "    Run: python -c \"from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights; efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)\""
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4b — FP correction data (negatives + mining results)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$FP_CORRECTION" -eq 1 ] && [ "$CODE_ONLY" -eq 0 ]; then
    echo ""
    echo "── Stage 4b: VisDrone negatives (~400 MB) ────────────────"
    NEG_COUNT=$(ls data/negatives/visdrone/*.jpg 2>/dev/null | wc -l || echo 0)
    echo "  $NEG_COUNT negative images"
    rsync "${RSYNC[@]}" \
        data/negatives/visdrone/ \
        "$REMOTE:$HPC_DIR/data/negatives/visdrone/"

    if [ -d logs/fp_correction/hard_negatives ]; then
        CROP_COUNT=$(find logs/fp_correction/hard_negatives -name "*.jpg" | wc -l)
        echo "  Transferring $CROP_COUNT hard-negative crops..."
        rsync "${RSYNC[@]}" \
            logs/fp_correction/hard_negatives/ \
            "$REMOTE:$HPC_DIR/logs/fp_correction/hard_negatives/"
    fi
    [ -f logs/fp_correction/threshold_sweep.json ] && \
        rsync "${RSYNC[@]}" \
            logs/fp_correction/threshold_sweep.json \
            "$REMOTE:$HPC_DIR/logs/fp_correction/"
    echo "  ✓ FP correction data transferred"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4c — Classifier crops
# ══════════════════════════════════════════════════════════════════════════════
if [ "$CLASSIFIER" -eq 1 ] && [ "$CODE_ONLY" -eq 0 ]; then
    echo ""
    echo "── Stage 4c: Classifier crops ────────────────────────────"
    for cls in knife pistol rifle; do
        COUNT=$(ls data/classifier_crops/$cls 2>/dev/null | wc -l || echo 0)
        echo "  $cls: $COUNT crops"
        rsync "${RSYNC[@]}" \
            data/classifier_crops/$cls/ \
            "$REMOTE:$HPC_DIR/data/classifier_crops/$cls/"
    done
    echo "  ✓ Classifier crops transferred"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4d — Temporal sequences (video clips for BiLSTM training)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$TEMPORAL" -eq 1 ] && [ "$CODE_ONLY" -eq 0 ]; then
    echo ""
    echo "── Stage 4d: Temporal sequences ─────────────────────────"
    SEQ_MB=$(du -sh data/sequences/ 2>/dev/null | cut -f1 || echo "?")
    SEQ_COUNT=$(find data/sequences -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)
    echo "  $SEQ_COUNT video files  ($SEQ_MB)"
    rsync "${RSYNC[@]}" \
        data/sequences/ \
        "$REMOTE:$HPC_DIR/data/sequences/"
    echo "  ✓ Temporal sequences transferred"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4e — Synthetic aerial images
# ══════════════════════════════════════════════════════════════════════════════
if [ "$SYNTHETIC" -eq 1 ] && [ "$CODE_ONLY" -eq 0 ]; then
    echo ""
    echo "── Stage 4e: Synthetic aerial images (~304 MB) ───────────"
    SYN_COUNT=$(ls data/synthetic/images/*.jpg 2>/dev/null | wc -l || echo 0)
    echo "  $SYN_COUNT synthetic images"
    rsync "${RSYNC[@]}" data/synthetic/images/ "$REMOTE:$HPC_DIR/data/synthetic/images/"
    rsync "${RSYNC[@]}" data/synthetic/labels/ "$REMOTE:$HPC_DIR/data/synthetic/labels/"
    echo "  ✓ Synthetic data transferred"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Stages 5-7 — Training images, validation images, labels
# ══════════════════════════════════════════════════════════════════════════════
if [ "$CODE_ONLY" -eq 0 ]; then
    if [ "$NEW_DATA" -eq 1 ]; then
        IMG_FILTER=(--include='drone_*' --include='visdrone_*' --exclude='*')
        LBL_FILTER=(--include='drone_*' --include='visdrone_*' --exclude='*')
        NEW_TRAIN=$(ls data/images/train/ | grep -cE '^(drone_|visdrone_)' || echo 0)
        NEW_VAL=$(ls data/images/val/   | grep -cE '^(drone_|visdrone_)' || echo 0)
        echo ""
        echo "── Stages 5-7: New drone images only ($NEW_TRAIN train / $NEW_VAL val) ──"
    else
        IMG_FILTER=(); LBL_FILTER=()
        TRAIN_COUNT=$(ls data/images/train | wc -l)
        VAL_COUNT=$(ls data/images/val   | wc -l)
        echo ""
        echo "── Stage 5: Training images — $TRAIN_COUNT files (~2.7 GB) ─────────"
        echo "  Largest stage. Safe to Ctrl-C and re-run — rsync resumes."
    fi

    rsync "${RSYNC[@]}" "${IMG_FILTER[@]}" \
        data/images/train/ "$REMOTE:$HPC_DIR/data/images/train/"

    echo ""
    echo "── Stage 6: Validation images ────────────────────────────"
    rsync "${RSYNC[@]}" "${IMG_FILTER[@]}" \
        data/images/val/ "$REMOTE:$HPC_DIR/data/images/val/"

    echo ""
    echo "── Stage 7: Labels ───────────────────────────────────────"
    rsync "${RSYNC[@]}" "${LBL_FILTER[@]}" \
        data/labels/train/ "$REMOTE:$HPC_DIR/data/labels/train/"
    rsync "${RSYNC[@]}" "${LBL_FILTER[@]}" \
        data/labels/val/   "$REMOTE:$HPC_DIR/data/labels/val/"

    # Delete stale Ultralytics label caches — they silently shadow any edits
    ssh "${SSH_MUX[@]}" -p "$SSH_PORT" "$REMOTE" \
        "rm -f $HPC_DIR/data/labels/train.cache $HPC_DIR/data/labels/val.cache" || true
    echo "  ✓ Labels transferred, stale caches cleared"
else
    echo ""
    echo "── Stages 5-7 SKIPPED (--code-only) ─────────────────────"
    echo "  Images + labels assumed already on HPC."
    ssh "${SSH_MUX[@]}" -p "$SSH_PORT" "$REMOTE" \
        "rm -f $HPC_DIR/data/labels/train.cache $HPC_DIR/data/labels/val.cache" 2>/dev/null || true
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ -n "$DRY_RUN" ]; then
    echo "  DRY-RUN complete — re-run without --dry-run to transfer"
else
    echo "  ✓ Transfer complete"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps on HPC:"
echo ""
echo "  1.  ssh $REMOTE"
echo "  2.  cd $HPC_DIR"
echo ""
echo "  3.  ONE-TIME setup (login node — compute nodes have no internet):"
echo "        source /apps/anaconda3/bin/activate deeplearning"
echo "        pip install --user -r requirements_hpc.txt"
echo ""

if [ "$FULL" -eq 1 ]; then
echo "  ── FULL pipeline ─────────────────────────────────────────"
echo "  4.  Submit all-in-one job (steps 0,1,2,3,4,5,9 sequentially):"
echo "        qsub run_all.pbs"
echo ""
echo "  OR submit steps individually for finer control:"
echo "        qsub run_detector.pbs         # step 2  (~16h)"
echo "        qsub run_part2.pbs            # step 3  (~12h)"
echo "        qsub run_temporal.pbs         # step 4  (~4h)"
echo "        qsub run_part4.pbs            # step 5  (~12h)"
echo "        qsub run_step9.pbs            # step 9  (~24h)"
echo ""
echo "  5.  Monitor: qstat -u $HPC_USER"
echo "              tail -f logs/pbs_all.out"
echo ""
echo "  6.  Pull weights back:"
echo "        scp $REMOTE:$HPC_DIR/logs/detector/best.pt              logs/detector/best.pt"
echo "        scp $REMOTE:$HPC_DIR/logs/classifier/best.pt            logs/classifier/best.pt"
echo "        scp $REMOTE:$HPC_DIR/logs/temporal/best.pt              logs/temporal/best.pt"
echo "        scp $REMOTE:$HPC_DIR/logs/fp_correction/detector_ft_best.pt  logs/fp_correction/"
echo "        scp $REMOTE:$HPC_DIR/runs/detect/logs/fp_correction/weights/best.pt  runs/detect/logs/fp_correction/weights/"
echo "        scp $REMOTE:$HPC_DIR/logs/student/best.pt               logs/student/best.pt"
else
    if [ "$FP_CORRECTION" -eq 1 ]; then
echo "  ── FP Correction ─────────────────────────────────────────"
echo "  4a. qsub run_part4.pbs"
echo "  5a. Monitor: tail -f logs/pbs_part4.out"
echo "  6a. scp $REMOTE:$HPC_DIR/logs/fp_correction/detector_ft_best.pt logs/fp_correction/"
echo ""
    fi
    if [ "$CLASSIFIER" -eq 1 ]; then
echo "  ── Classifier ────────────────────────────────────────────"
echo "  4b. qsub run_part2.pbs"
echo "  5b. Monitor: tail -f logs/pbs_part2.out"
echo "  6b. scp $REMOTE:$HPC_DIR/logs/classifier/best.pt logs/classifier/best.pt"
echo ""
    fi
echo "  ── Distillation (step 9) ─────────────────────────────────"
echo "  4c. qsub run_step9.pbs"
echo "  5c. Monitor: tail -f logs/pbs_step9.out"
echo "  6c. scp $REMOTE:$HPC_DIR/logs/student/best.pt logs/student/best.pt"
echo ""
fi

echo "  Cancel a job: qdel <job-id>"
echo "  All jobs:     qstat -u $HPC_USER"
echo ""
