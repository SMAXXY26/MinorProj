#!/usr/bin/env bash
# setup_pi5.sh — Set up Raspberry Pi 5 + Hailo-8L AI HAT for weapon detection
# ============================================================================
# Run once on the Pi 5 after:
#   1. Flashing Raspberry Pi OS Bookworm (64-bit, Desktop or Lite)
#   2. Copying weapon_detection/ to ~/weapon_detection/
#
# What this does:
#   1. System update + Hailo PCIe driver + HailoRT runtime (hailo-all)
#   2. Python packages: hailort, ultralytics, supervision, torch (CPU), etc.
#   3. Creates required log/output directories
#   4. Verifies Hailo chip is detected
#
# Usage:
#   bash scripts/setup_pi5.sh
#   bash scripts/setup_pi5.sh --skip-hailo   # Skip apt steps (already done)

set -euo pipefail

SKIP_HAILO_APT=false
for arg in "$@"; do
    [[ "$arg" == "--skip-hailo" ]] && SKIP_HAILO_APT=true
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ─── 0. Sanity checks ────────────────────────────────────────────────────────
[[ "$(uname -m)" == "aarch64" ]] || err "This script is for aarch64 (Pi 5). Got: $(uname -m)"

PI_MODEL=$(cat /proc/cpuinfo | grep "Model" | head -1 || true)
log "Platform : ${PI_MODEL:-unknown}"

# ─── 1. System update ────────────────────────────────────────────────────────
log "Updating system packages ..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ─── 2. Hailo-8L driver + runtime ────────────────────────────────────────────
if [[ "$SKIP_HAILO_APT" == false ]]; then
    log "Installing Hailo PCIe driver and HailoRT runtime ..."

    # hailo-all is in the standard Raspberry Pi OS repository since Bookworm
    # It installs: hailo-dkms (PCIe driver), hailort (runtime), libhailort-dev,
    # python3-hailort, and the hailo firmware.
    sudo apt-get install -y hailo-all

    # Verify the kernel module loaded
    if ! lsmod | grep -q hailo_pci; then
        warn "hailo_pci module not loaded yet. Rebooting may be required."
        warn "After reboot, re-run: bash scripts/setup_pi5.sh --skip-hailo"
    fi
else
    log "Skipping Hailo apt steps (--skip-hailo)"
fi

# ─── 3. System dependencies ──────────────────────────────────────────────────
log "Installing system dependencies ..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk-3-dev \
    libgl1 \
    git \
    curl

# picamera2 for Pi Camera Module support (optional)
sudo apt-get install -y python3-picamera2 --no-install-recommends || \
    warn "picamera2 not available — USB camera fallback will be used"

# ─── 4. Python virtual environment ───────────────────────────────────────────
VENV_DIR="$HOME/weapon_detection/.venv"
log "Creating Python venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR" --system-site-packages
# --system-site-packages gives venv access to python3-hailort installed by apt

source "$VENV_DIR/bin/activate"

# Upgrade pip inside venv
pip install --upgrade pip wheel --quiet

# ─── 5. Python packages ──────────────────────────────────────────────────────
log "Installing Python packages (this may take 5-10 min on Pi 5) ..."

# Core inference stack
pip install --quiet \
    "ultralytics>=8.3" \
    "supervision>=0.22" \
    "opencv-python-headless>=4.8" \
    "pyyaml>=6.0" \
    "numpy>=1.24" \
    "tqdm"

# ONNX Runtime CPU (fallback when HEF not available)
pip install --quiet onnxruntime

# PyTorch CPU for LSTM temporal smoother + PyTorch fallback
# Use the arm64 wheel from PyPI (CPU-only, smaller than CUDA build)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu

# timm for EfficientNet backbone (needed if using PyTorch fallback classifier)
pip install --quiet timm

# GPS / MAVLink / alert utilities
pip install --quiet \
    "pyserial>=3.5" \
    "requests>=2.28"

# hailort Python bindings — try pip first, fall back to apt-installed package
if ! python3 -c "import hailo_platform" 2>/dev/null; then
    log "hailort Python package not found via apt — installing from PyPI ..."
    pip install --quiet hailort || \
        warn "hailort pip install failed. It may be available only via hailo-all apt package."
fi

deactivate

# ─── 6. Create directory structure ───────────────────────────────────────────
log "Creating log directories ..."
mkdir -p ~/weapon_detection/logs/{hailo,alerts,temporal}

# ─── 7. Activation helper ────────────────────────────────────────────────────
ACTIVATE_SCRIPT="$HOME/weapon_detection/activate.sh"
cat > "$ACTIVATE_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# Source this to activate the weapon detection environment:
#   source ~/weapon_detection/activate.sh
source "$HOME/weapon_detection/.venv/bin/activate"
cd "$HOME/weapon_detection"
EOF
chmod +x "$ACTIVATE_SCRIPT"

# ─── 8. Hailo chip verification ──────────────────────────────────────────────
log "Verifying Hailo-8L chip ..."

if command -v hailortcli &>/dev/null; then
    hailortcli fw-control identify || warn "hailortcli identify failed — check PCIe connection"
elif python3 -c "
from hailo_platform import VDevice, HailoStreamInterface
try:
    d = VDevice()
    print('Hailo device detected OK')
    del d
except Exception as e:
    print(f'Hailo device error: {e}')
    exit(1)
" 2>&1; then
    :
else
    warn "Could not verify Hailo chip. Check:"
    warn "  1. AI HAT is firmly seated in the PCIe FFC slot"
    warn "  2. sudo raspi-config → Advanced → PCIe Speed → Gen 3"
    warn "  3. Reboot and re-run this script with --skip-hailo"
fi

# ─── 9. Quick smoke test ─────────────────────────────────────────────────────
log "Quick smoke test ..."
source "$VENV_DIR/bin/activate"

python3 - <<'PYEOF'
import sys
issues = []

try:
    import cv2
    print(f"  cv2          : {cv2.__version__}  ✓")
except ImportError:
    issues.append("opencv-python-headless")

try:
    import ultralytics
    print(f"  ultralytics  : {ultralytics.__version__}  ✓")
except ImportError:
    issues.append("ultralytics")

try:
    import supervision as sv
    print(f"  supervision  : {sv.__version__}  ✓")
except ImportError:
    issues.append("supervision")

try:
    import torch
    print(f"  torch        : {torch.__version__}  ✓ (CUDA={torch.cuda.is_available()})")
except ImportError:
    issues.append("torch")

try:
    import hailo_platform as hp
    print(f"  hailo_platform : ✓")
except ImportError:
    issues.append("hailo_platform — install hailo-all or pip install hailort")

try:
    import onnxruntime as ort
    print(f"  onnxruntime  : {ort.__version__}  ✓")
except ImportError:
    print("  onnxruntime  : not installed (optional CPU fallback)")

if issues:
    print(f"\n  Missing packages: {issues}")
    sys.exit(1)
else:
    print("\n  All core packages OK")
PYEOF

deactivate

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
log "Setup complete."
echo ""
echo "  Next steps:"
echo "  1. Copy HEF files from dev machine:"
echo "     (on dev machine)  rsync -avz logs/hailo/*.hef pi@<PI_IP>:~/weapon_detection/logs/hailo/"
echo ""
echo "  2. Activate environment and run:"
echo "     source ~/weapon_detection/activate.sh"
echo "     python3 inference_pi5.py --source 0 --hailo"
echo ""
echo "  3. For Pi Camera Module:"
echo "     python3 inference_pi5.py --source 0 --hailo --picam"
echo ""
echo "  4. Headless / SSH:"
echo "     python3 inference_pi5.py --source 0 --hailo --no-display"
echo ""
echo "  Power tip: set GPU memory split low (Pi 5 has dedicated RAM):"
echo "    sudo raspi-config → Performance → GPU Memory → 16"
