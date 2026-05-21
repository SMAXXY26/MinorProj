#!/bin/bash
# scripts/setup_jetson.sh
# Jetson Orin Nano (8GB) — JetPack 6.x — one-time setup
# Run as: bash scripts/setup_jetson.sh

set -e

echo "=== Jetson Orin Nano Setup ==="

# Confirm Python 3.10.x
python3 --version
PYVER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYVER" -lt 10 ]; then
    echo "[ERROR] Python 3.10+ required (JetPack 6 default). Got 3.$PYVER"
    exit 1
fi

# Set MAXN power mode and lock clocks
sudo nvpmodel -m 0
sudo jetson_clocks
echo "[OK] MAXN power mode + jetson_clocks"

# Install PyTorch + torchvision from Jetson-AI-Lab repo
echo "=== Installing PyTorch (Jetson wheel) ==="
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch torchvision

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA missing!'; \
    print(f'[OK] PyTorch {torch.__version__} CUDA={torch.version.cuda}')"

# Install onnxruntime-gpu from Jetson repo
echo "=== Installing onnxruntime-gpu ==="
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    onnxruntime-gpu

# Install remaining Python dependencies
echo "=== Installing requirements_jetson.txt ==="
pip install -r requirements_jetson.txt

# Verify TensorRT (pre-installed by JetPack)
python3 -c "import tensorrt as trt; print(f'[OK] TensorRT {trt.__version__}')"

# Verify cuDNN
python3 -c "import torch; print(f'[OK] cuDNN {torch.backends.cudnn.version()}')"

# Quick sanity check — all critical imports
python3 -c "
import torch, cv2, ultralytics, supervision, numpy as np, pyserial, pynmea2, requests
print('[OK] All required packages importable')
print(f'  torch      {torch.__version__}')
print(f'  numpy      {np.__version__}')
print(f'  cv2        {cv2.__version__}')
print(f'  supervision confirmed')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. python3 export_trt.py          # export models to TRT INT8"
echo "  2. python3 inference.py --source 0 --trt --config config/hyperparams.yaml"
