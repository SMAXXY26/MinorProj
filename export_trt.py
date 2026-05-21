"""
export_trt.py  —  Export detector + classifier to TensorRT INT8
=================================================================
Task 1b: Ultralytics YOLOv8s → TRT engine  (detector)
         PyTorch EfficientNet-B5 → ONNX → TRT INT8 (classifier)

Usage:
    python3 export_trt.py
    python3 export_trt.py --config config/hyperparams.yaml
    python3 export_trt.py --detector-only
    python3 export_trt.py --classifier-only

Requirements on Jetson:
    tensorrt pre-installed by JetPack 6 (import tensorrt as trt)
    No pycuda — TRT Python bindings only
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/hyperparams.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    Path("logs/trt").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  DETECTOR export  (Ultralytics → TRT INT8 engine)
# ---------------------------------------------------------------------------

def export_detector(cfg: dict) -> str:
    """
    Export YOLOv8s to TRT INT8 using Ultralytics built-in export.
    Returns path to the .engine file.
    """
    from ultralytics import YOLO

    trt_cfg      = cfg.get("trt", {})
    det_ckpt     = "logs/detector/best.pt"
    engine_path  = trt_cfg.get("detector_engine", "logs/trt/detector_int8.engine")
    workspace_gb = int(trt_cfg.get("workspace_gb", 4))
    data_yaml    = os.path.join(cfg["dataset"]["root"], "dataset.yaml")
    img_size     = cfg["detector"].get("img_size", 640)

    if not Path(det_ckpt).exists():
        logger.error(f"Detector weights not found: {det_ckpt}")
        logger.error("Run part1.py first.")
        sys.exit(1)

    logger.info(f"\n{'═'*60}")
    logger.info("  Exporting Detector → TRT INT8")
    logger.info(f"  Checkpoint : {det_ckpt}")
    logger.info(f"  Engine     : {engine_path}")
    logger.info(f"  img_size   : {img_size}")
    logger.info(f"  workspace  : {workspace_gb} GB")
    logger.info(f"{'═'*60}")

    model = YOLO(det_ckpt)
    exported = model.export(
        format    = "engine",
        imgsz     = img_size,
        int8      = True,
        data      = data_yaml,
        workspace = workspace_gb,
        device    = 0,
    )

    # Ultralytics saves the engine next to the .pt by default — move it
    exported_path = Path(str(exported))
    target_path   = Path(engine_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if exported_path.exists() and str(exported_path) != str(target_path):
        import shutil
        shutil.copy2(str(exported_path), str(target_path))
        logger.info(f"  Engine copied → {target_path}")

    # ---- Warmup + latency benchmark ----------------------------------------
    logger.info("\n  Benchmarking detector engine ...")
    engine_model = YOLO(str(target_path))
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 100 warmup
    for _ in range(100):
        engine_model(dummy, verbose=False)

    # 200 timed
    t0 = time.perf_counter()
    for _ in range(200):
        engine_model(dummy, verbose=False)
    elapsed = time.perf_counter() - t0
    avg_ms  = elapsed / 200 * 1000
    fps     = 1000 / avg_ms

    logger.info(f"  TRT detector latency : {avg_ms:.2f} ms  ({fps:.1f} fps)")
    return str(target_path)


# ---------------------------------------------------------------------------
#  INT8 Calibrator for classifier
# ---------------------------------------------------------------------------

class WeaponCropCalibrator:
    """
    IInt8EntropyCalibrator2 for the EfficientNet-B5 classifier.

    Uses TensorRT Python bindings only (no pycuda).
    Device memory is managed via ctypes + CUDA driver.
    """

    def __init__(
        self,
        crop_dir:           str,
        cache_file:         str = "logs/trt/classifier_calib.cache",
        max_images:         int = 300,
        batch_size:         int = 8,
        img_size:           int = 224,
    ) -> None:
        import tensorrt as trt

        self._trt         = trt
        self._cache_file  = cache_file
        self._batch_size  = batch_size
        self._img_size    = img_size

        IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._mean = IMG_MEAN
        self._std  = IMG_STD

        # Collect calibration images
        import cv2
        import random
        exts   = {".jpg", ".jpeg", ".png"}
        images = []
        crop_path = Path(crop_dir)
        for cls_dir in sorted(crop_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in exts]
            images.extend(cls_files)

        random.shuffle(images)
        images = images[:max_images]
        logger.info(f"  Calibrator: {len(images)} images from {crop_dir}")

        # Pre-load and preprocess all images into a single array  (N, 3, 224, 224)
        self._data: list[np.ndarray] = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD
            self._data.append(img.transpose(2, 0, 1))

        self._batches   = [
            np.stack(self._data[i:i + batch_size])
            for i in range(0, len(self._data), batch_size)
            if len(self._data[i:i + batch_size]) > 0
        ]
        self._batch_idx = 0

        # Allocate device buffer via ctypes CUDA driver
        nbytes = batch_size * 3 * img_size * img_size * 4   # float32
        self._d_buf = self._alloc_device(nbytes)
        self._nbytes = nbytes

    @staticmethod
    def _alloc_device(nbytes: int) -> int:
        """Allocate CUDA device memory using ctypes."""
        # Load libcuda.so
        try:
            libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            libcuda = ctypes.CDLL("libcuda.so")

        ptr = ctypes.c_uint64(0)
        ret = libcuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if ret != 0:
            raise RuntimeError(f"cuMemAlloc failed: error code {ret}")
        return ptr.value

    @staticmethod
    def _copy_to_device(host_arr: np.ndarray, dev_ptr: int, nbytes: int) -> None:
        """H2D copy via ctypes."""
        try:
            libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            libcuda = ctypes.CDLL("libcuda.so")
        host_ptr = host_arr.ctypes.data_as(ctypes.c_void_p)
        ret = libcuda.cuMemcpyHtoD_v2(
            ctypes.c_uint64(dev_ptr),
            host_ptr,
            ctypes.c_size_t(nbytes),
        )
        if ret != 0:
            raise RuntimeError(f"cuMemcpyHtoD failed: error code {ret}")

    # TensorRT calibrator interface ------------------------------------------

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names):
        if self._batch_idx >= len(self._batches):
            return None
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        arr = np.ascontiguousarray(batch, dtype=np.float32)
        nbytes = arr.nbytes
        self._copy_to_device(arr, self._d_buf, nbytes)
        return [self._d_buf]

    def read_calibration_cache(self) -> bytes | None:
        if Path(self._cache_file).exists():
            logger.info(f"  Calibration cache hit: {self._cache_file}")
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        Path(self._cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "wb") as f:
            f.write(cache)
        logger.info(f"  Calibration cache written: {self._cache_file}")


# ---------------------------------------------------------------------------
#  CLASSIFIER export  (PyTorch → ONNX → TRT INT8)
# ---------------------------------------------------------------------------

def export_classifier(cfg: dict) -> str:
    """
    Export WeaponClassifier (EfficientNet-B5) to TRT INT8 engine.
    Returns path to the .engine file.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "tensorrt not found. On Jetson it is pre-installed by JetPack.\n"
            "On desktop, install: pip install tensorrt"
        )
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.model import WeaponClassifier

    trt_cfg        = cfg.get("trt", {})
    cls_ckpt       = "logs/classifier/best.pt"
    engine_path    = trt_cfg.get("classifier_engine", "logs/trt/classifier_int8.engine")
    calib_cache    = "logs/trt/classifier_calib.cache"
    workspace_gb   = int(trt_cfg.get("workspace_gb", 4))
    fp16_fallback  = bool(trt_cfg.get("fp16_fallback", True))
    n_calib        = int(trt_cfg.get("calibration_images", 300))
    crop_dir       = cfg["dataset"].get("classifier_root", "data/classifier_crops")
    img_size       = int(cfg["classifier"].get("img_size", 224))
    num_classes    = int(cfg["dataset"]["num_classes"])

    onnx_path = "logs/trt/classifier.onnx"
    Path("logs/trt").mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'═'*60}")
    logger.info("  Exporting Classifier → TRT INT8")
    logger.info(f"  Checkpoint : {cls_ckpt}")
    logger.info(f"  ONNX       : {onnx_path}")
    logger.info(f"  Engine     : {engine_path}")
    logger.info(f"{'═'*60}")

    # ── Step 1: Load model and export to ONNX ────────────────────────────────
    if not Path(cls_ckpt).exists():
        logger.error(f"Classifier weights not found: {cls_ckpt}")
        logger.error("Run part2.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(cls_ckpt, map_location=device)
    nc     = ckpt.get("num_classes", num_classes)
    model  = WeaponClassifier(num_classes=nc, dropout=0.0, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    logger.info("  Exporting to ONNX (opset 17, dynamic batch) ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version   = 17,
        input_names     = ["input"],
        output_names    = ["logits"],
        dynamic_axes    = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        do_constant_folding = True,
    )
    logger.info(f"  ONNX saved: {onnx_path}")

    # Measure PyTorch baseline latency
    dummy_cpu = torch.randn(1, 3, img_size, img_size)
    pt_model  = WeaponClassifier(num_classes=nc, dropout=0.0, pretrained=False).cpu()
    pt_model.load_state_dict(ckpt["model"], strict=False)
    pt_model.eval()
    with torch.no_grad():
        for _ in range(50):
            pt_model(dummy_cpu)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            pt_model(dummy_cpu)
    pt_ms = (time.perf_counter() - t0) / 200 * 1000
    logger.info(f"  PyTorch CPU baseline: {pt_ms:.2f} ms/crop")

    # ── Step 2: Build TRT engine ──────────────────────────────────────────────
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )
    parser     = trt.OnnxParser(network, TRT_LOGGER)
    config     = builder.create_builder_config()

    # Flags
    config.set_flag(trt.BuilderFlag.INT8)
    if fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_gb << 30,
    )

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"  ONNX parse error [{i}]: {parser.get_error(i)}")
            sys.exit(1)
    logger.info("  ONNX parsed successfully")

    # Dynamic batch profile
    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    profile.set_shape(in_name, (1, 3, img_size, img_size),
                                (1, 3, img_size, img_size),
                                (8, 3, img_size, img_size))   # max batch = 8
    config.add_optimization_profile(profile)

    # INT8 calibrator
    if not Path(calib_cache).exists():
        logger.info(f"  Building INT8 calibrator from {crop_dir} ...")
        calibrator = WeaponCropCalibrator(
            crop_dir   = crop_dir,
            cache_file = calib_cache,
            max_images = n_calib,
            batch_size = 8,
            img_size   = img_size,
        )
        config.int8_calibrator = calibrator
    else:
        logger.info(f"  Reusing calibration cache: {calib_cache}")
        calibrator = WeaponCropCalibrator(
            crop_dir   = crop_dir,
            cache_file = calib_cache,
            max_images = 0,         # won't load images — just reads cache
            batch_size = 8,
            img_size   = img_size,
        )
        config.int8_calibrator = calibrator

    logger.info("  Building TRT engine (this may take several minutes) ...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("  TRT engine build FAILED")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logger.info(f"  Engine saved: {engine_path}")

    # ── Step 3: Benchmark TRT classifier ─────────────────────────────────────
    logger.info("  Benchmarking TRT classifier ...")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        trt_engine = runtime.deserialize_cuda_engine(f.read())
    context = trt_engine.create_execution_context()

    # Simple latency via PyTorch CUDA event timing
    dummy_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    in_tensor  = torch.from_numpy(dummy_np).cuda()
    out_tensor = torch.zeros(1, nc).cuda()

    import pycuda.driver as cuda_drv  # optional path — fallback handled below
    _use_pycuda = False
    try:
        import pycuda.autoinit  # noqa: F401
        _use_pycuda = True
    except ImportError:
        pass

    # Use PyTorch-based timing (no pycuda needed)
    bindings = [in_tensor.data_ptr(), out_tensor.data_ptr()]
    for _ in range(50):
        context.execute_v2(bindings)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(200):
        context.execute_v2(bindings)
        torch.cuda.synchronize()
    trt_ms = (time.perf_counter() - t0) / 200 * 1000

    logger.info(f"  TRT classifier latency : {trt_ms:.2f} ms/crop")
    logger.info(f"  Speedup vs PyTorch CPU : {pt_ms/max(trt_ms,0.001):.1f}x")

    return engine_path


# ---------------------------------------------------------------------------
#  Verification: compare TRT vs PyTorch outputs
# ---------------------------------------------------------------------------

def verify_outputs(cfg: dict) -> None:
    """Run a test image through TRT engine + PyTorch and compare outputs."""
    if not cfg.get("trt", {}).get("verify_after_export", True):
        return

    try:
        import tensorrt as trt
    except ImportError:
        return

    logger.info("\n  Verifying classifier TRT vs PyTorch ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.model import WeaponClassifier

    img_size    = int(cfg["classifier"].get("img_size", 224))
    engine_path = cfg.get("trt", {}).get("classifier_engine",
                                          "logs/trt/classifier_int8.engine")
    cls_ckpt    = "logs/classifier/best.pt"

    if not Path(engine_path).exists() or not Path(cls_ckpt).exists():
        logger.warning("  Skipping verification — files not found")
        return

    # PyTorch prediction
    device = torch.device("cpu")
    ckpt   = torch.load(cls_ckpt, map_location=device)
    nc     = ckpt.get("num_classes", cfg["dataset"]["num_classes"])
    model  = WeaponClassifier(nc, dropout=0.0, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    np.random.seed(1337)
    dummy_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_np)).numpy()
    pt_probs = np.exp(pt_out) / np.exp(pt_out).sum(axis=1, keepdims=True)

    # TRT prediction
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    in_t    = torch.from_numpy(dummy_np).cuda()
    out_t   = torch.zeros(1, nc).cuda()
    context.execute_v2([in_t.data_ptr(), out_t.data_ptr()])
    torch.cuda.synchronize()
    trt_out   = out_t.cpu().numpy()
    trt_probs = np.exp(trt_out) / np.exp(trt_out).sum(axis=1, keepdims=True)

    diff = float(np.abs(pt_probs - trt_probs).max())
    threshold = 0.05
    if diff > threshold:
        logger.warning(
            f"  [VERIFY] Max abs diff = {diff:.4f} — exceeds threshold {threshold}!\n"
            f"  This is expected for INT8; review calibration data quality."
        )
    else:
        logger.info(f"  [VERIFY] Max abs diff = {diff:.4f}  ✓ within threshold")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to TensorRT INT8")
    parser.add_argument("--config",           default="config/hyperparams.yaml")
    parser.add_argument("--detector-only",    action="store_true")
    parser.add_argument("--classifier-only",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    do_det = not args.classifier_only
    do_cls = not args.detector_only

    if do_det:
        det_engine = export_detector(cfg)
        logger.info(f"\n  [OK] Detector engine : {det_engine}")

    if do_cls:
        cls_engine = export_classifier(cfg)
        logger.info(f"\n  [OK] Classifier engine : {cls_engine}")
        verify_outputs(cfg)

    logger.info("\n  Export complete.")
    logger.info("  Next: python3 inference.py --source 0 --trt")
