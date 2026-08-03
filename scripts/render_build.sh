#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python -m pip install onnx==1.19.0
python scripts/export_onnx.py

# Build with the committed PyTorch model, then remove the export-only stack so
# the free 512 MB service imports ONNX Runtime without Torch or Ultralytics.
python -m pip uninstall -y torch torchvision ultralytics onnx opencv-python
python -m pip install --force-reinstall -r requirements-render.txt

if python -m pip show torch ultralytics opencv-python >/dev/null 2>&1; then
  echo "Export-only or GUI inference packages remain installed." >&2
  exit 1
fi

python - <<'PY'
from pathlib import Path

import cv2
import onnxruntime

model_path = Path("models/carton_yolov8n_best.onnx")
if not model_path.exists():
    raise SystemExit(f"Missing exported model: {model_path}")
print(f"Verified onnxruntime={onnxruntime.__version__}")
print(f"Verified opencv={cv2.__version__}, headless-only")
print(f"Verified model={model_path}, bytes={model_path.stat().st_size}")
PY
