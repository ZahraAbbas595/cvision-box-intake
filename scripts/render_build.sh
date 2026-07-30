#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

# Ultralytics declares the GUI OpenCV distribution as a dependency. Replace it
# after dependency resolution so the CPU-only service contains one cv2 wheel.
python -m pip uninstall -y opencv-python
python -m pip install \
  --no-deps \
  --force-reinstall \
  opencv-python-headless==5.0.0.93

if python -m pip show opencv-python >/dev/null 2>&1; then
  echo "opencv-python must not be installed on the Render service." >&2
  exit 1
fi

python - <<'PY'
import cv2
import torch

print(f"Verified torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"Verified opencv={cv2.__version__}, headless-only")
if torch.version.cuda is not None:
    raise SystemExit("Expected a CPU-only PyTorch build.")
PY
