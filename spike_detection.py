import os
import sys
from pathlib import Path

# Check torch first
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU - will be slow")
except ImportError:
    print("ERROR: torch not installed")
    sys.exit(1)

from ultralytics import YOLO
import cv2

# Paths
IMAGE_DIR = Path("eval/dataset/images")
OUTPUT_DIR = Path("eval/results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load pretrained COCO model
print("\nLoading YOLOv8n (COCO pretrained)...")
model = YOLO("yolov8n.pt")  # downloads automatically on first run

# COCO classes nearest to boxes - we check what actually fires
TARGET_CLASSES = None  # None = all classes, we want to see everything

print(f"\nRunning on {len(list(IMAGE_DIR.iterdir()))} images...\n")
print("=" * 60)

results_summary = []

for img_path in sorted(IMAGE_DIR.iterdir()):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    print(f"\nImage: {img_path.name}")

    # Run inference
    results = model(str(img_path), conf=0.25, iou=0.5, verbose=False)
    result = results[0]

    # Get all detections
    boxes = result.boxes
    detections = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"class": cls_name, "confidence": round(conf, 3)})

    # Print what fired
    if detections:
        print(f"  Detections ({len(detections)} total):")
        for d in detections:
            print(f"    {d['class']:20s}  conf={d['confidence']}")
    else:
        print("  NO DETECTIONS at conf=0.25")

    # Save annotated image
    annotated = result.plot()
    out_path = OUTPUT_DIR / f"annotated_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved: {out_path}")

    results_summary.append({
        "image": img_path.name,
        "detection_count": len(detections),
        "detections": detections
    })

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for r in results_summary:
    print(f"{r['image']:20s}  {r['detection_count']} detections")

print("\nDone. Check eval/results/ for annotated images.")
