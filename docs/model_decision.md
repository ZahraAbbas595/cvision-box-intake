# Model Decision

## What We Tested

### Attempt 1: YOLOv8n (COCO pretrained)
- Model: yolov8n.pt
- Classes: 80 COCO classes (no box, carton, or parcel class)
- Result: 0 detections on 4/6 images. Detected 'couch' and 'refrigerator' on box images. Detected boxes as 'suitcase' on img_006 only.
- Conclusion: COCO has no cardboard box class. Unusable for this task.

### Attempt 2: YOLO-World (yolov8s-worldv2.pt)
- Model: yolov8s-worldv2.pt (338MB)
- Classes: open-vocabulary via text prompt ['cardboard box', 'carton', 'package', 'parcel']
- Result: Detected cardboard boxes on all 6 images with conf=0.25 threshold
- Detections per image: img_001=1, img_002=3, img_003=13, img_004=2, img_005=1, img_006=23
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.1, inference ~100ms per image

## Decision: YOLO-World (Path A)

YOLO-World is chosen as the baseline detector.

Reason: It detects cardboard boxes zero-shot via text prompt with no fine-tuning required.
COCO YOLO failed completely on this task - it has no box/carton class and cannot be used.

## Known Issues Found on Day 1

- img_006 returned 23 detections - likely duplicate bounding boxes on same physical boxes.
  Needs IoU threshold and NMS tuning in Week 2.
- img_003 returned 13 detections - same overcounting pattern suspected.
- Model file is 338MB. Render free tier has 512MB RAM limit.
  Memory headroom is tight. Must measure on Day 2 before building further.

## Risk: Render Memory

YOLO-World (338MB) + FastAPI + Python runtime may exceed Render's 512MB RAM.
If it does, fallback options in order:
1. Shrink input resolution (640 -> 480)
2. Export to ONNX and serve with onnxruntime (saves ~100-150MB)

This will be measured on Day 2 with a real deployment.

## Model Source
- Model: yolov8s-worldv2.pt
- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
- Licence: AGPL-3.0
