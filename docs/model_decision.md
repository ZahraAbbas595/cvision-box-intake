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

## Render Memory Measurement
Render free tier does not expose memory or CPU metrics (paid plans only).
Memory usage could not be measured directly.
Model size: 338MB. No out-of-memory errors observed during deployment.
Cold start time: approximately 50-60 seconds on free tier (service spins down after inactivity).
This is expected behaviour and will be handled in the Streamlit UI with a loading message.

## CPU Latency

YOLO-World was benchmarked locally on CPU with PyTorch restricted to one
thread. Model loading and text-class setup happened before the timed region;
each value is wall-clock time for `model.predict`.

### Initial five-image run

| Evaluation image | Wall-clock latency |
| --- | ---: |
| `img_001.jpg` | 3.4940 s |
| `img_002.jpg` | 1.4680 s |
| `img_003.png` | 0.8762 s |
| `img_006.jpg` | 0.7393 s |
| `img_027.jpg` | 0.7914 s |
| **Mean** | **1.4738 s** |

### Full 31-image run

The follow-up run covered every evaluation image in filename order.

| Statistic | Wall-clock latency |
| --- | ---: |
| Mean | 0.2565 s |
| Median | 0.2315 s |
| p95 | 0.5061 s |
| Maximum (`img_011.jpg`) | 0.7790 s |
| First prediction (`img_001.jpg`) | 0.5061 s |
| Warm mean (predictions 2-31) | 0.2482 s |
| Warm median (predictions 2-31) | 0.2290 s |

The large difference between the two local runs indicates that initialization,
OS caching, and machine load materially affect these measurements. The slow
`img_001.jpg` result in the initial run is not evidence of scene complexity:
that image contains one box and was the first prediction after model setup.

These local results only show that YOLO-World can run with one PyTorch CPU
thread on the development laptop. One laptop thread does not simulate Render's
throttled, shared 0.1 vCPU, so CPU speed is not cleared as a deployment risk.

After Render deploys from `dev`, measure a wall-clock POST to the live
`/v1/box-intake/infer` endpoint with at least one easy and one crowded image.
Record cold-start time separately from warm inference time. Treat those live
measurements, along with whether the worker survives, as the decision gate for
ONNX or a smaller YOLOv8n model. ONNX remains worth testing for both memory and
latency until that live evidence is available.

## Live Render Inference Test

Commit `d3ebbc1` was deployed successfully from `dev` on the existing Render
Free instance. The service reached the `Live` state and `GET /health` returned
HTTP 200 in 0.563 seconds.

The first live inference request used the easy single-box `img_001.jpg` image.
It returned HTTP 502 after 20.359 seconds. The application produced no completed
request log, which is consistent with the worker being terminated while loading
YOLO-World under the free instance's memory limit.

A crowded-image request and warm-latency measurement were not run because the
worker did not survive the easy request. There is therefore no meaningful warm
inference latency for the current deployment.

This live result is the deployment decision gate: YOLO-World with the PyTorch
runtime does not fit the current free service. Proceed with a smaller model or
an ONNX runtime before further latency tuning. The build logs also show that the
unrestricted `torch` dependency installs CUDA packages on this CPU-only service,
creating an approximately 2.8 GB build cache; the replacement deployment should
remove those unnecessary dependencies.

## CPU-only Runtime Experiment

Started: 2026-07-30 12:50:13 +05:00.

This experiment changes dependencies only: CPU-only PyTorch and Torchvision,
plus headless OpenCV as the final installed OpenCV distribution. It does not
change input size, model weights, inference thresholds, or application logic.

Completed: 2026-07-30 13:05:55 +05:00 (15 minutes 42 seconds).

Render's build verification confirmed `torch==2.12.1+cpu` with
`torch.version.cuda == None` and OpenCV 5.0.0 from the headless distribution
only. The service deployed successfully and `GET /health` returned HTTP 200 in
0.592 seconds.

The same easy `img_001.jpg` inference request returned HTTP 502 after 38.368
seconds. Render then reported that the instance exited with status 137. Removing
unused CUDA libraries and GUI OpenCV therefore did not close the 512 MB memory
gap. The remaining blocker is the YOLO-World/model runtime footprint, which
justifies moving to the timeboxed ONNX attempt or the smaller YOLOv8n path.
