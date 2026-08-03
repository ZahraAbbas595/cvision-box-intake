# Carton Detector Fine-Tuning Findings

## Decision

The project now uses a fine-tuned YOLOv8n detector as the candidate replacement
for the YOLO-World baseline.

YOLO-World established that open-vocabulary detection could find cartons, but
it overcounted crowded scenes and exited with status 137 on Render's free
instance. The fine-tuned model is much smaller and improves local carton
detection, but its PyTorch runtime still exceeds the free instance's measured
memory budget.

This remains a research prototype. The results below do not establish
warehouse readiness.

## Training Data and Configuration

- Source: Roboflow Universe project `oscd-efzvm`, version `dataset`
- Dataset licence: CC BY 4.0
- Export date: 30 July 2026
- Class: `Carton`
- Splits: 5,846 train, 1,670 validation, 833 test images
- Training model: YOLOv8n pretrained weights
- Task served by the application: object detection
- Input size: 640 pixels
- Batch size: 8
- Epochs: 50
- Best epoch: 50
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

The Roboflow labels contain segmentation polygons. Ultralytics derived
bounding boxes from those polygons for detection training because the service
contract requires box locations and counts rather than masks.

The selected deployment artifact is
`models/carton_yolov8n_best.pt` (6,220,323 bytes). It is committed because it
is well below GitHub's 100 MB single-file limit and makes deployment
reproducible. Arbitrary `.pt` outputs and the complete `runs/` directory remain
ignored.

## Roboflow Split Metrics

Validation after training:

| Metric | Value |
| --- | ---: |
| Precision | 0.951 |
| Recall | 0.906 |
| mAP50 | 0.964 |
| mAP50-95 | 0.842 |

Test evaluation of `best.pt`:

| Metric | Value |
| --- | ---: |
| Images | 833 |
| Instances | 16,204 |
| Precision | 0.945 |
| Recall | 0.908 |
| mAP50 | 0.966 |
| mAP50-95 | 0.846 |
| GPU inference | 9.4 ms/image |

These metrics are optimistic because the downloaded split contains
near-duplicate scenes across train, validation, and test.

## Cross-Split Leakage Audit

`scripts/audit_dataset_leakage.py` fingerprints all 8,349 images. It found no
byte-identical files, but normalized visual comparison found:

| Comparison | Near-identical candidates (pixel MAE <= 3) |
| --- | ---: |
| Train to validation | 149 |
| Train to test | 91 |
| Validation to test | 29 |

Manual inspection confirmed that at least one closest train/test pair depicts
the same photograph with different encoded pixels and filenames. The
Roboflow test result must therefore not be presented as an independent
generalization estimate.

A future dataset release should group visually related source images before
splitting. The independent curated evaluation set is the primary operational
evidence for this sprint.

## Independent Count Evaluation

The primary metric set contains 24 reviewed images. The counting rule is:

> Annotate a carton whenever its visible region is sufficient to identify it
> confidently as a distinct physical carton. Include partial, border-cropped,
> damaged, and wrapped-but-visually-identifiable cartons regardless of the
> exact visible percentage, and annotate only the visible area. Exclude
> extremely small or ambiguous fragments that cannot be reliably separated as
> individual cartons.

Seven dense scenes are reserved for a second independent count and qualitative
error analysis. They are not silently assigned uncertain labels.

Initial result at confidence 0.35 and NMS IoU 0.50:

| Metric | Result |
| --- | ---: |
| Exact-count accuracy | 58.3% |
| Mean absolute count error | 1.13 |
| Overcount images | 8 |
| Undercount images | 2 |

The worst case was `img_003.png`: 27 predictions for 12 visible cartons. This
shows why strong mAP does not imply reliable counting.

## Threshold Selection

`eval/tune_thresholds.py` sweeps confidence from 0.20 to 0.60 and NMS IoU from
0.30 to 0.70. It performs one-image inference, then reapplies filtering and NMS
to mirror the API's one-request-at-a-time CPU behavior.

Selected thresholds:

- Confidence: 0.45
- NMS IoU: 0.30

Result:

| Metric | Result |
| --- | ---: |
| Exact-count accuracy | 79.2% |
| Mean absolute count error | 0.83 |
| Overcount images | 3 |
| Undercount images | 2 |

Confidence 0.55 lowered mean absolute error to 0.75 but reduced exact-count
accuracy to 70.8% and increased undercount images to 4. Missing physical boxes
is the more serious intake failure, so 0.45/0.30 is the current operational
choice.

## Runtime and Deployment Gate

`scripts/benchmark_model_runtime.py` measured the exact deployment weight over
all 31 curated images with one CPU thread:

| Measurement | Result |
| --- | ---: |
| RSS before model load | 412.7 MB |
| RSS after model load | 432.0 MB |
| Peak RSS during inference | 697.7 MB |
| Model load | 0.258 s |
| Cold inference | 8.208 s |
| Warm mean | 0.375 s |
| Warm p95 | 0.646 s |

The local FastAPI HTTP smoke test passed `/health`, `/version`, and a real
single-box `/infer` request. The cold request took approximately 5.9 seconds.

The measured 697.7 MB peak exceeds Render free's 512 MB budget. A direct
PyTorch deployment is therefore expected to repeat the previous status-137
failure. ONNX remains the next timeboxed deployment experiment. The first ONNX
dependency installation made no progress within ten minutes and was stopped;
no ONNX result is claimed.

## Annotation Warnings

Ultralytics ignored three training images with out-of-bounds coordinates and
removed one duplicate annotation from each of five images. The validation and
test scans reported zero corrupt annotations.

## Current Limitations

- Cross-split near-duplicates inflate the Roboflow validation and test metrics.
- Exact-count accuracy is 79.2% on the reviewed primary set.
- Dense stacks still produce severe duplicate detections.
- Seven crowded scenes need a second independent ground-truth count.
- PyTorch inference exceeds the current free Render memory budget.
- Physical size cannot be inferred reliably from a single image; application
  size classes remain relative to frame area.
- Damage recognition is outside the current model's scope.

## Next Evidence Gate

1. Obtain a second count for the seven crowded evaluation images.
2. Complete the timeboxed ONNX memory/latency comparison.
3. Deploy the reviewed detector branch to Render only after the runtime choice
   is made.
4. Record cold start, warm inference, and worker survival from real requests.
5. Keep human review enabled for crowded, overlapping, edge-truncated, and
   low-confidence scenes.
