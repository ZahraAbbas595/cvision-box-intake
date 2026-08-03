# Production Inference Validation

**Date:** August 3, 2026

## Constraint

The production backend must run on Render's free web-service plan. The deployment
is not allowed to fall back to a paid instance. The free-tier memory gate is
512 MB peak process RSS during a real inference request.

Generated ONNX files, evaluation CSVs, response captures, datasets, and arbitrary
weights remain uncommitted.

## Current Live Render Baseline

The existing `dev` deployment was tested before this branch was deployed:

| Check | Result |
| --- | ---: |
| `GET /health` | HTTP 200 after 52.909 s cold wake-up |
| `GET /version` | HTTP 200, service 0.1.0, PyTorch model contract |
| First `img_001.jpg` inference | HTTP 200 after 66.051 s |
| Visible cartons | 1 |
| Server memory telemetry | Not available in the deployed contract |

The worker survived the request, but the cold latency is not suitable evidence
that the service will remain reliable under the free memory limit.

## ONNX Export and Parity

The first export used a fixed `[1, 3, 640, 640]` input. The PyTorch predictor,
however, uses stride-aligned rectangular inputs for single images: examples in
the curated set include `[1, 3, 448, 640]`, `[1, 3, 640, 448]`, and
`[1, 3, 352, 640]`. That preprocessing mismatch explained the five count-parity
differences even though the exported weights and basic decoding were correct.

The easy single-carton image produced effectively identical detections:

- PyTorch: box `[205, 267, 411, 453]`, confidence `0.954`
- ONNX Runtime: box `[206, 267, 410, 452]`, confidence `0.955`

`scripts/export_onnx.py` now exports dynamic spatial dimensions, and the ONNX
runtime path applies the same stride-32 rectangular letterboxing. At the same
confidence `0.45` and IoU `0.30` operating point, all 31 curated-image counts
match PyTorch. Accuracy metrics below use the 24 images with reviewed counts. The dynamic ONNX contract is input
`[batch, 3, height, width]` with a dynamic anchor dimension in the output.

## ONNX Threshold Decision

After matching preprocessing, `eval/tune_onnx_thresholds.py` swept confidence
`0.35` through `0.55` and NMS IoU `0.20` through `0.50` again on the 24-image
reviewed count set.

The selected Render settings are confidence `0.47` and IoU `0.25`:

| Metric | ONNX result |
| --- | ---: |
| Exact-count images | 19 / 24 |
| Exact-count accuracy | 79.2% |
| Mean absolute count error | 0.79 |
| Overcount images | 3 |
| Undercount images | 2 |

This recovers the PyTorch operating point's 19 / 24 exact counts while retaining
the lightweight ONNX Runtime deployment. The remaining five errors are model or
scene failures shared with PyTorch: one crowded overcount, two fragmented-carton
overcounts, and two missed cartons. They are not caused by ONNX conversion.

## Local Isolated Runtime Benchmark

Each backend was measured in a fresh process over all 31 curated images:

| Measurement | Dynamic ONNX Runtime | PyTorch |
| --- | ---: | ---: |
| RSS before model load | 35.8 MB | 35.8 MB |
| RSS after model load | 70.7 MB | 424.0 MB |
| Peak RSS | 200.5 MB | 713.0 MB |
| Model load | 0.2436 s | 2.9532 s |
| Cold inference | 0.0979 s | 1.9191 s |
| Warm mean across mixed shapes | 0.2603 s | 0.0851 s |

The dynamic ONNX peak is 311.5 MB below the 512 MB free-tier gate. Mixed-shape
evaluation is slower than the fixed-shape benchmark because the runtime handles
several tensor shapes, but memory remains well within the free-tier constraint.
This local result does not replace a post-deployment live Render measurement.

## Targeted Count-Risk Experiment

The production response does not automatically merge detections. It adds review
signals only:

- `high_detection_count` at 12 or more detections
- `possible_fragmented_detections` only for two-detection scenes whose aligned
  boxes have IoU at or below `0.02`

At the selected ONNX thresholds, these signals flagged two of six count-error
images and zero exact-count images. They caught the window false-positive scene
and the confirmed two-fragment damaged-carton scene. The existing edge review
rule remains responsible for the confirmed `img_030` multi-fragment case.

## Free Render Deployment Gate

`render.yaml` pins `plan: free`, selects the ONNX backend, and sets the measured
ONNX thresholds. `scripts/render_build.sh` exports ONNX from the committed model,
then removes Torch and Ultralytics before the service starts.

The merged deployment was validated on the existing Render Free web service on
2026-08-03. The service reported backend `onnx`; `img_001.jpg` returned HTTP 200
with one carton and 169.4 MB peak RSS. Three consecutive requests using crowded
`img_003.png` returned 23 cartons with the expected edge and high-count review
reasons. Peak RSS stabilized at 194.4 MB, leaving 317.6 MB of headroom below the
512 MB gate. Server processing time for those warm crowded requests was 883-1060
ms, with no worker restart or memory growth after the second request.

The deployment gate can be repeated with:

```bash
python scripts/smoke_render.py \
  https://cvision-box-intake-api.onrender.com \
  eval/dataset/images/img_001.jpg
```

The smoke command fails if the deployed backend is not ONNX, Linux peak RSS is
unavailable, or peak RSS reaches 512 MB. If the gate fails, the next actions are
free-tier optimizations such as reducing inference size, disabling unnecessary
ONNX Runtime memory arenas, and removing remaining runtime dependencies. A paid
Render plan is not an accepted fallback.
