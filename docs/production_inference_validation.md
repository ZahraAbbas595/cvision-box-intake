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

`scripts/export_onnx.py` exported the committed 6.2 MB PyTorch detector to an
ignored 12,238,574-byte ONNX file with input shape `[1, 3, 640, 640]` and output
shape `[1, 5, 8400]`.

The easy single-carton image produced effectively identical detections:

- PyTorch: box `[205, 267, 411, 453]`, confidence `0.954`
- ONNX Runtime: box `[206, 267, 410, 452]`, confidence `0.955`

Across all 31 curated images, 26 counts matched and five differed. ONNX therefore
requires its own threshold evidence instead of inheriting PyTorch settings by
assumption.

## ONNX Threshold Decision

`eval/tune_onnx_thresholds.py` swept confidence `0.35` through `0.55` and NMS
IoU `0.20` through `0.50` on the 24-image reviewed count set.

The selected Render settings are confidence `0.48` and IoU `0.25`:

| Metric | ONNX result |
| --- | ---: |
| Exact-count images | 18 / 24 |
| Exact-count accuracy | 75.0% |
| Mean absolute count error | 0.75 |
| Overcount images | 3 |
| Undercount images | 3 |

No tested ONNX threshold recovered the PyTorch operating point's 19 / 24 exact
counts. The deployment tradeoff is explicit: lower count accuracy is accepted
only for the free-tier memory experiment, with human review routing preserved.

## Local Isolated Runtime Benchmark

Each backend was measured in a fresh process over all 31 curated images:

| Measurement | ONNX Runtime | PyTorch |
| --- | ---: | ---: |
| RSS before model load | 35.8 MB | 35.8 MB |
| RSS after model load | 71.1 MB | 424.0 MB |
| Peak RSS | 228.1 MB | 713.0 MB |
| Model load | 0.2395 s | 2.9532 s |
| Cold inference | 0.1401 s | 1.9191 s |
| Warm mean | 0.1257 s | 0.0851 s |

The local ONNX peak is 283.9 MB below the 512 MB free-tier gate. This makes
ONNX eligible for a live Render test; it does not replace the required live
measurement because operating-system and platform overhead differ.

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

After the branch is deployed, run:

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
