# Draft CVision R&D Case Study

## Visual carton intake before barcode confirmation

CVision investigated whether one truck image could create a useful first-stage
intake record before operators scan individual barcodes. The research question
was intentionally narrow: detect visibly identifiable cartons, count accepted
detections, assign image-relative size classes, expose uncertainty, and preserve
annotated evidence.

## Approach

The prototype uses a truck-carton YOLOv8n detector behind FastAPI. OpenCV handles
validation and overlays; deterministic rules identify unusable images and risky
detection geometry. Streamlit provides upload/camera capture, original and
annotated views, count and size summaries, review guidance, and JSON download.
Production uses ONNX Runtime to fit Render's free 512 MB instance. Optional
Gemini observations remain advisory and cannot change counts or force review.

## Evidence

- Public UI and separately hosted API are accessible outside localhost.
- The locked external set contains 105 images and 8,107 cartons with no exact
  overlap against the fine-tuning train or validation splits.
- External object metrics are precision 89.9%, recall 84.7%, mAP50 89.0%, and
  mAP50-95 66.7%.
- Operational count metrics are 27.6% exact-count accuracy and 14.41 MAE.
- ONNX and PyTorch accepted counts match on all 105 external images.
- Automated validation covers upload failures, review rules, visual-review
  failure safety, frontend response validation, and runtime telemetry.

## What the evidence means

The system is useful as a transparent research prototype, especially on
front-facing stacks with clear boundaries. It is not reliable enough to replace
barcode confirmation or a manifest. Dense scenes with hundreds of small cartons
can undercount severely, while repeated edges and side faces can overcount. The
gap between strong mAP and modest exact-count accuracy demonstrates why model
confidence and detection metrics are not substitutes for operational trust.

## Operational value demonstrated

The prototype provides a fast visual estimate and a reviewable evidence record
before item identity is known. It makes failures visible instead of silently
correcting counts, separates advisory observations from authoritative routing,
and gives another developer a reproducible deployment and evaluation path.

## Before warehouse use

Required work includes a larger warehouse-specific dataset, camera and scene
validation, multi-view or tiled inference for dense loads, calibrated physical
measurement, authentication, secure evidence storage, monitoring, retention
rules, WMS/POD/DANI integration, and a feedback process based on confirmed
operator review outcomes.
