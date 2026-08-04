# Changelog

All notable changes to this research prototype are recorded here so another
developer can understand or reproduce any major project stage.

The format follows Keep a Changelog. Entries identify the difference from the
preceding stage and reference the integration commit or pull request where
available.

## [Unreleased]

### Added

- Root project README with setup, operation, evaluation, deployment, Git, and
  handoff guidance.
- Reproducibility and cleanup inventory for periodic repository maintenance.
- Public-boundary docstrings for the application and supported utility scripts.
- Module-level docstrings across supported application, evaluation, and
  operational Python modules.
- Conservative resolution, blur, and exposure signals for review-only image
  quality routing.

### Changed

- Clarified which files are supported workflow components, generated artifacts,
  historical spikes, and unresolved sprint gaps.
- Completed the Week 2 error taxonomy, visual evidence links, review-rule
  coverage notes, and endpoint/review regression tests.

### Removed

- Obsolete root-level `spike_detection.py` and `spike_world.py` experiments;
  their history remains available through Git and this changelog.

## [0.7.0] - 2026-08-04

### Complete Human-Reviewed Evaluation

- Reviewed all 31 evaluation images and corrected the count manifest.
- Recorded 58.1% full-set and 60.0% supported-scene exact-count accuracy.
- Retained the reflection example as an unsupported scene instead of deleting
  it from the dataset.
- Added five representative sprint examples and updated error categories.

**Difference from 0.6.0:** replaces the optimistic partial 24-image evaluation
with complete reviewed evidence. Integrated by PR #12 (`3a6b7c42`).

## [0.6.0] - 2026-08-03

### Dynamic ONNX Parity

- Exported ONNX with dynamic spatial dimensions.
- Matched PyTorch stride-32 rectangular preprocessing in ONNX Runtime.
- Selected Render thresholds confidence `0.47` and IoU `0.25`.
- Verified live Render Free memory stayed far below the 512 MB limit.

**Difference from 0.5.0:** removes ONNX/PyTorch preprocessing count differences
without adding Torch to production. Integrated by PRs #10 and #11
(`d12999a6`, `265179c3`).

## [0.5.0] - 2026-08-03

### Free-Tier ONNX Production

- Added a pure ONNX Runtime production backend and build-time export.
- Added lightweight Render dependencies and lazy backend imports.
- Added request IDs, structured logs, model-failure responses, and RSS telemetry.
- Added conservative count-risk review signals and live smoke testing.

**Difference from 0.4.0:** replaces the memory-heavy PyTorch production runtime
with a Render Free-compatible ONNX path. Integrated by PRs #8 and #9
(`b89c11b5`, `6ab9fad6`).

## [0.4.0] - 2026-08-03

### Fine-Tuned Carton Detector and Quality Gate

- Added the selected fine-tuned YOLOv8n carton model.
- Added the curated evaluation dataset, threshold sweeps, leakage audit, and
  runtime benchmarks.
- Added reviewed error analysis and targeted count-risk tests.
- Added GitHub Actions lint, formatting, type-checking, and test gates.

**Difference from 0.3.0:** replaces generic/zero-shot detection with a
carton-specific model and reproducible evidence. Integrated by the fine-tuning
series and PR #7 (`01074bb4`, `2f71d015`).

## [0.3.0] - 2026-07-30

### CPU Deployment Recovery

- Switched deployment dependencies toward CPU-only inference.
- Documented Render failures and out-of-memory evidence.
- Preserved explicit limitations rather than claiming a successful deployment.

**Difference from 0.2.0:** focuses on deployment reliability and records why the
initial runtime could not fit the target service. Integrated by PRs #3–#5
(`39059efe`, `ca56e223`, `c06f942c`).

## [0.2.0] - 2026-07-30

### FastAPI Intake Service

- Added health, version, and image-inference endpoints.
- Added validation for empty, unsupported, corrupt, and oversized uploads.
- Added structured detections, size classes, review metadata, and annotated
  image evidence.
- Added initial endpoint tests and deployment configuration.

**Difference from 0.1.0:** turns the local detection experiment into a callable
service with an operational response contract. Integrated by PR #2
(`d3ebbc12`).

## [0.1.0] - 2026-07-24

### Initial Detection Spike

- Added YOLO-World zero-shot carton detection experiments.
- Established evaluation image and generated-result directories.

**Difference from repository initialization:** introduces the first working
computer-vision experiment. Commit `3b674639`.

## [0.0.1] - 2026-07-23

### Repository Initialization

- Added the initial repository structure, ignore rules, and sample environment.

Commit `db67f221`.
