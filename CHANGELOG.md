# Changelog

All notable changes to this research prototype are recorded here so another
developer can understand or reproduce any major project stage.

The format follows Keep a Changelog. Entries identify the difference from the
preceding stage and reference the integration commit or pull request where
available.

## [Unreleased]

### Changed

- Replaced weakest-box operational confidence with average accepted-detection
  confidence and clarified its operator-facing label and contract.
- Made optional Gemini visual observations advisory-only so speculative visual
  concerns do not automatically route otherwise clean results to human review.
- Aligned count evaluation with deployed confidence and IoU settings and added
  exact, overcount, and undercount example reporting.
- Replaced Streamlit's deprecated `use_container_width` image option with the
  supported stretch-width API.

### Added

- Final sprint handoff artifacts: operational error analysis with three
  annotated examples, draft R&D case study, and demonstration script.

### Removed

- Removed the redundant sprint-completion checklist; required handoff evidence
  remains in the README and dedicated technical documents.
- Removed automatic human-review routing based only on detecting 12 or more
  cartons, because high carton counts are normal in the truck-intake workflow.

## [0.9.0] - 2026-08-18

### Added

- Separate free-tier staging deployments for the FastAPI backend and Streamlit
  frontend, connected to the `stage` branch and verified with API and browser
  upload smoke tests.
- Environment-configurable model paths and identity metadata for safely testing
  candidate checkpoints without replacing the production model.
- A reproducible external truck-carton test-set builder that combines Roboflow
  splits, prevents filename collisions, and normalizes source labels to `Box`.
- Per-source external evaluation of exact carton counts and mean absolute error.
- An application-level PyTorch/ONNX count-parity gate for candidate models.

- An opt-in, fail-safe vision reviewer that can add allow-listed visual-risk
  reasons for occlusion, reflections or shadows, confusing scenes, damaged
  cartons, and non-carton lookalikes, plus concise operator guidance. It never
  changes detections or counts and falls back to deterministic review routing.
- A validated `other_visual_risk` escape hatch with a required free-text
  `novel_reason` and request-scoped structured logging for discovering and
  monitoring previously unknown visual failure modes.
- Gemini Flash-Lite free-tier integration for optional visual review, replacing
  the earlier OpenAI-specific API key and Responses API request path.
- Gemini-compatible schema constraints and safe structured provider-error logs
  containing HTTP status, provider status, model, and message without credentials.
- The established `responseMimeType` and `responseSchema` generation fields for
  compatibility with the Gemini `v1beta generateContent` endpoint.
- Application-side exact-field validation in place of the unsupported
  `additionalProperties` provider-schema keyword.
- Consolidated operator presentation with one visual-review explanation and one
  suggested action, while retaining individual signals under technical details.
- Calibrated reviewer instructions that prefer known categories, reserve
  `other_visual_risk` for genuinely novel cases, and allow human confirmation of
  false positives without letting the AI alter counts.
- A Streamlit intake interface with original and annotated image views, count,
  size summary, confidence, plain-language review guidance, and JSON download.
- Frontend upload validation and safe cold-start, timeout, unavailable-service,
  API-error, and malformed-response handling.
- Public Streamlit Community Cloud deployment plus documented clear-scene and
  difficult-scene live acceptance evidence.
- Targeted tests for high-count review routing and optional process-memory
  telemetry, including malformed and unreadable Linux status data.
- A pull request template covering rationale, risk, contracts, documentation,
  automated checks, and smoke testing.

### Changed

- Advanced the release gate from `dev` to `stage`; promotion to `main` now
  requires staging sign-off. Optional Gemini review remains disabled in staging
  until its secret is configured and verified there.
- Aligned the root README with the truck-only counting scope, optional Gemini
  review contract, current repository layout, CI commands, and staging gate.
- Selected the truck-specific YOLOv8n checkpoint as the default local and
  build-time model; Git history retains the earlier checkpoint for rollback.
- Aligned candidate inference defaults at confidence `0.45` and IoU `0.30`.
- Declared the ONNX exporter dependency required by the reproducible export
  script.
- Reconciled deterministic quality review with advisory model output in the UI:
  quality-only cases now show the specific retake message without contradictory
  no-risk summaries, routine-processing guidance, or duplicate technical signals.
- Clarified reviewer handling of loose detection rectangles so overlap with part of
  a nearby non-carton object does not incorrectly invalidate a real carton detection.

- Added on-the-spot camera capture and specific retake/re-upload guidance for
  blurry, dark, bright, or low-resolution images in the Streamlit interface.
- Constrained the pre-analysis image preview and placed original and annotated
  evidence side by side on wider screens for easier comparison.
- Made optional Linux memory telemetry fail safely when `/proc/self/status`
  cannot be read or contains malformed values.
- Extended CI to run on `stage` and type-check the stable pure-service modules.

### Removed

- The redundant handoff/cleanup document, obsolete example overlays, general
  carton evaluation dataset and utilities, superseded model checkpoint, and
  legacy evidence documents tied to the earlier scope. Git history retains
  these artifacts if historical investigation is required.

## [0.8.0] - 2026-08-04

### Project Handoff and Week 2 Completion

### Added

- Root project README with setup, operation, evaluation, deployment, Git, and
  handoff guidance.
- Reproducibility and cleanup inventory for periodic repository maintenance.
- Public-boundary docstrings for the application and supported utility scripts.
- Module-level docstrings across supported application, evaluation, and
  operational Python modules.
- Conservative resolution, blur, and exposure signals for review-only image
  quality routing.
- Five curated annotated baseline examples documenting the COCO and YOLO-World
  model-selection evidence.

### Changed

- Clarified which files are supported workflow components, generated artifacts,
  historical spikes, and unresolved sprint gaps.
- Completed the Week 2 error taxonomy, visual evidence links, review-rule
  coverage notes, and endpoint/review regression tests.
- Expanded the root README into a complete architecture, API, configuration,
  evaluation, deployment, troubleshooting, and contribution guide.
- Synchronized the package and API service version at `0.8.0` and documented
  the rule that meaningful README and changelog updates move together.

### Removed

- Obsolete root-level `spike_detection.py` and `spike_world.py` experiments;
  their history remains available through Git and this changelog.

**Difference from 0.7.0:** completes the Week 2 reliability evidence, adds
review-only image-quality routing, preserves curated baseline overlays, removes
obsolete experiments, and turns the repository documentation into a complete
developer handoff. Integrated by PRs #13 through #18 (`d6acc290`, `ebc1ff4a`,
`28c98490`, `3f8df64a`, `b47259ca`, `3ac201d3`) and PR #19 (`f6186a3`).

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
