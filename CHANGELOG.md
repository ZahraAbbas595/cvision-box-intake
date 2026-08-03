# Changelog

All notable changes to this research prototype are documented in this file.

The format follows Keep a Changelog, and releases use Semantic Versioning.

## [Unreleased]

### Added

- Fine-tuned YOLOv8n carton detector and selected model artifact.
- Curated count-evaluation dataset, threshold sweep, leakage audit, and runtime
  benchmark utilities.
- GitHub Actions quality gate for linting, formatting, scoped type checking, and
  targeted endpoint tests.

### Changed

- Replaced the YOLO-World detector with the fine-tuned carton detector.
- Set evidence-based operating thresholds to confidence 0.45 and NMS IoU 0.30.
