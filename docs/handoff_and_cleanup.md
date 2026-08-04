# Handoff and Cleanup

## Purpose

This document defines the supported reproducible workflow and separates it from
local artifacts or historical experiments. Cleanup must be evidence-based and
must not remove data that is still required for evaluation or project history.

## Supported Workflow

The handoff path is:

1. Create a Python 3.11 virtual environment.
2. Install `requirements.txt` and the `dev` optional dependencies.
3. Run the FastAPI backend with the committed PyTorch model for development.
4. Run lint, formatting, scoped type checks, and tests.
5. Use the reviewed manifest for count evaluation.
6. Export ONNX at build time for Render Free production.

Exact commands are maintained in the root `README.md`.

## Tracked Files That Must Remain

- `models/carton_yolov8n_best.pt`: selected reproducible model input.
- `eval/dataset/images/`: the 31-image evaluation evidence set.
- `eval/dataset/ground_truth.json`: current reviewed count source of truth.
- `scripts/export_onnx.py`: reproducible production export path.
- `docs/`: model, evaluation, deployment, and limitation evidence.

## Generated or Local-Only Files

The following must remain ignored and uncommitted:

- exported `.onnx` files;
- training runs and model checkpoints other than the selected model;
- `eval/results/` CSV files, overlays, contact sheets, and reports;
- API response captures;
- Python, Ruff, mypy, and pytest caches;
- virtual environments, editable-install metadata, and local `.env` files.

The current workstation also contains protected untracked response captures and
an untracked memory test. They are intentionally left untouched until their
owner explicitly decides their disposition.

## Historical Cleanup

The obsolete `spike_detection.py` and `spike_world.py` exploratory scripts were
removed after confirming they were not referenced by the supported inference or
evaluation workflow. Their implementation and purpose remain recoverable from
Git history and `CHANGELOG.md`.

## Known Repository Gap

The sprint requires a Streamlit interface, but no tracked Streamlit application
currently exists. The local `ui/` directory is empty and Git does not preserve
empty directories. The backend is reproducible; the frontend deliverable still
requires a focused implementation and deployment.

## Periodic Cleanup Checklist

Before each handoff or release:

1. Start from an updated `dev` branch and inspect `git status`.
2. Classify every untracked file before deleting or adding it.
3. Confirm generated artifacts remain covered by `.gitignore`.
4. Run the documented quality checks from a clean environment where practical.
5. Confirm the evaluation manifest parses and contains the expected image set.
6. Confirm no secrets, datasets outside the approved evaluation set, exported
   models, response captures, or generated metrics are staged.
7. Review root-level scripts and dependencies for obsolete experimental paths.
8. Update `README.md` and `CHANGELOG.md` together when behavior, setup,
   operation, evaluation, deployment, limitations, or project status changes.
9. Merge through a pull request; never clean up directly on `dev` or `main`.

## Release Readiness

Before promoting `dev` to `main`, verify:

- backend setup and tests work from the README;
- the Streamlit gap is either completed or explicitly accepted;
- deployment URLs and environment variables are current;
- full-set and supported-scene metrics are reported honestly;
- limitations do not describe the prototype as production-ready;
- the branch contains no generated evaluation or response artifacts.

