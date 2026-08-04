# CVision Box Intake

CVision Box Intake is a research prototype that detects and counts visible
cardboard cartons in an uploaded image, estimates rough size classes, flags
results that need human review, and returns an annotated evidence record.

The project demonstrates an end-to-end computer-vision workflow. It is not a
production warehouse system and must not be presented as client-ready.

## Current Status

- FastAPI backend: implemented and deployed on Render Free.
- Production inference: ONNX Runtime on CPU.
- Human-reviewed evaluation set: 31 images.
- Full-set exact-count accuracy: 18/31 (58.1%).
- Supported-scene exact-count accuracy: 18/30 (60.0%).
- Streamlit frontend: not currently tracked in this repository.
- Operational model: review-assisted prototype, not autonomous counting.

Live backend:

- API: <https://cvision-box-intake-api.onrender.com>
- Health: <https://cvision-box-intake-api.onrender.com/health>
- Version: <https://cvision-box-intake-api.onrender.com/version>
- Interactive API docs: <https://cvision-box-intake-api.onrender.com/docs>

## What the API Returns

`POST /v1/box-intake/infer` returns:

- visible carton count;
- bounding boxes and confidence values;
- rough `small`, `medium`, or `large` size classes;
- human-review status and reasons;
- request, service, model, timing, and memory metadata;
- a base64-encoded annotated evidence image.

One accepted, non-suppressed detection represents one visible carton. Review
rules warn an operator about risk; they never merge detections or silently
change the count.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | FastAPI application, schemas, inference, review rules, and telemetry. |
| `tests/` | Endpoint, detector, and count-risk tests. |
| `eval/` | Curated images, reviewed counts, and reproducible evaluation utilities. |
| `scripts/` | Training, export, benchmarking, leakage audit, and smoke-test tools. |
| `docs/` | Model decisions, error analysis, deployment evidence, and handoff notes. |
| `models/` | Selected committed PyTorch model; generated ONNX files stay ignored. |
| `render.yaml` | Render Free backend deployment blueprint. |

## Local Setup

Requirements:

- Python 3.11;
- Git;
- enough local memory for the PyTorch development dependencies.

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
Copy-Item sample.env .env
```

Environment variables are documented in `sample.env`. Local development uses
the committed PyTorch model by default. Render uses ONNX Runtime with
`MODEL_BACKEND=onnx`, confidence `0.47`, and IoU `0.25`.

## Run the Backend

```powershell
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, or submit an image with PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/box-intake/infer" `
  -F "file=@eval/dataset/images/img_001.jpg"
```

## Quality Checks

Run the same supported checks used by CI:

```powershell
python -m ruff check app tests
python -m ruff format --check app tests
python -m mypy app/config.py app/schemas.py
python -m pytest -q
```

## Evaluation

The source of truth for reviewed counts is
`eval/dataset/ground_truth.json`. Current findings and all incorrect counts are
documented in `docs/error_analysis.md`.

Generated CSV files, overlays, response captures, exported ONNX models, and
other evaluation artifacts belong under ignored paths and must not be
committed.

Important limitations:

- dense stacks can undercount small or occluded cartons;
- damaged cartons can fragment into multiple detections;
- rectangular backgrounds can create false positives;
- stylized lighting and open cartons can be missed;
- reflection-heavy scenes are unsupported;
- size classes are image-relative estimates, not physical measurements.

## ONNX Production Path

Render builds the production model from the committed PyTorch checkpoint:

```powershell
python scripts/export_onnx.py
```

`scripts/render_build.sh` performs the export, installs the lightweight Render
runtime, and removes Torch/Ultralytics before service startup. Do not commit the
exported `.onnx` file.

See `docs/production_inference_validation.md` for memory and deployment
evidence.

## Git Workflow

- `main`: stable demonstration branch.
- `dev`: integrated development branch.
- Feature work: descriptive branches such as `feat/...`, `fix/...`, or
  `docs/...`.
- All changes reach `dev` through reviewed pull requests.
- The stable sprint release moves from `dev` to `main` through a separate pull
  request.

Never commit directly to `dev` or `main`.

## Handoff

- `CHANGELOG.md` records project milestones and differences from the preceding
  stage.
- `docs/handoff_and_cleanup.md` documents reproducibility, cleanup decisions,
  retained legacy files, and remaining sprint gaps.
- `docs/error_analysis.md` is the current evaluation reference.

