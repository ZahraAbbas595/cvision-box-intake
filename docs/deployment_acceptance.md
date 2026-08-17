# Deployment Acceptance

Date: 2026-08-06

This document records the first Week 3 public deployment acceptance check for
the CVision Box Intake research prototype.

## Live Services

- Streamlit frontend: <https://cvision-box-intake.streamlit.app/>
- Render FastAPI backend: <https://cvision-box-intake-api.onrender.com>
- Frontend source: `dev`, entry point `ui/streamlit_app.py`
- Backend configuration: `BACKEND_URL` stored as a Streamlit secret

The GitHub repository was made public to support Community Cloud deployment.
No backend URL or credential is hardcoded in the frontend source.

## Acceptance Results

| Scenario | Image | HTTP | Visible count | Review reasons | Client time |
| --- | --- | ---: | ---: | --- | ---: |
| Clear supported scene | `img_010.jpg` | 200 | 4 | None | 7.325 s |
| Difficult backlit scene | `img_003.png` | 200 | 24 | Edge truncation; high count | 2.853 s |

The clear result matches the reviewed expected count of four. The difficult
result intentionally preserves the known overcount and exposes review signals
rather than silently correcting it.

The clear request included service wake-up overhead. Server processing times
were 4,475 ms and 609 ms respectively. Render reported a 400.8 MB peak RSS,
remaining below the free service's 512 MB limit.

## UI Verification

The public frontend rendered the upload control, research-prototype disclaimer,
and accepted JPEG/PNG constraints successfully after deployment. Community
Cloud initially exposed a script import-path failure; PR #22 corrected the
entry-point imports and passed CI before automatic redeployment.

Automated control of the iframe-native file chooser was unavailable in the
in-app browser. The same two tracked files were therefore submitted directly
to the configured live backend through `scripts/smoke_render.py`. A manual UI
upload remains in the final browser acceptance checklist.

## Final Manual Acceptance

The project owner confirmed completion of the remaining public-UI checks on
2026-08-12: manual clear and difficult uploads, the cold-start experience, and
the visible timeout/backend-unavailable handling.

Recorded manual observations:

| Check | Observed result |
| --- | --- |
| Cold backend | 27 seconds |
| Cold-start progress message | “Waking up the service if needed and analyzing the image…” |
| Warm analysis | 2 seconds |
| Timeout state | “Analysis took too long. The service may still be waking up.” |
| Backend-unavailable state | “The analysis service is not responding. Try again shortly.” |

These messages were visible to the operator and clearly distinguished a slow
cold start from an unavailable analysis service. The timings are manual browser
observations; the earlier automated values above remain the reproducible server
measurements.
