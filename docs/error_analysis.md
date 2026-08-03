# Error Analysis

## Scope

This analysis covers the 24-image primary count set at the selected operating
point of confidence `0.45`, NMS IoU `0.30`, and inference size `640`. Nineteen
images have exact counts. The five non-exact images are reviewed below against
the full-resolution source and saved prediction overlay.

The seven crowded scenes reserved for a second independent count are not used
here. The Roboflow validation and test metrics are also not used as count
evidence because the leakage audit found cross-split near-duplicates.

## Counting Rule

A carton counts whenever the visible region identifies it confidently as a
distinct physical carton. Partial, border-cropped, damaged, and wrapped but
visually identifiable cartons count. Only the visible area should be
annotated. Extremely small or ambiguous fragments that cannot be separated
reliably are excluded.

## Current Count Results

| Metric | Result |
| --- | ---: |
| Reviewed images | 24 |
| Exact-count images | 19 |
| Exact-count accuracy | 79.2% |
| Mean absolute count error | 0.83 |
| Overcount images | 3 |
| Undercount images | 2 |

## Failure Frequency

The taxonomy is multi-label: one image may expose more than one failure mode.
This prevents a secondary problem, such as damage or annotation disagreement,
from being hidden behind the signed count error.

| Failure category | Images | Frequency | Evidence |
| --- | --- | ---: | --- |
| Missed carton | `img_012`, `img_022` | 2 | One partially obscured carton and one stylized open carton are not detected. |
| Duplicate or fragmented carton | `img_023`, `img_030` | 2 | One physical carton is represented by adjacent detections. |
| Non-carton detected | `img_003` | 1 | Window panes are repeatedly classified as cartons. |
| Partially hidden carton | `img_012` | 1 | The small upper carton is partly occluded by the person. |
| Unsupported or out-of-distribution scene | `img_022` | 1 | Stylized red carton, saturated lighting, and very low resolution differ from typical training examples. |
| Possible visible damage | `img_023` | 1 | A torn carton is detected as two carton regions. Damage is context, not a separately supported model output. |
| No useful detection | `img_022` | 1 | Zero cartons are returned for a one-carton scene. |
| Ground-truth disagreement | `img_030` | 1 | The stored count of one conflicts with the current rule because multiple partial cartons appear identifiable. |
| Poor image quality | None confirmed | 0 | `img_022` is small and stylized, but the dominant issue is domain shift rather than decode corruption. |
| Incorrect size class | Not assessed | — | The current manifest does not contain independently reviewed size-class ground truth. |

## Image-Level Findings

### `img_003.png`: 12 expected, 25 predicted

Primary category: **non-carton false positives**.

The detector finds the physical cartons but also places carton boxes over many
rectangular window panes. The +13 error is therefore not evidence that ordinary
NMS failed to merge repeated boxes on the same carton. It is a texture and
shape confusion: bright rectangular window regions resemble the rectangular
carton examples learned by the single-class model.

Operational impact: severe phantom inventory in a plausible warehouse-like
scene. Raising the global confidence threshold is a poor first response because
several window false positives are high confidence and the change would increase
missed cartons elsewhere.

### `img_012.jpg`: 4 expected, 3 predicted

Primary category: **missed, partially occluded carton**.

The three larger cartons are detected. The small upper carton behind the
person's head is missed. Its visible area is sufficient to identify it under
the counting rule, but its scale and person occlusion make it unlike the larger,
fully visible cartons below it.

Operational impact: one physical item is absent from the intake record. This is
why the selected threshold favors recall over the slightly lower mean absolute
error available at confidence `0.55`.

### `img_022.jpg`: 1 expected, 0 predicted

Primary category: **unsupported appearance / missed carton**.

The image is a small, stylized red open carton with saturated backlighting. The
shape is identifiable to a human, but it differs strongly from the brown,
photographic cartons that dominate the data. No useful detection is returned.

Operational impact: a silent false negative unless the zero-detection review
rule routes the scan to a person.

### `img_023.jpg`: 1 expected, 2 predicted

Primary category: **fragmented detection on a damaged carton**.

The torn front face creates a strong vertical separation. The model places two
adjacent boxes over the left and right parts of one physical carton. Their
overlap is small, so standard IoU NMS is not designed to merge them.

Operational impact: one damaged item becomes two inventory units. A targeted
fragmentation heuristic may help, but must not merge genuinely adjacent cartons.

### `img_030.jpg`: 1 recorded, 5 predicted

Status: **ground truth requires human re-review before model scoring**.

The overlay contains fragmented regions on the dominant foreground carton, but
the source also shows partial cartons at the lower-left edge and in the
background. Under the current identity-based counting rule, at least some of
those partial cartons may count. Treating all four excess detections as model
false positives would overstate the model error before the expected count is
reviewed consistently.

Operational impact: inconsistent annotation policy can distort both threshold
selection and the apparent value of post-processing. This image remains in the
reported metrics for traceability, but any post-processing experiment must show
results both with and without it until its count is resolved.

## What Changed After Reviewing the Errors

1. The operating point remains confidence `0.45`, IoU `0.30`. The corrected
   24-image sweep gives the best exact-count accuracy, while a higher confidence
   threshold increases undercounts.
2. Global NMS tightening is not the next default change. It cannot remove the
   high-confidence window false positives in `img_003`, and adjacent fragments
   in `img_023` have little overlap.
3. Error analysis is now multi-label and distinguishes model errors from
   ground-truth disagreements.
4. `img_030` is flagged for human recount under the current rule rather than
   silently accepted as four false positives.

## Next Experiments

Run each experiment against the complete reviewed set and reject it if exact
count accuracy or undercount frequency worsens materially.

1. Re-review `img_030` and obtain independent counts for the seven reserved
   crowded scenes.
2. Measure box-pair geometry on `img_023` and `img_030` to test a conservative
   adjacent-fragment review flag before attempting automatic merging.
3. Add a review reason for suspicious repeated rectangular detections or a
   high predicted count in scenes like `img_003`; do not claim that this fixes
   the underlying detector confusion.
4. Expand future training data with labelled hard negatives such as windows and
   with stylized, open, damaged, partial, and strongly lit cartons. This is a
   future model-improvement recommendation, not part of the current sprint
   implementation.

## Limitations

- This is a 24-image reviewed count set, not warehouse-scale validation.
- One of the five error images has unresolved ground truth.
- Seven crowded images await a second independent count.
- Size-class errors cannot be measured until size ground truth is reviewed.
- Damage is not a supported detection task; it is contextual evidence for human
  review only.
