# Count Error Analysis

## Evaluation setup

The locked Carton Loading evaluation set contains 49 truck images and 2,537
annotated cartons. It is excluded from training and threshold tuning. Results
below use the deployed PyTorch-equivalent settings: confidence `0.45`, NMS IoU
`0.30`, image size 640, and a maximum of 300 detections.

| Images | Ground truth | Predicted | Exact | Within ±1 | Within ±2 | MAE | Overcounts | Undercounts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 2,537 | 2,497 | 49.0% | 63.3% | 71.4% | 2.00 | 9 | 16 |

The totals include 24 exact images, 9 overcounts, and 16 undercounts. Aggregate
predicted and annotated totals are not treated as accuracy because image-level
overcounts and undercounts can offset one another.

## Annotated examples

### Exact count: 63 expected, 63 predicted

![Exact-count example](./annotated_examples/example-1.jpg)

The regular front-facing stack has clear carton boundaries. This is the target
operating pattern and demonstrates that a dense image does not require review
merely because it contains many cartons.

### Overcount: 58 expected, 66 predicted

![Overcount example](./annotated_examples/example-2.jpg)

The narrow cartons on the left and repeated label/edge patterns create several
fragmented regions. The likely operational effect is accepting more cartons
than are actually visible. The fragmented-detection rule is intentionally
conservative; it cannot reliably repair a count and therefore only provides
review evidence.

### Undercount: 46 expected, 29 predicted

![Undercount example](./annotated_examples/example-3.jpg)

Several tightly packed cartons in the lower half are missed where boundaries
are small, repetitive, and partially obscured. A closer image, multiple
viewpoints, tiled inference, or a higher-resolution warehouse-specific model
would be required before operational use.

## Error categories and response

| Category | Observed effect | Current response | Remaining work |
| --- | --- | --- | --- |
| Missed small/occluded carton | Undercount | Preserve annotated evidence; deterministic quality and edge checks | Add warehouse data, tiled inference, and multi-view capture |
| Duplicate/fragmented carton | Overcount | Conservative geometric review signal | Train on damaged/partial faces and validate duplicate suppression |
| Non-carton lookalike | False positive | Advisory visual observation | Collect operator-confirmed negatives |
| Poor blur/exposure/resolution | Miss or unstable boundary | Deterministic retake guidance | Calibrate thresholds on target cameras |
| Incorrect relative size | Wrong small/medium/large bucket | Explicitly label size as image-relative | Add camera calibration for physical dimensions |
| Hidden cartons | Cannot be counted from one image | Do not infer invisible inventory | Compare multiple views and manifest/barcode evidence |

The evaluation is reproducible with:

```powershell
python scripts/evaluate_external_counts.py `
  --model models/carton_yolov8n_truck_best.pt `
  --dataset external_test/carton_loading_evaluation `
  --confidence 0.45 `
  --iou 0.30
```
