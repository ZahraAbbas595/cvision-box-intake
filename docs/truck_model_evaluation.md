# Truck Carton Model Evaluation

## Scope

The `ft-truck-v1` YOLOv8n candidate detects visible cartons inside trucks. It
was fine-tuned on the ignored local `dataset2/` Roboflow export and remains
separate from the earlier general carton model for rollback.

## Training Result

Training completed all 50 epochs. The selected `best.pt` checkpoint reached
validation precision `0.9976`, recall `0.9877`, mAP50 `0.9945`, and mAP50-95
`0.9482`. These in-dataset values are not treated as deployment evidence.

## Locked External Test

Two independent Roboflow truck datasets were normalized into one `Box` class.
The locked set contains 105 images and 8,107 carton instances, with no exact
image overlap against the fine-tuning train or validation splits.

| Metric | Result |
| --- | ---: |
| Precision | 0.899 |
| Recall | 0.847 |
| mAP50 | 0.890 |
| mAP50-95 | 0.667 |

The external datasets remain excluded from training and threshold tuning.

## Manual Acceptance

The candidate passed local API and Streamlit smoke testing on representative
truck images. Review confirmed that most sampled images produced visually
correct carton boxes and counts. This subjective check is recorded separately
from the locked quantitative metrics.

## ONNX Parity

The dynamic opset-17 ONNX export was exercised through the application's ONNX
backend on all 105 locked external images. At confidence `0.45` and IoU `0.30`,
its accepted carton count matched the PyTorch backend on every image: 105 of
105 matched, with zero count mismatches.

## Candidate Configuration

- PyTorch checkpoint: `models/carton_yolov8n_truck_best.pt`
- ONNX checkpoint: generated beside the PyTorch checkpoint at build time
- Model identity: `carton-yolov8n-truck`, version `ft-truck-v1`
- Confidence threshold: `0.45`
- IoU threshold: `0.30`

The earlier `models/carton_yolov8n_best.pt` remains available for rollback.
