# Model Decision

## Selected Model

The application uses the `ft-truck-v1` YOLOv8n detector, fine-tuned for visible
cartons inside trucks. The committed source checkpoint is
`models/carton_yolov8n_truck_best.pt`; production exports it to dynamic-shape
ONNX at build time.

This model was selected because it matches the current operating domain, fits
the free deployment memory budget through ONNX Runtime, and passed both locked
external evaluation and manual Streamlit acceptance. Quantitative evidence is
recorded in `docs/truck_model_evaluation.md`.

## Rejected Alternatives

- Generic COCO YOLOv8n has no carton class and did not provide usable carton
  detection.
- YOLO-World demonstrated zero-shot feasibility but produced unstable crowded
  counts and exceeded the free deployment memory budget.
- The earlier general-carton fine-tuned YOLOv8n did not match the narrowed
  truck-interior scope closely enough.

## Operating Configuration

- Model identity: `carton-yolov8n-truck`, version `ft-truck-v1`
- Confidence threshold: `0.45`
- NMS IoU threshold: `0.30`
- Local backend: PyTorch by default
- Production backend: dynamic-shape ONNX Runtime

The locked external truck datasets must remain excluded from training and
threshold tuning.
