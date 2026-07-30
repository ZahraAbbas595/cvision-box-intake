from ultralytics import YOLOWorld
import numpy as np
import torch

torch.set_num_threads(1)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLOWorld('yolov8s-worldv2.pt')
        _model.set_classes(['cardboard box', 'carton', 'package', 'parcel'])
        _model.to('cpu')
    return _model

def run_inference(image_array: np.ndarray) -> list:
    model = get_model()
    results = model(
        image_array,
        conf=0.35,
        iou=0.50,
        device='cpu',
        verbose=False,
    )
    result = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'bbox_xyxy': [round(x1), round(y1), round(x2), round(y2)],
                'confidence': round(float(box.conf[0]), 3),
            })
    return detections
