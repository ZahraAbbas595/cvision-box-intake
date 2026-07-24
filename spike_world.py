from ultralytics import YOLOWorld
import cv2
from pathlib import Path

IMAGE_DIR = Path('eval/dataset/images')
OUTPUT_DIR = Path('eval/results')
OUTPUT_DIR.mkdir(exist_ok=True)

model = YOLOWorld('yolov8s-worldv2.pt')
model.set_classes(['cardboard box', 'carton', 'package', 'parcel'])

print('=' * 60)
for img_path in sorted(IMAGE_DIR.iterdir()):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    
    results = model(str(img_path), conf=0.25, iou=0.5, verbose=False)
    result = results[0]
    boxes = result.boxes
    
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({'class': cls_name, 'confidence': round(conf, 3)})
    
    print(f'{img_path.name}: {len(detections)} detections')
    for d in detections:
        print(f'  {d["class"]:20s}  conf={d["confidence"]}')
    
    annotated = result.plot()
    cv2.imwrite(str(OUTPUT_DIR / f'world_{img_path.stem}.jpg'), annotated)

print('=' * 60)
print('Done. Check eval/results/')
