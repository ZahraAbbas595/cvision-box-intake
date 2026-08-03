from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("models/carton_yolov8n_best.pt")


def main() -> None:
    model = YOLO(MODEL_PATH)
    output_path = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=False,
        dynamic=True,
    )
    print(f"exported={output_path}")


if __name__ == "__main__":
    main()
