"""Export the selected PyTorch carton detector to dynamic-shape ONNX."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse the source checkpoint path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/carton_yolov8n_truck_best.pt",
    )
    return parser.parse_args()


def main() -> None:
    """Export the selected detector with dynamic ONNX spatial dimensions."""
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"PyTorch model not found at {args.model}.")
    model = YOLO(args.model)
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
