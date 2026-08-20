"""Generate compact annotated carton examples for project handoff evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.services.detector import run_inference


def parse_args() -> argparse.Namespace:
    """Parse source images and destination directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def annotate_image(image_path: Path, output_path: Path) -> int:
    """Run application inference and save one resized annotated image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    detections = run_inference(image)
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{index} {detection['confidence']:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
    height, width = image.shape[:2]
    if max(height, width) > 1200:
        scale = 1200 / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"Cannot write annotated image: {output_path}")
    return len(detections)


def main() -> None:
    """Generate one numbered evidence image for every supplied source image."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(args.images, start=1):
        output_path = args.output / f"example-{index}.jpg"
        count = annotate_image(image_path, output_path)
        print(f"{output_path}: detections={count} source={image_path.stem}")


if __name__ == "__main__":
    main()
