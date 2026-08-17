"""Build a normalized external truck-carton test set from Roboflow exports."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSource:
    """Describe a source dataset and the class IDs that represent cartons."""

    name: str
    path: Path
    carton_class_ids: frozenset[int]


def parse_args() -> argparse.Namespace:
    """Parse source-root and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "external_test", help="Export root"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "external_test" / "combined"
    )
    return parser.parse_args()


def normalized_rows(label_path: Path, class_ids: frozenset[int]) -> list[str]:
    """Return valid selected annotations remapped to the single Box class."""
    rows: list[str] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 values")
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(
                f"{label_path}:{line_number}: invalid YOLO annotation"
            ) from error
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinate outside [0, 1]")
        if class_id in class_ids:
            rows.append("0 " + " ".join(parts[1:]))
    return rows


def build_dataset(
    sources: tuple[DatasetSource, ...], output: Path
) -> dict[str, object]:
    """Copy and normalize all source splits into one collision-safe test set."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    images_output = output / "images"
    labels_output = output / "labels"
    images_output.mkdir(parents=True)
    labels_output.mkdir(parents=True)

    source_counts: dict[str, dict[str, int]] = {}
    total_images = 0
    total_instances = 0
    for source in sources:
        image_count = 0
        instance_count = 0
        for split in ("train", "valid", "test"):
            image_dir = source.path / split / "images"
            label_dir = source.path / split / "labels"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_SUFFIXES
                ):
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing label for {image_path}")
                rows = normalized_rows(label_path, source.carton_class_ids)
                destination_stem = f"{source.name}_{split}_{image_path.stem}"
                destination_image = (
                    images_output / f"{destination_stem}{image_path.suffix.lower()}"
                )
                shutil.copy2(image_path, destination_image)
                (labels_output / f"{destination_stem}.txt").write_text(
                    "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
                )
                image_count += 1
                instance_count += len(rows)
        source_counts[source.name] = {
            "images": image_count,
            "instances": instance_count,
        }
        total_images += image_count
        total_instances += instance_count

    manifest: dict[str, object] = {
        "class_names": ["Box"],
        "images": total_images,
        "instances": total_instances,
        "sources": source_counts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "data.yaml").write_text(
        "path: external_test/combined\n"
        "train: images\n"
        "val: images\n"
        "nc: 1\n"
        "names: ['Box']\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Build the normalized external evaluation dataset."""
    args = parse_args()
    sources = (
        DatasetSource("boxintake", args.root / "boxintake", frozenset({1, 4})),
        DatasetSource("carton_loading", args.root / "carton_loading", frozenset({0})),
    )
    print(json.dumps(build_dataset(sources, args.output), indent=2))


if __name__ == "__main__":
    main()
