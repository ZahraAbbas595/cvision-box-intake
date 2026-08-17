"""Validate PyTorch and ONNX application count parity on an image directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse the parity image directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "external_test/combined/images",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "eval/results/truck_backend_parity.json",
    )
    parser.add_argument(
        "--worker-backend", choices=("pytorch", "onnx"), default=None
    )
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser.parse_args()


def image_paths(images: Path) -> list[Path]:
    """Return supported images in deterministic order."""
    paths = sorted(
        path
        for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {images}.")
    return paths


def run_worker(backend: str, images: Path, output: Path) -> None:
    """Run one detector backend and persist per-image counts."""
    cv2: Any = import_module("cv2")
    detector_module: Any = import_module("app.services.detector")
    detector_class = (
        detector_module.PyTorchDetector
        if backend == "pytorch"
        else detector_module.OnnxDetector
    )
    detector = detector_class()
    counts: dict[str, int] = {}
    for image_path in image_paths(images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to decode {image_path}.")
        counts[image_path.name] = len(detector.predict(image))
    output.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")


def collect_counts(backend: str, images: Path, output: Path) -> dict[str, int]:
    """Run a backend in an isolated process and load its result."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--images",
            str(images),
            "--worker-backend",
            backend,
            "--worker-output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    raw_counts: object = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(raw_counts, dict):
        raise TypeError(f"Invalid {backend} worker output.")
    return {str(name): int(count) for name, count in raw_counts.items()}


def compare_backends(args: argparse.Namespace) -> None:
    """Compare isolated backend outputs and write the parity report."""
    paths = image_paths(args.images)
    with tempfile.TemporaryDirectory(prefix="cvision-parity-") as temp_dir:
        temporary_root = Path(temp_dir)
        pytorch_counts = collect_counts(
            "pytorch", args.images, temporary_root / "pytorch.json"
        )
        onnx_counts = collect_counts(
            "onnx", args.images, temporary_root / "onnx.json"
        )

    image_names = sorted(path.name for path in paths)
    mismatches: list[tuple[str, int, int]] = []
    for image_name in image_names:
        pytorch_count = pytorch_counts[image_name]
        onnx_count = onnx_counts[image_name]
        if pytorch_count != onnx_count:
            mismatches.append((image_name, pytorch_count, onnx_count))

    report = {
        "images": len(paths),
        "count_mismatches": len(mismatches),
        "mismatches": [
            {"image": name, "pytorch": pytorch_count, "onnx": onnx_count}
            for name, pytorch_count, onnx_count in mismatches
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"images={len(paths)} count_mismatches={len(mismatches)}")
    for name, pytorch_count, onnx_count in mismatches:
        print(f"{name}: pytorch={pytorch_count} onnx={onnx_count}")
    if mismatches:
        raise SystemExit("PyTorch/ONNX count parity failed.")


def main() -> None:
    """Run a backend worker or orchestrate the isolated parity comparison."""
    args = parse_args()
    if args.worker_backend is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required for worker mode.")
        run_worker(args.worker_backend, args.images, args.worker_output)
        return
    compare_backends(args)


if __name__ == "__main__":
    main()
