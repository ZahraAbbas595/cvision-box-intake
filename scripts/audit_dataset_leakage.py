"""Audit Roboflow splits for exact and visually near-duplicate images."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageFingerprint:
    """Comparable identity and perceptual features for one image."""

    path: Path
    sha256: str
    pixel_sha256: str
    dhash: int


def iter_images(directory: Path) -> list[Path]:
    """Return supported image files in deterministic path order."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def difference_hash(image: Image.Image, size: int = 8) -> int:
    """Calculate a compact horizontal perceptual difference hash."""
    grayscale = ImageOps.exif_transpose(image).convert("L")
    resized = grayscale.resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value = (value << 1) | (
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def fingerprint(path: Path) -> ImageFingerprint:
    """Build exact and perceptual fingerprints for an image file."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        dhash = difference_hash(normalized)
        pixel_digest = hashlib.sha256(normalized.tobytes()).hexdigest()
    return ImageFingerprint(
        path=path,
        sha256=digest,
        pixel_sha256=pixel_digest,
        dhash=dhash,
    )


def normalized_pixel_mae(first: Path, second: Path, size: int = 128) -> float:
    """Measure resized RGB pixel difference normalized to zero through one."""

    def normalized_pixels(path: Path) -> np.ndarray:
        """Load one image as an orientation-corrected normalized RGB array."""
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            resized = normalized.resize((size, size), Image.Resampling.LANCZOS)
            return np.asarray(resized, dtype=np.float32)

    return float(np.abs(normalized_pixels(first) - normalized_pixels(second)).mean())


def compare_splits(
    reference: list[ImageFingerprint],
    candidate: list[ImageFingerprint],
    max_distance: int,
) -> list[tuple[ImageFingerprint, ImageFingerprint, int, bool, bool]]:
    """Find exact and near-duplicate images across two dataset splits."""
    matches: list[tuple[ImageFingerprint, ImageFingerprint, int, bool, bool]] = []
    for candidate_image in candidate:
        best_reference = min(
            reference,
            key=lambda item: (candidate_image.dhash ^ item.dhash).bit_count(),
        )
        distance = (candidate_image.dhash ^ best_reference.dhash).bit_count()
        exact_file = candidate_image.sha256 == best_reference.sha256
        exact_pixels = candidate_image.pixel_sha256 == best_reference.pixel_sha256
        if exact_file or exact_pixels or distance <= max_distance:
            matches.append(
                (
                    candidate_image,
                    best_reference,
                    distance,
                    exact_file,
                    exact_pixels,
                )
            )
    return matches


def parse_args() -> argparse.Namespace:
    """Parse dataset location and perceptual duplicate threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--max-distance", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "leakage_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Audit configured dataset splits and report potential leakage pairs."""
    args = parse_args()
    splits = {
        split: [
            fingerprint(path) for path in iter_images(args.dataset / split / "images")
        ]
        for split in ("train", "valid", "test")
    }
    comparisons = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    rows: list[dict[str, str | int | bool]] = []
    for reference_name, candidate_name in comparisons:
        matches = compare_splits(
            splits[reference_name],
            splits[candidate_name],
            args.max_distance,
        )
        for candidate, reference, distance, exact_file, exact_pixels in matches:
            rows.append(
                {
                    "reference_split": reference_name,
                    "candidate_split": candidate_name,
                    "reference_image": reference.path.name,
                    "candidate_image": candidate.path.name,
                    "dhash_distance": distance,
                    "normalized_pixel_mae": round(
                        normalized_pixel_mae(candidate.path, reference.path),
                        4,
                    ),
                    "exact_file_match": exact_file,
                    "exact_pixel_match": exact_pixels,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "reference_split",
                "candidate_split",
                "reference_image",
                "candidate_image",
                "dhash_distance",
                "normalized_pixel_mae",
                "exact_file_match",
                "exact_pixel_match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    exact_file_count = sum(bool(row["exact_file_match"]) for row in rows)
    exact_pixel_count = sum(bool(row["exact_pixel_match"]) for row in rows)
    near_identical_count = sum(
        float(row["normalized_pixel_mae"]) <= 3.0 for row in rows
    )
    print(
        f"Audited train={len(splits['train'])}, valid={len(splits['valid'])}, "
        f"test={len(splits['test'])} images."
    )
    print(
        f"Found {exact_file_count} byte-identical and "
        f"{exact_pixel_count} pixel-identical cross-split matches, plus "
        f"{near_identical_count} near-identical candidates (pixel MAE <= 3), "
        f"from {len(rows)} perceptual candidates "
        f"(dHash distance <= {args.max_distance})."
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
