"""Geometry helpers for conservative carton count-risk review flags."""

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence


@dataclass(frozen=True)
class PairGeometry:
    """Normalized geometric relationship between two detection boxes."""

    gap_ratio: float
    axis_overlap: float
    area_ratio: float
    iou: float


def _area(box: Sequence[int]) -> int:
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def measure_pair_geometry(
    first: Sequence[int],
    second: Sequence[int],
    image_width: int,
    image_height: int,
) -> PairGeometry:
    """Measure normalized gap, overlap, size, and IoU for two boxes."""
    first_width = max(first[2] - first[0], 1)
    first_height = max(first[3] - first[1], 1)
    second_width = max(second[2] - second[0], 1)
    second_height = max(second[3] - second[1], 1)
    overlap_width = max(min(first[2], second[2]) - max(first[0], second[0]), 0)
    overlap_height = max(min(first[3], second[3]) - max(first[1], second[1]), 0)
    horizontal_overlap = overlap_width / min(first_width, second_width)
    vertical_overlap = overlap_height / min(first_height, second_height)
    horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0)
    vertical_gap = max(first[1] - second[3], second[1] - first[3], 0)
    gap_ratio = max(horizontal_gap, vertical_gap) / max(image_width, image_height, 1)
    first_area = _area(first)
    second_area = _area(second)
    smaller_area = max(min(first_area, second_area), 1)
    area_ratio = max(first_area, second_area) / smaller_area
    intersection = overlap_width * overlap_height
    union = max(first_area + second_area - intersection, 1)
    return PairGeometry(
        gap_ratio=gap_ratio,
        axis_overlap=max(horizontal_overlap, vertical_overlap),
        area_ratio=area_ratio,
        iou=intersection / union,
    )


def find_suspicious_fragment_pairs(
    boxes: Sequence[Sequence[int]],
    image_width: int,
    image_height: int,
    *,
    max_gap_ratio: float,
    min_axis_overlap: float,
    max_area_ratio: float,
    max_iou: float,
) -> list[tuple[int, int]]:
    """Return box-index pairs whose geometry suggests fragmentation risk."""
    suspicious_pairs: list[tuple[int, int]] = []
    for (first_index, first), (second_index, second) in combinations(
        enumerate(boxes), 2
    ):
        geometry = measure_pair_geometry(first, second, image_width, image_height)
        if (
            geometry.gap_ratio <= max_gap_ratio
            and geometry.axis_overlap >= min_axis_overlap
            and geometry.area_ratio <= max_area_ratio
            and geometry.iou <= max_iou
        ):
            suspicious_pairs.append((first_index, second_index))
    return suspicious_pairs
