from app.services.count_risk import (
    find_suspicious_fragment_pairs,
    measure_pair_geometry,
)


def test_adjacent_aligned_boxes_are_fragment_candidates() -> None:
    pairs = find_suspicious_fragment_pairs(
        [[10, 10, 40, 70], [42, 12, 72, 68]],
        100,
        100,
        max_gap_ratio=0.04,
        min_axis_overlap=0.60,
        max_area_ratio=4.0,
        max_iou=0.20,
    )

    assert pairs == [(0, 1)]


def test_widely_separated_boxes_are_not_fragment_candidates() -> None:
    pairs = find_suspicious_fragment_pairs(
        [[0, 10, 20, 40], [70, 10, 90, 40]],
        100,
        100,
        max_gap_ratio=0.04,
        min_axis_overlap=0.60,
        max_area_ratio=4.0,
        max_iou=0.20,
    )

    assert pairs == []


def test_pair_geometry_reports_low_iou_for_adjacent_fragments() -> None:
    geometry = measure_pair_geometry([10, 10, 40, 70], [42, 12, 72, 68], 100, 100)

    assert geometry.gap_ratio == 0.02
    assert geometry.axis_overlap > 0.9
    assert geometry.iou == 0.0
