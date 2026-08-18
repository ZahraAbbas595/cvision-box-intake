"""Tests for deterministic image-quality review signals."""

import cv2
import numpy as np

from app.services.quality import assess_image_quality


def test_quality_flags_low_resolution_blur_and_underexposure() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    flags = assess_image_quality(
        image,
        min_dimension=200,
        blur_variance_threshold=50.0,
        dark_mean_threshold=40.0,
        bright_mean_threshold=215.0,
    )

    assert flags == ["low_resolution", "possible_blur", "underexposed"]


def test_quality_accepts_detailed_mid_exposure_image() -> None:
    grayscale = np.indices((256, 256)).sum(axis=0) % 2 * 80 + 80
    image = cv2.cvtColor(grayscale.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    flags = assess_image_quality(
        image,
        min_dimension=200,
        blur_variance_threshold=50.0,
        dark_mean_threshold=40.0,
        bright_mean_threshold=215.0,
    )

    assert flags == []
