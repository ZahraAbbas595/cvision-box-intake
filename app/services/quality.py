"""Deterministic image-quality signals used for human-review routing."""

import cv2
import numpy as np


def assess_image_quality(
    image: np.ndarray,
    *,
    min_dimension: int,
    blur_variance_threshold: float,
    dark_mean_threshold: float,
    bright_mean_threshold: float,
) -> list[str]:
    """Return explainable resolution, blur, and exposure quality flags."""
    height, width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(grayscale.mean())
    laplacian_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    flags: list[str] = []
    if min(height, width) < min_dimension:
        flags.append("low_resolution")
    if laplacian_variance < blur_variance_threshold:
        flags.append("possible_blur")
    if mean_brightness < dark_mean_threshold:
        flags.append("underexposed")
    elif mean_brightness > bright_mean_threshold:
        flags.append("overexposed")
    return flags
