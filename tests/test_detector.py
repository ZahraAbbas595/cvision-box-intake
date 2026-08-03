import numpy as np

from app.services.detector import _decode_output, _nms, _prepare_input


def test_prepare_input_uses_stride_aligned_landscape_shape() -> None:
    image = np.zeros((501, 762, 3), dtype=np.uint8)

    tensor, scale, pad_x, pad_y = _prepare_input(image)

    assert tensor.shape == (1, 3, 448, 640)
    assert scale == 640 / 762
    assert pad_x == 0
    assert pad_y == 13


def test_prepare_input_uses_stride_aligned_portrait_shape() -> None:
    image = np.zeros((900, 598, 3), dtype=np.uint8)

    tensor, scale, pad_x, pad_y = _prepare_input(image)

    assert tensor.shape == (1, 3, 640, 448)
    assert scale == 640 / 900
    assert pad_x == 11
    assert pad_y == 0


def test_decode_output_filters_confidence_and_restores_image_coordinates() -> None:
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    raw_output = np.array(
        [
            [
                [320.0, 160.0, 0.0, 0.0, 0.0, 0.0],
                [320.0, 160.0, 0.0, 0.0, 0.0, 0.0],
                [200.0, 100.0, 1.0, 1.0, 1.0, 1.0],
                [100.0, 50.0, 1.0, 1.0, 1.0, 1.0],
                [0.90, 0.20, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, scores = _decode_output(
        raw_output,
        scale=1.0,
        pad_x=0,
        pad_y=160,
        image_array=image,
    )

    np.testing.assert_allclose(boxes, [[220.0, 110.0, 420.0, 210.0]])
    np.testing.assert_allclose(scores, [0.90])


def test_nms_keeps_highest_score_and_separate_box() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
            [5.0, 5.0, 105.0, 105.0],
            [200.0, 200.0, 260.0, 260.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.95, 0.80, 0.70], dtype=np.float32)

    assert _nms(boxes, scores, iou_threshold=0.30) == [0, 2]
