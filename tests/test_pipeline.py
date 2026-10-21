import numpy as np
from PIL import Image

from waste_sorting_vision.pipeline import predict_image, summarize_class_counts


class _RecordingDetector:
    """Stands in for a YOLO model and records what predict() receives."""

    def __init__(self) -> None:
        self.source = None
        self.kwargs = {}

    def predict(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        return [_EmptyResult(source)]


class _EmptyResult:
    boxes = None

    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def plot(self) -> np.ndarray:
        return self._image


def test_predict_image_passes_bgr_array_with_agnostic_nms() -> None:
    detector = _RecordingDetector()
    red = Image.new("RGB", (4, 4), (255, 0, 0))

    result = predict_image(detector, red, confidence=0.4, class_names={0: "Paper"})

    assert detector.source[0, 0].tolist() == [0, 0, 255]  # BGR order for Ultralytics
    assert detector.kwargs["agnostic_nms"] is True
    assert result.annotated_image.getpixel((0, 0)) == (255, 0, 0)  # displayed as RGB


def test_summarize_class_counts_sorts_by_count_then_name() -> None:
    class_names = {
        0: "Paper",
        1: "Can",
        2: "Battery",
    }

    summary = summarize_class_counts([1, 0, 1, 2, 2], class_names)

    assert list(summary.items()) == [
        ("Battery", 2),
        ("Can", 2),
        ("Paper", 1),
    ]
