from __future__ import annotations

import sys

import cv2

_MAX_PROBE_INDEX = 4


def _capture_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def probe_camera_indices(max_probe: int = _MAX_PROBE_INDEX) -> list[int]:
    """Return camera indices that OpenCV can open and read at least one frame."""
    backend = _capture_backend()
    available: list[int] = []

    for index in range(max(1, max_probe)):
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            success, _frame = capture.read()
            if success:
                available.append(index)
        finally:
            capture.release()

    return available


def format_camera_label(index: int) -> str:
    return f"Camera {index}"
