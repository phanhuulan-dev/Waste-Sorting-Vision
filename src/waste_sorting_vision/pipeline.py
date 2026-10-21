from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Mapping

import cv2
import numpy as np
from PIL import Image

from .class_names import get_class_name

_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}

# Each object has a single material, so suppress overlapping boxes across classes
# too; otherwise one bottle can be drawn as both "PET" and "Glass".
_PREDICT_KWARGS = {"agnostic_nms": True, "verbose": False}

# The cached YOLO model is shared by every session and by streamlit-webrtc's
# worker threads; Ultralytics predictors are not thread-safe.
_PREDICT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def select_device() -> str:
    """Fastest available torch device: CUDA GPU, Apple GPU (MPS) or CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _predict(detector: Any, source: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("device", select_device())
    with _PREDICT_LOCK:
        return detector.predict(source, **_PREDICT_KWARGS, **kwargs)[0]


@dataclass(**_DATACLASS_KWARGS)
class ImageInferenceResult:
    annotated_image: Image.Image | np.ndarray
    class_counts: dict[str, int]
    detection_rows: list[dict[str, Any]]
    raw: Any = None  # the Ultralytics Results object, when the caller asks for it


@dataclass(**_DATACLASS_KWARGS)
class VideoInferenceResult:
    processed_frames: int
    sampled_frames: int
    class_counts: dict[str, int]
    last_annotated_frame: Image.Image | None


def _to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Ultralytics treats numpy input as BGR (OpenCV order), so PIL images must be flipped."""
    if isinstance(image, Image.Image):
        return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    return image


def extract_label_ids(result: Any) -> list[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    return [int(label.item()) for label in boxes.cls]


def summarize_class_counts(
    label_ids: list[int],
    class_names: Mapping[int, str],
) -> dict[str, int]:
    counts = Counter(get_class_name(label_id, dict(class_names)) for label_id in label_ids)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_detection_rows(
    result: Any,
    class_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    confidences = getattr(boxes, "conf", [])
    coordinates = getattr(boxes, "xyxy", [])
    label_ids = extract_label_ids(result)
    rows: list[dict[str, Any]] = []

    for index, label_id in enumerate(label_ids):
        confidence = (
            round(float(confidences[index].item()), 4)
            if len(confidences) > index
            else None
        )
        bbox = (
            [round(float(value), 2) for value in coordinates[index].tolist()]
            if len(coordinates) > index
            else []
        )
        rows.append(
            {
                "class_id": label_id,
                "class_name": get_class_name(label_id, dict(class_names)),
                "confidence": confidence,
                "bbox_xyxy": bbox,
            }
        )
    return rows


def _build_image_inference_result(
    result: Any,
    class_names: Mapping[int, str],
    *,
    as_pil: bool = True,
    keep_raw: bool = False,
) -> ImageInferenceResult:
    detection_rows = build_detection_rows(result, class_names)
    label_ids = [row["class_id"] for row in detection_rows]
    annotated_rgb = result.plot()[:, :, ::-1]
    annotated_image: Image.Image | np.ndarray = (
        Image.fromarray(annotated_rgb) if as_pil else np.ascontiguousarray(annotated_rgb)
    )
    return ImageInferenceResult(
        annotated_image=annotated_image,  # type: ignore[arg-type]
        class_counts=summarize_class_counts(label_ids, class_names),
        detection_rows=detection_rows,
        raw=result if keep_raw else None,
    )


def predict_image(
    detector: Any,
    image: Image.Image | np.ndarray,
    confidence: float,
    class_names: Mapping[int, str],
) -> ImageInferenceResult:
    image_array = _to_bgr_array(image)
    result = _predict(detector, image_array, conf=confidence)
    return _build_image_inference_result(result, class_names)


def resize_frame_for_inference(
    frame_bgr: np.ndarray,
    max_width: int,
) -> np.ndarray:
    if max_width <= 0:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    if width <= max_width:
        return frame_bgr

    scale = max_width / width
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def predict_frame(
    detector: Any,
    frame_bgr: np.ndarray,
    confidence: float,
    class_names: Mapping[int, str],
    *,
    imgsz: int = 640,
    as_pil: bool = True,
    keep_raw: bool = False,
) -> ImageInferenceResult:
    result = _predict(detector, frame_bgr, conf=confidence, imgsz=imgsz)
    return _build_image_inference_result(result, class_names, as_pil=as_pil, keep_raw=keep_raw)


def run_video_inference(
    detector: Any,
    video_path: str | Path,
    confidence: float,
    class_names: Mapping[int, str],
    frame_stride: int = 12,
    max_frames: int | None = None,
    on_annotated_frame: Callable[[Image.Image, int, int], None] | None = None,
) -> VideoInferenceResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {video_path}")

    processed_frames = 0
    sampled_frames = 0
    collected_label_ids: list[int] = []
    last_annotated_frame: Image.Image | None = None

    try:
        while capture.isOpened():
            if max_frames is not None and processed_frames >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            processed_frames += 1

            if frame_stride > 1 and (processed_frames - 1) % frame_stride != 0:
                continue

            sampled_frames += 1
            result = _predict(detector, frame, conf=confidence)
            collected_label_ids.extend(extract_label_ids(result))
            annotated_rgb = result.plot()[:, :, ::-1]
            last_annotated_frame = Image.fromarray(annotated_rgb)
            if on_annotated_frame is not None:
                on_annotated_frame(last_annotated_frame, processed_frames, sampled_frames)
    finally:
        capture.release()

    return VideoInferenceResult(
        processed_frames=processed_frames,
        sampled_frames=sampled_frames,
        class_counts=summarize_class_counts(collected_label_ids, class_names),
        last_annotated_frame=last_annotated_frame,
    )
