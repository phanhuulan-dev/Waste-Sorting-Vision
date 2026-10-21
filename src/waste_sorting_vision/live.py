"""Shared live-stream processing: device choice, tracking and frame reading.

LiveProcessor turns camera frames into annotated frames and keeps per-session
statistics. It is used both from the Streamlit script thread (local OpenCV
stream) and from streamlit-webrtc's worker thread (browser video), so all
mutable state is guarded by a lock.

Counting uses ByteTrack: every physical object gets a track id, and an id is
counted once it has been seen in MIN_TRACK_HITS inferred frames, under the
class it was most often detected as. Without tracking, a bottle held in front
of the camera for ten seconds would be counted once per frame.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from typing import Any, Mapping

import cv2
import numpy as np

from .pipeline import predict_frame, resize_frame_for_inference, select_device

MIN_TRACK_HITS = 3


def _new_tracker() -> Any:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    return BYTETracker(args=args, frame_rate=30)


class UniqueObjectCounter:
    """Counts distinct track ids per class, ignoring short-lived tracks."""

    def __init__(self, min_hits: int = MIN_TRACK_HITS) -> None:
        self.min_hits = min_hits
        self._class_votes: dict[int, Counter] = defaultdict(Counter)

    def update(self, track_ids: list[int], class_ids: list[int]) -> None:
        for track_id, class_id in zip(track_ids, class_ids):
            self._class_votes[int(track_id)][int(class_id)] += 1

    def counts(self, class_names: Mapping[int, str]) -> dict[str, int]:
        totals: Counter = Counter()
        for votes in self._class_votes.values():
            if sum(votes.values()) >= self.min_hits:
                class_id = votes.most_common(1)[0][0]
                totals[class_names.get(class_id, str(class_id))] += 1
        return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


class LiveProcessor:
    """Runs detection (+ optional tracking) on frames; safe to call from any thread."""

    def __init__(
        self,
        detector: Any,
        class_names: Mapping[int, str],
        confidence: float,
        imgsz: int,
        max_width: int,
        inference_stride: int = 1,
        track: bool = True,
    ) -> None:
        self.detector = detector
        self.class_names = dict(class_names)
        self.confidence = confidence
        self.imgsz = imgsz
        self.max_width = max_width
        self.inference_stride = max(1, int(inference_stride))
        self.track = track
        self._lock = threading.Lock()
        self._tracker = _new_tracker() if track else None
        self._counter = UniqueObjectCounter()
        self._frame_index = 0
        self._last_annotated: np.ndarray | None = None
        self._last_rows: list[dict[str, Any]] = []
        self._inference_ms: deque[float] = deque(maxlen=30)
        self._frame_times: deque[float] = deque(maxlen=30)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an annotated BGR frame (the latest one when skipping inference)."""
        with self._lock:
            self._frame_index += 1
            self._frame_times.append(time.perf_counter())
            run_inference = self._last_annotated is None or self._frame_index % self.inference_stride == 0
            if not run_inference:
                return self._last_annotated

            frame = resize_frame_for_inference(frame_bgr, self.max_width)
            started = time.perf_counter()
            result = predict_frame(
                self.detector, frame, self.confidence, self.class_names,
                imgsz=self.imgsz, as_pil=False, keep_raw=True,
            )
            self._inference_ms.append((time.perf_counter() - started) * 1000)

            if self._tracker is not None and result.raw is not None:
                boxes = result.raw.boxes.cpu().numpy()
                tracks = self._tracker.update(boxes, frame) if len(boxes) else np.empty((0, 8))
                if len(tracks):
                    self._counter.update(tracks[:, 4].astype(int).tolist(), tracks[:, 6].astype(int).tolist())

            # annotated_image is RGB (for st.image); callers here work in BGR
            self._last_annotated = cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR)
            self._last_rows = result.detection_rows
            return self._last_annotated

    def stats(self) -> dict[str, Any]:
        with self._lock:
            times = list(self._frame_times)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
            inference = sum(self._inference_ms) / len(self._inference_ms) if self._inference_ms else 0.0
            return {
                "frames": self._frame_index,
                "fps": fps,
                "inference_ms": inference,
                "device": select_device(),
                "current_objects": len(self._last_rows),
                "current_rows": list(self._last_rows),
                "unique_counts": self._counter.counts(self.class_names) if self.track else {},
            }

    def reset_counts(self) -> None:
        with self._lock:
            self._counter = UniqueObjectCounter()
            self._tracker = _new_tracker() if self.track else None


class LatestFrameReader:
    """Reads a cv2.VideoCapture on a background thread and keeps only the newest frame.

    Reading on demand returns frames queued in the driver's buffer, which lags
    behind reality when inference is slower than the camera (macOS ignores
    CAP_PROP_BUFFERSIZE). A reader thread drains the buffer continuously.
    """

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._failed_reads = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            success, frame = self._capture.read()
            with self._lock:
                if success and frame is not None:
                    self._frame = frame
                    self._sequence += 1
                    self._failed_reads = 0
                else:
                    self._failed_reads += 1
            if not success:
                time.sleep(0.05)

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._sequence, self._frame

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._running and self._failed_reads < 40  # ~2 s of failed reads

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()
