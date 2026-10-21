import time

import numpy as np

from waste_sorting_vision.live import LatestFrameReader, UniqueObjectCounter

NAMES = {0: "Paper", 5: "Can", 6: "PET"}


def test_unique_counter_counts_each_track_once() -> None:
    counter = UniqueObjectCounter(min_hits=3)
    for _ in range(20):  # the same bottle over 20 frames
        counter.update([7], [6])

    assert counter.counts(NAMES) == {"PET": 1}


def test_unique_counter_ignores_short_lived_tracks_and_uses_majority_class() -> None:
    counter = UniqueObjectCounter(min_hits=3)
    counter.update([1, 2], [6, 5])  # track 2 appears once: noise
    counter.update([1], [5])
    counter.update([1], [6])

    assert counter.counts(NAMES) == {"PET": 1}


class _FakeCapture:
    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def read(self):
        self.frames += 1
        time.sleep(0.001)
        return True, np.full((2, 2, 3), self.frames % 255, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_latest_frame_reader_keeps_newest_frame_and_releases() -> None:
    capture = _FakeCapture()
    reader = LatestFrameReader(capture)
    time.sleep(0.05)
    first_sequence, frame = reader.latest()
    time.sleep(0.05)
    second_sequence, _ = reader.latest()
    reader.release()

    assert frame is not None and reader.healthy is False
    assert second_sequence > first_sequence
    assert capture.released
