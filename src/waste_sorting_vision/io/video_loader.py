from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def load_video_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def save_uploaded_video(uploaded_file: Any) -> Path:
    suffix = Path(getattr(uploaded_file, "name", "")).suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)
