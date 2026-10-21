from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_uploaded_image(uploaded_file: Any) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
