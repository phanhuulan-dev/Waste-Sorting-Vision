from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_app_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "app.yaml")


def load_model_sources() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "model_sources.yaml")


def get_model_catalog() -> dict[str, dict[str, Any]]:
    return load_model_sources().get("models", {})


def get_model_source(model_key: str) -> dict[str, Any]:
    model_catalog = get_model_catalog()
    if model_key not in model_catalog:
        available = ", ".join(sorted(model_catalog))
        raise KeyError(f"Unknown model key '{model_key}'. Available models: {available}")
    return model_catalog[model_key]


def get_default_image_path() -> Path | None:
    raw_path = load_app_config().get("assets", {}).get("default_image")
    if not raw_path:
        return None
    return resolve_path(raw_path)


def get_default_detected_image_path() -> Path | None:
    raw_path = load_app_config().get("assets", {}).get("default_detected_image")
    if not raw_path:
        return None
    return resolve_path(raw_path)


def get_demo_video_paths() -> dict[str, Path]:
    raw_videos = load_app_config().get("assets", {}).get("demo_videos", {})
    return {
        str(label): resolve_path(path)
        for label, path in raw_videos.items()
    }
