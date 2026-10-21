from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import get_model_catalog, get_model_source, resolve_path

try:
    import streamlit as st
except Exception:
    class _StreamlitShim:
        @staticmethod
        def cache_resource(func=None, **_kwargs):
            def decorator(inner):
                return inner

            if func is None:
                return decorator
            return decorator(func)

    st = _StreamlitShim()

if TYPE_CHECKING:
    from ultralytics import YOLO as YOLOModel
else:
    YOLOModel = Any


class ModelResolutionError(FileNotFoundError):
    """Raised when no configured checkpoint path can be resolved."""


def list_model_options() -> dict[str, str]:
    model_catalog = get_model_catalog()
    return {
        model_key: str(model_config.get("display_name", model_key))
        for model_key, model_config in model_catalog.items()
    }


def resolve_model_path(
    model_key: str,
    env: Mapping[str, str] | None = None,
    must_exist: bool = True,
) -> Path:
    source = get_model_source(model_key)
    env_map = env if env is not None else os.environ
    env_var = source.get("env_var")

    if env_var and env_map.get(env_var):
        override_path = resolve_path(env_map[env_var])
        if must_exist and not override_path.exists():
            raise ModelResolutionError(
                f"Environment override '{env_var}' points to a missing checkpoint: "
                f"{override_path}"
            )
        return override_path

    candidate_paths = [resolve_path(source["artifact_path"])]

    if not must_exist:
        return candidate_paths[0]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    searched_paths = ", ".join(str(path) for path in candidate_paths)
    raise ModelResolutionError(
        f"Unable to resolve checkpoint for '{model_key}'. Checked: {searched_paths}"
    )


def load_detector(model_path: str | Path) -> YOLOModel:
    from ultralytics import YOLO

    return YOLO(str(Path(model_path).expanduser()))


@st.cache_resource(show_spinner=False)
def load_detector_cached(model_path: str) -> YOLOModel:
    return load_detector(model_path)
