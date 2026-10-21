from __future__ import annotations

from .config import CONFIGS_DIR, load_yaml


def load_class_names(model_key: str | None = None) -> dict[int, str]:
    raw = load_yaml(CONFIGS_DIR / "class_names.yaml")
    if "class_names" in raw:
        return {int(k): str(v) for k, v in raw["class_names"].items()}

    model_class_names = raw.get("model_class_names", {})
    if not model_class_names:
        return {}

    resolved_model_key = model_key or raw.get("default_model_key")
    if resolved_model_key in model_class_names:
        return {
            int(k): str(v)
            for k, v in model_class_names[resolved_model_key].items()
        }

    if len(model_class_names) == 1:
        only_mapping = next(iter(model_class_names.values()))
        return {int(k): str(v) for k, v in only_mapping.items()}

    available = ", ".join(sorted(model_class_names))
    raise KeyError(
        f"Unknown class-name model key '{resolved_model_key}'. Available mappings: {available}"
    )


def get_class_name(class_id: int, class_names: dict[int, str] | None = None) -> str:
    mapping = class_names or load_class_names()
    return mapping.get(class_id, f"class_{class_id}")
