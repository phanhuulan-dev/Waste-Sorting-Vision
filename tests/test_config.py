from waste_sorting_vision.config import (
    get_default_image_path,
    get_demo_video_paths,
    load_app_config,
    load_model_sources,
)
from waste_sorting_vision.detector import list_model_options, resolve_model_path


def test_app_config_loads() -> None:
    data = load_app_config()
    assert "defaults" in data
    assert "assets" in data


def test_model_sources_loads() -> None:
    data = load_model_sources()
    assert "models" in data
    assert set(data["models"]) == {"waste_v2", "waste_v3", "waste_v5"}


def test_default_demo_assets_resolve() -> None:
    assert get_default_image_path().name == "default_input.png"
    assert len(get_demo_video_paths()) == 4


def test_model_options_expose_display_names() -> None:
    model_options = list_model_options()
    assert model_options["waste_v2"] == "waste_v2.pt (9 classes)"


def test_model_path_resolves_primary_artifact_path_without_existence_check() -> None:
    resolved_path = resolve_model_path("waste_v3", env={}, must_exist=False)
    assert resolved_path.name == "waste_v3.pt"


def test_model_path_prefers_environment_override(tmp_path) -> None:
    override_path = tmp_path / "override.pt"
    override_path.write_bytes(b"")

    resolved_path = resolve_model_path(
        "waste_v2",
        env={"WSV_MODEL_WASTE_V2": str(override_path)},
        must_exist=True,
    )

    assert resolved_path == override_path.resolve()
