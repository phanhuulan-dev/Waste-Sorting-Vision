from waste_sorting_vision.class_names import get_class_name, load_class_names


def test_waste_v5_class_names_load_expected_mapping() -> None:
    class_names = load_class_names("waste_v5")
    assert class_names[0] == "Paper"
    assert class_names[8] == "Battery"
    assert len(class_names) == 9


def test_unknown_class_name_uses_safe_fallback() -> None:
    assert get_class_name(99, {0: "Paper"}) == "class_99"
