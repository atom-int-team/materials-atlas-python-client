import pytest

from materials_atlas import CRYSTAL_SYSTEMS, RANGE_FIELDS, StructureFilters, describe_filters
from materials_atlas.filters import FILTER_NAMES, build_params


def test_every_range_field_has_min_and_max_keyword():
    for field in RANGE_FIELDS:
        assert f"{field.name}_min" in FILTER_NAMES
        assert f"{field.name}_max" in FILTER_NAMES


def test_passes_known_filters_through():
    params = build_params({"band_gap_min": 1, "band_gap_max": 3.5, "is_stable": True})
    assert params == {"band_gap_min": 1, "band_gap_max": 3.5, "is_stable": True}


def test_drops_none_values():
    assert build_params({"band_gap_min": None, "has_elastic": False}) == {"has_elastic": False}


def test_crystal_system_string_becomes_list():
    assert build_params({"crystal_system": "Cubic"}) == {"crystal_system": ["Cubic"]}
    assert build_params({"crystal_system": ("Cubic", "Hexagonal")}) == {
        "crystal_system": ["Cubic", "Hexagonal"]
    }


def test_unknown_crystal_system_is_rejected():
    with pytest.raises(ValueError, match="cubic"):
        build_params({"crystal_system": "cubic"})


def test_unknown_filter_name_lists_valid_ones():
    with pytest.raises(TypeError) as error:
        build_params({"bandgap_min": 1})
    assert "bandgap_min" in str(error.value)
    assert "band_gap_min" in str(error.value)


def test_min_above_max_is_rejected_before_request():
    with pytest.raises(ValueError, match="band_gap_min"):
        build_params({"band_gap_min": 3, "band_gap_max": 1})


def test_min_equal_max_is_fine():
    assert build_params({"n_atoms_min": 2, "n_atoms_max": 2}) == {
        "n_atoms_min": 2,
        "n_atoms_max": 2,
    }


def test_describe_filters_mentions_everything():
    text = describe_filters()
    for name in FILTER_NAMES:
        assert name in text
    for system in CRYSTAL_SYSTEMS:
        assert system in text


def test_typed_dict_matches_filter_names():
    assert set(StructureFilters.__annotations__) == FILTER_NAMES
