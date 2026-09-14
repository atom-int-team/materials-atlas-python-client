import pandas as pd
import pytest

from materials_atlas import Neighbor, Page, ProjectionCloud, Structure, StructureBrief, to_dataframe


def test_structure_parses_real_card(structure_json):
    structure = Structure.model_validate(structure_json)
    assert structure.id == "mp-149"
    assert structure.formula == "Si"
    assert structure.structural_properties.crystal_system == "Cubic"
    assert structure.electronic_properties.band_gap == pytest.approx(0.6105)
    assert structure.dielectric_properties.tensor is not None
    assert len(structure.dielectric_properties.tensor) == 3
    assert structure.availability.has_elastic is True


def test_to_pymatgen(structure_json):
    pymatgen_structure = Structure.model_validate(structure_json).to_pymatgen()
    assert pymatgen_structure.composition.reduced_formula == "Si"
    assert len(pymatgen_structure) == 2


def test_to_pymatgen_without_structure_json():
    with pytest.raises(ValueError, match="mp-x"):
        Structure(id="mp-x").to_pymatgen()


def test_page_behaves_like_a_list(brief_json):
    briefs = [StructureBrief.model_validate({**brief_json, "id": f"mp-{i}"}) for i in range(3)]
    page = Page(items=briefs, total=10, limit=3, offset=0)
    assert len(page) == 3
    assert page[0].id == "mp-0"
    assert [brief.id for brief in page] == ["mp-0", "mp-1", "mp-2"]
    assert page[1:].__len__() == 2
    assert page.has_more is True
    assert Page(items=briefs, total=3, limit=3, offset=0).has_more is False
    assert Page(items=briefs, total=10, limit=3, offset=7).has_more is False


def test_page_to_dataframe(brief_json):
    page = Page(items=[StructureBrief.model_validate(brief_json)], total=1, limit=1, offset=0)
    frame = page.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["id"]) == ["mp-149"]
    assert "band_gap" in frame.columns


def test_to_dataframe_on_neighbors():
    frame = to_dataframe([Neighbor(id="mp-1", similarity=99.5, formula_reduced="Si")])
    assert frame.loc[0, "similarity"] == 99.5


def test_projection_to_dataframe():
    cloud = ProjectionCloud(
        count=2, ids=["a", "b"], x=[0, 1], y=[0, 1], z=[0, 1], kind=["other", None]
    )
    frame = cloud.to_dataframe()
    assert list(frame.columns) == ["id", "x", "y", "z", "kind"]
    assert len(frame) == 2


def test_embedding_set():
    from materials_atlas import EmbeddingSet

    empty = EmbeddingSet(model="mace", ids=[], vectors=[])
    assert empty.dimension is None and len(empty) == 0
    assert empty.to_numpy().shape == (0,)
    full = EmbeddingSet(
        model="mace", ids=["a", "b"], vectors=[[1.0, 2.0], [3.0, 4.0]], missing=["c"]
    )
    assert full["b"] == [3.0, 4.0]
    with pytest.raises(KeyError):
        full["c"]
    assert full.to_numpy().shape == (2, 2)
    assert full.to_dataframe().loc["a", 1] == 2.0


def test_projection_cloud_to_numpy():
    cloud = ProjectionCloud(
        count=2, ids=["a", "b"], x=[0, 1], y=[2, 3], z=[4, 5], kind=[None, None]
    )
    assert cloud.to_numpy().tolist() == [[0, 2, 4], [1, 3, 5]]
    assert ProjectionCloud(count=0, ids=[], x=[], y=[], z=[], kind=[]).to_numpy().shape == (0, 3)
