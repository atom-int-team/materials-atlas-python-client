"""Against the production 3D Atlas API. Run with ``pytest -m integration``."""

import pytest

from materials_atlas import AsyncAtlasClient, AtlasClient, NotFoundError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def atlas(live_url):
    with AtlasClient(live_url) as client:
        yield client


def test_health_and_count(atlas):
    assert atlas.health() is True
    assert atlas.count() > 0


def test_silicon_card(atlas):
    silicon = atlas.get_structure("mp-149")
    assert silicon.formula == "Si"
    assert silicon.structural_properties.crystal_system == "Cubic"
    assert silicon.to_pymatgen().composition.reduced_formula == "Si"


def test_random_structure(atlas):
    assert atlas.get_random_structure().id


def test_formula_any_spelling(atlas):
    assert {b.id for b in atlas.find_by_formula("SiO2")} == {
        b.id for b in atlas.find_by_formula("O2Si")
    }
    with pytest.raises(NotFoundError):
        atlas.find_by_formula("!!!")


def test_search_count_ids_agree(atlas):
    filters = dict(
        band_gap_min=5, band_gap_max=6, crystal_system=["Cubic", "Hexagonal"], is_stable=True
    )
    page = atlas.search(sort="band_gap", order="desc", limit=10, **filters)
    assert page.total == atlas.count(**filters) == len(atlas.ids(**filters))
    gaps = [b.band_gap for b in page]
    assert gaps == sorted(gaps, reverse=True)
    assert all(5 <= gap <= 6 for gap in gaps)
    assert all(b.crystal_system in ("Cubic", "Hexagonal") for b in page)


def test_iter_search_covers_all_matches(atlas):
    filters = dict(band_gap_min=7)
    ids = [b.id for b in atlas.iter_search(page_size=100, **filters)]
    assert len(ids) == atlas.count(**filters)
    assert len(set(ids)) == len(ids)


def test_stats(atlas):
    stats = atlas.stats(bins=5, crystal_system="Cubic")
    assert stats.total == atlas.count(crystal_system="Cubic")
    assert set(stats.crystal_system) == {"Cubic"}
    assert stats.band_gap is not None and len(stats.band_gap.counts) == 5


def test_neighbors_both_models(atlas):
    for model in ("mace", "petmad"):
        neighbors = atlas.neighbors("mp-149", k=5, model=model)
        assert len(neighbors) == 5
        assert "mp-149" not in {n.id for n in neighbors}
        similarities = [n.similarity for n in neighbors]
        assert similarities == sorted(similarities, reverse=True)


def test_projection(atlas):
    cloud = atlas.projection()
    assert cloud.count == len(cloud.ids) == len(cloud.x) == len(cloud.y) == len(cloud.z)
    assert cloud.count > 0
    assert cloud.to_dataframe().shape == (cloud.count, 5)


def test_errors(atlas):
    with pytest.raises(NotFoundError):
        atlas.get_structure("does-not-exist")


async def test_async_client(live_url):
    async with AsyncAtlasClient(live_url) as atlas:
        assert (await atlas.get_structure("mp-149")).formula == "Si"
        assert (await atlas.count(is_stable=True)) > 0
        first = [b async for b in atlas.iter_search(band_gap_min=8, page_size=10)]
        assert len(first) == await atlas.count(band_gap_min=8)


def test_embeddings_in_card(atlas):
    plain = atlas.get_structure("mp-149")
    assert plain.embeddings is None
    card = atlas.get_structure("mp-149", include_embeddings=True)
    assert card.embeddings is not None
    assert len(card.embeddings.mace) == 256
    assert len(card.embeddings.petmad) == 512


def test_get_embeddings_by_formula(atlas):
    records = atlas.find_by_formula("Si")
    emb = atlas.get_embeddings(records)
    assert emb.ids == [r.id for r in records]
    assert emb.dimension == 256 and emb.missing == []
    assert emb.to_numpy().shape == (len(records), 256)
    petmad = atlas.get_embeddings(records, model="petmad")
    assert len(petmad) + len(petmad.missing) == len(records)


def test_neighbors_are_close_in_embedding_space(atlas):
    neighbors = atlas.neighbors("mp-149", k=3)
    emb = atlas.get_embeddings(["mp-149", *neighbors])
    import numpy as np

    matrix = emb.to_numpy()
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = unit[1:] @ unit[0]
    assert np.allclose(cosine * 100, [n.similarity for n in neighbors], atol=0.05)


def test_neighbors_of_own_vector_is_itself(atlas):
    silicon = atlas.get_structure("mp-149", include_embeddings=True)
    hits = atlas.neighbors(silicon.embeddings.mace, k=3)  # model inferred from 256 components
    assert hits[0].id == "mp-149" and hits[0].similarity == pytest.approx(100.0, abs=0.01)
    assert [h.id for h in hits[1:]] == [n.id for n in atlas.neighbors("mp-149", k=2)]
    if silicon.embeddings.petmad is not None:
        assert atlas.neighbors(silicon.embeddings.petmad, k=1)[0].id == "mp-149"


def test_neighbors_batch_of_embedding_set(atlas):
    emb = atlas.get_embeddings(atlas.find_by_formula("NaCl"))
    hits = atlas.neighbors_batch(emb, k=2)
    assert len(hits) == len(emb) and all(len(h) == 2 for h in hits)
    assert [h[0].id for h in hits] == emb.ids


def test_neighbors_of_a_mixed_vector(atlas):
    pytest.importorskip("numpy")
    pair = atlas.get_embeddings(["mp-149", "mp-32"])  # Si and Ge
    midpoint = pair.to_numpy().mean(axis=0)
    hits = atlas.neighbors(midpoint, k=5)
    assert "SiGe" in {h.formula_reduced for h in hits}  # halfway between Si and Ge sits SiGe


def test_neighbors_vector_rejected_by_server(atlas):
    from materials_atlas import ValidationError

    with pytest.raises(ValidationError, match="zeros"):
        atlas.neighbors([0.0] * 256)


def test_get_3d_coords_both_paths_agree(atlas):
    records = atlas.find_by_formula("SiO2")  # a few hundred, above the card limit
    via_cloud = atlas.get_3d_coords(records)
    sample = records[:5]
    via_cards = atlas.get_3d_coords(sample, via_projection=False)
    sample_ids = [r.id for r in sample]
    assert set(via_cards.ids) | set(via_cards.missing) == set(sample_ids)
    assert via_cards.ids == [i for i in sample_ids if i in via_cards.ids]  # requested order kept
    for structure_id, x in zip(via_cards.ids, via_cards.x, strict=True):
        assert via_cloud.x[via_cloud.ids.index(structure_id)] == pytest.approx(x)
    assert via_cloud.count + len(via_cloud.missing) == len(records)
    assert via_cloud.to_dataframe().shape == (via_cloud.count, 5)


def test_single_id_coords(atlas):
    coords = atlas.get_3d_coords("mp-149")
    card = atlas.get_structure("mp-149")
    assert coords.ids == ["mp-149"]
    assert coords.x == [pytest.approx(card.projection.x)]
