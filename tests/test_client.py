"""Unit tests for AtlasClient against a mocked server (respx)."""

import json

import httpx
import pytest

from materials_atlas import (
    AtlasAPIError,
    AtlasClient,
    AtlasConnectionError,
    EmbeddingSet,
    IndexNotReadyError,
    NotFoundError,
    StructureBrief,
    ValidationError,
    __version__,
)
from materials_atlas.client import DEFAULT_URL


def test_base_url_trailing_slash_is_stripped():
    assert AtlasClient("http://x/").base_url == "http://x"


def test_default_url_is_production():
    with AtlasClient() as atlas:
        assert atlas.base_url == DEFAULT_URL == "https://atlas3d.api.atom-int.com"


def test_user_agent_names_the_package(client, api):
    route = api.get("/health").respond(json={"status": "ok"})
    client.health()
    user_agent = route.calls.last.request.headers["User-Agent"]
    assert user_agent.startswith(f"materials-atlas/{__version__} python-httpx/")


def test_health(client, api):
    api.get("/health").respond(json={"status": "ok"})
    assert client.health() is True


def test_health_false_when_unreachable(client, api):
    api.get("/health").mock(side_effect=httpx.ConnectError("boom"))
    assert client.health() is False


def test_get_structure(client, api, structure_json):
    api.get("/structures/id/mp-149").respond(json=structure_json)
    structure = client.get_structure("mp-149")
    assert structure.id == "mp-149"
    assert structure.formula == "Si"


def test_get_structure_not_found(client, api):
    api.get("/structures/id/nope").respond(404, json={"detail": "Structure nope not found"})
    with pytest.raises(NotFoundError) as error:
        client.get_structure("nope")
    assert error.value.status_code == 404
    assert error.value.detail == "Structure nope not found"


def test_get_random_structure(client, api, structure_json):
    api.get("/structures/random_structure").respond(json=structure_json)
    assert client.get_random_structure().id == "mp-149"


def test_find_by_formula(client, api, brief_json):
    api.get("/structures/formula/SiO2").respond(json=[brief_json])
    briefs = client.find_by_formula("SiO2")
    assert [brief.id for brief in briefs] == ["mp-149"]


def test_find_by_formula_invalid(client, api):
    api.get("/structures/formula/!!!").respond(400, json={"detail": "Invalid formula"})
    with pytest.raises(ValidationError):
        client.find_by_formula("!!!")


def test_search_sends_filters_and_reads_total(client, api, brief_json):
    route = api.get("/structures").respond(json=[brief_json], headers={"X-Total-Count": "2772"})
    page = client.search(
        band_gap_min=1,
        band_gap_max=3,
        crystal_system=["Cubic", "Hexagonal"],
        is_stable=True,
        sort="band_gap",
        order="desc",
        limit=50,
        offset=100,
    )
    assert page.total == 2772
    assert page.limit == 50 and page.offset == 100
    assert page[0].formula_reduced == "Si"

    params = route.calls.last.request.url.params
    assert params["band_gap_min"] == "1"
    assert params["band_gap_max"] == "3"
    assert params.get_list("crystal_system") == ["Cubic", "Hexagonal"]
    assert params["is_stable"] == "true"
    assert params["sort"] == "band_gap"
    assert params["order"] == "desc"
    assert params["limit"] == "50"
    assert params["offset"] == "100"


def test_search_without_filters_sends_only_paging(client, api):
    route = api.get("/structures").respond(json=[], headers={"X-Total-Count": "0"})
    page = client.search()
    assert len(page) == 0 and page.total == 0
    assert dict(route.calls.last.request.url.params) == {
        "sort": "id",
        "order": "asc",
        "limit": "100",
        "offset": "0",
    }


def test_search_rejects_bad_paging_before_request(client, api):
    route = api.get("/structures")
    with pytest.raises(ValueError, match="limit"):
        client.search(limit=0)
    with pytest.raises(ValueError, match="limit"):
        client.search(limit=1001)
    with pytest.raises(ValueError, match="offset"):
        client.search(offset=-1)
    with pytest.raises(TypeError, match="unknown filter"):
        client.search(bandgap_min=1)
    with pytest.raises(ValueError, match="band_gap_min"):
        client.search(band_gap_min=3, band_gap_max=1)
    assert not route.called


def test_search_server_validation_error(client, api):
    api.get("/structures").respond(
        422, json={"detail": [{"loc": ["query", "limit"], "msg": "too big"}]}
    )
    with pytest.raises(ValidationError, match="query.limit: too big"):
        client.search()


def test_iter_search_walks_every_page(client, api, brief_json):
    def page(request):
        offset = int(request.url.params["offset"])
        items = [{**brief_json, "id": f"mp-{offset + i}"} for i in range(2) if offset + i < 5]
        return httpx.Response(200, json=items, headers={"X-Total-Count": "5"})

    route = api.get("/structures").mock(side_effect=page)
    ids = [brief.id for brief in client.iter_search(page_size=2, is_stable=True)]
    assert ids == ["mp-0", "mp-1", "mp-2", "mp-3", "mp-4"]
    assert route.call_count == 3
    assert all(call.request.url.params["is_stable"] == "true" for call in route.calls)


def test_count(client, api):
    route = api.get("/structures/count").respond(json=2734)
    assert client.count(is_stable=True, has_elastic=True, bulk_modulus_voigt_min=100) == 2734
    params = route.calls.last.request.url.params
    assert params["bulk_modulus_voigt_min"] == "100"
    assert params["has_elastic"] == "true"


def test_ids(client, api):
    route = api.get("/structures/ids").respond(json=["mp-1", "mp-2"])
    assert client.ids(energy_above_hull_max=0.05) == ["mp-1", "mp-2"]
    assert route.calls.last.request.url.params["energy_above_hull_max"] == "0.05"


def test_stats(client, api, stats_json):
    route = api.get("/structures/stats").respond(json=stats_json)
    stats = client.stats(bins=3, crystal_system="Cubic")
    assert stats.total == stats_json["total"]
    assert stats.band_gap is not None
    assert len(stats.band_gap.bin_edges) == len(stats.band_gap.counts) + 1
    assert stats.crystal_system == {"Cubic": stats_json["total"]}
    assert set(stats.is_stable) == {"true", "false"}
    params = route.calls.last.request.url.params
    assert params["bins"] == "3"
    assert params["crystal_system"] == "Cubic"


def test_stats_rejects_bad_bins(client, api):
    with pytest.raises(ValueError, match="bins"):
        client.stats(bins=0)


def test_neighbors(client, api):
    route = api.get("/structures/mp-149/neighbors").respond(
        json=[{"id": "mp-165", "similarity": 99.9983, "formula_reduced": "Si"}]
    )
    neighbors = client.neighbors("mp-149", k=1, model="petmad")
    assert neighbors[0].id == "mp-165"
    assert neighbors[0].similarity == pytest.approx(99.9983)
    params = route.calls.last.request.url.params
    assert params["k"] == "1" and params["model"] == "petmad"


def test_neighbors_defaults(client, api):
    route = api.get("/structures/mp-149/neighbors").respond(json=[])
    client.neighbors("mp-149")
    params = route.calls.last.request.url.params
    assert params["k"] == "50" and params["model"] == "mace"


def test_neighbors_rejects_bad_k(client, api):
    with pytest.raises(ValueError, match="k must"):
        client.neighbors("mp-149", k=501)


def test_neighbors_index_not_ready(client, api):
    api.get("/structures/mp-149/neighbors").respond(503, json={"detail": "'mace' index not built"})
    with pytest.raises(IndexNotReadyError):
        client.neighbors("mp-149")


def _hit(structure_id="mp-149", similarity=100.0):
    return {"id": structure_id, "similarity": similarity, "formula_reduced": "Si"}


def test_neighbors_by_vector_posts_body(client, api):
    route = api.post("/structures/neighbors").respond(json=[_hit()])
    neighbors = client.neighbors([0.5] * 256, k=3)
    assert neighbors[0].id == "mp-149" and neighbors[0].similarity == 100.0
    body = json.loads(route.calls.last.request.content)
    assert body == {"vector": [0.5] * 256, "model": "mace", "k": 3}


def test_neighbors_by_vector_infers_petmad_from_length(client, api):
    route = api.post("/structures/neighbors").respond(json=[])
    client.neighbors((1.0,) * 512)
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "petmad" and body["k"] == 50 and len(body["vector"]) == 512


def test_neighbors_by_vector_accepts_numpy(client, api):
    numpy = pytest.importorskip("numpy")
    route = api.post("/structures/neighbors").respond(json=[])
    client.neighbors(numpy.arange(256, dtype=numpy.float32), model="mace")
    body = json.loads(route.calls.last.request.content)
    assert body["vector"][:3] == [0.0, 1.0, 2.0] and body["model"] == "mace"


def test_neighbors_by_vector_rejects_bad_length_before_request(client, api):
    route = api.post("/structures/neighbors").respond(json=[])
    with pytest.raises(ValueError, match="'petmad' vectors have 512"):
        client.neighbors([1.0] * 256, model="petmad")
    with pytest.raises(ValueError, match="cannot tell the model"):
        client.neighbors([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="k must"):
        client.neighbors([1.0] * 256, k=0)
    assert not route.called


def test_neighbors_by_vector_rejects_2d_input(client, api):
    with pytest.raises(TypeError, match="neighbors_batch"):
        client.neighbors([[1.0] * 256, [2.0] * 256])


def test_neighbors_by_vector_server_validation_error(client, api):
    api.post("/structures/neighbors").respond(
        422, json={"detail": [{"loc": ["body"], "msg": "vector is all zeros"}]}
    )
    with pytest.raises(ValidationError, match="all zeros"):
        client.neighbors([0.0] * 256)


def test_neighbors_batch_keeps_order(client, api):
    def answer(request):
        first = json.loads(request.content)["vector"][0]
        return httpx.Response(200, json=[_hit(f"mp-{int(first)}")])

    route = api.post("/structures/neighbors").mock(side_effect=answer)
    hits = client.neighbors_batch([[1.0] * 256, [2.0] * 256, [3.0] * 256], k=1)
    assert [h[0].id for h in hits] == ["mp-1", "mp-2", "mp-3"]
    assert route.call_count == 3
    assert json.loads(route.calls.last.request.content)["k"] == 1


def test_neighbors_batch_from_embedding_set_uses_its_model(client, api):
    route = api.post("/structures/neighbors").respond(json=[])
    emb = EmbeddingSet(model="petmad", ids=["a", "b"], vectors=[[1.0] * 512, [2.0] * 512])
    assert client.neighbors_batch(emb) == [[], []]
    assert all(json.loads(c.request.content)["model"] == "petmad" for c in route.calls)


def test_neighbors_batch_from_2d_numpy(client, api):
    numpy = pytest.importorskip("numpy")
    route = api.post("/structures/neighbors").respond(json=[])
    client.neighbors_batch(numpy.ones((2, 256)))
    assert route.call_count == 2


def test_neighbors_batch_validates_everything_before_the_first_request(client, api):
    route = api.post("/structures/neighbors").respond(json=[])
    with pytest.raises(ValueError):
        client.neighbors_batch([[1.0] * 256, [1.0] * 7])
    assert not route.called
    assert client.neighbors_batch([]) == []


def test_projection(client, api):
    api.get("/projection").respond(
        json={"count": 1, "ids": ["mp-1"], "x": [0.1], "y": [0.2], "z": [0.3], "kind": ["other"]}
    )
    cloud = client.projection()
    assert cloud.count == 1 and cloud.ids == ["mp-1"] and cloud.x == [0.1]


def test_unknown_error_status(client, api):
    api.get("/health").respond(500, text="boom")
    with pytest.raises(AtlasAPIError) as error:
        client._get("/health")
    assert error.value.status_code == 500
    assert error.value.detail == "boom"


def test_connection_error(client, api):
    api.get("/structures/id/mp-149").mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(AtlasConnectionError, match="mp-149"):
        client.get_structure("mp-149")


# --- get_structure(include_embeddings=), get_embeddings, get_3d_coords ------------------


def _card_with_embeddings(structure_json, structure_id="mp-149", petmad=True):
    return {
        **structure_json,
        "id": structure_id,
        "embeddings": {"mace": [0.1] * 256, "petmad": [0.2] * 512 if petmad else None},
    }


def test_get_structure_include_embeddings(client, api, structure_json):
    route = api.get("/structures/id/mp-149").respond(json=_card_with_embeddings(structure_json))
    card = client.get_structure("mp-149", include_embeddings=True)
    assert route.calls.last.request.url.params["include"] == "embeddings"
    assert card.embeddings is not None
    assert len(card.embeddings.mace) == 256
    assert len(card.embeddings.petmad) == 512


def test_get_structure_default_has_no_embeddings(client, api, structure_json):
    route = api.get("/structures/id/mp-149").respond(json=structure_json)
    card = client.get_structure("mp-149")
    assert "include" not in route.calls.last.request.url.params
    assert card.embeddings is None


def test_get_embeddings_accepts_ids_and_records(client, api, structure_json, brief_json):
    def card(request, structure_id):
        return httpx.Response(200, json=_card_with_embeddings(structure_json, structure_id))

    route = api.get(path__regex=r"/structures/id/(?P<structure_id>.+)").mock(side_effect=card)
    embeddings = client.get_embeddings(
        ["mp-1", StructureBrief.model_validate({**brief_json, "id": "mp-2"})]
    )
    assert embeddings.model == "mace"
    assert embeddings.ids == ["mp-1", "mp-2"]
    assert embeddings.dimension == 256
    assert len(embeddings) == 2
    assert embeddings["mp-2"] == [0.1] * 256
    assert embeddings.missing == []
    assert all(call.request.url.params["include"] == "embeddings" for call in route.calls)
    single = client.get_embeddings("mp-1")
    assert single.ids == ["mp-1"]


def test_get_embeddings_reports_missing_petmad(client, api, structure_json):
    def card(request, structure_id):
        has_petmad = structure_id != "mp-2"
        return httpx.Response(
            200, json=_card_with_embeddings(structure_json, structure_id, has_petmad)
        )

    api.get(path__regex=r"/structures/id/(?P<structure_id>.+)").mock(side_effect=card)
    embeddings = client.get_embeddings(["mp-1", "mp-2", "mp-3"], model="petmad")
    assert embeddings.ids == ["mp-1", "mp-3"]
    assert embeddings.missing == ["mp-2"]
    assert embeddings.dimension == 512
    assert embeddings.to_numpy().shape == (2, 512)
    assert list(embeddings.to_dataframe().index) == ["mp-1", "mp-3"]


def test_get_embeddings_unknown_id_raises(client, api):
    api.get("/structures/id/nope").respond(404, json={"detail": "not found"})
    with pytest.raises(NotFoundError):
        client.get_embeddings(["nope"])


def test_get_embeddings_rejects_garbage(client):
    with pytest.raises(TypeError, match="record with an .id"):
        client.get_embeddings([42])


def test_get_3d_coords_via_cards(client, api, structure_json):
    def card(request, structure_id):
        body = {**structure_json, "id": structure_id}
        if structure_id == "mp-2":
            body["projection"] = {"x": None, "y": None, "z": None, "kind": None}
        return httpx.Response(200, json=body)

    cards = api.get(path__regex=r"/structures/id/(?P<structure_id>.+)").mock(side_effect=card)
    cloud_route = api.get("/projection")
    coords = client.get_3d_coords(["mp-1", "mp-2", "mp-3"])
    assert coords.count == 2
    assert coords.ids == ["mp-1", "mp-3"]
    assert coords.missing == ["mp-2"]
    assert coords.x[0] == pytest.approx(structure_json["projection"]["x"])
    assert coords.kind == ["other", "other"]
    assert coords.to_numpy().shape == (2, 3)
    assert cards.call_count == 3
    assert not cloud_route.called


def test_get_3d_coords_via_projection_above_limit(client, api):
    from materials_atlas.client import COORDS_CARD_LIMIT

    ids = [f"mp-{i}" for i in range(COORDS_CARD_LIMIT + 1)]
    cloud = {
        "count": len(ids),
        "ids": list(reversed(ids)),
        "x": [float(i) for i in range(len(ids))],
        "y": [0.0] * len(ids),
        "z": [1.0] * len(ids),
        "kind": ["other"] * len(ids),
    }
    cloud_route = api.get("/projection").respond(json=cloud)
    cards = api.get(path__regex=r"/structures/id/.+")
    coords = client.get_3d_coords(ids + ["mp-absent"])
    assert cloud_route.call_count == 1
    assert not cards.called
    assert coords.ids == ids  # requested order, not cloud order
    assert coords.x[0] == float(len(ids) - 1)
    assert coords.missing == ["mp-absent"]


def test_get_3d_coords_via_projection_forced(client, api):
    api.get("/projection").respond(
        json={"count": 1, "ids": ["mp-1"], "x": [1.0], "y": [2.0], "z": [3.0], "kind": [None]}
    )
    coords = client.get_3d_coords("mp-1", via_projection=True)
    assert coords.count == 1 and coords.x == [1.0] and coords.kind == [None]
