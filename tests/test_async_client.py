"""AsyncAtlasClient mirrors AtlasClient; check the async plumbing on a few endpoints."""

import json

import httpx
import pytest

from materials_atlas import (
    AtlasConnectionError,
    EmbeddingSet,
    NotFoundError,
    __version__,
)


async def test_user_agent_names_the_package(async_client, api):
    route = api.get("/health").respond(json={"status": "ok"})
    await async_client.health()
    user_agent = route.calls.last.request.headers["User-Agent"]
    assert user_agent.startswith(f"materials-atlas/{__version__} python-httpx/")


async def test_get_structure(async_client, api, structure_json):
    api.get("/structures/id/mp-149").respond(json=structure_json)
    structure = await async_client.get_structure("mp-149")
    assert structure.formula == "Si"


async def test_not_found(async_client, api):
    api.get("/structures/id/nope").respond(404, json={"detail": "nope"})
    with pytest.raises(NotFoundError):
        await async_client.get_structure("nope")


async def test_search(async_client, api, brief_json):
    route = api.get("/structures").respond(json=[brief_json], headers={"X-Total-Count": "7"})
    page = await async_client.search(band_gap_min=1, crystal_system="Cubic", limit=1)
    assert page.total == 7 and page[0].id == "mp-149"
    params = route.calls.last.request.url.params
    assert params["band_gap_min"] == "1" and params["crystal_system"] == "Cubic"


async def test_iter_search(async_client, api, brief_json):
    def page(request):
        offset = int(request.url.params["offset"])
        items = [{**brief_json, "id": f"mp-{offset + i}"} for i in range(2) if offset + i < 3]
        return httpx.Response(200, json=items, headers={"X-Total-Count": "3"})

    api.get("/structures").mock(side_effect=page)
    ids = [brief.id async for brief in async_client.iter_search(page_size=2)]
    assert ids == ["mp-0", "mp-1", "mp-2"]


async def test_count_ids_stats_neighbors_projection(async_client, api, stats_json):
    api.get("/structures/count").respond(json=5)
    api.get("/structures/ids").respond(json=["mp-1"])
    api.get("/structures/stats").respond(json=stats_json)
    api.get("/structures/mp-1/neighbors").respond(json=[{"id": "mp-2", "similarity": 90.0}])
    api.get("/projection").respond(
        json={"count": 1, "ids": ["mp-1"], "x": [0], "y": [0], "z": [0], "kind": [None]}
    )
    assert await async_client.count(is_stable=True) == 5
    assert await async_client.ids() == ["mp-1"]
    assert (await async_client.stats()).total == stats_json["total"]
    assert (await async_client.neighbors("mp-1"))[0].id == "mp-2"
    assert (await async_client.projection()).count == 1


async def test_health_and_connection_error(async_client, api):
    api.get("/health").mock(side_effect=httpx.ConnectError("down"))
    assert await async_client.health() is False
    api.get("/structures/id/mp-1").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(AtlasConnectionError):
        await async_client.get_structure("mp-1")


async def test_get_embeddings_and_coords(async_client, api, structure_json):
    def card(request, structure_id):
        body = {**structure_json, "id": structure_id}
        if "include" in request.url.params:
            body["embeddings"] = {"mace": [1.0] * 256, "petmad": None}
        return httpx.Response(200, json=body)

    api.get(path__regex=r"/structures/id/(?P<structure_id>.+)").mock(side_effect=card)
    ids = [f"mp-{i}" for i in range(20)]

    embeddings = await async_client.get_embeddings(ids)
    assert embeddings.ids == ids and embeddings.dimension == 256
    petmad = await async_client.get_embeddings(ids, model="petmad")
    assert petmad.ids == [] and petmad.missing == ids

    coords = await async_client.get_3d_coords(ids)
    assert coords.ids == ids and coords.count == 20

    api.get("/projection").respond(
        json={"count": 1, "ids": ["mp-0"], "x": [0.0], "y": [0.0], "z": [0.0], "kind": ["other"]}
    )
    subset = await async_client.get_3d_coords(ids, via_projection=True)
    assert subset.ids == ["mp-0"] and subset.missing == ids[1:]


async def test_neighbors_by_vector_and_batch(async_client, api):
    def answer(request):
        body = json.loads(request.content)
        return httpx.Response(
            200, json=[{"id": f"mp-{int(body['vector'][0])}", "similarity": 99.0}]
        )

    route = api.post("/structures/neighbors").mock(side_effect=answer)
    assert (await async_client.neighbors([7.0] * 256, k=1))[0].id == "mp-7"
    assert json.loads(route.calls.last.request.content)["model"] == "mace"

    emb = EmbeddingSet(model="petmad", ids=list("abc"), vectors=[[i] * 512 for i in (1, 2, 3)])
    hits = await async_client.neighbors_batch(emb, k=1)
    assert [h[0].id for h in hits] == ["mp-1", "mp-2", "mp-3"]  # input order despite concurrency
    assert route.call_count == 4
    assert json.loads(route.calls.last.request.content)["model"] == "petmad"
