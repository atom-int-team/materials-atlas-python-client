"""The client classes.

``AtlasClient`` is synchronous, ``AsyncAtlasClient`` is the same API with ``await``. Both keep
one ``httpx`` connection pool open, so use them as context managers or call ``close()``.

Every public method maps to exactly one endpoint of the 3D Atlas API; the shared pieces that
never touch the network (parameter building, response parsing) live in ``_BaseClient``.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any

import httpx

from ._version import __version__
from .exceptions import AtlasConnectionError, raise_for_status
from .filters import EmbeddingModel, SortField, SortOrder, StructureFilters, build_params
from .models import (
    EmbeddingSet,
    Neighbor,
    Page,
    ProjectionCloud,
    Structure,
    StructureBrief,
    StructureStats,
)

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

DEFAULT_URL = "https://atlas3d.api.atom-int.com"
DEFAULT_TIMEOUT = 60.0
# Sent with every request so the backend logs show which client version is calling.
USER_AGENT = f"materials-atlas/{__version__} python-httpx/{httpx.__version__}"
MAX_PAGE_SIZE = 1000  # the server rejects a larger `limit`
MAX_NEIGHBORS = 500  # the server rejects a larger `k`
EMBEDDING_DIMS: dict[str, int] = {"mace": 256, "petmad": 512}  # what the server expects
# `get_3d_coords` fetches one card per structure up to this many ids; beyond it, one call for
# the whole point cloud (~4 MB) is cheaper than that many round trips.
COORDS_CARD_LIMIT = 50
# How many card requests the async client keeps in flight at once for batch methods.
ASYNC_CONCURRENCY = 8

# What the batch methods accept: one id, or any iterable of ids / records with an `.id`
# (`StructureBrief`, `Structure`, `Neighbor`) — so `find_by_formula(...)` output plugs in as is.
IdsOrRecords = str | Iterable[str | StructureBrief | Structure | Neighbor]
# What `neighbors` / `neighbors_batch` accept as a vector: any 1-D sequence of numbers or an
# object with `.tolist()` (numpy, torch, jax). Typed loosely — there is no numpy at import time.
VectorLike = Iterable[float]


class _BaseClient:
    """Everything the sync and async clients share: settings, parameter building, parsing."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float | None = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- request parameters -------------------------------------------------------------

    @staticmethod
    def _search_params(
        sort: SortField, order: SortOrder, limit: int, offset: int, filters: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must not be negative, got {offset}")
        return {
            **build_params(filters),
            "sort": sort,
            "order": order,
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _neighbors_params(k: int, model: EmbeddingModel) -> dict[str, Any]:
        if not 1 <= k <= MAX_NEIGHBORS:
            raise ValueError(f"k must be between 1 and {MAX_NEIGHBORS}, got {k}")
        return {"k": k, "model": model}

    @staticmethod
    def _as_vector(vector: Any) -> list[float]:
        """A 1-D array-like → ``list[float]``; strings and nested sequences are rejected."""
        if isinstance(vector, str):
            raise TypeError("expected a structure id or a vector, got a string that is not an id")
        if hasattr(vector, "tolist"):  # numpy / torch / jax without importing them
            vector = vector.tolist()
        try:
            return [float(x) for x in vector]
        except TypeError:
            raise TypeError(
                "expected a 1-D vector of numbers; for several vectors use neighbors_batch"
            ) from None

    @staticmethod
    def _neighbor_query(
        values: list[float], k: int, model: EmbeddingModel | None
    ) -> dict[str, Any]:
        """Body of ``POST /structures/neighbors``; infers ``model`` from the length if not given."""
        if not 1 <= k <= MAX_NEIGHBORS:
            raise ValueError(f"k must be between 1 and {MAX_NEIGHBORS}, got {k}")
        if model is None:
            by_length = {dim: name for name, dim in EMBEDDING_DIMS.items()}
            if len(values) not in by_length:
                raise ValueError(
                    f"cannot tell the model from a vector of {len(values)} components: "
                    "mace has 256, petmad has 512"
                )
            model = by_length[len(values)]  # type: ignore[assignment]
        elif len(values) != EMBEDDING_DIMS[model]:
            raise ValueError(
                f"'{model}' vectors have {EMBEDDING_DIMS[model]} components, got {len(values)}"
            )
        return {"vector": values, "model": model, "k": k}

    @classmethod
    def _neighbor_queries(
        cls, vectors: Any, k: int, model: EmbeddingModel | None
    ) -> list[dict[str, Any]]:
        """Bodies for ``neighbors_batch``: a list of vectors, a 2-D array or an ``EmbeddingSet``.

        Every vector is checked before the first request goes out, so a bad batch leaves no
        half-finished work behind.
        """
        if isinstance(vectors, EmbeddingSet):
            model = model or vectors.model  # type: ignore[assignment]
            vectors = vectors.vectors
        elif hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [cls._neighbor_query(cls._as_vector(v), k, model) for v in vectors]

    @staticmethod
    def _stats_params(bins: int, filters: Mapping[str, Any]) -> dict[str, Any]:
        if not 1 <= bins <= 500:
            raise ValueError(f"bins must be between 1 and 500, got {bins}")
        return {**build_params(filters), "bins": bins}

    @staticmethod
    def _ids_of(ids_or_records: Any) -> list[str]:
        """Normalise ``get_embeddings`` / ``get_3d_coords`` input to a list of ids, order kept."""
        if isinstance(ids_or_records, str):
            return [ids_or_records]
        ids: list[str] = []
        for item in ids_or_records:
            if isinstance(item, str):
                ids.append(item)
            elif hasattr(item, "id") and isinstance(item.id, str):
                ids.append(item.id)
            else:
                raise TypeError(f"expected a structure id or a record with an .id, got {item!r}")
        return ids

    @staticmethod
    def _card_params(include_embeddings: bool) -> dict[str, Any] | None:
        return {"include": ["embeddings"]} if include_embeddings else None

    # --- response parsing ---------------------------------------------------------------

    @staticmethod
    def _parse_page(response: httpx.Response, limit: int, offset: int) -> Page:
        items = [StructureBrief.model_validate(item) for item in response.json()]
        total = int(response.headers.get("X-Total-Count", len(items)))
        return Page(items=items, total=total, limit=limit, offset=offset)

    @staticmethod
    def _parse_briefs(response: httpx.Response) -> list[StructureBrief]:
        return [StructureBrief.model_validate(item) for item in response.json()]

    @staticmethod
    def _parse_neighbors(response: httpx.Response) -> list[Neighbor]:
        return [Neighbor.model_validate(item) for item in response.json()]

    @staticmethod
    def _embedding_set(cards: list[Structure], model: EmbeddingModel) -> EmbeddingSet:
        """Collect one model's vectors from cards fetched with ``include_embeddings=True``."""
        found = EmbeddingSet(model=model, ids=[], vectors=[])
        for card in cards:
            vector = getattr(card.embeddings, model, None) if card.embeddings else None
            if vector is None:
                found.missing.append(card.id)
            else:
                found.ids.append(card.id)
                found.vectors.append(vector)
        return found

    @staticmethod
    def _coords_from_cards(cards: list[Structure]) -> ProjectionCloud:
        cloud = ProjectionCloud(count=0, ids=[], x=[], y=[], z=[], kind=[])
        for card in cards:
            point = card.projection
            if point.x is None or point.y is None or point.z is None:
                cloud.missing.append(card.id)
                continue
            cloud.ids.append(card.id)
            cloud.x.append(point.x)
            cloud.y.append(point.y)
            cloud.z.append(point.z)
            cloud.kind.append(point.kind)
        cloud.count = len(cloud.ids)
        return cloud

    @staticmethod
    def _coords_from_cloud(cloud: ProjectionCloud, ids: list[str]) -> ProjectionCloud:
        """Pick ``ids`` out of the whole point cloud, in the order they were asked for."""
        position = {structure_id: i for i, structure_id in enumerate(cloud.ids)}
        subset = ProjectionCloud(count=0, ids=[], x=[], y=[], z=[], kind=[])
        for structure_id in ids:
            i = position.get(structure_id)
            if i is None:
                subset.missing.append(structure_id)
                continue
            subset.ids.append(structure_id)
            subset.x.append(cloud.x[i])
            subset.y.append(cloud.y[i])
            subset.z.append(cloud.z[i])
            subset.kind.append(cloud.kind[i])
        subset.count = len(subset.ids)
        return subset


class AtlasClient(_BaseClient):
    """Synchronous client for the 3D Atlas API.

    Args:
        base_url: the API to talk to; the public 3D Atlas at
            ``https://atlas3d.api.atom-int.com`` by default.
        timeout: seconds to wait for a response (``None`` = no limit). ``projection()`` is the
            only heavy call — several megabytes — and gets three times this value.

    Example::

        with AtlasClient() as atlas:
            silicon = atlas.get_structure("mp-149")
            page = atlas.search(band_gap_min=1, band_gap_max=3, crystal_system="Cubic")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float | None = DEFAULT_TIMEOUT,
    ):
        super().__init__(base_url, timeout=timeout)
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )

    # --- plumbing -----------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise AtlasConnectionError(f"{method} {self.base_url}{path}: {error}") from error
        raise_for_status(response)
        return response

    def _get(
        self, path: str, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        return self._request("GET", path, params=params, **kwargs)

    def close(self) -> None:
        """Close the connection pool."""
        self._http.close()

    def __enter__(self) -> AtlasClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- single structures --------------------------------------------------------------

    def health(self) -> bool:
        """``True`` when the backend and its database answer."""
        try:
            return self._get("/health").json().get("status") == "ok"
        except AtlasConnectionError:
            return False

    def get_structure(self, structure_id: str, *, include_embeddings: bool = False) -> Structure:
        """The full card of one structure, e.g. ``"mp-149"``. Raises ``NotFoundError``.

        ``include_embeddings=True`` also fills ``card.embeddings`` (``mace`` and ``petmad``
        vectors); off by default because they add several kilobytes to the response.
        """
        params = self._card_params(include_embeddings)
        return Structure.model_validate(self._get(f"/structures/id/{structure_id}", params).json())

    def get_random_structure(self) -> Structure:
        """The full card of a random structure."""
        return Structure.model_validate(self._get("/structures/random_structure").json())

    def find_by_formula(self, formula: str) -> list[StructureBrief]:
        """Every structure with this composition; any spelling of the formula matches.

        Raises ``NotFoundError`` when nothing matches (this includes strings that are not a
        formula at all).
        """
        return self._parse_briefs(self._get(f"/structures/formula/{formula}"))

    # --- filtered lists -----------------------------------------------------------------

    def search(
        self,
        *,
        sort: SortField = "id",
        order: SortOrder = "asc",
        limit: int = 100,
        offset: int = 0,
        **filters: Unpack[StructureFilters],
    ) -> Page:
        """One page of structures matching the filters, as compact records.

        Args:
            sort: property to order by; missing values sort last in both directions.
            order: ``"asc"`` or ``"desc"``.
            limit: page size, 1..1000.
            offset: how many matches to skip.
            **filters: see ``StructureFilters`` / ``describe_filters()``. Numeric bounds
                ``<field>_min`` / ``<field>_max`` are inclusive, ``crystal_system`` takes a
                name or a list of names, the rest are booleans.

        Returns:
            ``Page`` — iterate it for the records, read ``.total`` for the overall match count.
        """
        params = self._search_params(sort, order, limit, offset, filters)
        return self._parse_page(self._get("/structures", params), limit, offset)

    def iter_search(
        self,
        *,
        sort: SortField = "id",
        order: SortOrder = "asc",
        page_size: int = MAX_PAGE_SIZE,
        **filters: Unpack[StructureFilters],
    ) -> Iterator[StructureBrief]:
        """Every structure matching the filters, fetched page by page as you iterate.

        Same arguments as ``search`` except for ``limit`` / ``offset``, which it manages itself.
        """
        offset = 0
        while True:
            page = self.search(sort=sort, order=order, limit=page_size, offset=offset, **filters)
            yield from page
            if not page.has_more:
                return
            offset += page_size

    def count(self, **filters: Unpack[StructureFilters]) -> int:
        """How many structures match the filters; without filters, the size of the database."""
        return int(self._get("/structures/count", build_params(filters)).json())

    def ids(self, **filters: Unpack[StructureFilters]) -> list[str]:
        """IDs of every structure matching the filters, sorted. ~210k strings without filters."""
        return list(self._get("/structures/ids", build_params(filters)).json())

    def stats(self, *, bins: int = 50, **filters: Unpack[StructureFilters]) -> StructureStats:
        """Min / max / histogram of every numeric property and facet counts over the matches."""
        params = self._stats_params(bins, filters)
        return StructureStats.model_validate(self._get("/structures/stats", params).json())

    # --- embeddings and the atlas -------------------------------------------------------

    def neighbors(
        self,
        query: str | VectorLike,
        *,
        k: int = 50,
        model: EmbeddingModel | None = None,
    ) -> list[Neighbor]:
        """The ``k`` structures closest to a structure or to an embedding vector, closest first.

        Args:
            query: a structure id (``"mp-149"``) or an embedding vector — a list of floats,
                a 1-D numpy array, anything with ``.tolist()``. With an id the structure
                itself is left out of the result; with a vector nothing is (a structure's own
                vector returns that structure first with similarity 100).
            k: how many neighbours, 1..500.
            model: ``"mace"`` (256d, every structure has one) or ``"petmad"`` (512d, ~5%
                missing). Default: ``"mace"`` for an id; for a vector the model is inferred
                from its length.

        Raises ``ValueError`` when the length does not fit the model, ``NotFoundError`` for an
        id without that embedding, ``ValidationError`` (422) when the server rejects the vector
        (NaN, all zeros) and ``IndexNotReadyError`` when the index for that model is not built.

        Example::

            atlas.neighbors("mp-149", k=10)
            atlas.neighbors(my_model_output, k=10)          # 256 floats → mace
        """
        if isinstance(query, str):
            params = self._neighbors_params(k, model or "mace")
            return self._parse_neighbors(self._get(f"/structures/{query}/neighbors", params))
        body = self._neighbor_query(self._as_vector(query), k, model)
        return self._parse_neighbors(self._request("POST", "/structures/neighbors", json=body))

    def neighbors_batch(
        self, vectors: Any, *, k: int = 50, model: EmbeddingModel | None = None
    ) -> list[list[Neighbor]]:
        """Neighbours of several vectors, one request per vector, results in input order.

        Args:
            vectors: a list of vectors, a 2-D array (rows are vectors) or an ``EmbeddingSet``
                (its ``model`` is used unless ``model`` is given).
            k: how many neighbours per vector, 1..500.
            model: as in ``neighbors``; inferred from the vector length when ``None``.

        Example::

            emb = atlas.get_embeddings(atlas.find_by_formula("SiO2"))
            hits = atlas.neighbors_batch(emb, k=5)          # hits[i] belongs to emb.ids[i]
        """
        return [
            self._parse_neighbors(self._request("POST", "/structures/neighbors", json=body))
            for body in self._neighbor_queries(vectors, k, model)
        ]

    def projection(self) -> ProjectionCloud:
        """The whole 3D point cloud of the atlas (~155k points, a few MB)."""
        timeout = None if self.timeout is None else self.timeout * 3
        return ProjectionCloud.model_validate(self._get("/projection", timeout=timeout).json())

    def get_embeddings(
        self, ids_or_records: IdsOrRecords, *, model: EmbeddingModel = "mace"
    ) -> EmbeddingSet:
        """Embedding vectors of several structures, one card request per structure.

        Args:
            ids_or_records: one id, a list of ids, or records with an ``.id`` — the output of
                ``find_by_formula`` / ``search`` / ``neighbors`` can be passed as is.
            model: ``"mace"`` (256d, every structure) or ``"petmad"`` (512d, ~5% missing).

        Returns:
            ``EmbeddingSet`` — ``vectors[i]`` belongs to ``ids[i]``; structures without a
            vector for this model are listed in ``missing``. Raises ``NotFoundError`` for an
            unknown id.

        Example::

            silicon_oxides = atlas.get_embeddings(atlas.find_by_formula("SiO2"))
            matrix = silicon_oxides.to_numpy()          # (n, 256)
        """
        ids = self._ids_of(ids_or_records)
        cards = [self.get_structure(structure_id, include_embeddings=True) for structure_id in ids]
        return self._embedding_set(cards, model)

    def get_3d_coords(
        self, ids_or_records: IdsOrRecords, *, via_projection: bool | None = None
    ) -> ProjectionCloud:
        """Atlas point-cloud coordinates of several structures.

        Args:
            ids_or_records: one id, a list of ids, or records with an ``.id`` — the output of
                ``find_by_formula`` / ``search`` / ``neighbors`` can be passed as is.
            via_projection: ``True`` downloads the whole cloud once and picks the ids out of
                it, ``False`` fetches one card per id. Left ``None``, cards are used up to
                ``COORDS_CARD_LIMIT`` ids and the cloud beyond that.

        Returns:
            ``ProjectionCloud`` in the requested order; structures without a point in the
            atlas (about a quarter of the database) are listed in ``missing``. Raises
            ``NotFoundError`` for an unknown id on the card path.

        Example::

            coords = atlas.get_3d_coords(atlas.find_by_formula("SiO2"))
            coords.to_dataframe()                       # id, x, y, z, kind
        """
        ids = self._ids_of(ids_or_records)
        if via_projection is None:
            via_projection = len(ids) > COORDS_CARD_LIMIT
        if via_projection:
            return self._coords_from_cloud(self.projection(), ids)
        return self._coords_from_cards([self.get_structure(structure_id) for structure_id in ids])


class AsyncAtlasClient(_BaseClient):
    """Asynchronous client — the same methods as ``AtlasClient``, awaited.

    Example::

        async with AsyncAtlasClient() as atlas:
            silicon = await atlas.get_structure("mp-149")
            async for brief in atlas.iter_search(is_stable=True):
                ...
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float | None = DEFAULT_TIMEOUT,
    ):
        super().__init__(base_url, timeout=timeout)
        self._http = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise AtlasConnectionError(f"{method} {self.base_url}{path}: {error}") from error
        raise_for_status(response)
        return response

    async def _get(
        self, path: str, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> httpx.Response:
        return await self._request("GET", path, params=params, **kwargs)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncAtlasClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def health(self) -> bool:
        """``True`` when the backend and its database answer."""
        try:
            return (await self._get("/health")).json().get("status") == "ok"
        except AtlasConnectionError:
            return False

    async def get_structure(
        self, structure_id: str, *, include_embeddings: bool = False
    ) -> Structure:
        """The full card of one structure. See ``AtlasClient.get_structure``."""
        params = self._card_params(include_embeddings)
        response = await self._get(f"/structures/id/{structure_id}", params)
        return Structure.model_validate(response.json())

    async def _get_cards(self, ids: list[str], *, include_embeddings: bool) -> list[Structure]:
        """Cards of several structures, at most ``ASYNC_CONCURRENCY`` requests in flight."""
        semaphore = asyncio.Semaphore(ASYNC_CONCURRENCY)

        async def one(structure_id: str) -> Structure:
            async with semaphore:
                return await self.get_structure(structure_id, include_embeddings=include_embeddings)

        return list(await asyncio.gather(*(one(structure_id) for structure_id in ids)))

    async def get_random_structure(self) -> Structure:
        """The full card of a random structure."""
        return Structure.model_validate((await self._get("/structures/random_structure")).json())

    async def find_by_formula(self, formula: str) -> list[StructureBrief]:
        """Every structure with this composition; any spelling of the formula matches."""
        return self._parse_briefs(await self._get(f"/structures/formula/{formula}"))

    async def search(
        self,
        *,
        sort: SortField = "id",
        order: SortOrder = "asc",
        limit: int = 100,
        offset: int = 0,
        **filters: Unpack[StructureFilters],
    ) -> Page:
        """One page of structures matching the filters. See ``AtlasClient.search``."""
        params = self._search_params(sort, order, limit, offset, filters)
        return self._parse_page(await self._get("/structures", params), limit, offset)

    async def iter_search(
        self,
        *,
        sort: SortField = "id",
        order: SortOrder = "asc",
        page_size: int = MAX_PAGE_SIZE,
        **filters: Unpack[StructureFilters],
    ) -> AsyncIterator[StructureBrief]:
        """Every matching structure, page by page. See ``AtlasClient.iter_search``."""
        offset = 0
        while True:
            page = await self.search(
                sort=sort, order=order, limit=page_size, offset=offset, **filters
            )
            for brief in page:
                yield brief
            if not page.has_more:
                return
            offset += page_size

    async def count(self, **filters: Unpack[StructureFilters]) -> int:
        """How many structures match the filters; without filters, the size of the database."""
        return int((await self._get("/structures/count", build_params(filters))).json())

    async def ids(self, **filters: Unpack[StructureFilters]) -> list[str]:
        """IDs of every structure matching the filters, sorted."""
        return list((await self._get("/structures/ids", build_params(filters))).json())

    async def stats(self, *, bins: int = 50, **filters: Unpack[StructureFilters]) -> StructureStats:
        """Min / max / histogram of every numeric property and facet counts over the matches."""
        params = self._stats_params(bins, filters)
        return StructureStats.model_validate((await self._get("/structures/stats", params)).json())

    async def neighbors(
        self,
        query: str | VectorLike,
        *,
        k: int = 50,
        model: EmbeddingModel | None = None,
    ) -> list[Neighbor]:
        """The ``k`` closest structures to an id or a vector. See ``AtlasClient.neighbors``."""
        if isinstance(query, str):
            params = self._neighbors_params(k, model or "mace")
            response = await self._get(f"/structures/{query}/neighbors", params)
        else:
            body = self._neighbor_query(self._as_vector(query), k, model)
            response = await self._request("POST", "/structures/neighbors", json=body)
        return self._parse_neighbors(response)

    async def neighbors_batch(
        self, vectors: Any, *, k: int = 50, model: EmbeddingModel | None = None
    ) -> list[list[Neighbor]]:
        """Neighbours of several vectors, at most ``ASYNC_CONCURRENCY`` requests in flight.

        See ``AtlasClient.neighbors_batch``.
        """
        bodies = self._neighbor_queries(vectors, k, model)
        semaphore = asyncio.Semaphore(ASYNC_CONCURRENCY)

        async def one(body: dict[str, Any]) -> list[Neighbor]:
            async with semaphore:
                response = await self._request("POST", "/structures/neighbors", json=body)
                return self._parse_neighbors(response)

        return list(await asyncio.gather(*(one(body) for body in bodies)))

    async def projection(self) -> ProjectionCloud:
        """The whole 3D point cloud of the atlas (~155k points, a few MB)."""
        timeout = None if self.timeout is None else self.timeout * 3
        response = await self._get("/projection", timeout=timeout)
        return ProjectionCloud.model_validate(response.json())

    async def get_embeddings(
        self, ids_or_records: IdsOrRecords, *, model: EmbeddingModel = "mace"
    ) -> EmbeddingSet:
        """Embedding vectors of several structures. See ``AtlasClient.get_embeddings``."""
        ids = self._ids_of(ids_or_records)
        cards = await self._get_cards(ids, include_embeddings=True)
        return self._embedding_set(cards, model)

    async def get_3d_coords(
        self, ids_or_records: IdsOrRecords, *, via_projection: bool | None = None
    ) -> ProjectionCloud:
        """Atlas coordinates of several structures. See ``AtlasClient.get_3d_coords``."""
        ids = self._ids_of(ids_or_records)
        if via_projection is None:
            via_projection = len(ids) > COORDS_CARD_LIMIT
        if via_projection:
            return self._coords_from_cloud(await self.projection(), ids)
        return self._coords_from_cards(await self._get_cards(ids, include_embeddings=False))
