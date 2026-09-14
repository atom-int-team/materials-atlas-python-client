# materials-atlas

[![PyPI](https://img.shields.io/pypi/v/materials-atlas)](https://pypi.org/project/materials-atlas/)
[![Python](https://img.shields.io/pypi/pyversions/materials-atlas)](https://pypi.org/project/materials-atlas/)
[![CI](https://github.com/atom-int-team/materials-atlas-python-client/actions/workflows/ci.yml/badge.svg)](https://github.com/atom-int-team/materials-atlas-python-client/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Python client for the 3D Atlas: 210k crystal structures from the Materials Project with their
properties, MACE and PET-MAD embeddings and a 3D map of the embedding space. One class,
`AtlasClient`, whose methods map one to one onto the REST endpoints and return typed
[pydantic](https://docs.pydantic.dev/) models.

```python
from materials_atlas import AtlasClient

with AtlasClient() as atlas:                       # https://atlas3d.api.atom-int.com
    silicon = atlas.get_structure("mp-149")
    print(silicon.formula, silicon.electronic_properties.band_gap)

    page = atlas.search(band_gap_min=1, band_gap_max=3, crystal_system="Cubic",
                        sort="band_gap", order="desc", limit=50)
    print(page.total, "matches;", page[0].formula_reduced, "first")

    for neighbor in atlas.neighbors("mp-149", k=5):
        print(neighbor.id, neighbor.similarity)
```

## Installation

Python 3.10+:

```
pip install materials-atlas
pip install "materials-atlas[all]"      # + pymatgen, pandas, numpy
```

Extras: `[pymatgen]` enables `Structure.to_pymatgen()`, `[pandas]` enables every
`to_dataframe()`, `[numpy]` enables every `to_numpy()`, `[all]` is all three. The core package
depends only on `httpx` and `pydantic`.

## Client

```python
AtlasClient(base_url="https://atlas3d.api.atom-int.com", *, timeout=60.0)
```

| Method | Endpoint | Returns |
|---|---|---|
| `health()` | `GET /health` | `bool` |
| `get_structure(id, include_embeddings=False)` | `GET /structures/id/{id}` | `Structure` — the full card; with the embedding vectors on request |
| `get_random_structure()` | `GET /structures/random_structure` | `Structure` |
| `find_by_formula(formula)` | `GET /structures/formula/{formula}` | `list[StructureBrief]`, any spelling of the composition matches |
| `search(sort=, order=, limit=, offset=, **filters)` | `GET /structures` | `Page` — a list of `StructureBrief` plus `.total` |
| `iter_search(sort=, order=, page_size=, **filters)` | `GET /structures`, page by page | iterator of `StructureBrief` |
| `count(**filters)` | `GET /structures/count` | `int` (whole database without filters) |
| `ids(**filters)` | `GET /structures/ids` | `list[str]` |
| `stats(bins=50, **filters)` | `GET /structures/stats` | `StructureStats` — min / max / histogram per property, facet counts |
| `neighbors(id_or_vector, k=50, model=None)` | `GET /structures/{id}/neighbors` or `POST /structures/neighbors` | `list[Neighbor]`, closest first, cosine similarity in percent |
| `neighbors_batch(vectors, k=50, model=None)` | `POST /structures/neighbors`, once per vector | `list[list[Neighbor]]`, in input order |
| `projection()` | `GET /projection` | `ProjectionCloud` — the whole 3D point cloud, column-oriented |
| `get_embeddings(ids_or_records, model="mace")` | one card per id | `EmbeddingSet` — vectors of several structures |
| `get_3d_coords(ids_or_records)` | cards, or `/projection` for big lists | `ProjectionCloud` — atlas coordinates of several structures |

The client keeps an HTTP connection pool open: use it as a context manager or call `close()`.

`AsyncAtlasClient` has the same methods with `await`; `iter_search` is an async generator.

```python
from materials_atlas import AsyncAtlasClient

async with AsyncAtlasClient() as atlas:
    stable = await atlas.count(is_stable=True)
    async for brief in atlas.iter_search(is_stable=True, has_elastic=True):
        ...
```

## Filters

`search`, `iter_search`, `count`, `ids` and `stats` accept the same keyword arguments, named
exactly like the query parameters of the API. All conditions are ANDed. Numeric bounds are
inclusive and can be given alone or as a pair; a structure whose value is missing never
matches a numeric filter. They are typed as `StructureFilters`, so an IDE autocompletes them,
and `print(materials_atlas.describe_filters())` prints this table in a notebook.

| Keyword | Unit | Meaning |
|---|---|---|
| `band_gap_min` / `band_gap_max` | eV | GGA/GGA+U band gap from Materials Project |
| `energy_above_hull_min` / `_max` | eV/atom | distance to the convex hull; 0 means stable |
| `density_min` / `_max` | g/cm³ | mass density |
| `n_atoms_min` / `_max` | atoms | number of atoms in the unit cell |
| `volume_min` / `_max` | Å³ | unit cell volume |
| `volume_per_atom_min` / `_max` | Å³/atom | unit cell volume per atom |
| `bulk_modulus_voigt_min` / `_max` | GPa | Voigt bulk modulus |
| `shear_modulus_voigt_min` / `_max` | GPa | Voigt shear modulus |
| `debye_temperature_min` / `_max` | K | Debye temperature |
| `thermal_conductivity_clarke_min` / `_max` | W/(m·K) | Clarke estimate |
| `thermal_conductivity_cahill_min` / `_max` | W/(m·K) | Cahill estimate |
| `piezoelectric_tensor_max_value_min` / `_max` | C/m² | largest piezoelectric tensor component |
| `e_total_min` / `_max` | — | total dielectric constant |
| `refractive_index_min` / `_max` | — | refractive index |
| `is_stable` | bool | on the convex hull |
| `has_magnetic`, `has_elastic`, `has_dielectric`, `has_piezoelectric` | bool | that group of properties is available |
| `crystal_system` | name or list | `Triclinic`, `Monoclinic`, `Orthorhombic`, `Tetragonal`, `Trigonal`, `Hexagonal`, `Cubic`; OR within a list |

Elastic, dielectric and piezoelectric values exist only for a few percent of the structures.
Sorting by one of those columns is meant to be combined with its `has_*` flag, otherwise the
missing values pile up at the end of the list.

Mistakes are caught before any request goes out: an unknown keyword raises `TypeError` listing
the valid names, `_min` above `_max` raises `ValueError`.

```python
atlas.count(is_stable=True, has_elastic=True, bulk_modulus_voigt_min=100, density_max=5)
atlas.ids(energy_above_hull_max=0.05, crystal_system=["Cubic", "Hexagonal"])
atlas.stats(bins=20, band_gap_min=1).band_gap.counts      # histogram of the band gap over the matches
```

### Sorting and paging

`sort` takes any numeric filter field plus `id` and `formula_reduced`; `order` is `asc` or
`desc`, missing values sort last either way. `limit` is at most 1000 per page.

```python
page = atlas.search(band_gap_min=2, sort="density", order="desc", limit=200, offset=400)
page.total       # all matches, from the X-Total-Count header
page.has_more    # whether offset + len(page) < total
len(page), page[0], list(page)

for brief in atlas.iter_search(band_gap_min=2, sort="density", order="desc"):
    ...          # walks every page for you, 1000 records per request
```

## Models

`Structure` is the full card: `structural_properties`, `thermodynamic_properties`,
`electronic_properties`, `magnetic_properties`, `dielectric_properties`, `elastic_properties`,
`piezo_properties`, `projection`, `availability`, plus `pymatgen_structure` (the crystal as a
pymatgen dict). `structure.formula` is a shortcut for the reduced formula. Every model is a
pydantic v2 model, so `model_dump()` / `model_dump_json()` work as usual.

```python
silicon = atlas.get_structure("mp-149")
silicon.structural_properties.sg_symbol           # 'Fd-3m'
silicon.elastic_properties.bulk_modulus_voigt     # 88.916
silicon.availability.has_piezoelectric            # False
silicon.to_pymatgen()                             # pymatgen.core.Structure   [pymatgen extra]
```

### Embeddings and atlas coordinates

`get_embeddings` and `get_3d_coords` take one id, a list of ids, or any records that carry an
`.id` — so the output of `find_by_formula`, `search` or `neighbors` plugs in directly. That
is how "by formula" or "by filter" lookups work: pick the structures first, then ask for
their vectors or coordinates.

```python
# one structure
silicon = atlas.get_structure("mp-149", include_embeddings=True)
silicon.embeddings.mace          # 256 floats
silicon.embeddings.petmad        # 512 floats, or None for ~5% of structures
silicon.projection               # x, y, z, kind

# several, by formula
emb = atlas.get_embeddings(atlas.find_by_formula("SiO2"))           # model="mace" by default
emb.ids, emb.vectors             # aligned lists
emb["mp-640917"]                 # one vector by id
emb.to_numpy()                   # (n, 256)                          [numpy extra]
emb.to_dataframe()               # rows indexed by id                [pandas extra]

emb = atlas.get_embeddings(atlas.ids(is_stable=True, crystal_system="Cubic"), model="petmad")
emb.missing                      # ids that have no petmad vector

coords = atlas.get_3d_coords(atlas.neighbors("mp-149", k=100))
coords.to_numpy()                # (n, 3)
coords.missing                   # ids without a point in the atlas (about a quarter of the database)
```

`get_embeddings` makes one card request per structure. `get_3d_coords` does the same up to
50 ids and switches to a single download of the whole point cloud beyond that; force either
path with `via_projection=True` / `False`. The async client runs the card requests eight at a
time.

### Neighbours of your own vector

`neighbors` takes either a structure id or an embedding vector. With a vector it calls
`POST /structures/neighbors`, so an embedding computed outside the atlas (by the same MACE or
PET-MAD model) can be placed among the atlas structures. The vector can be a list, a 1-D numpy
array or anything with `.tolist()`; the model is inferred from its length (256 → `mace`,
512 → `petmad`) unless `model=` says otherwise. Unlike the search by id, nothing is excluded
from the result: a structure's own vector returns that structure first with similarity 100.

```python
silicon, germanium = atlas.get_embeddings(["mp-149", "mp-32"]).to_numpy()
atlas.neighbors(silicon, k=3)                    # mp-149 itself, then its neighbours
atlas.neighbors((silicon + germanium) / 2, k=3)  # SiGe polymorphs
atlas.neighbors(my_petmad_vector, k=10)          # 512 floats → petmad index

emb = atlas.get_embeddings(atlas.find_by_formula("SiO2"))
hits = atlas.neighbors_batch(emb, k=5)           # hits[i] belongs to emb.ids[i]; model from emb
hits = atlas.neighbors_batch(matrix, k=5)        # any (n, 256) or (n, 512) array works too
```

A vector of the wrong length raises `ValueError` before any request; NaN or an all-zero
vector is rejected by the server with `ValidationError`. `neighbors_batch` sends one request
per vector (the async client keeps eight in flight) and checks every vector up front.

DataFrames (`[pandas]` extra):

```python
atlas.search(is_stable=True, limit=1000).to_dataframe()      # one row per StructureBrief
atlas.projection().to_dataframe()                            # columns id, x, y, z, kind

from materials_atlas import to_dataframe
to_dataframe(atlas.neighbors("mp-149", k=100))               # any list of flat models
```

## Errors

Everything raised by the client is an `AtlasError`:

| Exception | When |
|---|---|
| `AtlasConnectionError` | server unreachable, timeout |
| `NotFoundError` | 404 — unknown id, formula without matches, no `petmad` embedding |
| `ValidationError` | 400 / 422 — the server rejected the parameters or a query vector (NaN, all zeros) |
| `AuthError` | 401 / 403 |
| `ConflictError` | 409 |
| `IndexNotReadyError` | 503 — neighbour-search index not built for that model |
| `AtlasAPIError` | any other non-2xx answer; base class of the five above |

API errors carry `.status_code` and `.detail` (the server's message).

## Examples

[`examples/materials_atlas_tour.ipynb`](examples/materials_atlas_tour.ipynb) walks through every
method on live data and ends with 3D plots of the atlas point cloud and of the embedding space
(plotly). Install the extra and open it:

```
pip install "materials-atlas[examples]"
jupyter lab examples/materials_atlas_tour.ipynb
```

## Development

```
uv venv && uv pip install -e '.[dev]'
pytest                      # unit tests, mocked server
pytest -m integration       # against the production API
ruff check . && ruff format --check .
```

Releases are published to PyPI by GitHub Actions when a `v*` tag matching the version in
`pyproject.toml` is pushed.

## Data and citation

Structures and their properties come from the [Materials Project](https://materialsproject.org)
and are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use
them, please cite:

> A. Jain et al., *Commentary: The Materials Project: A materials genome approach to
> accelerating materials innovation*, APL Materials 1, 011002 (2013).
> [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

Embeddings are computed with MACE and PET-MAD:

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast
> and Accurate Force Fields*, NeurIPS 2022. [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)

> I. Batatia et al., *A foundation model for atomistic materials chemistry*,
> [arXiv:2401.00096](https://arxiv.org/abs/2401.00096)

> A. Mazitov et al., *PET-MAD, a universal interatomic potential for advanced materials
> modeling*, [arXiv:2503.14118](https://arxiv.org/abs/2503.14118)

## License

The client is released under the MIT License, see [LICENSE](LICENSE).
