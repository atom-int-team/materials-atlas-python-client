import json
from pathlib import Path

import pytest
import respx

from materials_atlas import AsyncAtlasClient, AtlasClient
from materials_atlas.client import DEFAULT_URL

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://atlas.test"


@pytest.fixture
def structure_json() -> dict:
    """The real card of mp-149 (silicon) as the server returns it."""
    return json.loads((FIXTURES / "mp-149.json").read_text())


@pytest.fixture
def stats_json() -> dict:
    return json.loads((FIXTURES / "stats.json").read_text())


@pytest.fixture
def brief_json() -> dict:
    return {
        "id": "mp-149",
        "formula_reduced": "Si",
        "crystal_system": "Cubic",
        "sg_number": 227,
        "n_atoms": 2,
        "volume": 40.33,
        "density": 2.31,
        "band_gap": 0.6105,
        "energy_above_hull": 0.0,
        "volume_per_atom": 20.16,
        "bulk_modulus_voigt": 88.9,
        "shear_modulus_voigt": 63.8,
        "debye_temperature": 630.0,
        "thermal_conductivity_clarke": 1.31,
        "thermal_conductivity_cahill": 1.43,
        "piezoelectric_tensor_max_value": None,
        "e_total": 13.0,
        "refractive_index": 3.6,
    }


@pytest.fixture
def api():
    """A respx router bound to BASE_URL; unmatched requests fail the test."""
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture
def client(api):
    with AtlasClient(BASE_URL) as atlas:
        yield atlas


@pytest.fixture
async def async_client(api):
    async with AsyncAtlasClient(BASE_URL) as atlas:
        yield atlas


@pytest.fixture(scope="session")
def live_url() -> str:
    """The production API; integration tests run only with ``pytest -m integration``."""
    return DEFAULT_URL
