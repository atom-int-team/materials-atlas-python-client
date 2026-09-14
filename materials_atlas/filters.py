"""Filter, sort and model parameters accepted by the list-style endpoints.

``search``, ``iter_search``, ``count``, ``ids`` and ``stats`` all take the same keyword
arguments, mirroring the query parameters of the 3d-atlas API one to one. They are listed in
``StructureFilters`` (so an IDE can autocomplete them), described with units in
``RANGE_FIELDS`` / ``FLAG_FIELDS`` and printed by ``describe_filters()``.

Rules shared by every filter:

* all conditions are ANDed together;
* numeric bounds are inclusive, ``<field>_min`` and ``<field>_max`` can be used alone or together;
* a structure whose value is missing (NULL) never matches a numeric filter;
* ``crystal_system`` takes one name or a list of names — OR within the list.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:  # `Unpack[TypedDict]` in function signatures needs the backport before 3.12
    from typing_extensions import TypedDict

# Spelled the way Materials Project spells them.
CrystalSystem = Literal[
    "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"
]
CRYSTAL_SYSTEMS: tuple[str, ...] = (
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
)

SortField = Literal[
    "id",
    "formula_reduced",
    "band_gap",
    "energy_above_hull",
    "density",
    "n_atoms",
    "volume",
    "volume_per_atom",
    "bulk_modulus_voigt",
    "shear_modulus_voigt",
    "debye_temperature",
    "thermal_conductivity_clarke",
    "thermal_conductivity_cahill",
    "piezoelectric_tensor_max_value",
    "e_total",
    "refractive_index",
]
SortOrder = Literal["asc", "desc"]

# Which embedding `neighbors` searches: `mace` (256d, every structure) or `petmad` (512d,
# missing for ~5% of structures).
EmbeddingModel = Literal["mace", "petmad"]


@dataclass(frozen=True)
class RangeField:
    """A numeric property filterable by an inclusive ``<name>_min`` / ``<name>_max`` pair."""

    name: str
    unit: str
    description: str


# Order matches the server (`backend/filters.py`) and the `stats` response.
RANGE_FIELDS: tuple[RangeField, ...] = (
    RangeField("band_gap", "eV", "GGA/GGA+U band gap from Materials Project"),
    RangeField("energy_above_hull", "eV/atom", "distance to the convex hull; 0 means stable"),
    RangeField("density", "g/cm³", "mass density"),
    RangeField("n_atoms", "atoms", "number of atoms in the unit cell"),
    RangeField("volume", "Å³", "unit cell volume"),
    RangeField("volume_per_atom", "Å³/atom", "unit cell volume per atom"),
    RangeField("bulk_modulus_voigt", "GPa", "Voigt bulk modulus (needs has_elastic)"),
    RangeField("shear_modulus_voigt", "GPa", "Voigt shear modulus (needs has_elastic)"),
    RangeField("debye_temperature", "K", "Debye temperature (needs has_elastic)"),
    RangeField("thermal_conductivity_clarke", "W/(m·K)", "Clarke estimate (needs has_elastic)"),
    RangeField("thermal_conductivity_cahill", "W/(m·K)", "Cahill estimate (needs has_elastic)"),
    RangeField(
        "piezoelectric_tensor_max_value",
        "C/m²",
        "largest piezoelectric tensor component (needs has_piezoelectric)",
    ),
    RangeField("e_total", "—", "total dielectric constant (needs has_dielectric)"),
    RangeField("refractive_index", "—", "refractive index (needs has_dielectric)"),
)

FLAG_FIELDS: dict[str, str] = {
    "is_stable": "True for structures on the convex hull (energy_above_hull == 0)",
    "has_magnetic": "magnetic properties are available",
    "has_elastic": "elastic moduli, Debye temperature and thermal conductivity are available",
    "has_dielectric": "dielectric constants and tensors are available",
    "has_piezoelectric": "piezoelectric data is available",
}


class StructureFilters(TypedDict, total=False):
    """Every keyword accepted by ``search``, ``iter_search``, ``count``, ``ids`` and ``stats``.

    Numeric bounds are inclusive; a missing (NULL) value never matches; all conditions are
    ANDed. See ``describe_filters()`` for units and meanings.
    """

    band_gap_min: float
    band_gap_max: float
    energy_above_hull_min: float
    energy_above_hull_max: float
    density_min: float
    density_max: float
    n_atoms_min: int
    n_atoms_max: int
    volume_min: float
    volume_max: float
    volume_per_atom_min: float
    volume_per_atom_max: float
    bulk_modulus_voigt_min: float
    bulk_modulus_voigt_max: float
    shear_modulus_voigt_min: float
    shear_modulus_voigt_max: float
    debye_temperature_min: float
    debye_temperature_max: float
    thermal_conductivity_clarke_min: float
    thermal_conductivity_clarke_max: float
    thermal_conductivity_cahill_min: float
    thermal_conductivity_cahill_max: float
    piezoelectric_tensor_max_value_min: float
    piezoelectric_tensor_max_value_max: float
    e_total_min: float
    e_total_max: float
    refractive_index_min: float
    refractive_index_max: float
    is_stable: bool
    crystal_system: CrystalSystem | Sequence[CrystalSystem]
    has_magnetic: bool
    has_elastic: bool
    has_dielectric: bool
    has_piezoelectric: bool


FILTER_NAMES: frozenset[str] = frozenset(StructureFilters.__annotations__)


def build_params(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Turn filter keyword arguments into HTTP query parameters.

    * an unknown name raises ``TypeError`` listing the valid ones — the server would silently
      ignore it and return everything;
    * ``None`` values are dropped, so ``search(band_gap_min=lo)`` with ``lo = None`` is fine;
    * ``crystal_system`` given as one string becomes a one-element list (httpx repeats the
      parameter for lists, which is what the server expects);
    * ``<field>_min`` above ``<field>_max`` raises ``ValueError`` before any request is made.
    """
    unknown = set(filters) - FILTER_NAMES
    if unknown:
        raise TypeError(
            f"unknown filter(s): {', '.join(sorted(unknown))}. "
            f"Valid filters: {', '.join(sorted(FILTER_NAMES))}"
        )

    params: dict[str, Any] = {}
    for name, value in filters.items():
        if value is None:
            continue
        if name == "crystal_system":
            value = [value] if isinstance(value, str) else list(value)
            bad = [system for system in value if system not in CRYSTAL_SYSTEMS]
            if bad:
                raise ValueError(
                    f"unknown crystal system(s): {', '.join(bad)}. "
                    f"Valid: {', '.join(CRYSTAL_SYSTEMS)}"
                )
        params[name] = value

    for field in RANGE_FIELDS:
        lo = params.get(f"{field.name}_min")
        hi = params.get(f"{field.name}_max")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{field.name}_min ({lo}) must not exceed {field.name}_max ({hi})")
    return params


def describe_filters() -> str:
    """A plain-text table of every filter keyword with its unit and meaning.

    Handy in a notebook or REPL: ``print(materials_atlas.describe_filters())``.
    """
    width = max(len(field.name) for field in RANGE_FIELDS) + len("_min")
    lines = ["Numeric ranges (use <name>_min and/or <name>_max, inclusive):"]
    for field in RANGE_FIELDS:
        lines.append(f"  {field.name + '_min':<{width}}  {field.unit:<8} {field.description}")
        lines.append(f"  {field.name + '_max':<{width}}  {field.unit:<8} {field.description}")
    lines.append("")
    lines.append("Flags (True / False):")
    for name, description in FLAG_FIELDS.items():
        lines.append(f"  {name:<{width}}  {'':<8} {description}")
    lines.append("")
    lines.append("Categories:")
    lines.append(
        f"  {'crystal_system':<{width}}  {'':<8} one name or a list (OR): "
        + ", ".join(CRYSTAL_SYSTEMS)
    )
    return "\n".join(lines)
