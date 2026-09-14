"""Response models — a one-to-one copy of the server's API contract.

The classes mirror ``backend/data/data_structures.py`` in 3d-atlas; the nesting groups
properties by domain (structural, thermodynamic, electronic, ...). ``Page`` is client-side
only: it carries one page of ``search`` results together with the total match count.

Optional conversions need an extra: ``Structure.to_pymatgen()`` needs ``materials-atlas[pymatgen]``,
every ``to_dataframe`` needs ``materials-atlas[pandas]``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # only for type hints; the packages are optional at runtime
    import numpy
    import pandas
    import pymatgen.core


def _import_pandas() -> Any:
    try:
        import pandas
    except ImportError as error:
        raise ImportError(
            "pandas is not installed — install the extra: pip install 'materials-atlas[pandas]'"
        ) from error
    return pandas


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise ImportError(
            "numpy is not installed — install the extra: pip install 'materials-atlas[numpy]'"
        ) from error
    return numpy


def to_dataframe(records: Sequence[BaseModel]) -> pandas.DataFrame:
    """Any list of flat models (``StructureBrief``, ``Neighbor``) as a DataFrame, one row each.

    Requires the ``pandas`` extra.
    """
    pandas = _import_pandas()
    return pandas.DataFrame([record.model_dump() for record in records])


class Projection(BaseModel):
    """3d coordinates for the atlas point cloud."""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    kind: str | None = None


class StructureGeometry(BaseModel):
    """Lattice parameters."""

    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    angle_alpha: float | None = None
    angle_beta: float | None = None
    angle_gamma: float | None = None


class StructuralProperties(BaseModel):
    """Composition, geometry and symmetry."""

    formula_reduced: str | None = None
    formula_cell: str | None = None
    elements: list[str] = []
    n_elements: int | None = None
    n_atoms: int | None = None
    geometry: StructureGeometry = StructureGeometry()
    volume: float | None = None
    volume_per_atom: float | None = None
    density: float | None = None
    crystal_system: str | None = None
    sg_number: int | None = None
    sg_symbol: str | None = None


class ThermodynamicProperties(BaseModel):
    """Energetics and stability."""

    energy_per_atom: float | None = None
    uncorrected_energy_per_atom: float | None = None
    formation_energy_per_atom: float | None = None
    energy_above_hull: float | None = None
    equilibrium_reaction_energy_per_atom: float | None = None
    is_stable: bool | None = None
    is_theoretical: bool | None = None


class ElectronicProperties(BaseModel):
    """Band structure summary. ``band_gap`` is the GGA/GGA+U value from Materials Project."""

    band_gap: float | None = None
    cbm: float | None = None
    vbm: float | None = None
    efermi: float | None = None
    is_gap_direct: bool | None = None
    is_metal: bool | None = None


class MagneticProperties(BaseModel):
    """Magnetic moments and ordering."""

    is_magnetic: bool | None = None
    ordering: str | None = None
    total_magnetization_per_cell: float | None = None
    total_magnetization_per_unit: float | None = None
    total_magnetization_normalized_vol: float | None = None
    n_magnetic_sites: int | None = None
    n_unique_magnetic_sites: int | None = None


class DielectricProperties(BaseModel):
    """Dielectric constants and 3x3 tensors."""

    refractive_index: float | None = None
    e_total: float | None = None
    e_ionic: float | None = None
    e_electronic: float | None = None
    tensor: list[list[float]] | None = None
    tensor_ionic: list[list[float]] | None = None
    tensor_electronic: list[list[float]] | None = None


class ElasticProperties(BaseModel):
    """Moduli, thermal conductivity estimates and Debye temperature."""

    bulk_modulus_voigt: float | None = None
    shear_modulus_voigt: float | None = None
    thermal_conductivity_clarke: float | None = None
    thermal_conductivity_cahill: float | None = None
    thermal_conductivity_slack: float | None = None
    debye_temperature: float | None = None
    universal_anisotropy: float | None = None
    homogeneous_poisson: float | None = None


class PiezoProperties(BaseModel):
    """Piezoelectric response."""

    piezoelectric_tensor_max_value: float | None = None


class Embeddings(BaseModel):
    """Neural-network embeddings; present in the card only when requested."""

    mace: list[float] | None = None  # 256d, every structure has one
    petmad: list[float] | None = None  # 512d, missing for ~5% of structures


class Availability(BaseModel):
    """What data actually exists for this structure."""

    has_magnetic: bool = False
    has_elastic: bool = False
    has_dielectric: bool = False
    has_piezoelectric: bool = False
    has_embedding: bool = False
    has_projection: bool = False
    has_bandstructure: bool | None = None
    has_dos: bool | None = None
    has_xas: bool | None = None


class Structure(BaseModel):
    """Full structure card — what ``get_structure`` returns."""

    id: str
    source: str = "mp"
    pymatgen_structure: dict | None = None
    structural_properties: StructuralProperties = StructuralProperties()
    thermodynamic_properties: ThermodynamicProperties = ThermodynamicProperties()
    electronic_properties: ElectronicProperties = ElectronicProperties()
    magnetic_properties: MagneticProperties = MagneticProperties()
    dielectric_properties: DielectricProperties = DielectricProperties()
    elastic_properties: ElasticProperties = ElasticProperties()
    piezo_properties: PiezoProperties = PiezoProperties()
    projection: Projection = Projection()
    availability: Availability = Availability()
    embeddings: Embeddings | None = (
        None  # filled only by ``get_structure(..., include_embeddings=True)``
    )

    @property
    def formula(self) -> str | None:
        """Shortcut for ``structural_properties.formula_reduced``."""
        return self.structural_properties.formula_reduced

    def to_pymatgen(self) -> pymatgen.core.Structure:
        """The crystal structure as a ``pymatgen.core.Structure``.

        Requires the ``pymatgen`` extra.
        """
        try:
            from pymatgen.core import Structure as PymatgenStructure
        except ImportError as error:
            raise ImportError(
                "pymatgen is not installed — install the extra: "
                "pip install 'materials-atlas[pymatgen]'"
            ) from error
        if self.pymatgen_structure is None:
            raise ValueError(f"Structure {self.id} carries no pymatgen_structure")
        return PymatgenStructure.from_dict(self.pymatgen_structure)


class StructureBrief(BaseModel):
    """Compact record for lists and search results."""

    id: str
    formula_reduced: str | None = None
    crystal_system: str | None = None
    sg_number: int | None = None
    n_atoms: int | None = None
    volume: float | None = None
    density: float | None = None
    band_gap: float | None = None
    energy_above_hull: float | None = None
    volume_per_atom: float | None = None
    bulk_modulus_voigt: float | None = None
    shear_modulus_voigt: float | None = None
    debye_temperature: float | None = None
    thermal_conductivity_clarke: float | None = None
    thermal_conductivity_cahill: float | None = None
    piezoelectric_tensor_max_value: float | None = None
    e_total: float | None = None
    refractive_index: float | None = None


class Page(BaseModel):
    """One page of ``search`` results.

    Behaves like a list of ``StructureBrief`` (``len``, indexing, iteration); ``total`` is the
    number of structures matching the filters across all pages.
    """

    items: list[StructureBrief]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        """Whether another page follows this one."""
        return self.offset + len(self.items) < self.total

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[StructureBrief]:  # type: ignore[override]
        return iter(self.items)

    @overload
    def __getitem__(self, index: int) -> StructureBrief: ...

    @overload
    def __getitem__(self, index: slice) -> list[StructureBrief]: ...

    def __getitem__(self, index):
        return self.items[index]

    def to_dataframe(self) -> pandas.DataFrame:
        """The page as a DataFrame, one row per structure. Requires the ``pandas`` extra."""
        return to_dataframe(self.items)


class Neighbor(BaseModel):
    """A structure close to the query in embedding space; ``similarity`` is cosine, in percent."""

    id: str
    similarity: float
    similarity_3d: float | None = None
    formula_reduced: str | None = None
    n_atoms: int | None = None
    sg_number: int | None = None
    volume: float | None = None


class ProjectionCloud(BaseModel):
    """Atlas point-cloud coordinates, column-oriented: ``ids[i]`` sits at ``(x[i], y[i], z[i])``.

    ``projection()`` returns the whole cloud; ``get_3d_coords`` returns the requested subset
    and lists the structures that have no point in the atlas in ``missing``.
    """

    count: int
    ids: list[str]
    x: list[float]
    y: list[float]
    z: list[float]
    kind: list[str | None]
    formula_reduced: list[str | None] | None = None
    n_atoms: list[int | None] | None = None
    sg_number: list[int | None] | None = None
    volume: list[float | None] | None = None
    missing: list[str] = Field(default_factory=list)

    def to_numpy(self) -> numpy.ndarray:
        """Shape ``(count, 3)`` array of ``x, y, z``. Requires the ``numpy`` extra."""
        numpy = _import_numpy()
        return numpy.column_stack([self.x, self.y, self.z]) if self.count else numpy.empty((0, 3))

    def to_dataframe(self) -> pandas.DataFrame:
        """Columns ``id, x, y, z, kind``, one row per point. Requires the ``pandas`` extra."""
        pandas = _import_pandas()
        return pandas.DataFrame(
            {"id": self.ids, "x": self.x, "y": self.y, "z": self.z, "kind": self.kind}
        )


class EmbeddingSet(BaseModel):
    """Embedding vectors of several structures: ``vectors[i]`` belongs to ``ids[i]``.

    Structures that have no vector for ``model`` (``petmad`` is missing for ~5%) are left out
    of ``ids`` / ``vectors`` and listed in ``missing``.
    """

    model: str
    ids: list[str]
    vectors: list[list[float]]
    missing: list[str] = Field(default_factory=list)

    @property
    def dimension(self) -> int | None:
        """Length of one vector: 256 for ``mace``, 512 for ``petmad``; ``None`` when empty."""
        return len(self.vectors[0]) if self.vectors else None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, structure_id: str) -> list[float]:
        """The vector of one structure by id."""
        try:
            return self.vectors[self.ids.index(structure_id)]
        except ValueError:
            raise KeyError(structure_id) from None

    def to_numpy(self) -> numpy.ndarray:
        """Shape ``(len(ids), dimension)`` array. Requires the ``numpy`` extra."""
        numpy = _import_numpy()
        return numpy.array(self.vectors, dtype=float)

    def to_dataframe(self) -> pandas.DataFrame:
        """One row per structure indexed by id, one column per dimension. Requires ``pandas``."""
        pandas = _import_pandas()
        return pandas.DataFrame(self.vectors, index=pandas.Index(self.ids, name="id"))


class NumericStats(BaseModel):
    """Range and histogram of one numeric property over the matching structures."""

    min: float
    max: float
    count: int  # structures where the value is present
    bin_edges: list[float]  # len == len(counts) + 1
    counts: list[int]


class StructureStats(BaseModel):
    """Facets of the structures matching the filters (or of everything without filters).

    A numeric field is ``None`` when no matching structure has a value for it.
    """

    total: int
    band_gap: NumericStats | None = None
    energy_above_hull: NumericStats | None = None
    density: NumericStats | None = None
    n_atoms: NumericStats | None = None
    volume: NumericStats | None = None
    volume_per_atom: NumericStats | None = None
    bulk_modulus_voigt: NumericStats | None = None
    shear_modulus_voigt: NumericStats | None = None
    debye_temperature: NumericStats | None = None
    thermal_conductivity_clarke: NumericStats | None = None
    thermal_conductivity_cahill: NumericStats | None = None
    piezoelectric_tensor_max_value: NumericStats | None = None
    e_total: NumericStats | None = None
    refractive_index: NumericStats | None = None
    crystal_system: dict[str, int]
    is_stable: dict[str, int]  # keys "true" / "false"
