"""Python client for the 3D Atlas materials database.

Quick start::

    from materials_atlas import AtlasClient

    with AtlasClient() as atlas:                       # https://atlas3d.api.atom-int.com
        silicon = atlas.get_structure("mp-149")
        page = atlas.search(band_gap_min=1, band_gap_max=3, crystal_system="Cubic",
                            sort="band_gap", order="desc", limit=50)
        print(page.total, page[0].formula_reduced)

Run ``print(materials_atlas.describe_filters())`` for the list of filter keywords.
"""

from ._version import __version__
from .client import AsyncAtlasClient, AtlasClient
from .exceptions import (
    AtlasAPIError,
    AtlasConnectionError,
    AtlasError,
    AuthError,
    ConflictError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .filters import (
    CRYSTAL_SYSTEMS,
    FLAG_FIELDS,
    RANGE_FIELDS,
    CrystalSystem,
    EmbeddingModel,
    SortField,
    SortOrder,
    StructureFilters,
    describe_filters,
)
from .models import (
    Availability,
    DielectricProperties,
    ElasticProperties,
    ElectronicProperties,
    Embeddings,
    EmbeddingSet,
    MagneticProperties,
    Neighbor,
    NumericStats,
    Page,
    PiezoProperties,
    Projection,
    ProjectionCloud,
    StructuralProperties,
    Structure,
    StructureBrief,
    StructureGeometry,
    StructureStats,
    ThermodynamicProperties,
    to_dataframe,
)

__all__ = [
    "__version__",
    "AtlasClient",
    "AsyncAtlasClient",
    # exceptions
    "AtlasError",
    "AtlasConnectionError",
    "AtlasAPIError",
    "NotFoundError",
    "ValidationError",
    "AuthError",
    "ConflictError",
    "IndexNotReadyError",
    # filters
    "StructureFilters",
    "CrystalSystem",
    "SortField",
    "SortOrder",
    "EmbeddingModel",
    "CRYSTAL_SYSTEMS",
    "RANGE_FIELDS",
    "FLAG_FIELDS",
    "describe_filters",
    # models
    "Structure",
    "StructureBrief",
    "Page",
    "Neighbor",
    "ProjectionCloud",
    "StructureStats",
    "NumericStats",
    "EmbeddingSet",
    "Embeddings",
    "Projection",
    "StructureGeometry",
    "StructuralProperties",
    "ThermodynamicProperties",
    "ElectronicProperties",
    "MagneticProperties",
    "DielectricProperties",
    "ElasticProperties",
    "PiezoProperties",
    "Availability",
    "to_dataframe",
]
