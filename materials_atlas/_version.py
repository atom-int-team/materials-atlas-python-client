"""The installed package version, read from its metadata (``version`` in pyproject.toml)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("materials-atlas")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0.0.0+unknown"
