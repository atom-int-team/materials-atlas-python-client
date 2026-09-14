"""Exceptions raised by the client.

Every error is an ``AtlasError``. Network problems become ``AtlasConnectionError``; any
non-2xx answer from the server becomes an ``AtlasAPIError`` (or one of its subclasses chosen
by status code) carrying ``status_code`` and the server's ``detail`` message.
"""

from __future__ import annotations

import httpx


class AtlasError(Exception):
    """Base class for every error raised by materials_atlas."""


class AtlasConnectionError(AtlasError):
    """The server could not be reached or the request timed out."""


class AtlasAPIError(AtlasError):
    """The server answered with an error status."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class NotFoundError(AtlasAPIError):
    """404 — no structure with that id / formula."""


class ValidationError(AtlasAPIError):
    """400 / 422 — the server rejected the parameters (bad formula, ``_min`` above ``_max`` ...)."""


class AuthError(AtlasAPIError):
    """401 / 403 — the server refused access."""


class ConflictError(AtlasAPIError):
    """409 — the request conflicts with the current state on the server."""


class IndexNotReadyError(AtlasAPIError):
    """503 — the neighbour-search index for the requested embedding model is not built."""


_BY_STATUS: dict[int, type[AtlasAPIError]] = {
    400: ValidationError,
    401: AuthError,
    403: AuthError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    503: IndexNotReadyError,
}


def _extract_detail(response: httpx.Response) -> str:
    """The ``detail`` field FastAPI puts in every error body, flattened to one string."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text or response.reason_phrase
    if isinstance(detail, list):  # pydantic validation errors: [{"loc": [...], "msg": ...}, ...]
        return "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg')}"
            for item in detail
        )
    return str(detail) if detail is not None else response.reason_phrase


def raise_for_status(response: httpx.Response) -> None:
    """Turn a non-2xx response into the matching ``AtlasAPIError`` subclass."""
    if response.is_success:
        return
    error_class = _BY_STATUS.get(response.status_code, AtlasAPIError)
    raise error_class(response.status_code, _extract_detail(response))
