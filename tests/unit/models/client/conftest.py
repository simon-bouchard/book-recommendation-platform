# tests/unit/models/client/conftest.py
"""
Shared fixtures and helpers for model server HTTP client unit tests.

Design contract
---------------
All client methods are async, so every test module in this package uses
async def tests. The ``anyio_backend`` fixture configures the asyncio
runner once here rather than in every module.

``respx_mock`` is provided automatically by the respx library as a pytest
fixture. It intercepts all httpx calls within a test and resets between
tests. The route configurator helpers below accept a ``respx_mock`` value
and configure routes on it, keeping individual test bodies concise.

``reset_registry`` is autouse so module-level globals in the registry
never bleed between tests, regardless of which module is running.

Plain functions vs fixtures
---------------------------
Route configurators and payload builders are plain functions (not fixtures)
because they carry no state and need no pytest lifecycle. Tests call them
directly inside the test body after acquiring ``respx_mock``.
"""

from __future__ import annotations

from typing import Generator

import httpx
import pytest

import models.client.registry as _registry_module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_BASE_URL = "http://test-server"


# ---------------------------------------------------------------------------
# Async backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def anyio_backend() -> str:
    """Run all async tests in this package under the asyncio backend."""
    return "asyncio"


# ---------------------------------------------------------------------------
# Registry reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry() -> Generator[None, None, None]:
    """
    Reset all registry module-level client globals before and after every test.

    Autouse ensures registry state never leaks between tests even when the
    test itself does not reference the registry directly.
    """
    _registry_module._embedder = None
    _registry_module._similarity = None
    _registry_module._als = None
    _registry_module._metadata = None
    yield
    _registry_module._embedder = None
    _registry_module._similarity = None
    _registry_module._als = None
    _registry_module._metadata = None


# ---------------------------------------------------------------------------
# Route configurators
# ---------------------------------------------------------------------------


def mock_post_200(respx_mock, url: str, json_body: dict) -> None:
    """Configure respx to return a 200 JSON response for a POST to ``url``."""
    respx_mock.post(url).mock(return_value=httpx.Response(200, json=json_body))


def mock_post_500(respx_mock, url: str, detail: str = "internal server error") -> None:
    """Configure respx to return a 500 JSON response for a POST to ``url``."""
    respx_mock.post(url).mock(return_value=httpx.Response(500, json={"detail": detail}))


def mock_post_503(respx_mock, url: str, detail: str = "server not initialized") -> None:
    """Configure respx to return a 503 JSON response for a POST to ``url``."""
    respx_mock.post(url).mock(return_value=httpx.Response(503, json={"detail": detail}))


def mock_post_422(respx_mock, url: str) -> None:
    """Configure respx to return a 422 validation error for a POST to ``url``."""
    respx_mock.post(url).mock(return_value=httpx.Response(422, json={"detail": "validation error"}))


def mock_post_connect_error(respx_mock, url: str) -> None:
    """Configure respx to raise ConnectError for a POST to ``url``."""
    respx_mock.post(url).mock(side_effect=httpx.ConnectError("connection refused"))


def mock_post_timeout(respx_mock, url: str) -> None:
    """Configure respx to raise TimeoutException for a POST to ``url``."""
    respx_mock.post(url).mock(side_effect=httpx.TimeoutException("timed out"))


def mock_post_remote_protocol_error(respx_mock, url: str) -> None:
    """Configure respx to raise RemoteProtocolError for a POST to ``url``."""
    respx_mock.post(url).mock(side_effect=httpx.RemoteProtocolError("peer closed connection"))


# ---------------------------------------------------------------------------
# JSON payload builders
# ---------------------------------------------------------------------------


def scored_items_json(pairs: list[tuple[int, float]]) -> dict:
    """
    Build a ``{"results": [...]}`` payload matching the SimResponse /
    AlsRecsResponse / SubjectRecsResponse contract.

    Args:
        pairs: Ordered list of ``(item_idx, score)`` tuples.
    """
    return {"results": [{"item_idx": idx, "score": score} for idx, score in pairs]}


def book_meta_json(item_idx: int, title: str, **kwargs) -> dict:
    """
    Build a minimal BookMeta JSON dict suitable for use in enrich / popular
    response payloads.

    Required contract fields (``num_ratings``) are defaulted so callers only
    need to supply the fields relevant to their assertion.
    """
    return {"item_idx": item_idx, "title": title, "num_ratings": 0, **kwargs}


def books_json(books: list[dict]) -> dict:
    """
    Wrap a list of BookMeta dicts in the ``{"books": [...]}`` envelope used
    by EnrichResponse and PopularResponse.
    """
    return {"books": books}
