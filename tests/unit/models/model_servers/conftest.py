# tests/unit/models/model_servers/conftest.py
"""
Shared fixtures and helpers for model server unit tests.

Design contract
---------------
- ``install_mock_singleton`` is a factory fixture: call it with a singleton
  class to get a configured MagicMock installed as ``_instance``. Teardown
  is automatic and covers all classes installed in a single test.
- ``reset_singleton`` is NOT here. It is autouse and class-specific, so it
  lives in each server test module where it references the correct class.
- ``make_test_client``, ``assert_health_ok``, ``assert_scored_items``, and
  ``make_scored_arrays`` are plain functions (not fixtures) because they
  carry no state and need no pytest lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Factory fixture: singleton installation + teardown
# ---------------------------------------------------------------------------


@pytest.fixture()
def install_mock_singleton() -> Generator:
    """
    Factory fixture that installs a MagicMock as the active singleton for
    any infrastructure class and tears it down after the test.

    Usage in a per-server test module::

        @pytest.fixture()
        def mock_model(install_mock_singleton) -> MagicMock:
            return install_mock_singleton(ALSModel)

    Multiple classes can be installed in one test (e.g. similarity server).
    All are reset to None on teardown.
    """
    installed: list[type] = []

    def _install(cls: type) -> MagicMock:
        mock = MagicMock(spec=cls)
        cls._instance = mock
        installed.append(cls)
        return mock

    yield _install

    for cls in installed:
        cls._instance = None


# ---------------------------------------------------------------------------
# Fixture: version pointer file
# ---------------------------------------------------------------------------


@pytest.fixture()
def version_pointer_file(tmp_path: Path) -> tuple[Path, str]:
    """
    Write a known version string to a temporary pointer file.

    Returns ``(path, version)`` so the test can set ``MODEL_VERSION_POINTER``
    and assert against the expected string.
    """
    version = "20260227-0625-607c6e"
    pointer = tmp_path / "active_version"
    pointer.write_text(version)
    return pointer, version


# ---------------------------------------------------------------------------
# Helper: TestClient construction
# ---------------------------------------------------------------------------


def make_test_client(app: FastAPI) -> TestClient:
    """
    Return a TestClient with server exceptions suppressed.

    ``raise_server_exceptions=False`` means 5xx responses are returned as
    response objects rather than being re-raised as Python exceptions,
    which is the correct behaviour for asserting on HTTP error codes.
    """
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper: health response assertion
# ---------------------------------------------------------------------------


def assert_health_ok(response, expected_server: str) -> None:
    """
    Assert that a /health response has the correct 200 shape.

    All four servers return the same HealthResponse contract, so this helper
    avoids repeating the same three assertions in every health test.
    """
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["server"] == expected_server
    assert "artifact_version" in body


# ---------------------------------------------------------------------------
# Helpers: scored item construction and assertion
# ---------------------------------------------------------------------------


def make_scored_arrays(
    item_ids: list[int],
    scores: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the ``(item_ids, scores)`` numpy tuple that infrastructure score
    methods return.

    Centralises dtype choice so tests don't scatter ``dtype=np.int64`` and
    ``dtype=np.float32`` throughout their bodies.
    """
    return (
        np.array(item_ids, dtype=np.int64),
        np.array(scores, dtype=np.float32),
    )


def assert_scored_items(
    results: list[dict],
    expected: list[tuple[int, float]],
    abs_tol: float = 1e-4,
) -> None:
    """
    Assert that a ``results`` list of ScoredItem dicts matches expected pairs.

    Args:
        results: The ``response.json()["results"]`` list from the server.
        expected: Ordered list of ``(item_idx, score)`` tuples to match against.
        abs_tol: Absolute tolerance for float comparison.
    """
    assert len(results) == len(expected)
    for result, (exp_idx, exp_score) in zip(results, expected):
        assert result["item_idx"] == exp_idx
        assert result["score"] == pytest.approx(exp_score, abs=abs_tol)
