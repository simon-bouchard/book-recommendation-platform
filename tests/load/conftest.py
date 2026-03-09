# tests/load/conftest.py
"""
Shared pytest configuration for all load test suites.

Skips every test in tests/load/ when PERF_TEST_BASE_URL is not set, so load
tests are silently bypassed in normal CI runs and only execute when a live
stack is explicitly targeted.
"""

import os
from pathlib import Path

import httpx
import pytest

_LOAD_DIR = Path(__file__).parent


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "load: requires a live server at PERF_TEST_BASE_URL",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Attach a skip marker to every test under tests/load/ when the target URL
    is absent, rather than failing at fixture-setup time.

    Filtering by path prevents this hook from accidentally skipping tests
    outside tests/load/ when this conftest is active in a combined run.
    """
    if os.environ.get("PERF_TEST_BASE_URL"):
        return

    reason = (
        "PERF_TEST_BASE_URL is not set. "
        "Point it at a running server before executing load tests, e.g.: "
        "export PERF_TEST_BASE_URL=http://localhost:8000"
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(str(item.fspath)).is_relative_to(_LOAD_DIR):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url() -> str:
    """Return the live server base URL, stripped of any trailing slash."""
    return os.environ["PERF_TEST_BASE_URL"].rstrip("/")


@pytest.fixture(scope="module")
async def http_client(base_url: str) -> httpx.AsyncClient:
    """
    Async HTTP client configured for the live server.

    Module scope pairs with pytestmark = pytest.mark.asyncio(loop_scope="module")
    in each test file so the client and all tests in a module share one event
    loop and one persistent connection pool.

    The timeout of 30 s is generous enough to absorb a cold-start request on
    the recommendation pipeline while still failing fast on hung connections.
    """
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
    ) as client:
        yield client
