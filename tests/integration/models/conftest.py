# tests/integration/models/conftest.py
"""
Pytest configuration for models integration/performance tests.
"""

import os
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    os.environ["TESTING"] = "1"
    yield


@pytest.fixture(scope="session")
def db() -> Session:
    """
    Provide a database session for the full test session.

    Reuses the application's database connection pool so tests do not pay
    repeated connection setup costs.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
async def client() -> httpx.AsyncClient:
    """
    Create an async ASGI test client scoped to the module.

    Uses httpx.AsyncClient with ASGITransport instead of TestClient. This
    eliminates the AnyIO blocking portal bridge that TestClient requires to
    run an async ASGI app from synchronous test code — that bridge added
    ~25-30ms of overhead to every request, masking real application latency.

    ASGITransport does not trigger lifespan events, so the app lifespan is
    run manually. Without this the model server HTTP clients are never
    initialised and every request fails.

    The async engine is disposed in the finally block while the event loop
    is still live, preventing RuntimeError: Event loop is closed from
    aiomysql's Connection.__del__ running after the loop shuts down.

    Module scope pairs with pytestmark = pytest.mark.asyncio(loop_scope="module")
    in the test file so the fixture and all tests share one event loop. With
    asyncio_default_test_loop_scope=function (the project default) a module-
    scoped async fixture and its tests would otherwise run on different loops
    and the fixture would be unusable.
    """
    from app.database import async_engine
    from main import app

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac

    if async_engine is not None:
        await async_engine.dispose()


@pytest.fixture(scope="session")
def performance_baselines_dir():
    """Create and return the performance baselines directory."""
    baselines_dir = Path(__file__).parent / "performance_baselines"
    baselines_dir.mkdir(exist_ok=True)
    return baselines_dir


@pytest.fixture(scope="session", autouse=True)
def ensure_model_servers_ready(test_env):
    """
    Block the test session until all model servers report ready.

    Raises RuntimeError if any server is unreachable after the timeout,
    which surfaces a clear message rather than cryptic 500s in every test.
    """
    import time

    server_urls = {
        "embedder": os.environ.get("EMBEDDER_URL", "http://embedder:8001"),
        "similarity": os.environ.get("SIMILARITY_URL", "http://similarity:8002"),
        "als": os.environ.get("ALS_URL", "http://als:8003"),
        "metadata": os.environ.get("METADATA_URL", "http://metadata:8004"),
    }

    timeout_seconds = 60
    poll_interval = 2.0
    deadline = time.monotonic() + timeout_seconds

    print("\nWaiting for model servers to be ready...")

    for name, url in server_urls.items():
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{url}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(f"  {name}: ready")
                    break
            except httpx.RequestError:
                pass

            time.sleep(poll_interval)
        else:
            raise RuntimeError(
                f"Model server '{name}' at {url} did not become ready within "
                f"{timeout_seconds}s. Ensure all containers are running before "
                f"executing integration tests."
            )

    print("All model servers ready.\n")
    yield


@pytest.fixture(scope="session", autouse=True)
def disable_route_caching():
    """
    Patch both cache decorators to identity functions for the full test session.

    The decorators are applied at module import time, so routes.models must be
    removed from sys.modules and re-imported after patching to pick up the
    inert versions. Without this, measurements reflect cache hits rather than
    real pipeline latency.
    """
    import sys
    from unittest.mock import patch

    import models.cache as cache_module

    with (
        patch.object(cache_module, "cached_recommendations", lambda f: f),
        patch.object(cache_module, "cached_similarity", lambda f: f),
    ):
        sys.modules.pop("routes.models", None)
        yield
        sys.modules.pop("routes.models", None)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_test_data: mark test as requiring configured test data"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require test data if not configured."""
    test_data_file = Path(__file__).parent / "test_data_config.json"

    if not test_data_file.exists():
        skip_marker = pytest.mark.skip(
            reason="Test data not configured. Run setup_test_data.py first."
        )
        for item in items:
            if "requires_test_data" in item.keywords:
                item.add_marker(skip_marker)
