# tests/integration/models/conftest.py
"""
Pytest configuration for models performance tests.
"""

import pytest
import os
from pathlib import Path
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    os.environ["TESTING"] = "1"
    # Add any other test-specific environment variables
    yield
    # Cleanup if needed


@pytest.fixture(scope="session")
def db() -> Session:
    """
    Provide a database session for tests.
    Reuses the application's database connection.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client(db: Session):
    """
    Create a FastAPI test client.
    Scope is module-level to reuse across tests in same file.
    """
    from main import app

    # Create test client
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def ensure_models_loaded():
    """
    Ensure all models are preloaded before tests.
    This warms up the cache so first test isn't penalized.
    """
    from models.data.loaders import preload_all_artifacts

    print("\nPreloading models for performance tests...")
    preload_all_artifacts()
    print("Models loaded\n")

    yield


@pytest.fixture(scope="session")
def performance_baselines_dir():
    """Create and return the performance baselines directory."""
    baselines_dir = Path(__file__).parent / "performance_baselines"
    baselines_dir.mkdir(exist_ok=True)
    return baselines_dir


# Auto-use fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(test_env, ensure_models_loaded):
    """
    Automatically set up test environment for all tests.
    This runs once per test session.
    """
    pass


# Optional: Skip tests if test data not configured
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_test_data: mark test as requiring configured test data"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require test data if not configured."""
    import sys
    from pathlib import Path

    # Check if test data is configured
    test_data_file = Path(__file__).parent / "test_data_config.json"

    if not test_data_file.exists():
        skip_marker = pytest.mark.skip(
            reason="Test data not configured. Run setup_test_data.py first."
        )
        for item in items:
            if "requires_test_data" in item.keywords:
                item.add_marker(skip_marker)
