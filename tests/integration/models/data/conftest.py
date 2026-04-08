# tests/integration/models/data/conftest.py
"""
Pytest configuration for data loader integration tests.

These tests require real artifact files on disk (embeddings, ALS factors,
Bayesian scores, metadata). They do NOT require model servers or a database.
All tests in this directory are skipped automatically when artifacts are absent.
"""

import pytest
from models.core.paths import PATHS


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_artifacts: mark test as requiring artifact files on disk"
    )


@pytest.fixture(scope="session", autouse=True)
def require_artifacts():
    """Skip the entire session if the active artifact version is not present."""
    if not PATHS.active_version_file.exists():
        pytest.skip(
            "Artifact files not available — run the training pipeline first "
            f"(expected: {PATHS.active_version_file})"
        )
