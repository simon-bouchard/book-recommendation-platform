# tests/load/models/_constants.py
"""
Test data constants and load test configuration for models load tests.

TEST_BOOK_IDS is populated from test_data_config.json when it exists (written
by setup_test_data.py). Every ID in that list is guaranteed to have ALS
factors, so similarity tests for modes 'als' and 'hybrid' will never return
422. Fall back to the hardcoded defaults when the config file is absent so
tests can still run without the setup step, with the caveat that some IDs
may produce 422 responses for ALS/hybrid modes.

Lives in its own module so locustfile.py can import it without pulling in
pytest, which conftest.py would introduce as a transitive dependency.

Run setup once before load testing:
    export PERF_TEST_BASE_URL=http://localhost:8000
    python tests/load/models/setup_test_data.py
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Test user identifiers (no verification needed — user warmth is stable)
# ---------------------------------------------------------------------------

WARM_USER_IDS: list[int] = [
    11676,
    98391,
    189835,
    153662,
    23902,
    171118,
    235105,
    76499,
    16795,
    248718,
]

COLD_WITH_SUBJECTS_USER_IDS: list[int] = [
    248965,
    249650,
    249939,
    250634,
    251575,
    251744,
    252628,
    253310,
    258352,
    259734,
]

COLD_WITHOUT_SUBJECTS_USER_IDS: list[int] = [278860, 278855, 52702]

# ---------------------------------------------------------------------------
# Book IDs — loaded from verified config when available
# ---------------------------------------------------------------------------

_CONFIG_FILE = Path(__file__).parent / "test_data_config.json"

_FALLBACK_BOOK_IDS: list[int] = [
    1666,
    45959,
    402,
    27,
    41636,
    166,
    44327,
    3240,
    45503,
    49865,
    43852,
    208,
    41810,
    12372,
    3158,
    729,
    2015,
    46695,
    46839,
    45820,
]


def _load_book_ids() -> list[int]:
    """
    Load verified ALS book IDs from test_data_config.json.

    Returns the hardcoded fallback list with a warning when the config file
    does not exist so tests degrade gracefully rather than failing at import.
    """
    if not _CONFIG_FILE.exists():
        import warnings

        warnings.warn(
            f"test_data_config.json not found at {_CONFIG_FILE}. "
            "Using unverified fallback book IDs — some ALS/hybrid similarity "
            "tests may receive 422 responses. "
            "Run setup_test_data.py to generate verified IDs.",
            stacklevel=2,
        )
        return _FALLBACK_BOOK_IDS

    with open(_CONFIG_FILE) as f:
        config = json.load(f)

    ids = config.get("verified_book_ids_with_als", [])
    if not ids:
        import warnings

        warnings.warn(
            "test_data_config.json exists but contains no verified ALS book IDs. "
            "Re-run setup_test_data.py.",
            stacklevel=2,
        )
        return _FALLBACK_BOOK_IDS

    return ids


TEST_BOOK_IDS: list[int] = _load_book_ids()

# ---------------------------------------------------------------------------
# Load test tuning parameters
# ---------------------------------------------------------------------------

WARMUP_RUNS: int = 10

# Ramp test: total requests fired at each concurrency level (distributed evenly
# across workers). Lower values give faster runs; raise for tighter percentiles.
REQUESTS_PER_LEVEL: int = 60

# Concurrency levels exercised by the ramp test.
CONCURRENCY_LEVELS: list[int] = [1, 2, 5, 10, 20]

# Sustained test: number of parallel workers and wall-clock duration.
SUSTAINED_WORKERS: int = 10
SUSTAINED_DURATION_S: int = 30
