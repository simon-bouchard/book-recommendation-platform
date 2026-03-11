# model_servers/_shared/server_utils.py
"""
Shared utilities for model server startup and health reporting.

Provides the artifact version reader and a lifespan factory used by all
four model servers, eliminating boilerplate that is identical across them.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI


def get_artifact_version() -> str:
    """
    Read the current artifact version string from the version pointer file.

    Reads the path from the MODEL_VERSION_POINTER environment variable, which
    is set per-container in docker-compose.yml and points to the active_version
    file on the host-mounted artifacts volume.

    Returns 'unknown' if the variable is unset, the file is absent, or the
    file cannot be read, so that a missing pointer never blocks a health check.
    """
    pointer_path = os.environ.get("MODEL_VERSION_POINTER", "")
    if not pointer_path or not os.path.exists(pointer_path):
        return "unknown"
    try:
        with open(pointer_path) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def make_lifespan(load_fn: Callable[[], None]) -> Callable[[FastAPI], AsyncGenerator]:
    """
    Return a FastAPI lifespan context manager that calls load_fn on startup.

    Gunicorn's when_ready hook calls load_fn in the master process before any
    workers are forked, so the lifespan call in each worker is a fast no-op
    (all singletons are already initialized). The lifespan exists to satisfy
    FastAPI's startup contract and to handle the direct-uvicorn case in tests.

    Args:
        load_fn: Zero-argument callable that initializes this server's artifacts.

    Returns:
        An async context manager suitable for use as FastAPI(lifespan=...).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        load_fn()
        yield

    return lifespan
