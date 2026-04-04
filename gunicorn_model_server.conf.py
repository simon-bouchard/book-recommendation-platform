# gunicorn.conf.py
"""
Gunicorn configuration for all four model servers.

Preload + copy-on-write memory sharing
---------------------------------------
``preload_app = True`` causes Gunicorn to import the FastAPI app in the
master process before any workers are forked. The ``when_ready`` hook then
calls the server's ``_load_artifacts()`` function while still in the master,
loading all numpy arrays, FAISS indices, and PyTorch weights into the
master's address space.

When workers fork from the master, they inherit all of that memory via
copy-on-write. Because workers only read from the arrays (never write to
them), the OS never needs to copy the pages. All workers share the same
physical memory ÔÇö the cost of loading multi-gigabyte artifacts is paid once,
not once per worker.

After forking, each worker's uvicorn event loop starts and the FastAPI
lifespan fires. The lifespan calls ``_load_artifacts()`` again, but since all
singletons are already set, every call is a fast no-op. The reload poller
(which is async) starts correctly in each worker's own event loop.

SERVER env var
--------------
``SERVER`` must be set to one of: embedder | similarity | als | metadata.
It is set as an ENV in the Dockerfile and as environment: in docker-compose,
so it is always available to this file at runtime.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger("gunicorn.error")

# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

_port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{_port}"

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

# With preload + CoW, memory cost does not scale linearly with worker count.
# Each numpy read releases the GIL, so 2 workers keeps a spare while one is
# blocked waiting for faiss or torch to finish. Raise via GUNICORN_WORKERS
# if profiling shows CPU underutilisation.
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"

# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------

# Load the ASGI app in the master process before forking so workers inherit
# the imported modules and singleton references via copy-on-write.
preload_app = True

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

# hybrid_sim and subject_recs involve large matrix multiplications.
# 60 s covers worst-case cold-cache inference on constrained hardware.
timeout = 60
graceful_timeout = 30
# Keep connections alive long enough to outlast the httpx client's idle
# pool on the main app side (main gunicorn keepalive=150s). 5s was too
# short: any lull longer than 5s caused httpx to attempt reuse of a dead
# socket, forcing a new TCP handshake on the next request and spiking p99.
keepalive = 75

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def when_ready(server) -> None:  # noqa: ANN001
    """
    Load artifacts in the master process before any workers are forked.

    Runs after Gunicorn has bound its socket but before the first worker is
    created. Any exception here aborts startup immediately, preventing workers
    from starting in a partially-initialized state.
    """
    _server_name = os.environ.get("SERVER")
    if not _server_name:
        raise RuntimeError(
            "SERVER environment variable is not set. "
            "Expected one of: embedder, similarity, als, metadata."
        )

    logger.info("Preloading artifacts for server: %s", _server_name)

    module = importlib.import_module(f"model_servers.{_server_name}.main")
    module._load_artifacts()

    logger.info("Artifact preload complete for server: %s", _server_name)


def worker_exit(server, worker) -> None:  # noqa: ANN001
    """Log worker exits so abnormal terminations are visible in the log stream."""
    logger.info("Worker %s exited (pid %s)", worker.age, worker.pid)

