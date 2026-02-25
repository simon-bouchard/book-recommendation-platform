# metrics.py
"""
Prometheus instrumentation for the FastAPI application.
Exposes /metrics endpoint and defines custom application-level metrics.
Call apply_metrics(app) from main.py before route registration.
"""

import os
from prometheus_client import CollectorRegistry, Counter, Histogram, multiprocess, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Multiprocess-aware registry
# ---------------------------------------------------------------------------


def _build_registry() -> CollectorRegistry:
    """
    Return a CollectorRegistry that aggregates across all Gunicorn workers
    when PROMETHEUS_MULTIPROC_DIR is set, or a plain registry for dev.
    """
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return CollectorRegistry(auto_describe=True)


REGISTRY = _build_registry()


# ---------------------------------------------------------------------------
# Custom metrics
# ---------------------------------------------------------------------------

CHAT_REQUESTS = Counter(
    "bookrec_chat_requests_total",
    "Total number of chatbot requests",
    ["target"],
    registry=REGISTRY,
)

RECSYS_REQUESTS = Counter(
    "bookrec_recsys_requests_total",
    "Total number of recommendation requests",
    ["mode"],
    registry=REGISTRY,
)

SEARCH_REQUESTS = Counter(
    "bookrec_search_requests_total",
    "Total number of search requests",
    ["search_mode"],
    registry=REGISTRY,
)

RATING_ACTIONS = Counter(
    "bookrec_rating_actions_total",
    "Total number of rating create/update/delete actions",
    ["action"],
    registry=REGISTRY,
)

RECSYS_LATENCY = Histogram(
    "bookrec_recsys_latency_seconds",
    "End-to-end recommendation pipeline latency in seconds",
    ["mode"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

CHAT_LATENCY = Histogram(
    "bookrec_chat_latency_seconds",
    "End-to-end chatbot response latency in seconds",
    ["target"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Instrumentation setup
# ---------------------------------------------------------------------------


def apply_metrics(app: FastAPI) -> None:
    """
    Attach Prometheus instrumentation to the FastAPI app.

    Mounts a /metrics endpoint that is served directly by prometheus_client,
    bypassing FastAPI middleware (including rate limiting and security headers).
    The standard instrumentator handles per-endpoint HTTP metrics automatically.
    """
    # HTTP request metrics: count, latency, in-progress — all labeled by
    # endpoint template and status code, normalised to avoid cardinality explosion.
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
        body_handlers=[],
        registry=REGISTRY,
    ).instrument(app).expose(app, include_in_schema=False, should_gzip=False)
