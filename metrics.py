# metrics.py
"""
Prometheus instrumentation for the FastAPI application.
Exposes /metrics endpoint and defines custom application-level metrics.
Call apply_metrics(app) from main.py before route registration.
"""

from prometheus_client import Counter, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Custom metrics
# Registered against the default REGISTRY. prometheus_client automatically
# uses the multiprocess shared directory when PROMETHEUS_MULTIPROC_DIR is set,
# so no manual registry wiring is needed here.
# ---------------------------------------------------------------------------

CHAT_REQUESTS = Counter(
    "bookrec_chat_requests_total",
    "Total number of chatbot requests",
    ["target"],
)

RECSYS_REQUESTS = Counter(
    "bookrec_recsys_requests_total",
    "Total number of recommendation requests",
    ["mode"],
)

SEARCH_REQUESTS = Counter(
    "bookrec_search_requests_total",
    "Total number of search requests",
    ["search_mode"],
)

RATING_ACTIONS = Counter(
    "bookrec_rating_actions_total",
    "Total number of rating create/update/delete actions",
    ["action"],
)

RECSYS_LATENCY = Histogram(
    "bookrec_recsys_latency_seconds",
    "End-to-end recommendation pipeline latency in seconds",
    ["mode"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

CHAT_LATENCY = Histogram(
    "bookrec_chat_latency_seconds",
    "End-to-end chatbot response latency in seconds",
    ["target"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
)


# ---------------------------------------------------------------------------
# Instrumentation setup
# ---------------------------------------------------------------------------

def apply_metrics(app: FastAPI) -> None:
    """
    Attach Prometheus instrumentation to the FastAPI app.

    Instruments all HTTP endpoints for request count and latency, and mounts
    the /metrics ASGI endpoint directly — bypassing FastAPI middleware so
    rate limiting and security headers do not interfere with Prometheus scraping.

    make_asgi_app() detects PROMETHEUS_MULTIPROC_DIR automatically and
    aggregates across all Gunicorn workers on each scrape when in prod.
    In dev (uvicorn, no env var set) it serves the single-process registry.
    """
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
        body_handlers=[],
    ).instrument(app)

    app.mount("/metrics", make_asgi_app())
