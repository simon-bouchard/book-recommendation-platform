# metrics.py
"""
Prometheus instrumentation for the FastAPI application.
Exposes /metrics endpoint and defines custom application-level metrics.
Call apply_metrics(app) from main.py before route registration.
"""

from fastapi import FastAPI
from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
)
from prometheus_fastapi_instrumentator import Instrumentator

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

SIMILARITY_REQUESTS = Counter(
    "bookrec_similarity_requests_total",
    "Total number of similarity requests",
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

SIMILARITY_LATENCY = Histogram(
    "bookrec_similarity_latency_seconds",
    "End-to-end similarity lookup latency in seconds",
    ["mode"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RECSYS_RESULT_COUNT = Histogram(
    "bookrec_recsys_result_count",
    "Number of recommendations returned after filtering, by mode",
    ["mode"],
    buckets=[0, 10, 25, 50, 100, 125, 150, 175, 190, 195, 200, 250, 300, 500],
)

RECSYS_SCORE = Histogram(
    "bookrec_recsys_score",
    "Distribution of recommendation scores, by mode",
    ["mode"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0],
)

RECSYS_EMPTY = Counter(
    "bookrec_recsys_empty_results_total",
    "Number of recommendation requests that returned zero results, by mode",
    ["mode"],
)

SIMILARITY_RESULT_COUNT = Histogram(
    "bookrec_similarity_result_count",
    "Number of similar books returned after enrichment, by mode",
    ["mode"],
    buckets=[0, 10, 25, 50, 100, 125, 150, 175, 190, 195, 200, 250, 300, 500],
)

SIMILARITY_SCORE = Histogram(
    "bookrec_similarity_score",
    "Distribution of similarity scores, by mode",
    ["mode"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0],
)

SIMILARITY_EMPTY = Counter(
    "bookrec_similarity_empty_results_total",
    "Number of similarity requests that returned zero results, by mode",
    ["mode"],
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
    the /metrics ASGI endpoint directly — bypassing FastAPI route handling so
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
