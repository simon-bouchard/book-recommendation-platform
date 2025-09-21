# app/telemetry/tracing.py
from __future__ import annotations
from contextlib import contextmanager

@contextmanager
def span(name: str, **attrs):
    """
    PR-1: no-op span so we can add tracing in PR-2 without touching call sites again.
    Usage:
        with span("conductor.run", conv_id=..., uid=...):
            ...
    """
    try:
        yield
    finally:
        pass
