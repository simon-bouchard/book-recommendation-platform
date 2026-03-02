# models/client/_exceptions.py
"""
Exception hierarchy for model server HTTP clients.

All errors raised by model server clients are subclasses of ModelServerError,
allowing callers to catch the full family with a single except clause or
handle specific failure modes individually.
"""


class ModelServerError(Exception):
    """Base class for all model server client errors."""


class ModelServerUnavailableError(ModelServerError):
    """
    Server could not be reached or returned a 5xx response.

    Raised on connection failures, timeouts, and server-side errors.
    Callers should treat this as a transient fault and handle gracefully
    (e.g. fallback to a degraded mode rather than surfacing a 500 to the user).
    """


class ModelServerRequestError(ModelServerError):
    """
    Server rejected the request with a 4xx response.

    Raised when the client sends a malformed or invalid request. This
    indicates a programming error rather than a transient infrastructure
    fault and should not be retried.
    """
