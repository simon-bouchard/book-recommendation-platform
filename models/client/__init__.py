# models/client/__init__.py
"""
HTTP clients for all model servers.

Each client is a thin async wrapper around httpx.AsyncClient. Instantiate
via the from_env() classmethod so base URLs are read from environment
variables, or pass a base_url directly for testing.

Usage:
    from models.client import AlsClient, EmbedderClient, MetadataClient, SimilarityClient
    from models.client import ModelServerError, ModelServerUnavailableError
"""

from models.client._exceptions import (
    ModelServerError,
    ModelServerRequestError,
    ModelServerUnavailableError,
)
from models.client.als import AlsClient
from models.client.embedder import EmbedderClient
from models.client.metadata import MetadataClient
from models.client.similarity import SimilarityClient

__all__ = [
    "AlsClient",
    "EmbedderClient",
    "MetadataClient",
    "ModelServerError",
    "ModelServerRequestError",
    "ModelServerUnavailableError",
    "SimilarityClient",
]
