# models/client/registry.py
"""
Singleton accessors for all model server HTTP clients.

Clients are initialized lazily on first access and reused across requests
for connection pooling. Call close_all() from the FastAPI lifespan on
application shutdown to release all connections cleanly.
"""

from __future__ import annotations

import logging

from models.client.als import AlsClient
from models.client.embedder import EmbedderClient
from models.client.metadata import MetadataClient
from models.client.similarity import SimilarityClient

logger = logging.getLogger(__name__)

_embedder: EmbedderClient | None = None
_similarity: SimilarityClient | None = None
_als: AlsClient | None = None
_metadata: MetadataClient | None = None


def get_embedder_client() -> EmbedderClient:
    """Get or create the singleton embedder client."""
    global _embedder
    if _embedder is None:
        _embedder = EmbedderClient.from_env()
    return _embedder


def get_similarity_client() -> SimilarityClient:
    """Get or create the singleton similarity client."""
    global _similarity
    if _similarity is None:
        _similarity = SimilarityClient.from_env()
    return _similarity


def get_als_client() -> AlsClient:
    """Get or create the singleton ALS client."""
    global _als
    if _als is None:
        _als = AlsClient.from_env()
    return _als


def get_metadata_client() -> MetadataClient:
    """Get or create the singleton metadata client."""
    global _metadata
    if _metadata is None:
        _metadata = MetadataClient.from_env()
    return _metadata


async def close_all() -> None:
    """
    Close all active client connections.

    Should be called from the FastAPI lifespan on application shutdown
    to release connection pool resources cleanly.
    """
    global _embedder, _similarity, _als, _metadata

    for name, client in [
        ("embedder", _embedder),
        ("similarity", _similarity),
        ("als", _als),
        ("metadata", _metadata),
    ]:
        if client is not None:
            try:
                await client.aclose()
                logger.info("Closed %s client", name)
            except Exception as e:
                logger.warning("Error closing %s client: %s", name, e)

    _embedder = _similarity = _als = _metadata = None
