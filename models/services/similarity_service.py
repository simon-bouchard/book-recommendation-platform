# models/services/similarity_service.py
"""
Book similarity service using shared FAISS indices for optimal performance.
Hybrid mode uses FAISS for initial retrieval then blends scores for accuracy.
"""

import logging
import time
from typing import List, Dict, Optional

from models.client.registry import get_similarity_client, get_metadata_client

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Book similarity service using singleton FAISS indices.

    All subject operations share the same FAISS index for memory efficiency.
    Hybrid mode uses FAISS for fast initial retrieval then exact scoring for accuracy.
    """

    def __init__(self):
        """Initialize similarity service."""
        pass

    async def get_similar(
        self,
        item_idx: int,
        mode: str = "subject",
        k: int = 200,
        alpha: float = 0.6,
    ) -> List[Dict]:
        """Find similar books with mode-specific filtering."""
        start_time = time.time()

        logger.info("Similarity search started", extra={"item_idx": item_idx, "mode": mode, "k": k})

        try:
            if mode == "subject":
                resp = await get_similarity_client().subject_sim(item_idx, k)
            elif mode == "als":
                resp = await get_similarity_client().als_sim(item_idx, k)
            elif mode == "hybrid":
                resp = await get_similarity_client().hybrid_sim(item_idx, k, alpha)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            enrich_resp = await get_metadata_client().enrich([r.item_idx for r in resp.results])
            meta = {b.item_idx: b for b in enrich_resp.books}

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Similarity search completed",
                extra={
                    "item_idx": item_idx,
                    "mode": mode,
                    "count": len(resp.results),
                    "latency_ms": latency_ms,
                },
            )

            return [
                {
                    "item_idx": b.item_idx,
                    "title": b.title,
                    "author": b.author,
                    "year": b.year,
                    "isbn": b.isbn,
                    "cover_id": b.cover_id,
                    "score": r.score,
                }
                for r in resp.results
                if (b := meta.get(r.item_idx)) is not None
            ]

        except Exception as e:
            logger.error(
                "Similarity search failed",
                extra={"item_idx": item_idx, "mode": mode, "error": str(e)},
                exc_info=True,
            )
            raise
