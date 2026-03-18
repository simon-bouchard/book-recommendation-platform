# app/semantic_index/subject_service.py
"""
Subject search service.

Resolves free-text subject phrases to candidate subject indices by calling
the semantic server once per phrase (concurrently), then groups the results
into the phrase -> candidates structure expected by the chatbot tools.
"""

import asyncio
import logging

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.client.registry import get_semantic_client

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_SCORE_THRESHOLD = 0.4


class SubjectSearchService:
    """
    Stateless service for resolving subject phrases to subject indices.

    Calls the semantic server's /subject_search endpoint once per phrase,
    running all phrases concurrently. Results below _SCORE_THRESHOLD are
    filtered out before returning.

    Output structure matches the old TF-IDF tool so the chatbot prompt
    and downstream tool calls are unchanged:
        [{"phrase": "...", "candidates": [{"subject_idx": int, "subject": str, "score": float}, ...]}, ...]
    """

    async def search(self, phrases: list[str], top_k: int = 5) -> list[dict]:
        """
        Resolve a list of subject phrases to candidate subject indices.

        Args:
            phrases: Free-text subject phrases from the chatbot.
            top_k: Number of candidates to request per phrase (max 20).

        Returns:
            List of {"phrase": str, "candidates": list[dict]} dicts, one per
            input phrase. Phrases with no matches above the score threshold
            return an empty candidates list.
        """
        with tracer.start_as_current_span("subject.search") as span:
            span.set_attribute("phrase_count", len(phrases))
            span.set_attribute("top_k", top_k)

            try:
                responses = await asyncio.gather(
                    *[self._search_one(phrase, top_k) for phrase in phrases]
                )
                results = list(responses)
                span.set_attribute(
                    "total_candidates",
                    sum(len(r["candidates"]) for r in results),
                )
                return results

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def _search_one(self, phrase: str, top_k: int) -> dict:
        response = await get_semantic_client().subject_search(phrase=phrase, top_k=top_k)
        candidates = [
            {
                "subject_idx": m.subject_idx,
                "subject": m.subject_name,
                "count": m.count,
                "score": m.score,
            }
            for m in response.matches
            if m.score >= _SCORE_THRESHOLD
        ]
        return {"phrase": phrase, "candidates": candidates}
