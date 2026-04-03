# models/domain/pipeline.py
"""
Recommendation pipeline orchestrating candidate generation and ranking.
Provides composable recommendation flow: generate -> rank.

Filtering and enrichment are handled in parallel by RecommendationService
after this pipeline returns ranked candidates, so the DB round-trip and the
metadata HTTP call overlap rather than running sequentially.
"""

from typing import List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.domain.candidate_generation import CandidateGenerator
from models.domain.rankers import NoOpRanker, Ranker
from models.domain.recommendation import Candidate
from models.domain.user import User

tracer = trace.get_tracer(__name__)


class RecommendationPipeline:
    """
    Orchestrates candidate generation and ranking.

    Pipeline stages:
    1. Generate candidates from primary generator (buffer_k = k + 50)
    2. If no candidates, use fallback generator
    3. Rank candidates
    4. Return all ranked candidates (unfiltered, unsliced)

    Filtering and slicing to k are intentionally left to the caller
    (RecommendationService) so the DB filter query can run concurrently
    with metadata enrichment rather than blocking the critical path.

    Each stage runs under a named OTel span.

    Example:
        pipeline = RecommendationPipeline(
            generator=ALSBasedGenerator(),
            fallback_generator=PopularityBasedGenerator(),
            ranker=NoOpRanker(),
        )
        candidates = await pipeline.recommend(user, k=200)
        # candidates is up to 250 ranked items; caller filters and slices
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        fallback_generator: Optional[CandidateGenerator] = None,
        ranker: Ranker = None,
    ):
        self.generator = generator
        self.fallback_generator = fallback_generator
        self.ranker = ranker or NoOpRanker()

    async def recommend(self, user: User, k: int) -> List[Candidate]:
        """
        Generate and rank candidates for a user.

        Returns up to k+50 ranked candidates. The caller is responsible
        for filtering read books and slicing to k.

        Args:
            user: User to generate recommendations for.
            k: Target number of recommendations (buffer_k = k + 50 is generated).

        Returns:
            Ranked candidates (unfiltered, up to buffer_k items).
        """
        if k <= 0:
            return []

        buffer_k = k + 50

        candidates = await self._generate(self.generator, user, buffer_k)

        if not candidates and self.fallback_generator is not None:
            candidates = await self._generate_fallback(user, buffer_k)

        if not candidates:
            return []

        return self.ranker.rank(candidates, user)

    async def _generate(
        self,
        generator: CandidateGenerator,
        user: User,
        k: int,
    ) -> List[Candidate]:
        """Run the primary candidate generator under a named span."""
        with tracer.start_as_current_span("pipeline.generate") as span:
            span.set_attribute("generator.name", generator.name)
            span.set_attribute("generator.k", k)
            try:
                candidates = await generator.generate(user, k)
                span.set_attribute("candidates.returned", len(candidates))
                return candidates
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def _generate_fallback(self, user: User, k: int) -> List[Candidate]:
        """
        Run the fallback generator under a span distinct from the primary.

        The presence of this span in a trace indicates the primary generator
        returned zero candidates for this request.
        """
        with tracer.start_as_current_span("pipeline.generate.fallback") as span:
            span.set_attribute("generator.name", self.fallback_generator.name)
            span.set_attribute("generator.k", k)
            try:
                candidates = await self.fallback_generator.generate(user, k)
                span.set_attribute("candidates.returned", len(candidates))
                return candidates
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
