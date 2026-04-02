# models/domain/pipeline.py
"""
Recommendation pipeline orchestrating candidate generation, filtering, and ranking.
Provides composable recommendation flow: generate -> filter -> rank -> top k.
"""

from typing import List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models.domain.candidate_generation import CandidateGenerator
from models.domain.filters import Filter, NoFilter
from models.domain.rankers import NoOpRanker, Ranker
from models.domain.recommendation import Candidate
from models.domain.user import User

tracer = trace.get_tracer(__name__)


class RecommendationPipeline:
    """
    Orchestrates the recommendation process.

    Pipeline stages:
    1. Generate candidates from primary generator
    2. If no candidates, use fallback generator
    3. Apply filter (async — native aiomysql DB call on the event loop)
    4. Rank remaining candidates
    5. Return top k

    Each stage runs under a named OTel span so the Jaeger waterfall shows
    exact wall-clock duration per stage. The httpx auto-instrumentor attaches
    model server POST calls as children of the generate spans; the SQLAlchemy
    auto-instrumentor attaches the DB query as a child of the filter span.
    No timing code is needed here — the spans capture it automatically.

    Example:
        pipeline = RecommendationPipeline(
            generator=ALSBasedGenerator(),
            fallback_generator=BayesianPopularityGenerator(),
            filter=ReadBooksFilter(),
            ranker=NoOpRanker()
        )

        recommendations = await pipeline.recommend(user, k=20, db=async_session)
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        fallback_generator: Optional[CandidateGenerator] = None,
        filter: Filter = None,
        ranker: Ranker = None,
    ):
        """
        Initialize recommendation pipeline.

        Args:
            generator: Primary candidate generator.
            fallback_generator: Generator to use if primary returns no candidates.
            filter: Filter to apply to candidates (default: NoFilter).
            ranker: Ranker for final ordering (default: NoOpRanker).
        """
        self.generator = generator
        self.fallback_generator = fallback_generator
        self.filter = filter or NoFilter()
        self.ranker = ranker or NoOpRanker()

    async def recommend(self, user: User, k: int) -> List[Candidate]:
        """
        Generate recommendations for a user.

        Args:
            user: User to generate recommendations for.
            k: Number of recommendations to return.

        Returns:
            Top k candidates after generation, filtering, and ranking.
        """
        if k <= 0:
            return []

        buffer_k = k + 50

        candidates = await self._generate(self.generator, user, buffer_k)

        if not candidates and self.fallback_generator is not None:
            candidates = await self._generate_fallback(user, buffer_k)

        if not candidates:
            return []

        candidates = await self._filter(candidates, user)

        if not candidates:
            return []

        candidates = self.ranker.rank(candidates, user)

        return candidates[:k]

    async def _generate(
        self,
        generator: CandidateGenerator,
        user: User,
        k: int,
    ) -> List[Candidate]:
        """
        Run the primary candidate generator under a named span.

        The httpx auto-instrumentor attaches the model server POST as a child
        of this span, giving the waterfall: generate -> POST /als_recs (or
        POST /embed + POST /subject_recs for the subject path).
        """
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

        The presence of this span in a trace is itself a signal: it indicates
        the primary generator returned zero candidates for this request.
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

    async def _filter(
        self,
        candidates: List[Candidate],
        user: User,
    ) -> List[Candidate]:
        """
        Apply the filter under a named span.

        The filter uses the raw aiomysql pool directly — no session argument needed.
        """
        with tracer.start_as_current_span("pipeline.filter") as span:
            span.set_attribute("filter.type", type(self.filter).__name__)
            span.set_attribute("filter.input_count", len(candidates))
            try:
                result = await self.filter.apply(candidates, user)
                span.set_attribute("filter.output_count", len(result))
                span.set_attribute("filter.removed_count", len(candidates) - len(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
