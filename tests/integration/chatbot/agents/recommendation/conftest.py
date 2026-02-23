# tests/integration/chatbot/agents/recommendation/conftest.py
"""
Fixtures for RecommendationAgent pipeline integration tests.

All sub-agent execute() methods are AsyncMock.
CurationAgent.execute_stream() is mocked as an async generator — it cannot
use AsyncMock directly because async generators and coroutines are
distinct protocol in Python; instead, a MagicMock with side_effect
that returns a fresh async generator on each call is used.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Optional, Dict, Any

from app.agents.domain.entities import BookRecommendation
from app.agents.domain.recsys_schemas import (
    PlannerStrategy,
    ExecutionContext,
    RetrievalOutput,
)
from app.agents.schemas import StreamChunk

from tests.integration.chatbot.recsys_fixtures import (
    test_user_warm,
    test_user_cold,
    test_user_new,
    test_user_with_profile,
    get_user_rating_count,
)


# ==============================================================================
# Test Data Factories
# ==============================================================================


class CandidateFactory:
    """
    Generates book candidates as dicts, matching the raw output of retrieval tools.

    The orchestrator converts these dicts to BookRecommendation objects via
    StandardResultProcessor._build_recommendations_from_objects(), so the
    dict keys must match the field names that method expects.
    """

    @staticmethod
    def create_batch(
        n: int,
        start_idx: int = 10000,
        source_tool: str = "als_recs",
    ) -> List[Dict[str, Any]]:
        """
        Generate n candidate dicts.

        Args:
            n: Number of candidates.
            start_idx: First item_idx (increments by 1 per candidate).
            source_tool: Logical source label stored in the dict.
        """
        return [
            {
                "item_idx": start_idx + i,
                "title": f"Test Book {start_idx + i}",
                "author": f"Author {i % 10}",
                "year": 1990 + (i % 30),
                "source_tool": source_tool,
            }
            for i in range(n)
        ]

    @staticmethod
    def create_with_metadata(
        item_idx: int,
        title: str,
        author: str,
        year: int,
        subjects: Optional[List[str]] = None,
        tones: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a single candidate dict with specific metadata.

        Used for tests that need to assert field-by-field preservation
        through the Retrieval → BookRecommendation conversion.
        """
        candidate: Dict[str, Any] = {
            "item_idx": item_idx,
            "title": title,
            "author": author,
            "year": year,
        }
        if subjects is not None:
            candidate["subjects"] = subjects
        if tones is not None:
            candidate["tones"] = tones
        return candidate

    @staticmethod
    def create_book_recommendations(
        n: int,
        start_idx: int = 20000,
    ) -> List[BookRecommendation]:
        """
        Generate n BookRecommendation objects.

        Used by MockSelectionBuilder which must return the post-conversion type.
        """
        return [
            BookRecommendation(
                item_idx=start_idx + i,
                title=f"Selected Book {start_idx + i}",
                author=f"Author {i % 10}",
                year=1990 + (i % 30),
            )
            for i in range(n)
        ]


class StrategyFactory:
    """Generates PlannerStrategy objects for common test scenarios."""

    @staticmethod
    def warm_user_strategy(profile_data: Optional[Dict] = None) -> PlannerStrategy:
        """ALS-first strategy for warm users."""
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            fallback_tools=["popular_books"],
            reasoning="User has sufficient rating history for collaborative filtering",
            profile_data=profile_data,
        )

    @staticmethod
    def cold_user_descriptive_strategy() -> PlannerStrategy:
        """Semantic-search strategy for cold users with a descriptive query."""
        return PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            fallback_tools=["popular_books"],
            reasoning="Descriptive query best served by semantic search",
            profile_data=None,
        )

    @staticmethod
    def cold_user_with_profile_strategy(profile_data: Dict) -> PlannerStrategy:
        """Subject-pool strategy for cold users who have a profile."""
        return PlannerStrategy(
            recommended_tools=["subject_hybrid_pool"],
            fallback_tools=["popular_books"],
            reasoning="Cold user with profile — use favourite subjects",
            profile_data=profile_data,
        )

    @staticmethod
    def custom_strategy(
        recommended_tools: List[str],
        fallback_tools: List[str],
        reasoning: str,
        profile_data: Optional[Dict] = None,
    ) -> PlannerStrategy:
        """Fully customisable strategy for edge-case testing."""
        return PlannerStrategy(
            recommended_tools=recommended_tools,
            fallback_tools=fallback_tools,
            reasoning=reasoning,
            profile_data=profile_data,
        )


# ==============================================================================
# Mock Builders
# ==============================================================================


class MockPlannerBuilder:
    """
    Fluent builder for PlannerAgent mocks.

    PlannerAgent.execute() is a coroutine — must use AsyncMock so the
    orchestrator can await it correctly.
    """

    def __init__(self):
        self._strategy = StrategyFactory.warm_user_strategy()
        self.mock = MagicMock()
        self._configure()

    def returns_strategy(self, strategy: PlannerStrategy) -> "MockPlannerBuilder":
        self._strategy = strategy
        self._configure()
        return self

    def returns_warm_user_strategy(self) -> "MockPlannerBuilder":
        self._strategy = StrategyFactory.warm_user_strategy()
        self._configure()
        return self

    def returns_cold_descriptive_strategy(self) -> "MockPlannerBuilder":
        self._strategy = StrategyFactory.cold_user_descriptive_strategy()
        self._configure()
        return self

    def returns_cold_with_profile_strategy(self, profile_data: Dict) -> "MockPlannerBuilder":
        self._strategy = StrategyFactory.cold_user_with_profile_strategy(profile_data)
        self._configure()
        return self

    def raises_error(self, error: Exception) -> "MockPlannerBuilder":
        self.mock.execute = AsyncMock(side_effect=error)
        return self

    def _configure(self):
        self.mock.execute = AsyncMock(return_value=self._strategy)

    def build(self):
        return self.mock


class MockRetrievalBuilder:
    """
    Fluent builder for RetrievalAgent mocks.

    RetrievalAgent.execute() is a coroutine that receives a RetrievalInput and
    returns a RetrievalOutput. The side_effect is an async function so it can
    inspect the actual input and propagate planner_reasoning / profile_data
    into the ExecutionContext — this makes stage-transition assertions accurate.
    """

    def __init__(self):
        self._candidates: List[Dict] = CandidateFactory.create_batch(60)
        self._tool_names: List[str] = ["als_recs"]
        self.mock = MagicMock()
        self._configure()

    def returns_candidates(
        self,
        candidates: List[Dict],
        tool_name: str = "als_recs",
    ) -> "MockRetrievalBuilder":
        self._candidates = candidates
        self._tool_names = [tool_name]
        self._configure()
        return self

    def returns_batch(self, n: int, source_tool: str = "als_recs") -> "MockRetrievalBuilder":
        self._candidates = CandidateFactory.create_batch(n, source_tool=source_tool)
        self._tool_names = [source_tool]
        self._configure()
        return self

    def returns_empty(self) -> "MockRetrievalBuilder":
        self._candidates = []
        self._tool_names = []
        self._configure()
        return self

    def raises_error(self, error: Exception) -> "MockRetrievalBuilder":
        self.mock.execute = AsyncMock(side_effect=error)
        return self

    def _configure(self):
        candidates = self._candidates
        tool_names = self._tool_names

        async def _execute(retrieval_input):
            return RetrievalOutput(
                candidates=candidates,
                execution_context=ExecutionContext(
                    planner_reasoning=retrieval_input.strategy.reasoning,
                    tools_used=list(tool_names),
                    profile_data=retrieval_input.profile_data,
                ),
                reasoning=f"Retrieved {len(candidates)} candidates",
            )

        self.mock.execute = AsyncMock(side_effect=_execute)

    def build(self):
        return self.mock


class MockSelectionBuilder:
    """
    Fluent builder for SelectionAgent mocks.

    SelectionAgent.execute() returns List[BookRecommendation] — the post-conversion
    type, not raw dicts. This is what curation receives.
    """

    def __init__(self):
        self._selected = CandidateFactory.create_book_recommendations(10)
        self.mock = MagicMock()
        self._configure()

    def returns_batch(self, n: int, start_idx: int = 20000) -> "MockSelectionBuilder":
        self._selected = CandidateFactory.create_book_recommendations(n, start_idx)
        self._configure()
        return self

    def returns_books(self, books: List[BookRecommendation]) -> "MockSelectionBuilder":
        self._selected = books
        self._configure()
        return self

    def returns_empty(self) -> "MockSelectionBuilder":
        self._selected = []
        self._configure()
        return self

    def raises_error(self, error: Exception) -> "MockSelectionBuilder":
        self.mock.execute = AsyncMock(side_effect=error)
        return self

    def _configure(self):
        self.mock.execute = AsyncMock(return_value=self._selected)

    def build(self):
        return self.mock


class MockCurationBuilder:
    """
    Fluent builder for CurationAgent mocks.

    CurationAgent.execute_stream() is an async generator — it cannot be mocked
    with AsyncMock (which produces a coroutine, not an async generator).
    Instead, execute_stream is a MagicMock whose side_effect returns a fresh
    async generator on each call, avoiding the single-iteration exhaustion
    problem that would occur with a stored generator instance.

    The success stream is dynamic: it reads the `candidates` kwarg passed at
    call time and uses those item_idx values as book_ids in the complete chunk,
    so assertions about book_ids reflect what the orchestrator actually passed.
    """

    def __init__(self):
        self._success = True
        self._n_books = 5
        self._error: Optional[Exception] = None
        self.mock = MagicMock()
        self._configure()

    def returns_success_with_books(self, n: int = 5) -> "MockCurationBuilder":
        self._success = True
        self._n_books = n
        self._error = None
        self._configure()
        return self

    def raises_error_on_stream(self, error: Exception) -> "MockCurationBuilder":
        self._success = False
        self._error = error
        self._configure()
        return self

    def _configure(self):
        n_books = self._n_books
        error = self._error

        if error is not None:

            async def _error_stream(*args, **kwargs):
                raise error
                yield  # required to make this an async generator function

            self.mock.execute_stream = MagicMock(side_effect=lambda *a, **kw: _error_stream())

        else:

            async def _success_stream(*args, **kwargs):
                candidates: List[BookRecommendation] = kwargs.get("candidates", [])
                book_ids = [c.item_idx for c in candidates[:n_books]]

                yield StreamChunk(type="status", content="Curating personalized recommendations...")
                yield StreamChunk(
                    type="token",
                    content="Here are some great book recommendations for you.",
                )
                yield StreamChunk(
                    type="complete",
                    data={
                        "target": "recsys",
                        "success": True,
                        "book_ids": book_ids,
                        "tool_calls": [],
                        "policy_version": "recsys.curation.md",
                        "elapsed_ms": 100,
                    },
                )

            self.mock.execute_stream = MagicMock(
                side_effect=lambda *a, **kw: _success_stream(*a, **kw)
            )

    def build(self):
        return self.mock


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture
def mock_planner_builder() -> MockPlannerBuilder:
    """Fresh MockPlannerBuilder per test."""
    return MockPlannerBuilder()


@pytest.fixture
def mock_retrieval_builder() -> MockRetrievalBuilder:
    """Fresh MockRetrievalBuilder per test."""
    return MockRetrievalBuilder()


@pytest.fixture
def mock_selection_builder() -> MockSelectionBuilder:
    """Fresh MockSelectionBuilder per test."""
    return MockSelectionBuilder()


@pytest.fixture
def mock_curation_builder() -> MockCurationBuilder:
    """Fresh MockCurationBuilder per test."""
    return MockCurationBuilder()


@pytest.fixture
def candidate_factory() -> CandidateFactory:
    """CandidateFactory instance (stateless, safe to share)."""
    return CandidateFactory()


@pytest.fixture
def strategy_factory() -> StrategyFactory:
    """StrategyFactory instance (stateless, safe to share)."""
    return StrategyFactory()
