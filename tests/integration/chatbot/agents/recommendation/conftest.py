# tests/integration/chatbot/agents/recommendation/conftest.py
"""
Fixtures for RecommendationAgent pipeline integration tests.
Provides mock builders for sub-agents and factories for test data generation.
"""

import pytest
from unittest.mock import Mock
from typing import List, Optional, Dict, Any

from app.agents.domain.entities import (
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
    ToolExecution,
    BookRecommendation,
)
from app.agents.domain.recsys_schemas import (
    PlannerStrategy,
    ExecutionContext,
    RetrievalOutput,
)

# Import shared recsys user fixtures from parent
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
    """Factory for generating test book candidates."""

    @staticmethod
    def create_batch(
        n: int,
        start_idx: int = 10000,
        source_tool: str = "als_recommendations",
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of book candidates as dictionaries.

        Args:
            n: Number of candidates to generate
            start_idx: Starting book ID (increments from here)
            source_tool: Tool that generated these candidates

        Returns:
            List of candidate dictionaries (NOT BookRecommendation objects)
            These will be converted to BookRecommendation by the orchestrator
        """
        candidates = []
        for i in range(n):
            candidates.append(
                {
                    "item_idx": start_idx + i,
                    "title": f"Test Book {start_idx + i}",
                    "author": f"Author {i % 10}",
                    "year": 1990 + (i % 30),
                    "tool_source": source_tool,
                }
            )
        return candidates

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
        Create a candidate with full metadata as a dictionary.

        Used for testing metadata preservation through pipeline.
        """
        candidate = {
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


class StrategyFactory:
    """Factory for generating test PlannerStrategy objects."""

    @staticmethod
    def warm_user_strategy(
        profile_data: Optional[Dict] = None,
    ) -> PlannerStrategy:
        """Strategy for warm user (ALS available)."""
        return PlannerStrategy(
            recommended_tools=["als_recommendations"],
            fallback_tools=["popular_books"],
            reasoning="User has sufficient rating history for collaborative filtering",
            profile_data=profile_data,
        )

    @staticmethod
    def cold_user_descriptive_strategy() -> PlannerStrategy:
        """Strategy for cold user with descriptive query."""
        return PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            fallback_tools=["popular_books"],
            reasoning="Descriptive query best served by semantic search",
            profile_data=None,
        )

    @staticmethod
    def cold_user_with_profile_strategy(
        profile_data: Dict,
    ) -> PlannerStrategy:
        """Strategy for cold user with profile."""
        return PlannerStrategy(
            recommended_tools=["subject_hybrid_pool"],
            fallback_tools=["popular_books"],
            reasoning="User has profile data, use subject-based recommendations",
            profile_data=profile_data,
        )

    @staticmethod
    def custom_strategy(
        recommended_tools: List[str],
        fallback_tools: List[str],
        reasoning: str,
        profile_data: Optional[Dict] = None,
    ) -> PlannerStrategy:
        """Create custom strategy for edge case testing."""
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
    Builder for configuring mock PlannerAgent behavior.

    Provides fluent API for setting up planner responses in tests.
    """

    def __init__(self):
        self.mock = Mock()
        # Default: warm user strategy
        self._strategy = StrategyFactory.warm_user_strategy()
        self._configure_default()

    def returns_strategy(self, strategy: PlannerStrategy):
        """Configure planner to return specific strategy."""
        self._strategy = strategy
        self._configure_default()
        return self

    def returns_warm_user_strategy(self):
        """Configure planner for warm user scenario."""
        self._strategy = StrategyFactory.warm_user_strategy()
        self._configure_default()
        return self

    def returns_cold_descriptive_strategy(self):
        """Configure planner for cold user with descriptive query."""
        self._strategy = StrategyFactory.cold_user_descriptive_strategy()
        self._configure_default()
        return self

    def returns_cold_with_profile_strategy(self, profile_data: Dict):
        """Configure planner for cold user with profile."""
        self._strategy = StrategyFactory.cold_user_with_profile_strategy(profile_data)
        self._configure_default()
        return self

    def raises_error(self, error: Exception):
        """Configure planner to raise error."""
        self.mock.execute.side_effect = error
        return self

    def _configure_default(self):
        """Set default return value."""
        self.mock.execute.return_value = self._strategy

    def build(self):
        """Return configured mock."""
        return self.mock


class MockRetrievalBuilder:
    """
    Builder for configuring mock RetrievalAgent behavior.

    Provides fluent API for setting up retrieval responses in tests.
    """

    def __init__(self):
        self.mock = Mock()
        # Default: 60 candidates
        self._candidates = CandidateFactory.create_batch(60)
        self._tool_executions = [
            ToolExecution(
                tool_name="als_recommendations",
                arguments={"user_id": 278859, "n": 60},
                result={"candidates": 60},
                execution_time_ms=150,
            )
        ]
        self._configure_default()

    def returns_candidates(
        self,
        candidates: List[BookRecommendation],
        tool_name: str = "als_recommendations",
    ):
        """Configure retrieval to return specific candidates."""
        self._candidates = candidates
        self._tool_executions = [
            ToolExecution(
                tool_name=tool_name,
                arguments={},
                result={"candidates": len(candidates)},
                execution_time_ms=100,
            )
        ]
        self._configure_default()
        return self

    def returns_empty(self):
        """Configure retrieval to return no candidates (failure scenario)."""
        self._candidates = []
        self._tool_executions = []
        self._configure_default()
        return self

    def returns_batch(self, n: int, source_tool: str = "als_recommendations"):
        """Configure retrieval to return n candidates from specified tool."""
        self._candidates = CandidateFactory.create_batch(n, source_tool=source_tool)
        self._tool_executions = [
            ToolExecution(
                tool_name=source_tool,
                arguments={},
                result={"candidates": n},
                execution_time_ms=100,
            )
        ]
        self._configure_default()
        return self

    def raises_error(self, error: Exception):
        """Configure retrieval to raise error."""
        self.mock.execute.side_effect = error
        return self

    def _configure_default(self):
        """Set default return value as RetrievalOutput object."""
        # Create ExecutionContext
        execution_context = ExecutionContext(
            planner_reasoning="Test planner reasoning",
            tools_used=[te.tool_name for te in self._tool_executions],
            profile_data=None,
        )

        # Create RetrievalOutput object (not tuple!)
        output = RetrievalOutput(
            candidates=self._candidates,
            execution_context=execution_context,
            reasoning=f"Retrieved {len(self._candidates)} candidates using {len(self._tool_executions)} tools",
        )

        self.mock.execute.return_value = output

    def build(self):
        """Return configured mock."""
        return self.mock


class MockCurationBuilder:
    """
    Builder for configuring mock CurationAgent behavior.

    Provides fluent API for setting up curation responses in tests.
    """

    def __init__(self):
        self.mock = Mock()
        # Default: successful response with 5 books
        self._response = self._default_response()
        self._configure_default()

    def returns_response(self, response: AgentResponse):
        """Configure curation to return specific response."""
        self._response = response
        self._configure_default()
        return self

    def returns_success_with_books(self, n: int = 5):
        """Configure curation to return successful response with n books."""
        # Create candidate dictionaries
        candidate_dicts = CandidateFactory.create_batch(n)

        # Convert to BookRecommendation objects (what curation actually returns)
        books = [
            BookRecommendation(
                item_idx=c["item_idx"],
                title=c["title"],
                author=c["author"],
                year=c["year"],
            )
            for c in candidate_dicts
        ]

        execution_state = AgentExecutionState(
            status=ExecutionStatus.COMPLETED,
            input_text="",
        )
        execution_state.mark_completed()

        self._response = AgentResponse(
            text=f"Here are {n} great book recommendations based on your reading history.",
            target_category="recsys",
            success=True,
            book_recommendations=books,
            execution_state=execution_state,
            policy_version="recsys.curation.v1",
        )
        self._configure_default()
        return self

    def raises_error(self, error: Exception):
        """Configure curation to raise error."""
        self.mock.execute.side_effect = error
        return self

    def _default_response(self) -> AgentResponse:
        """Create default successful response."""
        # Create candidate dictionaries
        candidate_dicts = CandidateFactory.create_batch(5)

        # Convert to BookRecommendation objects (what curation actually returns)
        books = [
            BookRecommendation(
                item_idx=c["item_idx"],
                title=c["title"],
                author=c["author"],
                year=c["year"],
            )
            for c in candidate_dicts
        ]

        execution_state = AgentExecutionState(
            status=ExecutionStatus.COMPLETED,
            input_text="",
        )
        execution_state.mark_completed()

        return AgentResponse(
            text="Here are 5 great book recommendations based on your reading history.",
            target_category="recsys",
            success=True,
            book_recommendations=books,
            execution_state=execution_state,
            policy_version="recsys.curation.v1",
        )

    def _configure_default(self):
        """Set default return value."""
        self.mock.execute.return_value = self._response

    def build(self):
        """Return configured mock."""
        return self.mock


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture
def mock_planner_builder():
    """
    Provides MockPlannerBuilder for configuring planner behavior.

    Usage:
        def test_something(mock_planner_builder):
            mock_planner = mock_planner_builder.returns_warm_user_strategy().build()
    """
    return MockPlannerBuilder()


@pytest.fixture
def mock_retrieval_builder():
    """
    Provides MockRetrievalBuilder for configuring retrieval behavior.

    Usage:
        def test_something(mock_retrieval_builder):
            mock_retrieval = mock_retrieval_builder.returns_batch(60).build()
    """
    return MockRetrievalBuilder()


@pytest.fixture
def mock_curation_builder():
    """
    Provides MockCurationBuilder for configuring curation behavior.

    Usage:
        def test_something(mock_curation_builder):
            mock_curation = mock_curation_builder.returns_success_with_books(5).build()
    """
    return MockCurationBuilder()


@pytest.fixture
def candidate_factory():
    """
    Provides CandidateFactory for generating test candidates.

    Usage:
        def test_something(candidate_factory):
            candidates = candidate_factory.create_batch(60)
    """
    return CandidateFactory()


@pytest.fixture
def strategy_factory():
    """
    Provides StrategyFactory for generating test strategies.

    Usage:
        def test_something(strategy_factory):
            strategy = strategy_factory.warm_user_strategy()
    """
    return StrategyFactory()
