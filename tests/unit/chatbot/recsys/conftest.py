# tests/unit/chatbot/recsys/conftest.py
"""
Shared fixtures for recsys unit tests.

Provides lightweight agent instances with all external dependencies
(LLM, prompt loading, logging) patched out so tests run without
network calls, API keys, or filesystem access.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.agents.domain.entities import BookRecommendation

# ==============================================================================
# Agent fixtures
# ==============================================================================


@pytest.fixture
def curation_agent():
    """
    CurationAgent with constructor dependencies patched.

    Patches read_prompt, get_llm, create_react_agent, and append_chatbot_log
    so the instance can be constructed and its pure methods called freely.
    """
    with (
        patch("app.agents.infrastructure.recsys.curation_agent.read_prompt", return_value="prompt"),
        patch("app.agents.infrastructure.base_langgraph_agent.get_llm", return_value=MagicMock()),
        patch(
            "app.agents.infrastructure.base_langgraph_agent.create_react_agent",
            return_value=MagicMock(),
        ),
        patch("app.agents.infrastructure.recsys.curation_agent.append_chatbot_log"),
    ):
        from app.agents.infrastructure.recsys.curation_agent import CurationAgent

        return CurationAgent()


@pytest.fixture
def planner_agent():
    """
    PlannerAgent with constructor dependencies patched.

    Patches get_llm and append_chatbot_log so the instance can be
    constructed and its pure methods (_parse_strategy_response,
    _build_prompt) called freely.
    """
    with (
        patch("app.agents.infrastructure.recsys.planner_agent.get_llm", return_value=MagicMock()),
        patch(
            "app.agents.infrastructure.recsys.planner_agent.read_prompt",
            return_value="[system prompt]",
        ),
        patch("app.agents.infrastructure.recsys.planner_agent.append_chatbot_log"),
    ):
        from app.agents.infrastructure.recsys.planner_agent import PlannerAgent

        return PlannerAgent(
            user_num_ratings=5,
            has_als_recs_available=False,
            allow_profile=False,
        )


@pytest.fixture
def orchestrator(mock_sub_agents):
    """
    RecommendationAgent with all four sub-agents injected as MagicMocks
    and logging patched, for testing orchestrator-level pure methods.
    """
    planner, retrieval, selection, curation = mock_sub_agents
    with patch("app.agents.infrastructure.recsys.orchestrator.append_chatbot_log"):
        from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent

        return RecommendationAgent(
            user_num_ratings=15,
            planner_agent=planner,
            retrieval_agent=retrieval,
            selection_agent=selection,
            curation_agent=curation,
        )


@pytest.fixture
def cold_orchestrator(mock_sub_agents):
    """RecommendationAgent configured as a cold user (no ALS)."""
    planner, retrieval, selection, curation = mock_sub_agents
    with patch("app.agents.infrastructure.recsys.orchestrator.append_chatbot_log"):
        from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent

        return RecommendationAgent(
            user_num_ratings=3,
            planner_agent=planner,
            retrieval_agent=retrieval,
            selection_agent=selection,
            curation_agent=curation,
        )


@pytest.fixture
def mock_sub_agents():
    """Four MagicMock sub-agents for orchestrator injection."""
    return MagicMock(), MagicMock(), MagicMock(), MagicMock()


# ==============================================================================
# BookRecommendation factories
# ==============================================================================


@pytest.fixture
def make_book():
    """
    Factory function that builds a BookRecommendation from keyword arguments.

    Usage:
        book = make_book(item_idx=1001, title="Dune", author="Herbert")
    """

    def _make(
        item_idx: int,
        title: str = "Test Book",
        author: str = "Test Author",
        year: int = 2000,
        **kwargs,
    ) -> BookRecommendation:
        return BookRecommendation(
            item_idx=item_idx, title=title, author=author, year=year, **kwargs
        )

    return _make


@pytest.fixture
def book_pool(make_book):
    """
    Five distinct BookRecommendation objects for use in ordering / filtering tests.
    item_idx values: 1001–1005.
    """
    return [
        make_book(item_idx=1000 + i, title=f"Book {i}", author=f"Author {i}") for i in range(1, 6)
    ]
