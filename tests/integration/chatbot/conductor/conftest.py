# tests/integration/chatbot/conductor/conftest.py
"""
Fixtures for Conductor integration tests.
Provides mocked agents for testing orchestration logic without LLM calls.
"""

import pytest
from unittest.mock import Mock
from typing import Dict, List, Optional

from app.agents.domain.entities import (
    AgentResponse,
    AgentExecutionState,
    ExecutionStatus,
    ToolExecution,
    BookRecommendation,
)

# Import shared recsys user fixtures
from tests.integration.chatbot.recsys_fixtures import (
    test_user_warm,
    test_user_cold,
    test_user_new,
    test_user_with_profile,
    get_user_rating_count,
)


@pytest.fixture
def mock_recsys_agent():
    """
    Mock RecommendationAgent that returns successful responses with books.

    Returns a mock with a pre-configured execute() method that returns
    a successful AgentResponse with 5 book recommendations.
    """
    agent = Mock()

    # Create realistic book recommendations
    books = [
        BookRecommendation(
            item_idx=12345,
            title="The Three-Body Problem",
            author="Cixin Liu",
            year=2008,
        ),
        BookRecommendation(
            item_idx=12346,
            title="Foundation",
            author="Isaac Asimov",
            year=1951,
        ),
        BookRecommendation(
            item_idx=12347,
            title="Dune",
            author="Frank Herbert",
            year=1965,
        ),
        BookRecommendation(
            item_idx=12348,
            title="Hyperion",
            author="Dan Simmons",
            year=1989,
        ),
        BookRecommendation(
            item_idx=12349,
            title="Neuromancer",
            author="William Gibson",
            year=1984,
        ),
    ]

    # Create execution state with tool calls
    execution_state = AgentExecutionState(
        status=ExecutionStatus.COMPLETED,
        input_text="",
        tool_executions=[
            ToolExecution(
                tool_name="als_recommendations",
                arguments={"user_id": 278859, "n": 60},
                result={"candidates": 60},
                execution_time_ms=150,
            )
        ],
    )
    execution_state.mark_completed()

    agent.execute.return_value = AgentResponse(
        text="Here are 5 great science fiction books I think you'll enjoy based on your reading history.",
        target_category="recsys",
        success=True,
        book_recommendations=books,
        execution_state=execution_state,
        policy_version="recsys.v1",
    )

    return agent


@pytest.fixture
def mock_web_agent():
    """
    Mock WebAgent that returns successful responses with web search results.

    Returns a mock with a pre-configured execute() method that returns
    a successful AgentResponse with web search information.
    """
    agent = Mock()

    # Create execution state with web search tool call
    execution_state = AgentExecutionState(
        status=ExecutionStatus.COMPLETED,
        input_text="",
        tool_executions=[
            ToolExecution(
                tool_name="web_search",
                arguments={"query": "best books 2024"},
                result={"results": 10},
                execution_time_ms=500,
            )
        ],
    )
    execution_state.mark_completed()

    agent.execute.return_value = AgentResponse(
        text="Based on web search results, here's what I found about the best books of 2024...",
        target_category="web",
        success=True,
        citations=[
            {
                "source": "web",
                "ref": "https://example.com/best-books-2024",
                "meta": {"title": "Best Books of 2024"},
            }
        ],
        execution_state=execution_state,
        policy_version="web.v1",
    )

    return agent


@pytest.fixture
def mock_docs_agent():
    """
    Mock DocsAgent that returns successful responses with documentation.

    Returns a mock with a pre-configured execute() method that returns
    a successful AgentResponse with documentation information.
    """
    agent = Mock()

    # Create execution state with docs search tool call
    execution_state = AgentExecutionState(
        status=ExecutionStatus.COMPLETED,
        input_text="",
        tool_executions=[
            ToolExecution(
                tool_name="docs_search",
                arguments={"query": "how to rate books"},
                result={"docs": 3},
                execution_time_ms=100,
            )
        ],
    )
    execution_state.mark_completed()

    agent.execute.return_value = AgentResponse(
        text="To rate a book in our system, you can click the star rating next to any book in your library...",
        target_category="docs",
        success=True,
        citations=[
            {
                "source": "docs",
                "ref": "user-guide/rating-books",
                "meta": {"section": "Rating Books"},
            }
        ],
        execution_state=execution_state,
        policy_version="docs.v1",
    )

    return agent


@pytest.fixture
def mock_response_agent():
    """
    Mock ResponseAgent that returns successful conversational responses.

    Returns a mock with a pre-configured execute() method that returns
    a successful AgentResponse with a generic conversational reply.
    """
    agent = Mock()

    # Create execution state (no tool calls for pure conversation)
    execution_state = AgentExecutionState(
        status=ExecutionStatus.COMPLETED,
        input_text="",
    )
    execution_state.mark_completed()

    agent.execute.return_value = AgentResponse(
        text="Hello! I'm here to help you discover great books. What kind of books are you interested in?",
        target_category="respond",
        success=True,
        execution_state=execution_state,
        policy_version="respond.v1",
    )

    return agent


@pytest.fixture
def mock_agent_factory(
    mock_recsys_agent,
    mock_web_agent,
    mock_docs_agent,
    mock_response_agent,
):
    """
    Mock AgentFactory that returns mocked agents instead of real ones.

    This factory can be injected into Conductor to replace real agents
    with mocks for testing orchestration logic.

    Usage:
        conductor = Conductor()
        conductor.factory = mock_agent_factory
        result = conductor.run(...)  # Uses mocked agents

    Returns:
        Mock factory with create_agent() method that returns appropriate
        mocked agent based on target type
    """
    factory = Mock()

    # Map of agent types to mocked agents
    agent_map = {
        "recsys": mock_recsys_agent,
        "web": mock_web_agent,
        "docs": mock_docs_agent,
        "respond": mock_response_agent,
    }

    def create_agent(target, **kwargs):
        """Return appropriate mocked agent based on target."""
        agent = agent_map.get(target)
        if agent is None:
            raise ValueError(f"Unknown agent type: {target}")
        return agent

    factory.create_agent = create_agent

    return factory


@pytest.fixture
def mock_router():
    """
    Mock RouterLLM for tests that need to control routing decisions.

    By default, routes to recsys. Configure with:
        mock_router.classify.return_value = RoutePlan(target="web", reason="...")

    Returns:
        Mock router with classify() method
    """
    from app.agents.schemas import RoutePlan

    router = Mock()
    router.classify.return_value = RoutePlan(
        target="recsys", reason="Default mock routing to recsys"
    )

    return router
