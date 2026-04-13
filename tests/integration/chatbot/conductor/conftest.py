# tests/integration/chatbot/conductor/conftest.py
"""
Fixtures for Conductor integration tests.
Provides mocked agents and a pre-wired Conductor for testing orchestration
logic without any LLM calls.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.schemas import AgentResult, StreamChunk

from tests.integration.chatbot.recsys_fixtures import (  # noqa: F401
    test_user_cold,
    test_user_new,
    test_user_warm,
    test_user_with_profile,
)

# ---------------------------------------------------------------------------
# Async generator mock
# ---------------------------------------------------------------------------


class AsyncGenMock:
    """
    Callable async generator mock that records call arguments.

    Use instead of unittest.mock.AsyncMock when the callable must behave
    as an async generator (i.e. ``async for chunk in agent.execute_stream(req)``).

    Exposes the same inspection attributes as Mock:
        .called       – True after first invocation
        .call_count   – number of times called
        .call_args    – ((positional_args,), {keyword_args}) of the last call,
                        so ``mock.call_args[0][0]`` returns the first positional arg
        .reset_mock() – resets all tracking state
    """

    def __init__(self, chunks: list):
        self._chunks = chunks
        self.called = False
        self.call_count = 0
        self.call_args = None

    async def __call__(self, *args, **kwargs):
        self.called = True
        self.call_count += 1
        # Mirror Mock's call_args interface: (positional_tuple, kwargs_dict)
        self.call_args = (args, kwargs)
        for chunk in self._chunks:
            yield chunk

    def reset_mock(self):
        self.called = False
        self.call_count = 0
        self.call_args = None


# ---------------------------------------------------------------------------
# collect_result fixture
# ---------------------------------------------------------------------------

# Targets that are valid for AgentResult; "error" is used by conductor on failure.
_VALID_TARGETS = {"web", "docs", "recsys", "respond"}


@pytest.fixture
def collect_result():
    """
    Drive conductor.run_stream() to completion and return an AgentResult.

    Collects all StreamChunk objects, locates the final 'complete' chunk,
    and reconstructs an AgentResult from its data dict.  Pydantic coerces
    nested dicts (tool_calls, citations) automatically.

    Usage::

        async def test_something(self, collect_result, ...):
            result = await collect_result(conductor, history=[], user_text="...", ...)
            assert result.success
    """

    async def _collect(conductor, **kwargs) -> AgentResult:
        chunks = []
        async for chunk in conductor.run_stream(**kwargs):
            chunks.append(chunk)

        complete = next((c for c in chunks if c.type == "complete"), None)
        if complete is None:
            types_seen = [c.type for c in chunks]
            raise AssertionError(
                f"Stream ended without a 'complete' chunk. Chunk types seen: {types_seen}"
            )

        data = dict(complete.data or {})

        # Conductor uses target="error" for failure paths; normalise to a valid literal.
        if data.get("target") not in _VALID_TARGETS:
            data["target"] = "respond"

        return AgentResult(**data)

    return _collect


# ---------------------------------------------------------------------------
# Mock agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_recsys_agent():
    """
    Mock RecommendationAgent with execute_stream returning 5 book recommendations.
    """
    agent = Mock()

    complete_chunk = StreamChunk(
        type="complete",
        data={
            "target": "recsys",
            "text": (
                "Here are 5 great science fiction books I think you'll enjoy "
                "based on your reading history."
            ),
            "success": True,
            "book_ids": [12345, 12346, 12347, 12348, 12349],
            "tool_calls": [
                {
                    "name": "als_recommendations",
                    "args": {"user_id": 278859, "n": 60},
                    "ok": True,
                    "elapsed_ms": 150,
                }
            ],
            "policy_version": "recsys.v1",
        },
    )

    agent.execute_stream = AsyncGenMock([complete_chunk])
    return agent


@pytest.fixture
def mock_web_agent():
    """
    Mock WebAgent with execute_stream returning a web-search response.
    """
    agent = Mock()

    complete_chunk = StreamChunk(
        type="complete",
        data={
            "target": "web",
            "text": (
                "Based on web search results, here's what I found about the best books of 2024..."
            ),
            "success": True,
            "citations": [
                {
                    "source": "web",
                    "ref": "https://example.com/best-books-2024",
                    "meta": {"title": "Best Books of 2024"},
                }
            ],
            "tool_calls": [
                {
                    "name": "web_search",
                    "args": {"query": "best books 2024"},
                    "ok": True,
                    "elapsed_ms": 500,
                }
            ],
            "policy_version": "web.v1",
        },
    )

    agent.execute_stream = AsyncGenMock([complete_chunk])
    return agent


@pytest.fixture
def mock_docs_agent():
    """
    Mock DocsAgent with execute_stream returning a documentation response.
    """
    agent = Mock()

    complete_chunk = StreamChunk(
        type="complete",
        data={
            "target": "docs",
            "text": (
                "To rate a book in our system, you can click the star rating "
                "next to any book in your library..."
            ),
            "success": True,
            "citations": [
                {
                    "source": "docs",
                    "ref": "user-guide/rating-books",
                    "meta": {"section": "Rating Books"},
                }
            ],
            "tool_calls": [
                {
                    "name": "docs_search",
                    "args": {"query": "how to rate books"},
                    "ok": True,
                    "elapsed_ms": 100,
                }
            ],
            "policy_version": "docs.v1",
        },
    )

    agent.execute_stream = AsyncGenMock([complete_chunk])
    return agent


@pytest.fixture
def mock_response_agent():
    """
    Mock ResponseAgent with execute_stream returning a conversational reply.
    """
    agent = Mock()

    complete_chunk = StreamChunk(
        type="complete",
        data={
            "target": "respond",
            "text": (
                "Hello! I'm here to help you discover great books. "
                "What kind of books are you interested in?"
            ),
            "success": True,
            "policy_version": "respond.v1",
        },
    )

    agent.execute_stream = AsyncGenMock([complete_chunk])
    return agent


# ---------------------------------------------------------------------------
# Factory and router fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_factory(
    mock_recsys_agent,
    mock_web_agent,
    mock_docs_agent,
    mock_response_agent,
):
    """
    Mock AgentFactory that returns mocked agents instead of real ones.
    """
    factory = Mock()

    agent_map = {
        "recsys": mock_recsys_agent,
        "web": mock_web_agent,
        "docs": mock_docs_agent,
        "respond": mock_response_agent,
    }

    def create_agent(target, **kwargs):
        agent = agent_map.get(target)
        if agent is None:
            raise ValueError(f"Unknown agent type: {target}")
        return agent

    factory.create_agent = create_agent
    return factory


@pytest.fixture
def mock_router():
    """
    Mock RouterLLM for controlling routing decisions without LLM calls.

    Defaults to routing to recsys.  Override per-test with::

        mock_router.classify.return_value = RoutePlan(target="web", reason="...")

    Because conductor awaits classify(), this uses AsyncMock.
    """
    from app.agents.schemas import RoutePlan

    router = Mock()
    router.classify = AsyncMock(
        return_value=RoutePlan(target="recsys", reason="Default mock routing to recsys")
    )
    return router


@pytest.fixture
def conductor(mock_agent_factory, mock_router):
    """
    Pre-wired Conductor with both the factory and router mocked.

    Eliminates the need to manually inject mocks in every test, and
    ensures no real LLM calls are ever made during infrastructure tests.

    Usage::

        async def test_something(self, conductor, collect_result, ...):
            result = await collect_result(conductor, ...)

    To control routing for a specific test, configure mock_router before
    calling collect_result::

        mock_router.classify.return_value = RoutePlan(target="docs", reason="...")
        result = await collect_result(conductor, ...)
    """
    from app.agents.orchestrator.conductor import Conductor

    c = Conductor()
    c.factory = mock_agent_factory
    c.router = mock_router
    return c
