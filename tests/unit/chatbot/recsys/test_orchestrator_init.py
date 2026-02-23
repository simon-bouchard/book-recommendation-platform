# tests/unit/chatbot/recsys/test_orchestrator_init.py
"""
Unit tests for RecommendationAgent init logic and pure helper methods.

Targets:
    __init__: warm/cold threshold, _has_als_recs, available tools list
    _build_fallback_strategy()           → PlannerStrategy
    _build_curation_fallback_text()      → str
    _no_candidates_complete_chunk()      → StreamChunk (terminal error)
"""

import pytest
from unittest.mock import MagicMock, patch

from app.agents.domain.entities import BookRecommendation
from app.agents.domain.recsys_schemas import PlannerStrategy
from app.agents.schemas import StreamChunk


# ==============================================================================
# Helpers
# ==============================================================================


def _make_orchestrator(num_ratings, **kwargs):
    """Build a RecommendationAgent with all sub-agents mocked."""
    with patch("app.agents.infrastructure.recsys.orchestrator.append_chatbot_log"):
        from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent

        return RecommendationAgent(
            user_num_ratings=num_ratings,
            planner_agent=MagicMock(),
            retrieval_agent=MagicMock(),
            selection_agent=MagicMock(),
            curation_agent=MagicMock(),
            **kwargs,
        )


def _make_books(n: int, start_idx: int = 1000) -> list[BookRecommendation]:
    return [
        BookRecommendation(item_idx=start_idx + i, title=f"Book {i}", author=f"Author {i}")
        for i in range(n)
    ]


# ==============================================================================
# Init — warm/cold threshold
# ==============================================================================


class TestOrchestratorInit:
    """Verify warm/cold determination and ALS availability at construction time."""

    def test_warm_user_has_als(self):
        agent = _make_orchestrator(num_ratings=10, warm_threshold=10)
        assert agent._has_als_recs is True

    def test_user_one_below_threshold_is_cold(self):
        agent = _make_orchestrator(num_ratings=9, warm_threshold=10)
        assert agent._has_als_recs is False

    def test_user_well_above_threshold_is_warm(self):
        agent = _make_orchestrator(num_ratings=100)
        assert agent._has_als_recs is True

    def test_zero_ratings_is_cold(self):
        agent = _make_orchestrator(num_ratings=0)
        assert agent._has_als_recs is False

    def test_none_ratings_defaults_to_zero_cold(self):
        agent = _make_orchestrator(num_ratings=None)
        assert agent._user_num_ratings == 0
        assert agent._has_als_recs is False

    def test_custom_warm_threshold_respected(self):
        agent = _make_orchestrator(num_ratings=5, warm_threshold=5)
        assert agent._has_als_recs is True

    def test_custom_warm_threshold_one_below_is_cold(self):
        agent = _make_orchestrator(num_ratings=4, warm_threshold=5)
        assert agent._has_als_recs is False

    def test_allow_profile_stored(self):
        agent = _make_orchestrator(num_ratings=15, allow_profile=True)
        assert agent._allow_profile is True

    def test_allow_profile_defaults_false(self):
        agent = _make_orchestrator(num_ratings=15)
        assert agent._allow_profile is False

    def test_injected_sub_agents_are_used(self):
        mock_planner = MagicMock()
        with patch("app.agents.infrastructure.recsys.orchestrator.append_chatbot_log"):
            from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent

            agent = RecommendationAgent(
                user_num_ratings=15,
                planner_agent=mock_planner,
                retrieval_agent=MagicMock(),
                selection_agent=MagicMock(),
                curation_agent=MagicMock(),
            )
        assert agent.planner_agent is mock_planner


# ==============================================================================
# _build_fallback_strategy
# ==============================================================================


class TestBuildFallbackStrategy:
    """
    _build_fallback_strategy returns a hardcoded PlannerStrategy when
    PlannerAgent fails. Warm users get ALS, cold users get popular_books.
    """

    def test_warm_user_recommends_als(self):
        agent = _make_orchestrator(num_ratings=15)
        strategy = agent._build_fallback_strategy()
        assert strategy.recommended_tools == ["als_recs"]

    def test_warm_user_fallback_is_popular_books(self):
        agent = _make_orchestrator(num_ratings=15)
        strategy = agent._build_fallback_strategy()
        assert strategy.fallback_tools == ["popular_books"]

    def test_cold_user_recommends_popular_books(self):
        agent = _make_orchestrator(num_ratings=3)
        strategy = agent._build_fallback_strategy()
        assert strategy.recommended_tools == ["popular_books"]

    def test_cold_user_fallback_is_semantic_search(self):
        agent = _make_orchestrator(num_ratings=3)
        strategy = agent._build_fallback_strategy()
        assert "book_semantic_search" in strategy.fallback_tools

    def test_returns_planner_strategy_type(self):
        agent = _make_orchestrator(num_ratings=15)
        assert isinstance(agent._build_fallback_strategy(), PlannerStrategy)

    def test_profile_data_is_none(self):
        """Fallback strategy never includes profile data — keep it simple and fast."""
        agent = _make_orchestrator(num_ratings=15, allow_profile=True)
        strategy = agent._build_fallback_strategy()
        assert strategy.profile_data is None

    def test_reasoning_is_non_empty(self):
        agent = _make_orchestrator(num_ratings=15)
        assert len(agent._build_fallback_strategy().reasoning) > 0

    def test_exactly_at_threshold_is_warm(self):
        agent = _make_orchestrator(num_ratings=10, warm_threshold=10)
        strategy = agent._build_fallback_strategy()
        assert strategy.recommended_tools == ["als_recs"]


# ==============================================================================
# _build_curation_fallback_text
# ==============================================================================


class TestBuildCurationFallbackText:
    """
    _build_curation_fallback_text generates markdown prose when CurationAgent fails.

    The output must:
    - Start with a header line.
    - Include each book as "[Title](item_idx) by Author".
    - Use the [Title](item_idx) citation format so the frontend can parse book_ids.
    """

    def test_output_starts_with_header(self):
        agent = _make_orchestrator(num_ratings=15)
        books = _make_books(3)
        text = agent._build_curation_fallback_text(books)
        assert "Here are some book recommendations" in text

    def test_contains_all_titles(self):
        agent = _make_orchestrator(num_ratings=15)
        books = _make_books(3)
        text = agent._build_curation_fallback_text(books)
        for book in books:
            assert book.title in text

    def test_contains_all_authors(self):
        agent = _make_orchestrator(num_ratings=15)
        books = _make_books(3)
        text = agent._build_curation_fallback_text(books)
        for book in books:
            assert book.author in text

    def test_uses_citation_format(self):
        """[Title](item_idx) format must be present so frontend can extract IDs."""
        import re

        agent = _make_orchestrator(num_ratings=15)
        books = [BookRecommendation(item_idx=9001, title="Dune", author="Herbert")]
        text = agent._build_curation_fallback_text(books)
        assert re.search(r"\[Dune\]\(9001\)", text)

    def test_empty_candidate_list(self):
        agent = _make_orchestrator(num_ratings=15)
        text = agent._build_curation_fallback_text([])
        assert isinstance(text, str)

    def test_book_without_author_does_not_crash(self):
        agent = _make_orchestrator(num_ratings=15)
        book = BookRecommendation(item_idx=1, title="No Author", author=None)
        text = agent._build_curation_fallback_text([book])
        assert "No Author" in text

    def test_returns_string(self):
        agent = _make_orchestrator(num_ratings=15)
        result = agent._build_curation_fallback_text(_make_books(1))
        assert isinstance(result, str)


# ==============================================================================
# _no_candidates_complete_chunk
# ==============================================================================


class TestNoCandidatesCompleteChunk:
    """
    _no_candidates_complete_chunk builds the terminal StreamChunk emitted
    when retrieval returns zero candidates.

    The catalog-limit hint (mentioning 2004) is triggered by:
    - Query containing a year in range 2004–2025.
    - Query containing the words 'recent', 'new', or 'latest'.

    Generic queries get a "try broader terms" message with no year mentioned.
    """

    def test_success_is_false(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("any query", start_time=0.0)
        assert chunk.data["success"] is False

    def test_book_ids_is_empty(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("any query", start_time=0.0)
        assert chunk.data["book_ids"] == []

    def test_returns_stream_chunk(self):
        agent = _make_orchestrator(num_ratings=15)
        result = agent._no_candidates_complete_chunk("anything", start_time=0.0)
        assert isinstance(result, StreamChunk)

    def test_generic_query_no_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("obscure xyz topic", start_time=0.0)
        assert "2004" not in chunk.data["text"]

    def test_year_2022_triggers_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("books from 2022", start_time=0.0)
        assert "2004" in chunk.data["text"]

    def test_year_2004_triggers_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("books published in 2004", start_time=0.0)
        assert "2004" in chunk.data["text"]

    def test_year_2003_does_not_trigger_catalog_hint(self):
        """2003 is before the cutoff window — no catalog hint."""
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("books from 2003", start_time=0.0)
        assert "2004" not in chunk.data["text"]

    def test_keyword_recent_triggers_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("recent sci-fi novels", start_time=0.0)
        assert "2004" in chunk.data["text"]

    def test_keyword_new_triggers_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("new fantasy releases", start_time=0.0)
        assert "2004" in chunk.data["text"]

    def test_keyword_latest_triggers_catalog_hint(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("latest books by Pratchett", start_time=0.0)
        assert "2004" in chunk.data["text"]

    def test_text_field_is_non_empty_for_generic_query(self):
        agent = _make_orchestrator(num_ratings=15)
        chunk = agent._no_candidates_complete_chunk("xyz123", start_time=0.0)
        assert len(chunk.data["text"]) > 0

    def test_elapsed_ms_is_present(self):
        import time

        agent = _make_orchestrator(num_ratings=15)
        start = time.time()
        chunk = agent._no_candidates_complete_chunk("query", start_time=start)
        assert "elapsed_ms" in chunk.data
        assert isinstance(chunk.data["elapsed_ms"], int)
