# tests/unit/chatbot/tools/recsys/test_tool_gating.py
"""
Unit tests for tool availability gating logic in InternalTools.

Tests get_retrieval_tools() and get_context_tools() conditional logic only.
All _create_*_tool() methods are mocked to return lightweight stubs so no
database, FAISS index, or other I/O is involved.
"""

from unittest.mock import MagicMock

import pytest

from app.agents.tools.recsys.native_tools import InternalTools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub(name: str) -> MagicMock:
    """Return a minimal tool stub with only a .name attribute."""
    stub = MagicMock()
    stub.name = name
    return stub


def _tool_names(tools: list) -> list[str]:
    return [t.name for t in tools]


# All _create_*_tool patch targets, keyed by the attribute name they return.
_RETRIEVAL_PATCHES = {
    "_create_semantic_search_tool": "book_semantic_search",
    "_create_subject_id_search_tool": "subject_id_search",
    "_create_popular_books_tool": "popular_books",
    "_create_subject_hybrid_tool": "subject_hybrid_pool",
    "_create_als_recs_tool": "als_recs",
}

_CONTEXT_PATCHES = {
    "_create_user_context_tools": ["user_profile", "recent_interactions"],
}


def _patch_retrieval_tools(instance: InternalTools) -> None:
    """Replace all _create_*_tool methods on instance with stub-returning mocks."""
    for method_name, tool_name in _RETRIEVAL_PATCHES.items():
        setattr(instance, method_name, lambda n=tool_name: _stub(n))


def _patch_context_tools(instance: InternalTools) -> None:
    """Replace _create_user_context_tools on instance with stub-returning mock."""
    names = _CONTEXT_PATCHES["_create_user_context_tools"]
    instance._create_user_context_tools = lambda: [_stub(n) for n in names]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.user_id = 42
    return user


@pytest.fixture
def mock_db():
    return MagicMock()


# ---------------------------------------------------------------------------
# Retrieval tool gating
# ---------------------------------------------------------------------------


class TestRetrievalToolGating:
    """Tests for get_retrieval_tools() conditional inclusion logic."""

    ALWAYS_PRESENT = {"book_semantic_search", "subject_id_search", "popular_books"}

    def test_always_present_tools_require_no_conditions(self):
        """The three core tools appear even with no user, no db, and cold state."""
        tools = InternalTools(current_user=None, db=None)
        _patch_retrieval_tools(tools)

        names = set(_tool_names(tools.get_retrieval_tools(is_warm=False)))

        assert self.ALWAYS_PRESENT.issubset(names)

    def test_always_present_tools_unaffected_by_warm_flag(self, mock_user, mock_db):
        """Warm flag does not remove the three core tools."""
        tools = InternalTools(current_user=mock_user, db=mock_db)
        _patch_retrieval_tools(tools)

        cold_names = set(_tool_names(tools.get_retrieval_tools(is_warm=False)))
        warm_names = set(_tool_names(tools.get_retrieval_tools(is_warm=True)))

        assert self.ALWAYS_PRESENT.issubset(cold_names)
        assert self.ALWAYS_PRESENT.issubset(warm_names)

    def test_subject_hybrid_present_with_db(self, mock_db):
        """subject_hybrid_pool included when db is provided."""
        tools = InternalTools(current_user=None, db=mock_db)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=False))

        assert "subject_hybrid_pool" in names

    def test_subject_hybrid_absent_without_db(self):
        """subject_hybrid_pool excluded when db is None."""
        tools = InternalTools(current_user=None, db=None)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=False))

        assert "subject_hybrid_pool" not in names

    def test_als_recs_present_when_all_conditions_met(self, mock_user, mock_db):
        """als_recs included when is_warm=True, user provided, and db provided."""
        tools = InternalTools(current_user=mock_user, db=mock_db)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=True))

        assert "als_recs" in names

    def test_als_recs_absent_when_cold(self, mock_user, mock_db):
        """als_recs excluded when is_warm=False even with user and db."""
        tools = InternalTools(current_user=mock_user, db=mock_db)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=False))

        assert "als_recs" not in names

    def test_als_recs_absent_without_user(self, mock_db):
        """als_recs excluded when current_user is None even when warm."""
        tools = InternalTools(current_user=None, db=mock_db)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=True))

        assert "als_recs" not in names

    def test_als_recs_absent_without_db(self, mock_user):
        """als_recs excluded when db is None even when warm and authenticated."""
        tools = InternalTools(current_user=mock_user, db=None)
        _patch_retrieval_tools(tools)

        names = _tool_names(tools.get_retrieval_tools(is_warm=True))

        assert "als_recs" not in names


# ---------------------------------------------------------------------------
# Context tool gating
# ---------------------------------------------------------------------------


class TestContextToolGating:
    """Tests for get_context_tools() conditional inclusion logic."""

    def test_context_tools_present_when_all_conditions_met(self, mock_user, mock_db):
        """Both context tools returned when allow_profile=True, user, and db all provided."""
        tools = InternalTools(
            current_user=mock_user,
            db=mock_db,
            allow_profile=True,
        )
        _patch_context_tools(tools)

        names = _tool_names(tools.get_context_tools())

        assert "user_profile" in names
        assert "recent_interactions" in names

    def test_context_tools_absent_when_profile_not_allowed(self, mock_user, mock_db):
        """No context tools returned when allow_profile=False."""
        tools = InternalTools(
            current_user=mock_user,
            db=mock_db,
            allow_profile=False,
        )
        _patch_context_tools(tools)

        assert tools.get_context_tools() == []

    @pytest.mark.parametrize(
        "current_user,db",
        [
            (None, "USE_DB"),  # missing user
            ("USE_USER", None),  # missing db
        ],
    )
    def test_context_tools_absent_when_missing_dependency(
        self, current_user, db, mock_user, mock_db
    ):
        """No context tools returned when any required dependency is missing."""
        actual_user = mock_user if current_user == "USE_USER" else current_user
        actual_db = mock_db if db == "USE_DB" else db

        tools = InternalTools(
            current_user=actual_user,
            db=actual_db,
            allow_profile=True,
        )
        _patch_context_tools(tools)

        assert tools.get_context_tools() == []
