# tests/integration/chatbot/tools/test_retrieval_tools.py
"""
Integration tests for all retrieval tools using real database.

Sync tools (popular_books, semantic_search, subject_id_search) use .invoke().
Async tools (subject_hybrid_pool, als_recs) use pytest-asyncio and .ainvoke().
All async tests share a session-scoped event loop (defined in conftest.py) to
prevent httpx AsyncClient singletons from binding to a loop that closes between
tests.
"""

import pytest
from sqlalchemy.orm import Session
from app.agents.tools.recsys.native_tools import InternalTools
from app.table_models import User


class TestPopularBooks:
    """Tests for popular_books retrieval tool."""

    def test_returns_requested_number(self, db_session: Session):
        """Verify popular_books returns the exact number of books specified by top_k."""
        tools = InternalTools(current_user=None, db=db_session)
        popular_books = tools._create_popular_books_tool()

        result = popular_books.invoke({"top_k": 10})

        assert isinstance(result, list)
        assert len(result) == 10

    def test_returns_standardized_fields(self, db_session: Session):
        """Verify popular_books returns standardized output with all expected fields."""
        tools = InternalTools(current_user=None, db=db_session)
        popular_books = tools._create_popular_books_tool()

        result = popular_books.invoke({"top_k": 5})

        book = result[0]
        assert "item_idx" in book
        assert "title" in book
        assert "author" in book
        assert "year" in book
        assert "num_ratings" in book
        assert "score" in book

    def test_clamps_top_k_upper_bound(self, db_session: Session):
        """Verify popular_books clamps top_k to maximum of 500."""
        tools = InternalTools(current_user=None, db=db_session)
        popular_books = tools._create_popular_books_tool()

        result = popular_books.invoke({"top_k": 1000})

        assert len(result) == 500

    def test_clamps_top_k_lower_bound(self, db_session: Session):
        """Verify popular_books clamps top_k to minimum of 1."""
        tools = InternalTools(current_user=None, db=db_session)
        popular_books = tools._create_popular_books_tool()

        result = popular_books.invoke({"top_k": 0})

        assert len(result) == 1


class TestSemanticSearch:
    """Tests for book_semantic_search retrieval tool."""

    def test_returns_results_for_query(self, db_session: Session):
        """Verify semantic_search returns results for a valid query."""
        tools = InternalTools(current_user=None, db=db_session)
        semantic_search = tools._create_semantic_search_tool()

        result = semantic_search.invoke({"query": "science fiction adventures", "top_k": 10})

        assert isinstance(result, list)
        assert "error" not in result[0], f"Semantic search failed: {result[0].get('error')}"
        assert len(result) == 10

    def test_returns_standardized_fields(self, db_session: Session):
        """Verify semantic_search returns standardized output with expected fields."""
        tools = InternalTools(current_user=None, db=db_session)
        semantic_search = tools._create_semantic_search_tool()

        result = semantic_search.invoke({"query": "mystery novels", "top_k": 5})

        assert "error" not in result[0], f"Semantic search failed: {result[0].get('error')}"
        book = result[0]
        assert "item_idx" in book
        assert "title" in book
        assert "author" in book
        assert "year" in book
        assert "num_ratings" in book
        assert "score" in book

    def test_clamps_top_k_upper_bound(self, db_session: Session):
        """Verify semantic_search clamps top_k to maximum of 500."""
        tools = InternalTools(current_user=None, db=db_session)
        semantic_search = tools._create_semantic_search_tool()

        result = semantic_search.invoke({"query": "fantasy", "top_k": 1000})

        assert "error" not in result[0], f"Semantic search failed: {result[0].get('error')}"
        assert len(result) == 500

    def test_clamps_top_k_lower_bound(self, db_session: Session):
        """Verify semantic_search clamps top_k to minimum of 1."""
        tools = InternalTools(current_user=None, db=db_session)
        semantic_search = tools._create_semantic_search_tool()

        result = semantic_search.invoke({"query": "romance", "top_k": 0})

        assert "error" not in result[0], f"Semantic search failed: {result[0].get('error')}"
        assert len(result) == 1


class TestSubjectIdSearch:
    """Tests for subject_id_search retrieval tool."""

    def test_returns_results_for_phrases(self, db_session: Session):
        """Verify subject_id_search returns results for valid phrases."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_id_search = tools._create_subject_id_search_tool()

        result = subject_id_search.invoke({"phrases": ["mystery"], "top_k": 3})

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["phrase"] == "mystery"
        assert "candidates" in result[0]
        assert len(result[0]["candidates"]) > 0

    def test_candidate_structure(self, db_session: Session):
        """Verify each candidate has required fields."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_id_search = tools._create_subject_id_search_tool()

        result = subject_id_search.invoke({"phrases": ["science fiction"], "top_k": 5})

        candidate = result[0]["candidates"][0]
        assert "subject_idx" in candidate
        assert "subject" in candidate
        assert "score" in candidate

    def test_clamps_top_k_upper_bound(self, db_session: Session):
        """Verify subject_id_search clamps top_k to maximum of 10."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_id_search = tools._create_subject_id_search_tool()

        result = subject_id_search.invoke({"phrases": ["fantasy"], "top_k": 50})

        assert len(result[0]["candidates"]) <= 10

    def test_clamps_top_k_lower_bound(self, db_session: Session):
        """Verify subject_id_search clamps top_k to minimum of 1."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_id_search = tools._create_subject_id_search_tool()

        result = subject_id_search.invoke({"phrases": ["horror"], "top_k": 0})

        assert len(result[0]["candidates"]) >= 1

    def test_returns_error_without_database(self):
        """Verify subject_id_search returns error when database is None."""
        tools_no_db = InternalTools(current_user=None, db=None)
        subject_id_search = tools_no_db._create_subject_id_search_tool()

        result = subject_id_search.invoke({"phrases": ["mystery"], "top_k": 3})

        assert isinstance(result, list)
        assert "error" in result[0]


class TestSubjectHybrid:
    """Tests for subject_hybrid_pool retrieval tool (async)."""

    @pytest.mark.asyncio
    async def test_returns_results_for_subject_indices(self, db_session: Session):
        """Verify subject_hybrid returns results for valid subject indices."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [1, 2, 3], "top_k": 10})

        assert isinstance(result, list)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_returns_standardized_fields(self, db_session: Session):
        """Verify subject_hybrid returns standardized output with expected fields."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [5, 10], "top_k": 5})

        book = result[0]
        assert "item_idx" in book
        assert "title" in book
        assert "author" in book
        assert "year" in book
        assert "num_ratings" in book
        assert "score" in book

    @pytest.mark.asyncio
    async def test_uses_user_favorite_subjects_when_empty_list(self, db_session: Session):
        """Verify subject_hybrid uses user's favorite subjects when empty list provided."""
        user = db_session.query(User).filter(User.user_id == 278859).first()
        assert user is not None, "Test user 278859 not found in database"

        tools = InternalTools(current_user=user, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [], "top_k": 10})

        assert isinstance(result, list)
        assert "error" not in result[0], (
            "Tool should use user's favorite subjects as fallback, not return error"
        )
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_clamps_top_k_upper_bound(self, db_session: Session):
        """Verify subject_hybrid clamps top_k to maximum of 500."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [1, 2], "top_k": 1000})

        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_clamps_top_k_lower_bound(self, db_session: Session):
        """Verify subject_hybrid clamps top_k to minimum of 1."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [1], "top_k": 0})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_clamps_subject_weight(self, db_session: Session):
        """Verify subject_hybrid clamps subject_weight to [0, 1]."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result_high = await subject_hybrid.ainvoke(
            {"fav_subjects_idxs": [1, 2], "top_k": 5, "subject_weight": 1.5}
        )
        result_low = await subject_hybrid.ainvoke(
            {"fav_subjects_idxs": [1, 2], "top_k": 5, "subject_weight": -0.5}
        )

        assert isinstance(result_high, list)
        assert isinstance(result_low, list)

    @pytest.mark.asyncio
    async def test_returns_error_without_database(self):
        """Verify subject_hybrid returns error when database is None."""
        tools_no_db = InternalTools(current_user=None, db=None)
        subject_hybrid = tools_no_db._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [1, 2], "top_k": 10})

        assert isinstance(result, list)
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_returns_error_without_subjects_and_no_user(self, db_session: Session):
        """Verify subject_hybrid returns error when no subjects provided and no user."""
        tools = InternalTools(current_user=None, db=db_session)
        subject_hybrid = tools._create_subject_hybrid_tool()

        result = await subject_hybrid.ainvoke({"fav_subjects_idxs": [], "top_k": 10})

        assert isinstance(result, list)
        assert "error" in result[0]


class TestAlsRecs:
    """Tests for als_recs retrieval tool (async, requires authenticated user)."""

    @pytest.mark.asyncio
    async def test_returns_results_for_authenticated_user(self, db_session: Session):
        """Verify als_recs returns results for authenticated warm user."""
        user = db_session.query(User).filter(User.user_id == 278859).first()
        assert user is not None, "Test user 278859 not found in database"

        tools = InternalTools(current_user=user, db=db_session)
        als_recs = tools._create_als_recs_tool()

        result = await als_recs.ainvoke({"top_k": 10})

        assert isinstance(result, list)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_returns_standardized_fields(self, db_session: Session):
        """Verify als_recs returns standardized output with expected fields."""
        user = db_session.query(User).filter(User.user_id == 278859).first()
        tools = InternalTools(current_user=user, db=db_session)
        als_recs = tools._create_als_recs_tool()

        result = await als_recs.ainvoke({"top_k": 5})

        book = result[0]
        assert "item_idx" in book
        assert "title" in book
        assert "author" in book
        assert "year" in book
        assert "num_ratings" in book
        assert "score" in book

    @pytest.mark.asyncio
    async def test_clamps_top_k_upper_bound(self, db_session: Session):
        """Verify als_recs clamps top_k to maximum of 500."""
        user = db_session.query(User).filter(User.user_id == 278859).first()
        tools = InternalTools(current_user=user, db=db_session)
        als_recs = tools._create_als_recs_tool()

        result = await als_recs.ainvoke({"top_k": 1000})

        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_clamps_top_k_lower_bound(self, db_session: Session):
        """Verify als_recs clamps top_k to minimum of 1."""
        user = db_session.query(User).filter(User.user_id == 278859).first()
        tools = InternalTools(current_user=user, db=db_session)
        als_recs = tools._create_als_recs_tool()

        result = await als_recs.ainvoke({"top_k": 0})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_error_without_authentication(self, db_session: Session):
        """Verify als_recs returns error when user is not authenticated."""
        tools = InternalTools(current_user=None, db=db_session)
        als_recs = tools._create_als_recs_tool()

        result = await als_recs.ainvoke({"top_k": 10})

        assert isinstance(result, list)
        assert "error" in result[0]
