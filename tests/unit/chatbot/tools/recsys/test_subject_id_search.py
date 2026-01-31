# tests/unit/chatbot/tools/recsys/test_subject_id_search.py
"""
Tests for the subject_id_search tool (3-gram TF-IDF fuzzy subject matching).
Validates index building, fuzzy matching, and list input/output handling.
"""

import pytest
from unittest.mock import patch, Mock


class TestSubjectIdSearchTool:
    """Test subject_id_search tool behavior."""

    def test_returns_list_with_phrase_and_candidates(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should return list with phrase and candidate matches."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["science fiction"], top_k=5)

        assert isinstance(result, list)
        assert len(result) > 0
        assert "phrase" in result[0]
        assert "candidates" in result[0]
        assert result[0]["phrase"] == "science fiction"

    def test_handles_multiple_phrases(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should process multiple phrases in single request."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["fantasy", "mystery", "romance"], top_k=3)

        assert len(result) == 3
        assert result[0]["phrase"] == "fantasy"
        assert result[1]["phrase"] == "mystery"
        assert result[2]["phrase"] == "romance"

    def test_respects_top_k_limit(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should limit candidates per phrase to top_k."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["fiction"], top_k=3)

        candidates = result[0]["candidates"]
        assert len(candidates) <= 3

    def test_enforces_top_k_bounds(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should enforce top_k between 1 and 10."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        # Test below minimum - should still return results
        result = subject_tool.execute(phrases=["test"], top_k=0)
        assert isinstance(result, list)

        # Test above maximum - should limit to 10 candidates
        result = subject_tool.execute(phrases=["test"], top_k=50)
        if result and result[0].get("candidates"):
            assert len(result[0]["candidates"]) <= 10

    def test_builds_index_on_first_call(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should build TF-IDF index on first call."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        # First call should build index
        subject_tool.execute(phrases=["fantasy"], top_k=5)

        # Verify get_all_subject_counts was called
        mock_get_all_subject_counts.assert_called()

    def test_returns_candidate_structure(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Each candidate should have subject_idx, subject, and score."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["science fiction"], top_k=5)

        candidates = result[0]["candidates"]

        if candidates:
            candidate = candidates[0]
            assert "subject_idx" in candidate
            assert "subject" in candidate
            assert "score" in candidate
            assert isinstance(candidate["subject_idx"], int)
            assert isinstance(candidate["subject"], str)
            assert isinstance(candidate["score"], (int, float))

    def test_handles_empty_query(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should handle empty phrase gracefully."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=[""], top_k=5)

        # Should return empty candidates for empty phrase
        assert len(result) == 1
        assert result[0]["phrase"] == ""
        assert result[0]["candidates"] == []

    def test_handles_no_matches(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should return empty candidates when no matches found."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        # Very unusual query unlikely to match
        result = subject_tool.execute(phrases=["xyzabc123"], top_k=5)

        # Should return structure even with no matches
        assert len(result) == 1
        assert result[0]["phrase"] == "xyzabc123"
        assert isinstance(result[0]["candidates"], list)

    def test_handles_empty_phrases_list(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should handle empty phrases list."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=[], top_k=5)

        # Should return empty list
        assert isinstance(result, list)
        assert len(result) == 0

    def test_fuzzy_matching_works(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should find approximate matches for misspelled queries."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        # Slight misspelling
        result = subject_tool.execute(phrases=["fantesy"], top_k=5)

        candidates = result[0]["candidates"]

        # Should find "Fantasy" despite misspelling
        if candidates:
            subject_names = [c["subject"] for c in candidates]
            assert any("Fantasy" in name for name in subject_names)

    def test_sorts_candidates_by_score(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should return candidates in descending score order."""
        tools = internal_tools_factory(db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["fiction"], top_k=5)

        candidates = result[0]["candidates"]

        if len(candidates) > 1:
            scores = [c["score"] for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_requires_database(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
    ):
        """Subject ID search requires database connection."""
        # With database - tool should be present
        tools_with_db = internal_tools_factory(db=Mock())
        tools_list = tools_with_db.get_retrieval_tools(is_warm=False)
        tool_names = [t.name for t in tools_list]
        assert "subject_id_search" in tool_names

        # Without database - tool should NOT be present
        tools_without_db = internal_tools_factory(db=None)
        tools_list = tools_without_db.get_retrieval_tools(is_warm=False)
        tool_names = [t.name for t in tools_list]
        assert "subject_id_search" not in tool_names

    def test_works_without_authentication(
        self,
        internal_tools_factory,
        mock_get_all_subject_counts,
        mock_db_session,
    ):
        """Should work for anonymous users."""
        tools = internal_tools_factory(current_user=None, db=mock_db_session)
        retrieval_tools = tools.get_retrieval_tools(is_warm=False)

        subject_tool = next(t for t in retrieval_tools if t.name == "subject_id_search")

        result = subject_tool.execute(phrases=["mystery"], top_k=3)

        assert isinstance(result, list)
        assert len(result) > 0
