# tests/unit/chabot/recsys/test_curation_logic.py
"""
Unit tests for CurationAgent pure methods.

Targets three methods that contain real logic and have no dependencies:
    _extract_book_ids_from_citations(text)        → List[int]
    _order_books_by_citations(candidates, ids)    → List[BookRecommendation]
    _prepare_candidates(candidates)               → List[Dict]

All tests are synchronous and make no LLM or I/O calls.
"""

import pytest
from app.agents.domain.entities import BookRecommendation


# ==============================================================================
# _extract_book_ids_from_citations
# ==============================================================================


class TestExtractBookIdsFromCitations:
    """
    Pattern under test: [Any text](integer) → integer extracted as book ID.

    Non-integer targets, plain markdown links, and URL-style targets must
    all be ignored.
    """

    def test_single_citation(self, curation_agent):
        text = "I recommend [Dune](4521) for science fiction lovers."
        assert curation_agent._extract_book_ids_from_citations(text) == [4521]

    def test_multiple_distinct_citations(self, curation_agent):
        text = "[Foundation](100) is a classic. So is [Dune](200) and [Neuromancer](300)."
        assert curation_agent._extract_book_ids_from_citations(text) == [100, 200, 300]

    def test_order_of_first_appearance_preserved(self, curation_agent):
        text = "[Book C](300) then [Book A](100) then [Book B](200)"
        assert curation_agent._extract_book_ids_from_citations(text) == [300, 100, 200]

    def test_duplicate_citations_deduplicated(self, curation_agent):
        """Same book cited twice — should appear only once, at first position."""
        text = "[Dune](4521) is great. Have you read [Dune](4521) yet?"
        assert curation_agent._extract_book_ids_from_citations(text) == [4521]

    def test_duplicate_preserves_first_occurrence_order(self, curation_agent):
        """When duplicates exist, second occurrence is dropped but others keep their order."""
        text = "[A](1) [B](2) [A](1) [C](3)"
        assert curation_agent._extract_book_ids_from_citations(text) == [1, 2, 3]

    def test_empty_text(self, curation_agent):
        assert curation_agent._extract_book_ids_from_citations("") == []

    def test_text_with_no_citations(self, curation_agent):
        assert curation_agent._extract_book_ids_from_citations("No books here.") == []

    def test_url_style_links_ignored(self, curation_agent):
        """Standard markdown URLs must not be mistaken for book citations."""
        text = "Visit [Google](https://google.com) for more info."
        assert curation_agent._extract_book_ids_from_citations(text) == []

    def test_non_integer_target_ignored(self, curation_agent):
        text = "[Book](some-slug) is not a valid citation."
        assert curation_agent._extract_book_ids_from_citations(text) == []

    def test_mixed_valid_and_url_citations(self, curation_agent):
        text = "Read [Dune](4521) and check [Wikipedia](https://en.wikipedia.org) for context."
        assert curation_agent._extract_book_ids_from_citations(text) == [4521]

    def test_citation_with_special_characters_in_title(self, curation_agent):
        """Titles containing brackets or punctuation should still parse correctly."""
        text = "[The Hitchhiker's Guide to the Galaxy](9999) is unmissable."
        assert curation_agent._extract_book_ids_from_citations(text) == [9999]

    def test_large_id(self, curation_agent):
        text = "[Book](1234567890)"
        assert curation_agent._extract_book_ids_from_citations(text) == [1234567890]


# ==============================================================================
# _order_books_by_citations
# ==============================================================================


class TestOrderBooksByCitations:
    """
    _order_books_by_citations reorders a candidate list to match citation order.

    Books not cited are dropped; cited IDs not in the candidate list are
    silently skipped.
    """

    def test_reorders_to_match_citation_order(self, curation_agent, book_pool):
        # Citation order: 1003, 1001, 1005
        cited_ids = [1003, 1001, 1005]
        result = curation_agent._order_books_by_citations(book_pool, cited_ids)
        assert [b.item_idx for b in result] == [1003, 1001, 1005]

    def test_uncited_books_are_dropped(self, curation_agent, book_pool):
        cited_ids = [1001, 1002]
        result = curation_agent._order_books_by_citations(book_pool, cited_ids)
        assert len(result) == 2
        assert {b.item_idx for b in result} == {1001, 1002}

    def test_cited_id_not_in_candidates_is_silently_skipped(self, curation_agent, book_pool):
        """ID 9999 is cited but not in the candidate pool — must not raise."""
        cited_ids = [1001, 9999, 1002]
        result = curation_agent._order_books_by_citations(book_pool, cited_ids)
        assert [b.item_idx for b in result] == [1001, 1002]

    def test_empty_cited_ids_returns_empty_list(self, curation_agent, book_pool):
        result = curation_agent._order_books_by_citations(book_pool, [])
        assert result == []

    def test_empty_candidates_returns_empty_list(self, curation_agent):
        result = curation_agent._order_books_by_citations([], [1001, 1002])
        assert result == []

    def test_returns_book_recommendation_objects(self, curation_agent, book_pool):
        result = curation_agent._order_books_by_citations(book_pool, [1001])
        assert all(isinstance(b, BookRecommendation) for b in result)

    def test_single_citation_single_result(self, curation_agent, book_pool):
        result = curation_agent._order_books_by_citations(book_pool, [1003])
        assert len(result) == 1
        assert result[0].item_idx == 1003

    def test_all_books_cited_in_reverse_order(self, curation_agent, book_pool):
        reversed_ids = [b.item_idx for b in reversed(book_pool)]
        result = curation_agent._order_books_by_citations(book_pool, reversed_ids)
        assert [b.item_idx for b in result] == reversed_ids


# ==============================================================================
# _prepare_candidates
# ==============================================================================


class TestPrepareCandidates:
    """
    _prepare_candidates converts BookRecommendation objects to dicts for the LLM.

    Key behaviours:
    - Uses to_curation_dict() when available (the normal path).
    - Falls back to manual extraction when the method is absent.
    - Truncates vibe at 200 characters and appends "...".
    - Never includes enrichment fields if they are None/empty.
    """

    def test_uses_to_curation_dict_when_available(self, curation_agent, make_book):
        book = make_book(item_idx=1, title="Dune", author="Herbert", year=1965)
        result = curation_agent._prepare_candidates([book])
        assert len(result) == 1
        assert result[0]["item_idx"] == 1
        assert result[0]["title"] == "Dune"

    def test_fallback_when_to_curation_dict_absent(self, curation_agent):
        """Object without to_curation_dict() triggers manual extraction."""

        class MinimalBook:
            item_idx = 42
            title = "Minimal"
            author = "Author"
            year = 1990
            num_ratings = 5
            subjects = None
            tones = None
            vibe = None
            genre = None

        result = curation_agent._prepare_candidates([MinimalBook()])
        assert result[0]["item_idx"] == 42
        assert result[0]["title"] == "Minimal"

    def test_vibe_within_limit_not_truncated(self, curation_agent, make_book):
        short_vibe = "A" * 199
        book = make_book(item_idx=1, vibe=short_vibe)
        result = curation_agent._prepare_candidates([book])
        assert result[0]["vibe"] == short_vibe
        assert not result[0]["vibe"].endswith("...")

    def test_vibe_at_exactly_200_not_truncated(self, curation_agent, make_book):
        vibe_200 = "B" * 200
        book = make_book(item_idx=1, vibe=vibe_200)
        result = curation_agent._prepare_candidates([book])
        assert result[0]["vibe"] == vibe_200

    def test_vibe_over_200_truncated_with_ellipsis(self, curation_agent, make_book):
        long_vibe = "C" * 300
        book = make_book(item_idx=1, vibe=long_vibe)
        result = curation_agent._prepare_candidates([book])
        assert result[0]["vibe"] == "C" * 200 + "..."

    def test_none_vibe_not_included(self, curation_agent, make_book):
        book = make_book(item_idx=1)  # vibe defaults to None
        result = curation_agent._prepare_candidates([book])
        assert "vibe" not in result[0]

    def test_enrichment_fields_present_when_populated(self, curation_agent, make_book):
        book = make_book(
            item_idx=1,
            subjects=["Science Fiction"],
            tones=["tense"],
            genre="SF",
        )
        result = curation_agent._prepare_candidates([book])
        assert result[0]["subjects"] == ["Science Fiction"]
        assert result[0]["tones"] == ["tense"]
        assert result[0]["genre"] == "SF"

    def test_enrichment_fields_absent_when_none(self, curation_agent, make_book):
        book = make_book(item_idx=1)
        result = curation_agent._prepare_candidates([book])
        for field in ("subjects", "tones", "genre", "vibe"):
            assert field not in result[0], f"Empty field '{field}' should not appear in output"

    def test_empty_candidate_list(self, curation_agent):
        assert curation_agent._prepare_candidates([]) == []

    def test_preserves_order(self, curation_agent, make_book):
        books = [make_book(item_idx=i) for i in [30, 10, 20]]
        result = curation_agent._prepare_candidates(books)
        assert [r["item_idx"] for r in result] == [30, 10, 20]
