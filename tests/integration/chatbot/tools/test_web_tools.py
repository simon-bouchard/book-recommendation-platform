# tests/integration/chatbot/tools/test_web_tools.py
"""
Integration tests for web tools using real external APIs.
"""

from app.agents.tools.external.web_tools import WebTools


class TestWebSearch:
    """Tests for web_search tool."""

    def test_returns_results_for_query(self):
        """
        Verify web_search returns results for a valid query.
        """
        web_tools = WebTools()
        web_search = web_tools._create_web_search_tool()

        result = web_search.invoke({"query": "python programming"})

        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "[Search] No results found."

    def test_deduplicates_repeat_queries(self):
        """
        Verify web_search returns guardrail message for duplicate queries.
        """
        web_tools = WebTools()
        web_search = web_tools._create_web_search_tool()

        web_search.invoke({"query": "artificial intelligence"})
        result = web_search.invoke({"query": "artificial intelligence"})

        assert "[Guardrail]" in result
        assert "already searched" in result


class TestWikipediaLookup:
    """Tests for wikipedia_lookup tool."""

    def test_returns_article_for_topic(self):
        """
        Verify wikipedia_lookup returns article content for valid topic.
        """
        web_tools = WebTools()
        wikipedia = web_tools._create_wikipedia_tool()

        result = wikipedia.invoke({"query": "Python programming language"})

        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "[Wikipedia] No article found."

    def test_deduplicates_repeat_queries(self):
        """
        Verify wikipedia_lookup returns guardrail message for duplicate queries.
        """
        web_tools = WebTools()
        wikipedia = web_tools._create_wikipedia_tool()

        wikipedia.invoke({"query": "Machine learning"})
        result = wikipedia.invoke({"query": "Machine learning"})

        assert "[Guardrail]" in result
        assert "already made" in result


class TestOpenLibrarySearch:
    """Tests for openlibrary_search tool."""

    def test_returns_books_for_query(self):
        """
        Verify openlibrary_search returns book results.
        """
        web_tools = WebTools()
        ol_search = web_tools._create_openlibrary_search_tool()

        result = ol_search.invoke({"query": "Isaac Asimov Foundation"})

        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "[OpenLibrary] No books found."

    def test_deduplicates_repeat_queries(self):
        """
        Verify openlibrary_search returns guardrail for duplicate queries.
        """
        web_tools = WebTools()
        ol_search = web_tools._create_openlibrary_search_tool()

        ol_search.invoke({"query": "Dune Frank Herbert"})
        result = ol_search.invoke({"query": "Dune Frank Herbert"})

        assert "[Guardrail]" in result
        assert "already performed" in result


class TestOpenLibraryWork:
    """Tests for openlibrary_work tool."""

    def test_returns_work_details(self):
        """
        Verify openlibrary_work returns details for valid work key.
        """
        web_tools = WebTools()
        ol_work = web_tools._create_openlibrary_work_tool()

        result = ol_work.invoke({"work_key": "/works/OL45804W"})

        assert isinstance(result, str)
        assert "Title:" in result
        assert "Description:" in result

    def test_handles_work_key_without_prefix(self):
        """
        Verify openlibrary_work normalizes work keys without /works/ prefix.
        """
        web_tools = WebTools()
        ol_work = web_tools._create_openlibrary_work_tool()

        result = ol_work.invoke({"work_key": "OL45804W"})

        assert isinstance(result, str)
        assert "Title:" in result

    def test_deduplicates_repeat_queries(self):
        """
        Verify openlibrary_work returns guardrail for duplicate queries.
        """
        web_tools = WebTools()
        ol_work = web_tools._create_openlibrary_work_tool()

        ol_work.invoke({"work_key": "/works/OL45804W"})
        result = ol_work.invoke({"work_key": "/works/OL45804W"})

        assert "[Guardrail]" in result
        assert "already fetched" in result
