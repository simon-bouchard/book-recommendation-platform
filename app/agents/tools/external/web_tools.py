# app/agents/tools/external/web_tools.py
"""
Modernized web tools - native functions with clean interfaces.
No LangChain dependencies, just pure Python.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


@dataclass
class WebToolState:
    """
    Deduplication state for web tools.
    Prevents repeated queries within a conversation turn.
    """

    seen_web_searches: set[str] = field(default_factory=set)
    seen_wiki_queries: set[str] = field(default_factory=set)
    seen_ol_searches: set[str] = field(default_factory=set)
    seen_ol_works: set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Clear all deduplication state."""
        self.seen_web_searches.clear()
        self.seen_wiki_queries.clear()
        self.seen_ol_searches.clear()
        self.seen_ol_works.clear()


def _normalize_query(query: str) -> str:
    """Normalize query for deduplication."""
    tokens = re.findall(r"\w+", query.lower())
    tokens.sort()
    return " ".join(tokens)


class WebTools:
    """Factory for web tools with shared state."""

    def __init__(self, state: Optional[WebToolState] = None):
        self.state = state or WebToolState()
        self._ddg = DuckDuckGoSearchResults(num_results=5)
        self._wiki = WikipediaAPIWrapper(lang="en")
        self._session = requests.Session()

    def get_tools(self) -> list[Callable]:
        """Get all web tool definitions."""
        return [
            self._create_web_search_tool(),
            self._create_wikipedia_tool(),
            self._create_openlibrary_search_tool(),
            self._create_openlibrary_work_tool(),
        ]

    def _create_web_search_tool(self) -> Callable:
        """General web search with deduplication."""

        @tool
        def web_search(query: str) -> str:
            """
            Search the web using DuckDuckGo.

            Args:
                query: Search query string

            Returns:
                Search results as formatted text
            """
            normalized = _normalize_query(query)

            # Check for duplicates
            if normalized in self.state.seen_web_searches:
                return "[Guardrail] This query was already searched. Use existing results."

            self.state.seen_web_searches.add(normalized)

            try:
                result = self._ddg.run(query)
                return result if result and result.strip() else "[Search] No results found."
            except Exception as e:
                return f"[Search Error] {str(e)}"

        web_search.metadata = {"status_message": "Searching the web..."}
        return web_search

    def _create_wikipedia_tool(self) -> Callable:
        """Wikipedia lookup with deduplication."""

        @tool
        def wikipedia_lookup(query: str) -> str:
            """
            Look up information on Wikipedia.

            Best for: Background info, definitions, historical context.
            Avoid for: 'Best of' lists, current events, comparisons.

            Args:
                query: Topic to look up

            Returns:
                Wikipedia article summary
            """
            normalized = _normalize_query(query)

            if normalized in self.state.seen_wiki_queries:
                return "[Guardrail] This Wikipedia query was already made. Use existing results."

            self.state.seen_wiki_queries.add(normalized)

            try:
                result = self._wiki.run(query)
                return result if result and result.strip() else "[Wikipedia] No article found."
            except Exception as e:
                return f"[Wikipedia Error] {str(e)}"

        wikipedia_lookup.metadata = {"status_message": "Looking up on Wikipedia..."}
        return wikipedia_lookup

    def _create_openlibrary_search_tool(self) -> Callable:
        """OpenLibrary search with deduplication."""

        @tool
        def openlibrary_search(query: str) -> str:
            """
            Search OpenLibrary for real books by title, author, or subject.

            Returns up to 5 books with metadata.

            Args:
                query: Search query (title, author, or topic)

            Returns:
                Formatted list of books with metadata
            """
            normalized = _normalize_query(query)

            if normalized in self.state.seen_ol_searches:
                return "[Guardrail] This OpenLibrary search was already performed. Use existing results."

            self.state.seen_ol_searches.add(normalized)

            try:
                response = self._session.get(
                    "https://openlibrary.org/search.json",
                    params={
                        "q": query,
                        "fields": "title,author_name,first_publish_year,key,edition_count",
                        "limit": 5,
                    },
                    timeout=12,
                )
                response.raise_for_status()
                data = response.json()

                docs = data.get("docs", [])[:5]
                if not docs:
                    return "[OpenLibrary] No books found."

                lines = []
                for doc in docs:
                    title = doc.get("title", "Untitled")
                    authors = doc.get("author_name", [])
                    author = ", ".join(authors[:2]) if authors else "Unknown"
                    year = doc.get("first_publish_year")
                    work_key = doc.get("key")
                    editions = doc.get("edition_count")

                    meta_parts = []
                    if year:
                        meta_parts.append(str(year))
                    if editions:
                        meta_parts.append(f"{editions} editions")

                    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
                    lines.append(f"- {title} â€” {author}{meta_str}  [openlibrary.org{work_key}]")

                return "\n".join(lines)

            except Exception as e:
                return f"[OpenLibrary Error] {str(e)}"

        openlibrary_search.metadata = {"status_message": "Searching OpenLibrary..."}
        return openlibrary_search

    def _create_openlibrary_work_tool(self) -> Callable:
        """Fetch specific OpenLibrary work details."""

        @tool
        def openlibrary_work(work_key: str) -> str:
            """
            Fetch detailed information about a specific OpenLibrary work.

            Args:
                work_key: Work identifier like '/works/OL12345W' or 'OL12345W'

            Returns:
                Work details including title and description
            """
            normalized = _normalize_query(work_key)

            if normalized in self.state.seen_ol_works:
                return (
                    "[Guardrail] This OpenLibrary work was already fetched. Use existing results."
                )

            self.state.seen_ol_works.add(normalized)

            # Normalize work key
            key = work_key.strip()
            if not key.startswith("/works/"):
                key = f"/works/{key}"

            try:
                response = self._session.get(
                    f"https://openlibrary.org{key}.json",
                    timeout=12,
                )
                response.raise_for_status()
                work = response.json()

                title = work.get("title", "Untitled")
                description = work.get("description")

                # Handle description being a dict or string
                if isinstance(description, dict):
                    description = description.get("value")

                if not description:
                    description = "[No description available]"

                # Truncate long descriptions
                if len(description) > 800:
                    description = description[:800] + "â€¦"

                return (
                    f"Title: {title}\n"
                    f"Work: {key}\n"
                    f"Description: {description}\n"
                    f"[source: openlibrary.org{key}]"
                )

            except Exception as e:
                return f"[OpenLibrary Error] {str(e)}"

        openlibrary_work.metadata = {"status_message": "Fetching book details..."}
        return openlibrary_work
