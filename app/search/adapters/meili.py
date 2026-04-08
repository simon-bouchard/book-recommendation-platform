# app/search/adapters/meili.py
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from meilisearch import Client

from ..models import SearchMode, SearchRequest, SearchResult
from .base import SearchAdapter


class MeiliSearchAdapter(SearchAdapter):
    """
    Optimized MeiliSearch adapter that pushes everything to Meili.
    No application-layer filtering/sorting/pagination.
    """

    def __init__(self):
        load_dotenv()
        self.client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
        self.index = self.client.index("books")

    def _convert_hit(self, hit: dict) -> SearchResult:
        return SearchResult(
            item_idx=hit.get("item_idx"),
            title=hit.get("title"),
            author=hit.get("author"),
            isbn=hit.get("isbn"),
            year=hit.get("year"),
            cover_id=hit.get("cover_id"),
            _score=hit.get("_rankingScore"),
        )

    def search(self, request: SearchRequest) -> Tuple[List[SearchResult], int, Dict]:
        params = {
            "limit": request.page_size,
            "offset": request.page * request.page_size,
            "showRankingScore": True,
        }

        if request.sort:
            params["sort"] = [request.sort]

        if request.filters:
            params["filter"] = self._build_meili_filter(request.filters)

        if request.facets:
            params["facets"] = request.facets

        if request.attributes_to_retrieve is not None:
            params["attributesToRetrieve"] = request.attributes_to_retrieve
        else:
            # Explicit sensible default — never return the kitchen sink
            params["attributesToRetrieve"] = [
                "item_idx",
                "title",
                "author",
                "cover_id",
                "year",
                "isbn",
            ]

        if request.min_score is not None:
            params["rankingScoreThreshold"] = request.min_score

        # Highlight + crop
        if request.highlight:
            params["attributesToHighlight"] = ["title", "author"]
            params["highlightPreTag"] = "<b>"
            params["highlightPostTag"] = "</b>"

        if request.crop and isinstance(request.crop, int):
            params["attributesToCrop"] = ["description"]
            params["cropLength"] = request.crop
            params["cropMarker"] = "…"

        result = self.index.search(request.query or "", params)

        # Use _formatted when available
        use_formatted = request.highlight or (request.crop and isinstance(request.crop, int))
        hits = result.get("hits", [])

        results = []
        for hit in hits:
            formatted = hit.get("_formatted", hit)
            r = SearchResult(
                item_idx=hit["item_idx"],
                title=formatted.get("title", hit["title"]),
                author=formatted.get("author", hit["author"]),
                cover_id=hit.get("cover_id"),
                year=hit.get("year"),
                isbn=hit.get("isbn"),
                _score=hit.get("_rankingScore"),
                description_snippet=formatted.get("description") if use_formatted else None,
            )
            results.append(r)

        return results, result.get("estimatedTotalHits", 0), result

    def _build_meili_params(self, request: SearchRequest) -> dict:
        """Push ALL filtering/sorting/pagination to Meili"""
        params = {
            "limit": request.page_size,
            "offset": request.page * request.page_size,
        }

        if request.sort:
            params["sort"] = [request.sort]  # Meili handles sorting

        if request.filters:
            # Convert our simple filters to Meili filter syntax
            params["filter"] = self._build_meili_filter(request.filters)

        return params

    def _build_meili_filter(self, filters: Dict[str, Any]) -> List[str]:
        """Convert simple dict filters to Meili filter syntax"""
        filter_strings = []
        for field, value in filters.items():
            if isinstance(value, list):
                # subject_idx = [1,2,3] → (subject_idx = 1 OR subject_idx = 2)
                conditions = [f'{field} = "{v}"' for v in value]
                filter_strings.append(f"({' OR '.join(conditions)})")
            else:
                filter_strings.append(f'{field} = "{value}"')
        return filter_strings

    def is_available(self) -> bool:
        try:
            # Simple health check
            self.client.health()
            return True
        except Exception:
            return False

    @property
    def mode(self) -> SearchMode:
        return SearchMode.MEILI
