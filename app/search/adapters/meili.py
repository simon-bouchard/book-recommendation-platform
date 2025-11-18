# app/search/adapters/meili.py
from meilisearch import Client
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Tuple
from ..models import SearchRequest, SearchResult, SearchMode
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
            cover_id=hit.get("cover_id"),
            _score=hit.get("_rankingScore"),
        )

    def search(self, request: SearchRequest) -> Tuple[List[SearchResult], int]:
        offset = request.page * request.page_size
        params = {
            "limit": request.page_size,
            "offset": offset,
        }
        
        if request.sort:
            params["sort"] = [request.sort]
            
        if request.filters:
            filter_strings = []
            for key, value in request.filters.items():
                if isinstance(value, list):
                    joined = " OR ".join(f'{key} = "{v}"' for v in value)
                    filter_strings.append(f"({joined})")
                else:
                    filter_strings.append(f'{key} = "{value}"')
            params["filter"] = filter_strings

        result = self.index.search(request.query, params)
        hits = result.get("hits", [])
        total = result.get("estimatedTotalHits", 0)
        
        return [self._convert_hit(h) for h in hits], total
       
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
            except:
                return False

    @property
    def mode(self) -> SearchMode:
        return SearchMode.MEILI
