# app/search/engine.py
from typing import Dict, List

from fastapi import HTTPException

from .adapters.base import SearchAdapter
from .adapters.meili import MeiliSearchAdapter
from .models import SearchMode, SearchRequest, SearchResponse


class SearchEngine:
    """
    Thin orchestrator that routes to appropriate adapter.
    No business logic - just delegation and error handling.
    """

    def __init__(self):
        self._adapters = self._initialize_adapters()

    def _initialize_adapters(self) -> Dict[SearchMode, SearchAdapter]:
        """Factory method - easily add new adapters here"""
        return {
            SearchMode.MEILI: MeiliSearchAdapter(),
            # SearchMode.CLASSIC: ClassicSearchAdapter(),  # Add later
            # SearchMode.SEMANTIC: SemanticSearchAdapter(), # Add later
        }

    def search(self, request: SearchRequest) -> SearchResponse:
        """Main entry point - pure delegation"""
        adapter = self._adapters.get(request.mode)
        if not adapter:
            raise ValueError(f"Unsupported search mode: {request.mode}")

        if not adapter.is_available():
            # No other adapter is currently registered to fall back to
            # (see _initialize_adapters) — surface a clean 503 instead of
            # silently returning empty/wrong results.
            raise HTTPException(
                status_code=503, detail=f"Search backend '{request.mode}' is unavailable"
            )

        # Delegate ALL search logic to the adapter
        results, total, raw_response = adapter.search(request)

        return SearchResponse(
            results=results,
            total=total,
            page=request.page,
            page_size=request.page_size,
            raw_response=raw_response,
        )

    def get_available_modes(self) -> List[SearchMode]:
        """Dynamic discovery of available search modes"""
        return [mode for mode, adapter in self._adapters.items() if adapter.is_available()]
