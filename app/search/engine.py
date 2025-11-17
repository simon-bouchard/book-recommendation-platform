# app/search/engine.py
from typing import Dict, List
from .adapters.meili import MeiliSearchAdapter
from .models import SearchRequest, SearchMode, SearchResult
from .adapters.base import SearchAdapter

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
    
    def search(self, request: SearchRequest) -> SearchResult:
        """Main entry point - pure delegation"""
        adapter = self._adapters.get(request.mode)
        if not adapter:
            raise ValueError(f"Unsupported search mode: {request.mode}")
            
        if not adapter.is_available():
            # Optional: fallback to another mode
            return self._fallback_search(request)
        
        # Delegate ALL search logic to the adapter
        results, total = adapter.search(request)
        
        return SearchResult(
            results=results,
            total=total,
            page=request.page,
            page_size=request.page_size,
            # ... other metadata
        )
    
    def get_available_modes(self) -> List[SearchMode]:
        """Dynamic discovery of available search modes"""
        return [mode for mode, adapter in self._adapters.items() 
                if adapter.is_available()]
