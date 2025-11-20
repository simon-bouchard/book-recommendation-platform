# app/search/adapters/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from ..models import SearchRequest, SearchResult, SearchMode

class SearchAdapter(ABC):
    """Base interface all search adapters must implement"""
    
    @abstractmethod
    def search(self, request: SearchRequest) -> Tuple[List[SearchResult], int, Optional[Dict[str, Any]]]:
        """Return (results, total_count) - adapter handles its own pagination"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this search mode is available/healthy"""
        pass
    
    @property
    @abstractmethod
    def mode(self) -> SearchMode:
        """Return which search mode this adapter handles"""
        pass
