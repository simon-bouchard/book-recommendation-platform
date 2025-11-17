#app/search/models.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum

class SearchMode(str, Enum):
    MEILI = "meili"
    CLASSIC = "classic"
    SEMANTIC = "semantic"

class SearchRequest(BaseModel):
    query: str = ""
    mode: SearchMode = SearchMode.MEILI
    filters: Dict[str, Any] = {}  # Simple key-value filters
    sort: Optional[str] = None    # "year:desc", "bayes_pop:desc"
    page: int = 0
    page_size: int = 50

class SearchResult(BaseModel):
    item_idx: int
    title: str
    author: str
    cover_id: Optional[int]
    # ... other fields
    _score: Optional[float] = None  # Engine-specific score

DEFAULT_SEARCH_CONFIG = {
    "page_size": 60,
    "default_mode": SearchMode.MEILI,
    "available_modes": [SearchMode.MEILI],  # Add others as implemented
}
