#app/search/models.py
from typing import List, Optional, Dict, Any, Union
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
    sort: Optional[str] = "bayes_pop:desc"    # "year:desc", "bayes_pop:desc"
    page: int = 0
    page_size: int = 50
    highlight: bool = False
    crop: Union[bool, int] = 25          # false → no crop, int → crop length
    facets: Optional[List[str]] = None
    attributes_to_retrieve: Optional[List[str]] = None
    min_score: Optional[float] = None    # ranking_score_threshold

class SearchResult(BaseModel):
    item_idx: int
    title: str
    author: Optional[str]
    cover_id: Optional[int]
    _score: Optional[float] = None  # Engine-specific score
    isbn: Optional[str] = None
    year: Optional[int] = None
    description_snippet: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    page: int
    page_size: int
    raw_response: Optional[Dict[str, Any]] = None

DEFAULT_SEARCH_CONFIG = {
    "page_size": 60,
    "default_mode": SearchMode.MEILI,
    "available_modes": [SearchMode.MEILI],  # Add others as implemented
}
