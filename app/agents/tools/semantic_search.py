from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.semantic_index.search import SemanticSearcher
from app.agents.settings import settings  # if you have a shared settings object

class SemanticSearchInput(BaseModel):
    query: str = Field(..., description="Free-text query")
    top_k: int = Field(10, ge=1, le=50)

_searcher = None

def _get_searcher():
    global _searcher
    if _searcher is None:
        # point to models/data or a new models/semantic/
        _searcher = SemanticSearcher(dir_path="models/data", embedder=settings.embedder)
    return _searcher

def book_semantic_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    s = _get_searcher()
    return s.search(query, top_k=top_k)
