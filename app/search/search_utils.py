# app/search/search_utils.py

from typing import Optional, Union
from sqlalchemy.orm import Session
from app.table_models import Subject

from app.search.engine import SearchEngine
from app.search.models import SearchRequest, SearchMode

search_engine = SearchEngine()

def get_search_results(query, subject, page, page_size, db):
    filters = {}
    if subject:
        filters["subject_ids"] = subject

    search_request = SearchRequest(
        query=query,
        mode=SearchMode.MEILI,
        sort="bayes_pop:desc", 
        filters=filters,
        page=page,
        page_size=page_size + 1
    )


    search_response = search_engine.search(search_request)
    results = search_response.results
    total = search_response.total
    
    has_next = len(results) > page_size
    results = results[:page_size]
    
    return results, has_next, total

def _build_search_request(
    query: str,
    subjects: Optional[str],
    page: int,
    page_size: int,
    mode: SearchMode,
    sort: Optional[str],
    highlight: bool,
    crop: Union[bool, int],
    min_score: Optional[float],
    db: Session,
) -> SearchRequest:
    filters = {}
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
        subject_rows = (
            db.query(Subject.subject_idx)
            .filter(Subject.subject.in_(subject_list))
            .all()
        )
        subject_idxs = [row.subject_idx for row in subject_rows]
        if subject_idxs:
            filters["subject_ids"] = subject_idxs  # Meili expects subject_ids array

    return SearchRequest(
        query=query,
        mode=mode,
        filters=filters,
        sort=sort,
        page=page,
        page_size=page_size,
        highlight=highlight,
        crop=crop if isinstance(crop, int) or crop is False else 25,
        min_score=min_score,
        attributes_to_retrieve=[
            "item_idx", "title", "author", "cover_id", "year", "isbn", "description"
        ],  # optional: be explicit
    )
