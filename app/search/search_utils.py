# app/search/search_utils.py

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

