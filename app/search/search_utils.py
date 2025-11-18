# app/search/search_utils.py

from app.search.engine import SearchEngine
from app.search.models import SearchRequest, SearchMode

search_engine = SearchEngine()

# In search_utils.py - enhanced version
def get_search_results(query, subject, page, page_size, db):
    filters = {}
    if subject:
        filters["subject_ids"] = subject
        print(f"🔍 DEBUG: Filtering by subject: '{subject}'")
        print(f"🔍 DEBUG: Full filters: {filters}")

    search_request = SearchRequest(
        query=query,
        mode=SearchMode.MEILI,
        sort="bayes_pop:desc", 
        filters=filters,
        page=page,
        page_size=page_size + 1
    )

    print(f"🔍 DEBUG: SearchRequest: {search_request.dict()}")

    search_response = search_engine.search(search_request)
    results = search_response.results
    total = search_response.total
    
    print(f"🔍 DEBUG: Raw results: {len(results)}, total: {total}")
    
    # Debug first few results to see their subjects
    for i, result in enumerate(results[:3]):
        print(f"🔍 DEBUG: Result {i}: item_idx={result.item_idx}, title={result.title[:50]}...")
    
    has_next = len(results) > page_size
    results = results[:page_size]
    
    return results, has_next, total

"""
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
"""
