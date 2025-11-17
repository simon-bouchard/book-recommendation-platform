# app/search/search_utils.py

from app.search.engine import SearchEngine

search_engine = SearchEngine()

def get_search_results(query, subject_idxs, page, page_size, db):
    """
    Unified interface used by the API.
    This wraps the new search engine so existing API logic doesn't break.
    """

    offset = page * page_size

    # Filters should be passed as simple dict to SearchEngine
    filters = {}
    if subject_idxs:
        filters["subject_idx"] = subject_idxs

    # Call SearchEngine
    results = search_engine.search(
        query=query,
        mode="meili",      # default mode; could be "classic"
        sort="bayes_pop:desc",  # or None
        filters=filters,
        limit=page_size + 1,
    )

    # Determine pagination
    has_next = len(results) > page_size
    results = results[:page_size]

    return results, has_next

