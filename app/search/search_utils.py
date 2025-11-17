# app/search/search_utils.py

from app.search.engine import SearchEngine

search_engine = SearchEngine()

def get_search_results(query, subject_idxs, page, page_size, db):
    filters = {}
    if subject_idxs:
        filters["subject_idx"] = subject_idxs

    results, total = search_engine.search(
        query=query,
        mode="meili",
        sort="bayes_pop:desc", 
        filters=filters,
        page=page,
        page_size=page_size + 1
    )

    has_next = len(results) > page_size
    results = results[:page_size]
    
    return results, has_next, total
