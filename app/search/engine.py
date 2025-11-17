# app/search/engine.py

from .meilisearch import MeiliSearch
from .classic_search import ClassicSearch
# from .semantic_search import SemanticSearch


class SearchEngine:
    """
    Routes queries to the correct search mode.
    API should only interact with this class.
    """

    def __init__(self):
        self.meili = MeiliSearch()
        #self.classic = ClassicSearch()
        # self.semantic = SemanticSearch()

    def search(
        self,
        query: str,
        mode: str = "meili",
        sort: str | None = None,
        filters: dict | None = None,
        limit: int = 50,
    ):
        mode = mode.lower()

        if mode == "meili":
            return self.meili.search(query, sort=sort, filters=filters, limit=limit)

        elif mode == "classic":
            #return self.classic.search(query, sort=sort, filters=filters, limit=limit)
            raise ValueError("Mode not implemented yet")

        elif mode == "semantic":
            return self.semantic.search(query, limit=limit)

        else:
            raise ValueError(f"Unknown search mode: {mode}")

