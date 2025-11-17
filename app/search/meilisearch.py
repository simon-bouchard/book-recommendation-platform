# app/search/meilisearch.py

from meilisearch import Client
from typing import Optional, Dict, Any
import sys, os
from dotenv import load_dotenv

load_dotenv()

secret_key = os.getenv("MEILI_MASTER_KEY")

class MeiliSearch:
    """
    Meilisearch query interface.
    Encapsulates all Meili-specific query logic.
    """

    def __init__(
        self,
        host: str = "http://localhost:7700",
        api_key: str = secret_key,
        index_name: str = "books",
    ):
        self.client = Client(host, api_key)
        self.index = self.client.index(index_name)

    # --------------------------------------------------------------
    # Main entrypoint
    # --------------------------------------------------------------
    def search(
        self,
        query: str,
        sort: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
    ):
        params = {
            "limit": limit,
        }

        # Sorting
        if sort:
            params["sort"] = [sort]

        # Filtering
        if filters:
            filter_strings = []
            for key, value in filters.items():
                if isinstance(value, list):
                    joined = " OR ".join(f'{key} = "{v}"' for v in value)
                    filter_strings.append(f"({joined})")
                else:
                    filter_strings.append(f'{key} = "{value}"')

            params["filter"] = filter_strings

        # Query Meilisearch
        result = self.index.search(query, params)
        hits = result.get("hits", [])

        return [self._convert_hit(h) for h in hits]

    # --------------------------------------------------------------
    # Convert Meili hit → book card format
    # --------------------------------------------------------------
    def _convert_hit(self, hit: dict) -> dict:
        return {
            "item_idx": hit.get("item_idx"),
            "title": hit.get("title"),
            "author": hit.get("author"),
            "cover_id": hit.get("cover_id"),
            "isbn": hit.get("isbn"),
            "description": hit.get("description"),
            "year": hit.get("year"),
            "subjects": hit.get("subjects"),
            "bayes_pop": hit.get("bayes_pop"),
        }

