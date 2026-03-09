# models/infrastructure/metadata_enrichment.py
"""
Book metadata enrichment helpers for the metadata model server.

At startup the metadata DataFrame is converted to a dict[int, str] where each
value is the pre-serialized JSON string for that book. Per-request enrichment
is then purely string operations — dict lookups and a join — with no Pydantic
or json.dumps calls on the hot path.
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from model_servers._shared.contracts import BookMeta

logger = logging.getLogger(__name__)


def _row_to_book_meta(item_idx: int, row: pd.Series) -> BookMeta:
    """
    Convert a book metadata DataFrame row to a BookMeta contract object.

    Called once per book at startup during lookup construction. All optional
    fields are guarded against NaN and empty strings, both of which can appear
    in the raw DataFrame for books with incomplete records.

    Args:
        item_idx: The book's item index, used as the primary key.
        row: A pandas Series representing one row from the metadata DataFrame.

    Returns:
        A fully validated BookMeta instance.
    """

    def _opt_str(key: str) -> Optional[str]:
        val = row.get(key)
        return str(val) if val and not (isinstance(val, float) and np.isnan(val)) else None

    def _opt_int(key: str) -> Optional[int]:
        val = row.get(key)
        try:
            return (
                int(val)
                if val is not None and not (isinstance(val, float) and np.isnan(val))
                else None
            )
        except (ValueError, TypeError):
            return None

    def _opt_float(key: str) -> Optional[float]:
        val = row.get(key)
        try:
            f = float(val)
            return f if np.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    return BookMeta(
        item_idx=item_idx,
        title=str(row["title"]),
        author=_opt_str("author"),
        year=_opt_int("year"),
        isbn=_opt_str("isbn"),
        cover_id=_opt_str("cover_id"),
        avg_rating=_opt_float("book_avg_rating"),
        num_ratings=int(row["book_num_ratings"]) if "book_num_ratings" in row else 0,
        bayes_score=_opt_float("bayes"),
    )


def build_lookup(df: pd.DataFrame) -> dict[int, str]:
    """
    Convert the metadata DataFrame to a pre-serialized JSON string lookup dict.

    Each book is converted to a BookMeta instance (for contract validation),
    then immediately serialized to a JSON string and stored. Request-time
    enrichment becomes pure dict lookups and string joins with no serialization.

    Rows that fail conversion are logged and skipped rather than aborting the
    entire build.

    Args:
        df: Book metadata DataFrame indexed by item_idx.

    Returns:
        Mapping of item_idx -> JSON string for all successfully converted rows.
    """
    lookup: dict[int, str] = {}
    for row in df.itertuples():
        item_idx = int(row.Index)
        try:
            book_meta = _row_to_book_meta(item_idx, df.loc[item_idx])
            lookup[item_idx] = json.dumps(book_meta.model_dump())
        except Exception as e:
            logger.warning("Skipping item %d during lookup build: %s", item_idx, e)
    return lookup


def enrich_items(lookup: dict[int, str], item_indices: list[int]) -> str:
    """
    Return a complete JSON response body for the requested item indices.

    Collects pre-serialized JSON strings from the lookup and joins them into
    a valid EnrichResponse JSON body. Items absent from the lookup are silently
    omitted.

    Args:
        lookup: Pre-built item_idx -> JSON string mapping from build_lookup.
        item_indices: Ordered list of item indices to enrich.

    Returns:
        A JSON string of the form {"books": [...]} ready to return as a
        raw Response body, with no further serialization required.
    """
    parts = [lookup[idx] for idx in item_indices if idx in lookup]
    return '{"books":[' + ",".join(parts) + "]}"
