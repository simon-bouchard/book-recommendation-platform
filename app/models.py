import bcrypt
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.table_models import BookSubject, Subject
from time import time
from typing import Any, Dict, List
import math
from dotenv import load_dotenv
from meilisearch import Client
import os

_subject_cache = []
_last_subject_fetch = 0
SUBJECT_CACHE_TTL = 100000


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_all_subject_counts(db: Session):
    global _subject_cache, _last_subject_fetch

    now = time()
    if _subject_cache and (now - _last_subject_fetch) < SUBJECT_CACHE_TTL:
        return _subject_cache

    try:
        load_dotenv()
        client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
        index = client.index("books")

        # Minimal facet query: empty search, facets only, no hits
        result = index.search(
            "",  # Empty query = match everything
            {
                "facets": ["subject_ids"],
                "limit": 0,  # No hits needed, just counts
            },
        )

        facet_dist = result.get("facetDistribution", {}).get("subject_ids", {})
        if not facet_dist:
            raise ValueError("No facet data returned")

        # Convert {str(idx): count} to sorted list of (int(idx), count)
        subject_counts = sorted(
            [(int(idx), count) for idx, count in facet_dist.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Batch-resolve names with one SQL query (fast)
        if subject_counts:
            subject_idxs = [idx for idx, _ in subject_counts]  # Top 100 max
            name_rows = (
                db.query(Subject.subject_idx, Subject.subject)
                .filter(Subject.subject_idx.in_(subject_idxs))
                .all()
            )
            name_map = {row.subject_idx: row.subject for row in name_rows}

        # Preserve the count-based order from subject_counts
        _subject_cache = []
        for idx, count in subject_counts:
            if idx in name_map and name_map[idx] != "[NO_SUBJECT]":
                _subject_cache.append(
                    {"subject_idx": idx, "subject": name_map.get(idx, "[Unknown]"), "count": count}
                )

        _subject_cache.sort(key=lambda x: (-x["count"], x["subject"]))

    except Exception as e:
        # Fallback to your original SQL (logs warning if you add logging)
        print(f"Meili facet fetch failed, using SQL fallback: {e}")  # Or use logger
        counts = (
            db.query(BookSubject.subject_idx, func.count().label("c"))
            .group_by(BookSubject.subject_idx)
            .order_by(func.count().desc())
            .all()
        )
        if not counts:
            _subject_cache, _last_subject_fetch = [], now
            return _subject_cache

        subject_ids = [sid for sid, _ in counts]
        names = (
            db.query(Subject.subject_idx, Subject.subject)
            .filter(Subject.subject_idx.in_(subject_ids))
            .all()
        )
        name_map = {i: s for i, s in names}

        _subject_cache = [
            {"subject_idx": int(sid), "subject": name_map.get(sid, ""), "count": int(c)}
            for sid, c in counts
            if name_map.get(sid) and name_map[sid] != "[NO_SUBJECT]"
        ]

    _last_subject_fetch = now
    return _subject_cache


def clean_float_values(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean NaN, Inf, and other non-serializable values so | tojson works perfectly.
    """

    def clean_value(val: Any) -> Any:
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if val is None:
            return None
        if isinstance(val, (int, str, bool)):
            return val
        # Last resort: convert to str (should never happen)
        try:
            return str(val)
        except:
            return None

    cleaned = []
    for item in data:
        cleaned_item = {k: clean_value(v) for k, v in item.items()}
        cleaned.append(cleaned_item)
    return cleaned
