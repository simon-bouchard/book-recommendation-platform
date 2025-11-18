from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.table_models import BookSubject, Subject
from time import time
from typing import Any, Dict, List
import math

_subject_cache = []
_last_subject_fetch = 0
SUBJECT_CACHE_TTL = 100000

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_all_subject_counts(db: Session):
    global _subject_cache, _last_subject_fetch

    now = time()
    if _subject_cache and (now - _last_subject_fetch) < SUBJECT_CACHE_TTL:
        return _subject_cache

    # Group by subject_idx and order by count descending
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
    names = db.query(Subject.subject_idx, Subject.subject)\
              .filter(Subject.subject_idx.in_(subject_ids))\
              .all()
    name_map = {i: s for i, s in names}

    # return subject_idx too (and drop placeholders)
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
