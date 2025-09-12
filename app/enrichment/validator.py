from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set, Optional

class EnrichmentOut(BaseModel):
    subjects: List[str] = Field(default_factory=list, max_items=8)
    tone_ids: List[int] = Field(default_factory=list, max_items=3)
    genre: Optional[str] = None
    vibe: str = Field(default="")

def validate_payload(payload: Dict[str, Any],
                     valid_tone_ids: Set[int],
                     valid_genre_slugs: Set[str]) -> EnrichmentOut:
    data = EnrichmentOut(**payload)

    # subjects: keep clean strings only
    data.subjects = [s.strip() for s in data.subjects if isinstance(s, str) and s.strip()]

    # tones: must be valid IDs
    if not all(t in valid_tone_ids for t in data.tone_ids):
        raise ValueError("Invalid tone_ids present.")

    # genre: exactly one valid slug
    if not data.genre or data.genre not in valid_genre_slugs:
        raise ValueError("Invalid or missing genre slug.")

    # vibe token cap (≤ 20)
    if len(data.vibe.split()) > 20:
        raise ValueError("Vibe exceeds 20 tokens.")

    return data
