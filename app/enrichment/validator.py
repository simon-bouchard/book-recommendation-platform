# app/enrichment/validator.py
"""
Validation logic for enrichment payloads with tier-aware rules.
"""
from typing import List, Set
from pydantic import BaseModel, Field, field_validator

from app.enrichment.quality_classifier import get_tier_requirements


class EnrichmentPayload(BaseModel):
    """Validated enrichment output."""
    subjects: List[str] = Field(..., min_length=0, max_length=15)
    tone_ids: List[int] = Field(..., min_length=0, max_length=5)
    genre_id: int = Field(..., ge=1)  # Changed from genre slug to genre_id
    genre: str = Field(..., min_length=1)  # Keep slug for database storage
    vibe: str = Field(default="", max_length=500)


def validate_payload(
    data: dict,
    valid_tone_ids: Set[int],
    valid_genre_ids: Set[int],  # Changed from valid_genre_slugs to valid_genre_ids
    tier: str,
    ontology_version: str = "v2"
) -> EnrichmentPayload:
    """
    Validate enrichment payload with tier-specific requirements.
    
    Args:
        data: Raw dict with subjects, tone_ids, genre_id, genre (slug), vibe
        valid_tone_ids: Set of valid tone IDs
        valid_genre_ids: Set of valid genre IDs
        tier: Quality tier (RICH, SPARSE, MINIMAL, BASIC)
        ontology_version: Ontology version
    
    Returns:
        Validated EnrichmentPayload
    
    Raises:
        ValueError: If validation fails with specific error message
    """
    # Get tier-specific requirements
    reqs = get_tier_requirements(tier)
    
    # Create payload model (will do basic type checking)
    try:
        payload = EnrichmentPayload(**data)
    except Exception as e:
        raise ValueError(f"[{tier}] Invalid payload structure: {e}")
    
    # ========================================================================
    # SUBJECTS VALIDATION
    # ========================================================================
    
    subject_count = len(payload.subjects)
    subj_min = reqs['subjects']['min']
    subj_max = reqs['subjects']['max']
    subj_required = reqs['subjects']['required']
    
    if subj_required and subject_count == 0:
        raise ValueError(
            f"[{tier}] subjects cannot be empty (minimum: {subj_min})"
        )
    
    if subject_count < subj_min:
        raise ValueError(
            f"[{tier}] subjects count ({subject_count}) below minimum ({subj_min})"
        )
    
    if subject_count > subj_max:
        raise ValueError(
            f"[{tier}] subjects count ({subject_count}) exceeds maximum ({subj_max})"
        )
    
    # Check for non-empty strings
    if any(not s or not s.strip() for s in payload.subjects):
        raise ValueError(
            f"[{tier}] subjects contain empty strings"
        )
    
    # ========================================================================
    # TONES VALIDATION
    # ========================================================================
    
    tone_count = len(payload.tone_ids)
    tone_min = reqs['tones']['min']
    tone_max = reqs['tones']['max']
    tone_required = reqs['tones']['required']
    
    if tone_required and tone_count == 0:
        raise ValueError(
            f"[{tier}] tone_ids cannot be empty (minimum: {tone_min})"
        )
    
    if tone_count < tone_min:
        raise ValueError(
            f"[{tier}] tone_ids count ({tone_count}) below minimum ({tone_min})"
        )
    
    if tone_count > tone_max:
        raise ValueError(
            f"[{tier}] tone_ids count ({tone_count}) exceeds maximum ({tone_max})"
        )
    
    # Validate tone IDs are in valid set
    if payload.tone_ids:
        invalid_tones = [t for t in payload.tone_ids if t not in valid_tone_ids]
        if invalid_tones:
            raise ValueError(
                f"[{tier}] Invalid tone_ids: {invalid_tones}. "
                f"Must be from ontology {ontology_version}"
            )
    
    # ========================================================================
    # GENRE VALIDATION (now validates genre_id instead of slug)
    # ========================================================================
    
    if reqs['genre']['required']:
        # Validate genre_id (integer)
        if payload.genre_id not in valid_genre_ids:
            raise ValueError(
                f"[{tier}] Invalid genre_id: {payload.genre_id}. "
                f"Must be from valid genre list."
            )
        
        # Also validate that genre slug is present (for database storage)
        if not payload.genre or not payload.genre.strip():
            raise ValueError(
                f"[{tier}] genre slug is required but missing or empty"
            )
    
    # ========================================================================
    # VIBE VALIDATION
    # ========================================================================
    
    vibe_min_words = reqs['vibe']['min_words']
    vibe_max_words = reqs['vibe']['max_words']
    vibe_required = reqs['vibe']['required']
    
    # Handle empty vibe
    if not payload.vibe or not payload.vibe.strip():
        if vibe_required:
            raise ValueError(
                f"[{tier}] vibe is required but empty"
            )
        # Empty vibe is OK for non-required tiers
        payload.vibe = ""
    else:
        # Vibe is present - validate length
        vibe_word_count = len(payload.vibe.split())
        
        # Check minimum (only if vibe is present)
        if vibe_min_words > 0 and vibe_word_count < vibe_min_words:
            raise ValueError(
                f"[{tier}] vibe too short ({vibe_word_count} words, minimum: {vibe_min_words})"
            )
        
        # Check maximum
        if vibe_max_words > 0 and vibe_word_count > vibe_max_words:
            raise ValueError(
                f"[{tier}] vibe too long ({vibe_word_count} words, maximum: {vibe_max_words})"
            )
        
        # Special case: MINIMAL and BASIC must have empty vibe
        if tier in ("MINIMAL", "BASIC") and payload.vibe.strip():
            raise ValueError(
                f"[{tier}] vibe must be empty (got: '{payload.vibe[:50]}...')"
            )
    
    # ========================================================================
    # ADDITIONAL QUALITY CHECKS
    # ========================================================================
    
    # Check for suspiciously similar subjects (only for higher tiers)
    if tier in ("RICH", "SPARSE") and len(payload.subjects) > 1:
        _check_subject_uniqueness(payload.subjects, tier)
    
    return payload


def _check_subject_uniqueness(subjects: List[str], tier: str) -> None:
    """
    Check that subjects are sufficiently distinct.
    Warns about near-duplicates but doesn't fail validation.
    """
    # Simple check for exact duplicates (case-insensitive)
    seen = set()
    for s in subjects:
        s_lower = s.lower()
        if s_lower in seen:
            # This should have been caught by clean_subjects, but double-check
            raise ValueError(
                f"[{tier}] Duplicate subject detected: '{s}'"
            )
        seen.add(s_lower)
    
    # Check for stem conflicts (e.g., "mythology" and "mythological")
    # This is a soft warning - log but don't fail
    stems = {}
    for s in subjects:
        # Extract first word as crude stem
        first_word = s.lower().split()[0]
        if len(first_word) > 4:  # Only check substantial words
            stem = first_word[:5]  # First 5 chars as proxy for stem
            if stem in stems:
                # Just a warning in logs, don't fail validation
                pass
            stems[stem] = s
