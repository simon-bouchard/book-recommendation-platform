# app/enrichment/validator.py
"""
Tier-aware validation for enrichment outputs.
Different rules per quality tier to prevent hallucination in sparse metadata cases.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Set, Optional


class EnrichmentOut(BaseModel):
    """
    Validated enrichment output.
    Fields are validated based on quality tier.
    """
    subjects: List[str] = Field(default_factory=list, max_length=8)
    tone_ids: List[int] = Field(default_factory=list, max_length=3)
    genre: Optional[str] = None
    vibe: str = Field(default="")
    
    @field_validator('subjects')
    @classmethod
    def clean_subjects(cls, v):
        """Clean and deduplicate subjects"""
        if not v:
            return []
        # Strip whitespace and filter empty
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for s in cleaned:
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                unique.append(s)
        return unique


def validate_payload(
    payload: Dict[str, Any],
    valid_tone_ids: Set[int],
    valid_genre_slugs: Set[str],
    tier: str,
    ontology_version: str = "v2"
) -> EnrichmentOut:
    """
    Validate enrichment payload with tier-specific rules.
    
    Tier-specific validation:
    - RICH: Requires 5-8 subjects, 2-3 tones, non-empty vibe (8-12 words)
    - SPARSE: Requires 3-5 subjects, 0-2 tones (optional), vibe optional (4-8 words)
    - MINIMAL: Requires 1-3 subjects, 0-1 tone (optional), vibe must be empty
    - BASIC: Allows 0-1 subject, no tones, vibe must be empty
    
    Args:
        payload: Raw JSON payload from LLM
        valid_tone_ids: Set of valid tone IDs for validation
        valid_genre_slugs: Set of valid genre slugs for validation
        tier: Quality tier ("RICH" | "SPARSE" | "MINIMAL" | "BASIC")
        ontology_version: Which ontology version (v1 or v2)
        
    Returns:
        Validated EnrichmentOut object
        
    Raises:
        ValueError: If payload doesn't meet tier requirements
    """
    # Parse with Pydantic (basic structure validation)
    data = EnrichmentOut(**payload)
    
    # Get tier-specific requirements
    from app.enrichment.quality_classifier import get_tier_requirements
    reqs = get_tier_requirements(tier)
    
    # ========================================================================
    # SUBJECTS VALIDATION
    # ========================================================================
    
    subject_count = len(data.subjects)
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
    if any(not s or not s.strip() for s in data.subjects):
        raise ValueError(
            f"[{tier}] subjects contain empty strings"
        )
    
    # ========================================================================
    # TONES VALIDATION
    # ========================================================================
    
    tone_count = len(data.tone_ids)
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
    if data.tone_ids:
        invalid_tones = [t for t in data.tone_ids if t not in valid_tone_ids]
        if invalid_tones:
            raise ValueError(
                f"[{tier}] Invalid tone_ids: {invalid_tones}. "
                f"Must be from ontology {ontology_version}"
            )
    
    # ========================================================================
    # GENRE VALIDATION
    # ========================================================================
    
    if reqs['genre']['required']:
        if not data.genre or not data.genre.strip():
            raise ValueError(
                f"[{tier}] genre is required but missing or empty"
            )
        
        if data.genre not in valid_genre_slugs:
            raise ValueError(
                f"[{tier}] Invalid genre slug: '{data.genre}'. "
                f"Must be from valid genre list."
            )
    
    # ========================================================================
    # VIBE VALIDATION
    # ========================================================================
    
    vibe_min_words = reqs['vibe']['min_words']
    vibe_max_words = reqs['vibe']['max_words']
    vibe_required = reqs['vibe']['required']
    
    # Handle empty vibe
    if not data.vibe or not data.vibe.strip():
        if vibe_required:
            raise ValueError(
                f"[{tier}] vibe is required but empty"
            )
        # Empty vibe is OK for non-required tiers
        data.vibe = ""
    else:
        # Vibe is present - validate length
        vibe_word_count = len(data.vibe.split())
        
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
        if tier in ("MINIMAL", "BASIC") and data.vibe.strip():
            raise ValueError(
                f"[{tier}] vibe must be empty (got: '{data.vibe[:50]}...')"
            )
    
    # ========================================================================
    # ADDITIONAL QUALITY CHECKS
    # ========================================================================
    
    # Check for suspiciously similar subjects (only for higher tiers)
    if tier in ("RICH", "SPARSE") and len(data.subjects) > 1:
        _check_subject_uniqueness(data.subjects, tier)
    
    return data


def _check_subject_uniqueness(subjects: List[str], tier: str) -> None:
    """
    Check that subjects are sufficiently distinct.
    Warns about near-duplicates but doesn't fail validation.
    
    Args:
        subjects: List of subject strings
        tier: Quality tier (for error message)
        
    Raises:
        ValueError: If clear duplicates detected
    """
    # Convert to lowercase for comparison
    subjects_lower = [s.lower() for s in subjects]
    
    # Check for exact duplicates (after Pydantic cleaning, shouldn't happen)
    if len(subjects_lower) != len(set(subjects_lower)):
        raise ValueError(
            f"[{tier}] Duplicate subjects detected (case-insensitive)"
        )
    
    # Check for near-duplicates (e.g., "Greek gods" vs "Greek deities")
    # This is a soft check - we look for significant word overlap
    for i, subj1 in enumerate(subjects_lower):
        words1 = set(subj1.split())
        for j, subj2 in enumerate(subjects_lower[i+1:], start=i+1):
            words2 = set(subj2.split())
            
            # If subjects share >75% of words, they're too similar
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2)
                min_len = min(len(words1), len(words2))
                
                if overlap / min_len > 0.75:
                    raise ValueError(
                        f"[{tier}] Near-duplicate subjects: '{subjects[i]}' and '{subjects[j]}'. "
                        f"Choose more distinct subjects."
                    )


def validate_tier_consistency(
    tier: str,
    output: EnrichmentOut
) -> List[str]:
    """
    Check if output is consistent with tier expectations.
    Returns list of warnings (not errors).
    
    This is used for monitoring and quality assessment, not validation.
    
    Args:
        tier: Quality tier
        output: Validated enrichment output
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # RICH tier should have substantial output
    if tier == "RICH":
        if len(output.subjects) < 6:
            warnings.append(
                f"RICH tier book only has {len(output.subjects)} subjects (expected 6-8)"
            )
        if len(output.tone_ids) < 2:
            warnings.append(
                f"RICH tier book only has {len(output.tone_ids)} tones (expected 2-3)"
            )
        if not output.vibe or len(output.vibe.split()) < 8:
            warnings.append(
                f"RICH tier book has short/empty vibe (expected 8-12 words)"
            )
    
    # BASIC tier should have minimal output
    if tier == "BASIC":
        if len(output.subjects) > 0:
            warnings.append(
                f"BASIC tier book has {len(output.subjects)} subjects (expected 0-1)"
            )
        if len(output.tone_ids) > 0:
            warnings.append(
                f"BASIC tier book has tones (expected none)"
            )
    
    return warnings
