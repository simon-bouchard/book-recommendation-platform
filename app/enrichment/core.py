# app/enrichment/core.py
"""
Shared enrichment logic for both runner and backfill.
Provides tier-aware enrichment with automatic retry on validation failures.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.enrichment.llm_client import call_enrichment_llm
from app.enrichment.postprocess import clean_subjects, render_genre_slugs, render_tone_slugs
from app.enrichment.prompts import SYSTEM, build_retry_prompt, build_user_prompt
from app.enrichment.quality_classifier import assess_book_quality, get_tier_requirements
from app.enrichment.validator import validate_payload

logger = logging.getLogger(__name__)

# Ontology paths
ROOT = Path(__file__).resolve().parents[2]
TONES_V1_CSV = ROOT / "ontology" / "tones_v1.csv"
TONES_V2_CSV = ROOT / "ontology" / "tones_v2.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"


# ============================================================================
# ONTOLOGY LOADERS
# ============================================================================


def load_tones(ontology_version: str = "v2") -> Tuple[list, str, set, dict]:
    """
    Load tones from CSV for specified ontology version.

    Args:
        ontology_version: Which ontology version to load (v1 or v2)

    Returns:
        (rows, slugs_str, valid_ids_set, slug_to_id_map)
    """
    csv_path = TONES_V2_CSV if ontology_version == "v2" else TONES_V1_CSV

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    slugs = render_tone_slugs(rows)
    valid_ids = {int(r["tone_id"]) for r in rows}
    slug2id = {r["slug"]: int(r["tone_id"]) for r in rows}

    logger.info(
        f"Loaded {len(rows)} tones from {ontology_version}: IDs {min(valid_ids)}-{max(valid_ids)}"
    )

    return rows, slugs, valid_ids, slug2id


def load_genres() -> Tuple[list, str, set, dict, dict]:
    """
    Load genres from CSV.

    Returns:
        (rows, id_slugs_str, valid_ids_set, id_to_slug_map, slug_to_id_map)
    """
    rows = []
    with open(GENRES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Render with IDs: "1=fantasy, 2=science-fiction, ..."
    id_slugs_line = render_genre_slugs(rows)

    # Build mappings
    valid_ids = {int(r["genre_idx"]) for r in rows}
    id_to_slug = {int(r["genre_idx"]): r["slug"] for r in rows}
    slug_to_id = {r["slug"]: int(r["genre_idx"]) for r in rows}

    logger.info(f"Loaded {len(rows)} genres")

    return rows, id_slugs_line, valid_ids, id_to_slug, slug_to_id


# ============================================================================
# QUALITY ASSESSMENT
# ============================================================================


def assess_quality(rec: dict) -> Tuple[Optional[str], Optional[float], Optional[Any]]:
    """
    Assess book quality tier.

    Returns:
        (tier, score, assessment) or (None, None, None) if not enrichable
    """
    assessment = assess_book_quality(
        title=rec["title"],
        author=rec["author"],
        description=rec["description"],
        ol_subjects=rec.get("ol_subjects", []),
    )

    if not assessment.is_enrichable:
        return None, None, None

    return assessment.tier, assessment.score, assessment


def normalize_tone_ids(raw: dict, slug2id: dict) -> list[int]:
    """
    Normalize tone_ids (convert slugs to IDs if needed for backward compatibility).
    """
    tone_ids = raw.get("tone_ids", [])
    if not tone_ids or any(isinstance(t, str) for t in tone_ids):
        mapped = []
        for t in tone_ids:
            if isinstance(t, int):
                mapped.append(t)
            elif isinstance(t, str) and t in slug2id:
                mapped.append(slug2id[t])
        return mapped
    return tone_ids


def normalize_genre(raw: dict, id_to_slug: dict, slug_to_id: dict) -> Tuple[int, str]:
    """
    Map genre between ID and slug formats (no validation - validator handles that).

    Args:
        raw: LLM response dict
        id_to_slug: genre_id -> slug mapping
        slug_to_id: slug -> genre_id mapping

    Returns:
        (genre_id, genre_slug) tuple

    Raises:
        ValueError only for missing fields or type errors
    """
    # Try new format first (genre_id)
    if "genre_id" in raw:
        genre_id = raw["genre_id"]
        if not isinstance(genre_id, int):
            raise ValueError(f"genre_id must be integer, got {type(genre_id)}")

        # Map to slug (validator will check if ID is valid)
        genre_slug = id_to_slug.get(genre_id, f"invalid-{genre_id}")
        return genre_id, genre_slug

    # Fall back to legacy format (genre slug)
    elif "genre" in raw:
        genre_slug = raw["genre"]
        if not isinstance(genre_slug, str):
            raise ValueError(f"genre must be string, got {type(genre_slug)}")

        # Map to ID (validator will check if slug is valid)
        genre_id = slug_to_id.get(genre_slug, 0)
        return genre_id, genre_slug

    else:
        raise ValueError("Missing both genre_id and genre fields")


def extract_retry_feedback(
    error_msg: str, tier: str, raw_response: dict, score: float, genre_id_slugs_line: str
) -> Optional[Dict[str, Any]]:
    """
    Determine if error is retryable and build feedback.

    Returns:
        Retry feedback dict or None if not retryable
    """
    error_msg_lower = error_msg.lower()

    # Determine error type and required changes
    if "vibe too short" in error_msg_lower:
        error_type = "vibe_too_short"
        required_changes = _extract_vibe_requirement(error_msg, tier, "short")
    elif "vibe too long" in error_msg_lower:
        error_type = "vibe_too_long"
        required_changes = _extract_vibe_requirement(error_msg, tier, "long")
    elif "subjects count" in error_msg_lower:
        error_type = "subject_count_wrong"
        required_changes = _extract_subject_requirement(error_msg, tier)
    elif "tone_ids count" in error_msg_lower:
        error_type = "tone_count_wrong"
        required_changes = _extract_tone_requirement(error_msg, tier)
    elif "invalid genre" in error_msg_lower:
        error_type = "invalid_genre_id"
        required_changes = _extract_genre_requirement(error_msg, tier, genre_id_slugs_line)
    else:
        # Not retryable
        return None

    return {
        "error_type": error_type,
        "error_msg": error_msg,
        "original_response": raw_response,
        "tier": tier,
        "score": score,
        "required_changes": required_changes,
    }


def _extract_vibe_requirement(error_msg: str, tier: str, issue_type: str) -> str:
    """Extract vibe length requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_w = reqs["vibe"]["min_words"]
    max_w = reqs["vibe"]["max_words"]

    if issue_type == "short":
        return f"Your vibe needs to be at least {min_w} words (maximum {max_w}). Expand it with more descriptive language."
    else:
        return (
            f"Your vibe needs to be at most {max_w} words (minimum {min_w}). Make it more concise."
        )


def _extract_subject_requirement(error_msg: str, tier: str) -> str:
    """Extract subject count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_s = reqs["subjects"]["min"]
    max_s = reqs["subjects"]["max"]

    if "below minimum" in error_msg.lower():
        return f"You need at least {min_s} subjects (maximum {max_s}). Add more relevant subjects."
    else:
        return f"You have too many subjects (maximum {max_s}). Remove the least important ones."


def _extract_tone_requirement(error_msg: str, tier: str) -> str:
    """Extract tone count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_t = reqs["tones"]["min"]
    max_t = reqs["tones"]["max"]

    if "below minimum" in error_msg.lower():
        return f"You need at least {min_t} tones (maximum {max_t}). Add more tones that fit."
    else:
        return f"You have too many tones (maximum {max_t}). Remove the least fitting ones."


def _extract_genre_requirement(error_msg: str, tier: str, genre_id_slugs_line: str) -> str:
    """Build retry feedback for invalid genre errors."""
    import re

    # Try to extract the invalid ID
    match = re.search(r"Invalid genre_id: (\d+)", error_msg)
    if match:
        invalid_id = match.group(1)
        return f"""The genre_id {invalid_id} is not valid.

Valid genres: {genre_id_slugs_line}

Pick ONE exact ID from the list above."""
    else:
        return f"""The genre you selected is not valid.

Valid genres: {genre_id_slugs_line}

Pick ONE exact ID from the list above."""


def enrich_one(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_ids: set,
    genre_id_to_slug: dict,
    genre_slug_to_id: dict,
    tone_slugs: str,
    genre_id_slugs_line: str,
    ontology_version: str = "v2",
    tags_version: str = "v2",
    retry_feedback: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Enrich a single book with optional retry feedback.

    Args:
        rec: Book record with title, author, description, ol_subjects
        slug2id: Tone slug to ID mapping
        valid_tone_ids: Valid tone IDs
        valid_genre_ids: Valid genre IDs
        genre_id_to_slug: Genre ID to slug mapping
        genre_slug_to_id: Genre slug to ID mapping (for legacy support)
        tone_slugs: Comma-separated tone ID=slug pairs
        genre_id_slugs_line: Comma-separated genre ID=slug pairs
        ontology_version: Ontology version
        tags_version: Tags version for output
        retry_feedback: Optional feedback from previous attempt

    Returns:
        (result_dict, error_dict): One will be None
    """
    # ========================================================================
    # STEP 1: ASSESS QUALITY (skip if retrying)
    # ========================================================================

    if retry_feedback:
        # Use tier from original attempt
        tier = retry_feedback["tier"]
        score = retry_feedback.get("score", 0)
        assessment = None  # Already assessed
    else:
        tier, score, assessment = assess_quality(rec)

        if tier is None:
            error = {
                "stage": "quality_assessment",
                "error_code": "INSUFFICIENT_METADATA",
                "error_msg": "Missing title or author - cannot enrich",
                "attempted": None,
            }
            return None, error

        logger.info(f"item_idx={rec['item_idx']}: {tier} tier (score={score})")

    # ========================================================================
    # STEP 2: BUILD PROMPT (with retry feedback if present)
    # ========================================================================

    if retry_feedback:
        prompt = build_retry_prompt(
            title=rec["title"],
            author=rec["author"],
            description=rec["description"],
            ol_subjects=rec.get("ol_subjects", []),
            tier=tier,
            tone_slugs=tone_slugs,
            genre_id_slugs=genre_id_slugs_line,
            ontology_version=ontology_version,
            feedback=retry_feedback,
        )
    else:
        prompt = build_user_prompt(
            title=rec["title"],
            author=rec["author"],
            description=rec["description"],
            ol_subjects=rec.get("ol_subjects", []),
            tier=tier,
            tone_slugs=tone_slugs,
            genre_id_slugs=genre_id_slugs_line,
            ontology_version=ontology_version,
        )

    # ========================================================================
    # STEP 3: CALL LLM
    # ========================================================================

    try:
        raw, usage, latency_ms = call_enrichment_llm(SYSTEM, prompt)
    except Exception as e:
        error = {
            "stage": "llm_invoke",
            "error_code": "LLM_ERROR",
            "error_msg": str(e),
            "attempted": None,
        }
        return None, error

    # ========================================================================
    # STEP 4: NORMALIZE & MAP GENRE
    # ========================================================================

    try:
        # Normalize tone_ids (backward compatibility for slugs)
        raw["tone_ids"] = normalize_tone_ids(raw, slug2id)

        # Normalize and map genre: genre_id → genre_slug
        genre_id, genre_slug = normalize_genre(raw, genre_id_to_slug, genre_slug_to_id)

        # Store both for validation context, but database uses slug
        raw["genre_id"] = genre_id
        raw["genre"] = genre_slug

    except Exception as e:
        error = {
            "stage": "postprocess",
            "error_code": "NORMALIZATION_ERROR",
            "error_msg": str(e),
            "attempted": {
                "raw_response": raw,
                "tier": tier,
                "score": score,
            },
        }
        return None, error

    # ========================================================================
    # STEP 5: VALIDATE
    # ========================================================================

    try:
        validated = validate_payload(
            raw,
            valid_tone_ids=valid_tone_ids,
            valid_genre_ids=valid_genre_ids,  # Now validates against IDs
            tier=tier,
            ontology_version=ontology_version,
        )
    except ValueError as e:
        error = {
            "stage": "validate",
            "error_code": "VALIDATION_ERROR",
            "error_msg": str(e),
            "attempted": {
                "raw_response": raw,
                "tier": tier,
                "score": score,
                "error_type": _classify_validation_error(str(e)),
                "required_changes": str(e),  # Placeholder, will be enriched by retry logic
            },
        }
        return None, error

    # ========================================================================
    # STEP 6: BUILD RESULT (store slug in database, not ID)
    # ========================================================================

    result = {
        "item_idx": rec["item_idx"],
        "subjects": clean_subjects(validated.subjects),
        "tone_ids": validated.tone_ids,
        "genre": validated.genre,  # This is the slug for database storage
        "vibe": validated.vibe,
        "tags_version": tags_version,
        "enrichment_quality": tier,  # For tier stats tracking
        "scores": {
            "quality_score": score,
            "tier": tier,
        },
    }

    return result, None


def _classify_validation_error(error_msg: str) -> str:
    """Quick classification of validation error type."""
    error_lower = error_msg.lower()
    if "vibe too short" in error_lower:
        return "vibe_too_short"
    elif "vibe too long" in error_lower:
        return "vibe_too_long"
    elif "subjects count" in error_lower:
        return "subject_count_wrong"
    elif "tone_ids count" in error_lower:
        return "tone_count_wrong"
    elif "invalid genre" in error_lower:
        return "invalid_genre_id"
    else:
        return "validation_error"


def enrich_with_retry(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_ids: set,
    genre_id_to_slug: dict,
    genre_slug_to_id: dict,
    tone_slugs: str,
    genre_id_slugs_line: str,
    ontology_version: str = "v2",
    tags_version: str = "v2",
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Enrich with automatic retry on validation failures.

    Returns:
        (result_dict, error_dict): One will be None
    """
    # First attempt
    result, error = enrich_one(
        rec,
        slug2id,
        valid_tone_ids,
        valid_genre_ids,
        genre_id_to_slug,
        genre_slug_to_id,
        tone_slugs,
        genre_id_slugs_line,
        ontology_version,
        tags_version,
    )

    if result:
        return result, None

    # Check if this is a retryable validation error
    if error and error.get("stage") == "validate":
        attempted = error.get("attempted", {})
        error_type = attempted.get("error_type")
        raw_response = attempted.get("raw_response")

        # Only retry if we have the original response
        if raw_response and error_type in [
            "vibe_too_short",
            "vibe_too_long",
            "subject_count_wrong",
            "tone_count_wrong",
            "invalid_genre_id",
        ]:
            logger.info(f"Validation failed ({error_type}), retrying with feedback...")

            # Build retry feedback
            retry_feedback = extract_retry_feedback(
                error_msg=error["error_msg"],
                tier=attempted["tier"],
                raw_response=raw_response,
                score=attempted["score"],
                genre_id_slugs_line=genre_id_slugs_line,
            )

            if retry_feedback:
                # Retry with feedback
                result, retry_error = enrich_one(
                    rec,
                    slug2id,
                    valid_tone_ids,
                    valid_genre_ids,
                    genre_id_to_slug,
                    genre_slug_to_id,
                    tone_slugs,
                    genre_id_slugs_line,
                    ontology_version,
                    tags_version,
                    retry_feedback=retry_feedback,
                )

                if result:
                    logger.info(f"✓ Retry succeeded for item_idx={rec['item_idx']}")
                    return result, None
                else:
                    logger.warning(f"✗ Retry also failed for item_idx={rec['item_idx']}")
                    # Return the retry error (more informative)
                    return None, retry_error

    # Not retryable or retry failed
    return None, error
