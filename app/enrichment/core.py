# app/enrichment/core.py
"""
Shared enrichment logic for both runner and backfill.
Provides tier-aware enrichment with automatic retry on validation failures.
"""
import csv
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.enrichment.prompts import SYSTEM, build_user_prompt, build_retry_prompt
from app.enrichment.postprocess import clean_subjects, render_tone_slugs, render_genre_slugs
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm
from app.enrichment.quality_classifier import assess_book_quality, get_tier_requirements

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
    
    logger.info(f"Loaded {len(rows)} tones from {ontology_version}: IDs {min(valid_ids)}-{max(valid_ids)}")
    
    return rows, slugs, valid_ids, slug2id


def load_genres() -> Tuple[list, str, set]:
    """
    Load genres from CSV.
    
    Returns:
        (rows, slugs_str, valid_slugs_set)
    """
    rows = []
    with open(GENRES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    
    slugs_line = render_genre_slugs(rows)
    valid_slugs = {r["slug"] for r in rows}
    
    logger.info(f"Loaded {len(rows)} genres")
    
    return rows, slugs_line, valid_slugs


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
        ol_subjects=rec.get("ol_subjects", [])
    )
    
    if not assessment.is_enrichable:
        return None, None, None
    
    return assessment.tier, assessment.score, assessment


def normalize_tone_ids(raw: dict, slug2id: dict) -> list[int]:
    """
    Normalize tone_ids (convert slugs to IDs if needed).
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


def extract_retry_feedback(
    error_msg: str,
    tier: str,
    raw_response: dict,
    score: float,
    genre_slugs_line: str
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
        error_type = "invalid_genre_slug"
        required_changes = _extract_genre_requirement(error_msg, tier, genre_slugs_line)
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
    min_w = reqs['vibe']['min_words']
    max_w = reqs['vibe']['max_words']
    
    if issue_type == "short":
        return f"Your vibe needs to be at least {min_w} words (maximum {max_w}). Expand it with more descriptive language."
    else:
        return f"Your vibe needs to be at most {max_w} words (minimum {min_w}). Make it more concise."


def _extract_subject_requirement(error_msg: str, tier: str) -> str:
    """Extract subject count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_s = reqs['subjects']['min']
    max_s = reqs['subjects']['max']
    
    if "below minimum" in error_msg.lower():
        return f"You need at least {min_s} subjects (maximum {max_s}). Add more relevant subjects."
    else:
        return f"You have too many subjects (maximum {max_s}). Remove the least important ones."


def _extract_tone_requirement(error_msg: str, tier: str) -> str:
    """Extract tone count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_t = reqs['tones']['min']
    max_t = reqs['tones']['max']
    
    if "below minimum" in error_msg.lower():
        return f"You need at least {min_t} tones (maximum {max_t}). Add more tones that fit."
    else:
        return f"You have too many tones (maximum {max_t}). Remove the least fitting ones."


def _extract_genre_requirement(error_msg: str, tier: str, genre_slugs_line: str) -> str:
    """Build retry feedback for invalid genre errors."""
    import re
    match = re.search(r"Invalid genre slug: '([^']+)'", error_msg)
    invalid_genre = match.group(1) if match else "your genre"
    
    return f"""The genre '{invalid_genre}' is not valid.

Valid genres: {genre_slugs_line}

Pick ONE exact slug from the list above (case-sensitive)."""


def enrich_one(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str,
    ontology_version: str = "v2",
    tags_version: str = "v2",
    retry_feedback: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Enrich a single book with optional retry feedback.
    
    Args:
        rec: Book record with title, author, description, ol_subjects
        slug2id: Tone slug to ID mapping
        valid_tone_ids: Valid tone IDs
        valid_genre_slugs: Valid genre slugs
        tone_slugs: Comma-separated tone slugs
        genre_slugs_line: Comma-separated genre slugs
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
        user_prompt = build_retry_prompt(
            title=rec["title"],
            author=rec["author"],
            description=rec["description"],
            ol_subjects=rec.get("ol_subjects", []),
            tier=tier,
            tone_slugs=tone_slugs,
            genre_slugs=genre_slugs_line,
            ontology_version=ontology_version,
            feedback=retry_feedback
        )
        logger.info(f"Retrying with feedback: {retry_feedback['error_type']}")
    else:
        user_prompt = build_user_prompt(
            title=rec["title"],
            author=rec["author"],
            description=rec["description"],
            ol_subjects=rec.get("ol_subjects", []),
            tier=tier,
            tone_slugs=tone_slugs,
            genre_slugs=genre_slugs_line,
            ontology_version=ontology_version
        )
    
    start_time = time.time()
    
    try:
        # ====================================================================
        # STEP 3: LLM CALL
        # ====================================================================
        
        raw = call_enrichment_llm(SYSTEM, user_prompt)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Normalize tone_ids (slug -> id)
        raw["tone_ids"] = normalize_tone_ids(raw, slug2id)
        
        # ====================================================================
        # STEP 4: TIER-AWARE VALIDATION
        # ====================================================================
        
        data = validate_payload(
            raw, 
            valid_tone_ids, 
            valid_genre_slugs,
            tier=tier,
            ontology_version=ontology_version
        )
        
        # Additional subject cleaning
        data.subjects = clean_subjects(data.subjects)
        
        # ====================================================================
        # STEP 5: BUILD RESULT
        # ====================================================================
        
        result = {
            "item_idx": rec["item_idx"],
            "subjects": data.subjects,
            "tone_ids": data.tone_ids,
            "genre": data.genre,
            "vibe": data.vibe,
            "tags_version": tags_version,
            "scores": {},
            "enrichment_quality": tier,
            "quality_score": score,
            "ontology_version": ontology_version,
            "metadata": {
                "latency_ms": latency_ms,
                "model": "unknown",  # Caller can override
                "quality_signals": assessment.signals.to_dict() if assessment else {},
                "retry_count": 1 if retry_feedback else 0,
            }
        }
        
        return result, None
        
    except ValueError as e:
        # Validation error - prepare for potential retry
        error_msg = str(e)
        
        logger.warning(f"âœ— Enrichment validation failed for item_idx={rec['item_idx']}: {error_msg}")

        error = {
            "stage": "validate",
            "error_code": "VALIDATION_FAILED",
            "error_msg": error_msg[:512],
            "error_field": None,
            "attempted": {
                "tier": tier,
                "score": score,
                "raw_response": raw if 'raw' in locals() else None,
                "error_type": None,
                "required_changes": None,
            }
        }
        
        # Try to extract retry feedback
        if 'raw' in locals():
            retry_info = extract_retry_feedback(
                error_msg, tier, raw, score, genre_slugs_line
            )
            if retry_info:
                error["attempted"]["error_type"] = retry_info["error_type"]
                error["attempted"]["required_changes"] = retry_info["required_changes"]
        
        return None, error
        
    except Exception as e:
        # LLM or parsing error
        error_str = str(e)
        if "timeout" in error_str.lower():
            stage, code = "llm_invoke", "TIMEOUT"
        elif "parse" in error_str.lower() or "json" in error_str.lower():
            stage, code = "llm_parse", "JSON_PARSE"
        else:
            stage, code = "llm_invoke", "LLM_ERROR"
        
        error = {
            "stage": stage,
            "error_code": code,
            "error_msg": error_str[:512],
            "attempted": {
                "tier": tier if 'tier' in locals() else "UNKNOWN",
                "score": score if 'score' in locals() else 0,
            }
        }
        
        return None, error


def enrich_with_retry(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str,
    ontology_version: str = "v2",
    tags_version: str = "v2"
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Enrich with automatic retry on validation failures.
    
    Returns:
        (result_dict, error_dict): One will be None
    """
    # First attempt
    result, error = enrich_one(
        rec, slug2id, valid_tone_ids, valid_genre_slugs,
        tone_slugs, genre_slugs_line, ontology_version, tags_version
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
            "vibe_too_short", "vibe_too_long", 
            "subject_count_wrong", "tone_count_wrong",
            "invalid_genre_slug"
        ]:
            logger.info(f"Validation failed ({error_type}), retrying with feedback...")
            
            # Build retry feedback
            retry_feedback = {
                "error_type": error_type,
                "error_msg": error["error_msg"],
                "original_response": raw_response,
                "tier": attempted["tier"],
                "score": attempted["score"],
                "required_changes": attempted["required_changes"],
            }
            
            # Retry with feedback
            result, retry_error = enrich_one(
                rec, slug2id, valid_tone_ids, valid_genre_slugs,
                tone_slugs, genre_slugs_line, ontology_version, tags_version,
                retry_feedback=retry_feedback
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
