# app/enrichment/runner_kafka.py
"""
Enhanced enrichment with retry feedback for validation failures.
"""
import csv, time, os, sys
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

# Generate unique run identifier
RUN_ID = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
logger.info(f"Run ID: {RUN_ID}")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.enrichment.prompts import SYSTEM, build_user_prompt, build_retry_prompt
from app.enrichment.postprocess import render_tone_slugs, render_genre_slugs, clean_subjects
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm, ensure_enrichment_ready
from app.enrichment.kafka_producer import EnrichmentProducer
from app.enrichment.quality_classifier import assess_book_quality, get_tier_requirements

from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TONES_CSV = ROOT / "ontology" / "tones_v2.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"

# Behavior flags
STOP_ON_FIRST_LLM_ERROR = os.getenv("ENRICH_STOP_ON_FIRST_LLM_ERROR", "0") == "1"

# Versioning
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2")
ONTOLOGY_VERSION = os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")


# ============================================================================
# ONTOLOGY LOADERS
# ============================================================================

def load_tones(ontology_version: str = "v2"):
    """
    Load tones from CSV for specified ontology version.
    
    Args:
        ontology_version: Which ontology version to load (v1 or v2)
    
    Returns:
        (rows, slugs_str, valid_ids_set, slug_to_id_map)
    """
    if ontology_version == "v2":
        csv_path = ROOT / "ontology" / "tones_v2.csv"
    else:
        csv_path = ROOT / "ontology" / "tones_v1.csv"
    
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    
    from app.enrichment.postprocess import render_tone_slugs
    slugs = render_tone_slugs(rows)
    valid_ids = {int(r["tone_id"]) for r in rows}
    slug2id = {r["slug"]: int(r["tone_id"]) for r in rows}
    
    logger.info(f"Loaded {len(rows)} tones from {ontology_version}: IDs {min(valid_ids)}-{max(valid_ids)}")
    
    return rows, slugs, valid_ids, slug2id


def load_genres():
    """Load genres from CSV"""
    rows = []
    with open(GENRES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    
    from app.enrichment.postprocess import render_genre_slugs
    slugs_line = render_genre_slugs(rows)
    valid_slugs = {r["slug"] for r in rows}
    
    logger.info(f"Loaded {len(rows)} genres")
    
    return rows, slugs_line, valid_slugs


# ============================================================================
# DATABASE QUERIES
# ============================================================================

def fetch_book_with_ol_subjects(db: Session, item_idx: int) -> Optional[Dict[str, Any]]:
    """
    Fetch book metadata including OL subjects.
    
    Args:
        db: Database session
        item_idx: Book item_idx
        
    Returns:
        Dict with book metadata or None if not found
    """
    # Fetch book and author
    result = db.query(Book, Author).outerjoin(
        Author, Book.author_idx == Author.author_idx
    ).filter(Book.item_idx == item_idx).first()
    
    if not result:
        return None
    
    book, author = result
    
    # Fetch OL subjects
    ol_subjects_query = db.query(OLSubject.subject).join(
        BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx
    ).filter(BookOLSubject.item_idx == item_idx).all()
    
    ol_subjects = [subj for (subj,) in ol_subjects_query]
    
    return {
        "item_idx": int(book.item_idx),
        "title": book.title or "",
        "author": author.name if author else "",
        "description": book.description or "",
        "ol_subjects": ol_subjects,
    }


def iter_books_from_db(db: Session, limit: int | None = None):
    """
    Query books with item_idx and OL subjects.
    
    Args:
        db: Database session
        limit: Maximum number of books to return
        
    Yields:
        Dict with book metadata
    """
    q = db.query(Book.item_idx).filter(Book.item_idx.isnot(None))
    
    if limit:
        q = q.limit(limit)
    
    for (item_idx,) in q:
        book_data = fetch_book_with_ol_subjects(db, item_idx)
        if book_data:
            yield book_data


def enrich_one(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str,
    ontology_version: str = "v2",
    retry_feedback: Optional[Dict[str, Any]] = None
) -> tuple[dict | None, dict | None]:
    """
    Enrich a single book with optional retry feedback.
    
    Args:
        rec: Book record
        slug2id: Tone slug to ID mapping
        valid_tone_ids: Valid tone IDs
        valid_genre_slugs: Valid genre slugs
        tone_slugs: Comma-separated tone slugs
        genre_slugs_line: Comma-separated genre slugs
        ontology_version: Ontology version
        retry_feedback: Optional feedback from previous attempt {
            "error_type": "vibe_too_short" | "validation_failed",
            "error_msg": "...",
            "original_response": {...},  # Original JSON from LLM
            "tier": "RICH",
            "required_changes": "..."
        }
    
    Returns:
        (result_dict, error_dict)
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
        assessment = assess_book_quality(
            title=rec["title"],
            author=rec["author"],
            description=rec["description"],
            ol_subjects=rec.get("ol_subjects", [])
        )
        
        if not assessment.is_enrichable:
            error = {
                "stage": "quality_assessment",
                "error_code": "INSUFFICIENT_METADATA",
                "error_msg": "Missing title or author - cannot enrich",
                "attempted": None,
            }
            return None, error
        
        tier = assessment.tier
        score = assessment.score
        
        logger.info(
            f"item_idx={rec['item_idx']}: {tier} tier (score={score})"
        )
    
    # ========================================================================
    # STEP 2: BUILD PROMPT (with retry feedback if present)
    # ========================================================================
    
    if retry_feedback:
        # Build retry prompt with feedback
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
        # Normal prompt
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
        tone_ids = raw.get("tone_ids", [])
        if not tone_ids or any(isinstance(t, str) for t in tone_ids):
            mapped = []
            for t in tone_ids:
                if isinstance(t, int):
                    mapped.append(t)
                elif isinstance(t, str) and t in slug2id:
                    mapped.append(slug2id[t])
            raw["tone_ids"] = mapped
        
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
            "tags_version": os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2"),
            "scores": {},
            "enrichment_quality": tier,
            "quality_score": score,
            "ontology_version": ontology_version,
            "metadata": {
                "latency_ms": latency_ms,
                "model": os.getenv("DEEPINFRA_MODEL", "unknown"),
                "quality_signals": assessment.signals.to_dict() if assessment else {},
                "retry_count": 1 if retry_feedback else 0,
            }
        }
        
        return result, None
        
    except ValueError as e:
        # Validation error - prepare for potential retry
        error_msg = str(e)
        
        # Determine error type
        if "vibe too short" in error_msg.lower():
            error_type = "vibe_too_short"
            required_changes = extract_vibe_requirement(error_msg, tier)
        elif "vibe too long" in error_msg.lower():
            error_type = "vibe_too_long"
            required_changes = extract_vibe_requirement(error_msg, tier)
        elif "subjects count" in error_msg.lower():
            error_type = "subject_count_wrong"
            required_changes = extract_subject_requirement(error_msg, tier)
        elif "tone_ids count" in error_msg.lower():
            error_type = "tone_count_wrong"
            required_changes = extract_tone_requirement(error_msg, tier)
        else:
            error_type = "validation_failed"
            required_changes = error_msg
        
        error = {
            "stage": "validate",
            "error_code": "VALIDATION_FAILED",
            "error_msg": error_msg[:512],
            "error_field": None,
            "attempted": {
                "tier": tier,
                "score": score,
                "raw_response": raw if 'raw' in locals() else None,
                "error_type": error_type,
                "required_changes": required_changes,
            }
        }
        
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
                "tier": tier,
                "score": score,
            }
        }
        
        return None, error


def extract_vibe_requirement(error_msg: str, tier: str) -> str:
    """Extract vibe length requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_w = reqs['vibe']['min_words']
    max_w = reqs['vibe']['max_words']
    
    if "too short" in error_msg.lower():
        return f"Your vibe needs to be at least {min_w} words (maximum {max_w}). Expand it with more descriptive language."
    else:
        return f"Your vibe needs to be at most {max_w} words (minimum {min_w}). Make it more concise."


def extract_subject_requirement(error_msg: str, tier: str) -> str:
    """Extract subject count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_s = reqs['subjects']['min']
    max_s = reqs['subjects']['max']
    
    if "below minimum" in error_msg.lower():
        return f"You need at least {min_s} subjects (maximum {max_s}). Add more relevant subjects."
    else:
        return f"You have too many subjects (maximum {max_s}). Remove the least important ones."


def extract_tone_requirement(error_msg: str, tier: str) -> str:
    """Extract tone count requirement from error message."""
    reqs = get_tier_requirements(tier)
    min_t = reqs['tones']['min']
    max_t = reqs['tones']['max']
    
    if "below minimum" in error_msg.lower():
        return f"You need at least {min_t} tones (maximum {max_t}). Add more tones that fit."
    else:
        return f"You have too many tones (maximum {max_t}). Remove the least fitting ones."


def enrich_with_retry(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str,
    ontology_version: str = "v2"
) -> tuple[dict | None, dict | None]:
    """
    Enrich with automatic retry on validation failures.
    
    Returns:
        (result_dict, error_dict): One will be None
    """
    # First attempt
    result, error = enrich_one(
        rec, slug2id, valid_tone_ids, valid_genre_slugs,
        tone_slugs, genre_slugs_line, ontology_version
    )
    
    if result:
        return result, None
    
    # Check if this is a retryable validation error
    if error and error.get("stage") == "validate":
        attempted = error.get("attempted", {})
        error_type = attempted.get("error_type")
        raw_response = attempted.get("raw_response")
        
        # Only retry if we have the original response
        if raw_response and error_type in ["vibe_too_short", "vibe_too_long", 
                                            "subject_count_wrong", "tone_count_wrong"]:
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
                tone_slugs, genre_slugs_line, ontology_version,
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


# Update main() to use enrich_with_retry instead of enrich_one
def main(limit: int | None = None, sleep_s: float = 0.0, workers: int = 1):
    """Main enrichment runner with retry logic."""
    ensure_enrichment_ready()
    
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones(
        os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")
    )
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    producer = EnrichmentProducer(enable_kafka=True)
    
    tier_stats = {
        "RICH": {"ok": 0, "err": 0, "retry_success": 0},
        "SPARSE": {"ok": 0, "err": 0, "retry_success": 0},
        "MINIMAL": {"ok": 0, "err": 0, "retry_success": 0},
        "BASIC": {"ok": 0, "err": 0, "retry_success": 0},
    }
    count_ok, count_err = 0, 0
    start_time = time.time()
    
    try:
        with SessionLocal() as db:
            for rec in iter_books_from_db(db, limit):
                # Use retry-enabled enrichment
                result, error = enrich_with_retry(
                    rec, slug2id, valid_tone_ids, valid_genre_slugs,
                    tone_slugs, genre_slugs_line,
                    os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")
                )
                
                if result:
                    tier = result["enrichment_quality"]
                    retry_count = result.get("metadata", {}).get("retry_count", 0)
                    
                    if retry_count > 0:
                        tier_stats[tier]["retry_success"] += 1
                    
                    success = producer.send_result(
                        item_idx=result["item_idx"],
                        subjects=result["subjects"],
                        tone_ids=result["tone_ids"],
                        genre=result["genre"],
                        vibe=result["vibe"],
                        tags_version=result["tags_version"],
                        scores=result["scores"],
                        metadata=result.get("metadata", {})
                    )
                    
                    if success:
                        count_ok += 1
                        tier_stats[tier]["ok"] += 1
                        # Compact one-liner
                        retry_msg = " (retry)" if retry_count > 0 else ""
                        logger.info(f"✓ item_idx={result['item_idx']} ({tier}){retry_msg}")
                    else:
                        count_err += 1
                        tier_stats[tier]["err"] += 1
                        logger.error(f"✗ Kafka send failed: item_idx={result['item_idx']}")
                    
                    if sleep_s:
                        time.sleep(sleep_s)

                else:
                    # Both attempts failed
                    attempted = error.get("attempted", {})
                    tier = attempted.get("tier", "UNKNOWN")
                    
					producer.send_error(
						item_idx=rec["item_idx"],
						error_msg=error["error_msg"],
						stage=error["stage"],
						error_code=error["error_code"],
						error_field=error.get("error_field"),
						title=rec["title"][:256],
						author=rec["author"][:256],
						tags_version=os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2"),
						attempted=attempted,
						run_metadata={
							"run_id": RUN_ID,  
							"model": os.getenv("DEEPINFRA_MODEL", "unknown"),
							"timestamp": int(time.time() * 1000),
						}
					)
                    
                    count_err += 1
                    if tier in tier_stats:
                        tier_stats[tier]["err"] += 1
                    
                    # Compact one-liner with brief error
                    logger.error(f"✗ item_idx={rec['item_idx']} ({tier}): {error['error_code']} - {error['error_msg'][:80]}")

                # Progress indicator (keep this as-is)
                total = count_ok + count_err
                if total % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {total} ({count_ok} ok, {count_err} err) | {rate:.1f}/s")   

    finally:
        producer.flush()
        producer.close()
    
    elapsed = time.time() - start_time
    
    return {
        "ok": count_ok,
        "error": count_err,
        "elapsed_s": elapsed,
        "tier_stats": tier_stats
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrichment runner with quality tiering and multithreading")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of books to enrich (default: all)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep duration between enrichments in seconds (for rate limiting)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker threads (default: 2)"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Tags version (overrides ENRICHMENT_JOB_TAG_VERSION env var)"
    )
    parser.add_argument(
        "--ontology",
        default="v2",
        help="Ontology version (v1 or v2, default: v2)"
    )
    
    args = parser.parse_args()
    
    # Override version if specified
    if args.version:
        os.environ["ENRICHMENT_JOB_TAG_VERSION"] = args.version
        VERSION_TAG = args.version
    
    if args.ontology:
        os.environ["ENRICHMENT_ONTOLOGY_VERSION"] = args.ontology
        ONTOLOGY_VERSION = args.ontology
    
    logger.info("="*70)
    logger.info("ENRICHMENT RUNNER (QUALITY TIERING - MULTITHREADED)")
    logger.info("="*70)
    logger.info(f"Tags Version: {VERSION_TAG}")
    logger.info(f"Ontology Version: {ONTOLOGY_VERSION}")
    logger.info(f"Limit: {args.limit or 'unlimited'}")
    logger.info(f"Sleep: {args.sleep}s")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Stop on LLM error: {STOP_ON_FIRST_LLM_ERROR}")
    logger.info("="*70)
    
    res = main(limit=args.limit, sleep_s=args.sleep, workers=args.workers)
    
    logger.info("="*70)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"Success: {res['ok']}")
    logger.info(f"Errors: {res['error']}")
    logger.info(f"Elapsed: {res.get('elapsed_s', 0):.1f}s")
    
    # Show tier breakdown
    logger.info("\nTier Breakdown:")
    for tier, stats in res.get('tier_stats', {}).items():
        total = stats['ok'] + stats['err']
        if total > 0:
            success_rate = stats['ok'] / total * 100
            logger.info(f"  {tier:12} {stats['ok']:>6} ok, {stats['err']:>6} err ({success_rate:.1f}% success)")
    
    if res.get("aborted"):
        logger.info("Status: ABORTED")
    logger.info("="*70)
    
    # Exit code
    code = 0
    if res.get("aborted") or (res.get("ok", 0) == 0 and res.get("error", 0) > 0):
        code = 2
    
    sys.exit(code)
