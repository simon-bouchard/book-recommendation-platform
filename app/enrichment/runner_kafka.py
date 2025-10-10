# app/enrichment/runner_kafka.py - COMPLETE MULTITHREADED VERSION

import csv, time, os, sys
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.enrichment.prompts import SYSTEM, build_user_prompt, render_tone_instructions, render_genre_instructions
from app.enrichment.postprocess import render_tone_slugs, render_genre_slugs, clean_subjects
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm, ensure_enrichment_ready
from app.enrichment.kafka_producer import EnrichmentProducer
from app.enrichment.quality_classifier import assess_book_quality

from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TONES_CSV = ROOT / "ontology" / "tones_v2.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"

STOP_ON_FIRST_LLM_ERROR = os.getenv("ENRICH_STOP_ON_FIRST_LLM_ERROR", "0") == "1"
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2")
ONTOLOGY_VERSION = os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")


def load_tones(ontology_version: str = "v2"):
    """Load tones from CSV for specified ontology version."""
    if ontology_version == "v2":
        csv_path = ROOT / "ontology" / "tones_v2.csv"
    else:
        csv_path = ROOT / "ontology" / "tones_v1.csv"
    
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    
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
    
    slugs_line = render_genre_slugs(rows)
    valid_slugs = {r["slug"] for r in rows}
    
    logger.info(f"Loaded {len(rows)} genres")
    
    return rows, slugs_line, valid_slugs


def fetch_book_with_ol_subjects(db: Session, item_idx: int) -> Optional[Dict[str, Any]]:
    """Fetch book metadata including OL subjects."""
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
    """Query books with item_idx and OL subjects."""
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
    ontology_version: str = "v2"
) -> tuple[dict | None, dict | None]:
    """Enrich a single book record with quality tiering."""
    # Assessment
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
        logger.warning(f"Skipping item_idx={rec['item_idx']}: INSUFFICIENT metadata")
        return None, error
    
    tier = assessment.tier
    score = assessment.score
    
    logger.info(
        f"item_idx={rec['item_idx']}: {tier} tier (score={score}, "
        f"desc_words={assessment.signals.desc_words}, "
        f"ol_subjects={assessment.signals.ol_subject_count})"
    )
    
    # Build prompt
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
        # LLM call
        raw = call_enrichment_llm(SYSTEM, user_prompt)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Normalize tone_ids
        tone_ids = raw.get("tone_ids", [])
        if not tone_ids or any(isinstance(t, str) for t in tone_ids):
            mapped = []
            for t in tone_ids:
                if isinstance(t, int):
                    mapped.append(t)
                elif isinstance(t, str) and t in slug2id:
                    mapped.append(slug2id[t])
            raw["tone_ids"] = mapped
        
        # Validate
        data = validate_payload(
            raw, 
            valid_tone_ids, 
            valid_genre_slugs,
            tier=tier,
            ontology_version=ontology_version
        )
        
        data.subjects = clean_subjects(data.subjects)
        
        # Build result
        result = {
            "item_idx": rec["item_idx"],
            "subjects": data.subjects,
            "tone_ids": data.tone_ids,
            "genre": data.genre,
            "vibe": data.vibe,
            "tags_version": VERSION_TAG,
            "scores": {},
            "enrichment_quality": tier,
            "quality_score": score,
            "ontology_version": ontology_version,
            "metadata": {
                "latency_ms": latency_ms,
                "model": os.getenv("DEEPINFRA_MODEL", "unknown"),
                "quality_signals": assessment.signals.to_dict(),
            }
        }
        
        return result, None
        
    except ValueError as e:
        error = {
            "stage": "validate",
            "error_code": "VALIDATION_FAILED",
            "error_msg": str(e)[:512],
            "error_field": None,
            "attempted": {
                "tier": tier,
                "score": score,
				"raw_subjects": raw.get("subjects", []),      # All subjects, not just 3
                "raw_tone_ids": raw.get("tone_ids", []),      # All tones
                "raw_genre": raw.get("genre", ""),            # Genre
                "raw_vibe": raw.get("vibe", ""),
            }
        }
        logger.warning(f"Validation failed for item_idx={rec['item_idx']}: {e}")
        return None, error
        
    except Exception as e:
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
        logger.error(f"LLM error for item_idx={rec['item_idx']}: {e}")
        return None, error


def main(limit: int | None = None, sleep_s: float = 0.0, workers: int = 2):
    """
    Main enrichment runner with Kafka support, quality tiering, and multithreading.
    
    Args:
        limit: Maximum number of books to enrich
        sleep_s: Sleep duration between enrichments (for rate limiting)
        workers: Number of parallel worker threads
    """
    # Fail fast
    ensure_enrichment_ready()
    
    # Load ontology (shared, read-only - thread-safe)
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones(ONTOLOGY_VERSION)
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    # Initialize Kafka producer (thread-safe)
    producer = EnrichmentProducer(enable_kafka=True)
    
    # Collect books upfront (avoid DB session sharing issues)
    with SessionLocal() as db:
        books_list = list(iter_books_from_db(db, limit))
    
    logger.info(f"Found {len(books_list)} books to enrich")
    logger.info(f"Using {workers} worker threads")
    
    # Thread-safe counters and stats
    stats_lock = Lock()
    tier_stats = {
        "RICH": {"ok": 0, "err": 0},
        "SPARSE": {"ok": 0, "err": 0},
        "MINIMAL": {"ok": 0, "err": 0},
        "BASIC": {"ok": 0, "err": 0},
        "INSUFFICIENT": {"ok": 0, "err": 0},
    }
    count_ok = 0
    count_err = 0
    start_time = time.time()
    should_abort = False  # For STOP_ON_FIRST_LLM_ERROR
    
    def process_book(rec: dict) -> tuple[str, str, Optional[str]]:
        """
        Process a single book (thread worker function).
        
        Returns:
            (status, tier, error_stage) where status is "ok" or "err"
        """
        nonlocal should_abort
        
        # Check abort flag
        if should_abort:
            return ("aborted", "UNKNOWN", None)
        
        # First attempt
        result, error = enrich_one(
            rec, slug2id, valid_tone_ids, valid_genre_slugs,
            tone_slugs, genre_slugs_line, ONTOLOGY_VERSION
        )
        
        # Determine tier for stats
        if error and error.get("attempted"):
            tier = error["attempted"].get("tier", "UNKNOWN")
        elif result:
            tier = result["enrichment_quality"]
        else:
            tier = "UNKNOWN"
        
        if result:
            # Success - send to results topic
            success = producer.send_result(
                item_idx=result["item_idx"],
                subjects=result["subjects"],
                tone_ids=result["tone_ids"],
                genre=result["genre"],
                vibe=result["vibe"],
                tags_version=result["tags_version"],
                scores=result["scores"],
                metadata={
                    **result.get("metadata", {}),
                    "enrichment_quality": result["enrichment_quality"],
                    "quality_score": result["quality_score"],
                    "ontology_version": result["ontology_version"],
                }
            )
            
            if success:
                logger.info(f"✓ Enriched item_idx={result['item_idx']} ({tier} tier)")
                if sleep_s:
                    time.sleep(sleep_s)
                return ("ok", tier, None)
            else:
                logger.warning(f"Failed to send result for item_idx={result['item_idx']}")
                return ("err", tier, "produce")
        
        # First attempt failed - retry once
        logger.warning(f"First attempt failed for item_idx={rec['item_idx']}, retrying...")
        result, error = enrich_one(
            rec, slug2id, valid_tone_ids, valid_genre_slugs,
            tone_slugs, genre_slugs_line, ONTOLOGY_VERSION
        )
        
        if result:
            success = producer.send_result(
                item_idx=result["item_idx"],
                subjects=result["subjects"],
                tone_ids=result["tone_ids"],
                genre=result["genre"],
                vibe=result["vibe"],
                tags_version=result["tags_version"],
                scores=result["scores"],
                metadata={
                    **result.get("metadata", {}),
                    "enrichment_quality": result["enrichment_quality"],
                    "quality_score": result["quality_score"],
                    "ontology_version": result["ontology_version"],
                }
            )
            
            if success:
                logger.info(f"✓ Enriched item_idx={result['item_idx']} (retry, {tier} tier)")
                return ("ok", tier, None)
            else:
                return ("err", tier, "produce")
        else:
            # Both attempts failed - send to DLQ
            producer.send_error(
                item_idx=rec["item_idx"],
                error_msg=error["error_msg"],
                stage=error["stage"],
                error_code=error["error_code"],
                error_field=error.get("error_field"),
                title=rec["title"][:256],
                author=rec["author"][:256],
                tags_version=VERSION_TAG,
                attempted=error.get("attempted"),
            )
            logger.error(f"✗ Failed item_idx={rec['item_idx']}: {error['error_msg']}")
            
            # Check if we should abort
            if STOP_ON_FIRST_LLM_ERROR and error["stage"] in ("llm_invoke", "llm_parse"):
                should_abort = True
                logger.error("Setting abort flag due to LLM error")
            
            return ("err", tier, error["stage"])
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_book, rec): rec for rec in books_list}
            
            # Process results as they complete
            for future in as_completed(futures):
                rec = futures[future]
                
                try:
                    status, tier, error_stage = future.result()
                    
                    # Update stats (thread-safe)
                    with stats_lock:
                        if status == "ok":
                            count_ok += 1
                            if tier in tier_stats:
                                tier_stats[tier]["ok"] += 1
                        elif status == "err":
                            count_err += 1
                            if tier in tier_stats:
                                tier_stats[tier]["err"] += 1
                        elif status == "aborted":
                            # Don't count aborted tasks
                            pass
                        
                        # Progress indicator (every 10 books)
                        total = count_ok + count_err
                        if total % 10 == 0 and total > 0:
                            elapsed = time.time() - start_time
                            rate = total / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"Progress: {total}/{len(books_list)} "
                                f"({count_ok} ✓, {count_err} ✗) | {rate:.1f}/s"
                            )
                        
                        # Check if we should abort remaining tasks
                        if should_abort:
                            logger.warning("Aborting remaining tasks due to LLM error")
                            # Cancel pending futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                
                except Exception as e:
                    logger.error(f"Task exception for item_idx={rec['item_idx']}: {e}")
                    with stats_lock:
                        count_err += 1
    
    finally:
        # Ensure all messages are sent before exiting
        logger.info("Flushing Kafka producer...")
        producer.flush()
        producer.close()
    
    elapsed = time.time() - start_time
    
    return {
        "ok": count_ok,
        "error": count_err,
        "aborted": should_abort,
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
