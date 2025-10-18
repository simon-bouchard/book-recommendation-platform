# app/enrichment/runner_kafka.py
"""
Main enrichment runner - processes ALL books from database.
Uses shared core logic from app.enrichment.core
"""
import time
import os
import sys
import logging
import uuid
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Shared enrichment logic
from app.enrichment.core import enrich_with_retry, load_tones, load_genres

# Infrastructure
from app.enrichment.llm_client import ensure_enrichment_ready
from app.enrichment.kafka_producer import EnrichmentProducer

# Database
from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate unique run identifier
RUN_ID = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
logger.info(f"Run ID: {RUN_ID}")

# Versioning
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2")
ONTOLOGY_VERSION = os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")


# ============================================================================
# DATABASE QUERIES (runner-specific: query ALL books)
# ============================================================================

def fetch_book_with_ol_subjects(db: Session, item_idx: int) -> Optional[Dict[str, Any]]:
    """
    Fetch book metadata including OL subjects.
    """
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


def iter_books_from_db(db: Session, limit: Optional[int] = None):
    """
    Query ALL books from database (main runner behavior).
    
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


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main(limit: Optional[int] = None, sleep_s: float = 0.0, workers: int = 1):
    """Main enrichment runner with retry logic."""
    ensure_enrichment_ready()
    
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones(ONTOLOGY_VERSION)
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
                # Use shared retry-enabled enrichment
                result, error = enrich_with_retry(
                    rec, slug2id, valid_tone_ids, valid_genre_slugs,
                    tone_slugs, genre_slugs_line,
                    ontology_version=ONTOLOGY_VERSION,
                    tags_version=VERSION_TAG
                )
                
                if result:
                    # Add run metadata
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"]["run_id"] = RUN_ID
                    result["metadata"]["model"] = os.getenv("DEEPINFRA_MODEL", "unknown")
                    
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
                        attempted=error.get("attempted"),
                        tags_version=VERSION_TAG,
                        run_id=RUN_ID,
                    )
                    
                    count_err += 1
                    if tier in tier_stats:
                        tier_stats[tier]["err"] += 1
                    
                    logger.error(
                        f"✗ item_idx={rec['item_idx']} ({tier}): "
                        f"{error['error_code']} - {error['error_msg'][:80]}"
                    )
                
                # Progress indicator
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
    
    parser = argparse.ArgumentParser(description="Enrichment runner with quality tiering")
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
        help="Sleep duration between enrichments in seconds"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker threads (not yet implemented)"
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
    
    # Override versions if specified
    if args.version:
        os.environ["ENRICHMENT_JOB_TAG_VERSION"] = args.version
        VERSION_TAG = args.version
    
    if args.ontology:
        os.environ["ENRICHMENT_ONTOLOGY_VERSION"] = args.ontology
        ONTOLOGY_VERSION = args.ontology
    
    logger.info("="*70)
    logger.info("ENRICHMENT RUNNER (QUALITY TIERING)")
    logger.info("="*70)
    logger.info(f"Run ID: {RUN_ID}")
    logger.info(f"Tags Version: {VERSION_TAG}")
    logger.info(f"Ontology Version: {ONTOLOGY_VERSION}")
    logger.info(f"Limit: {args.limit or 'unlimited'}")
    logger.info(f"Sleep: {args.sleep}s")
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
            retry_success = stats.get('retry_success', 0)
            logger.info(
                f"  {tier:12} {stats['ok']:>6} ok, {stats['err']:>6} err "
                f"({success_rate:.1f}% success, {retry_success} retries)"
            )
    
    logger.info("="*70)
    
    # Exit code
    code = 0
    if res.get("ok", 0) == 0 and res.get("error", 0) > 0:
        code = 2
    
    sys.exit(code)
