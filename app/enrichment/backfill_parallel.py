# app/enrichment/backfill_parallel.py
"""
Backfill runner - processes SPECIFIC failed books.
Uses shared core logic from app.enrichment.core

Backfill strategy:
  1. SQL enrichment_errors → discovery (which books failed)
  2. SQL books table → source of truth (fetch fresh data)
  3. Kafka DLQ → optional forensics (preserve context)
"""

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal

# Shared enrichment logic (includes loaders!)
from app.enrichment.core import enrich_with_retry, load_genres, load_tones
from app.enrichment.kafka_producer import KAFKA_BOOTSTRAP, EnrichmentProducer

# Infrastructure
from app.enrichment.llm_client import ensure_enrichment_ready
from app.table_models import Author, Book, BookGenre, BookOLSubject, EnrichmentError, OLSubject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v2")
ONTOLOGY_VERSION = os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")
ROOT = Path(__file__).resolve().parents[2]


class BackfillStrategy(Enum):
    """Backfill source selection"""

    SQL_ERRORS = "sql"
    KAFKA_DLQ = "kafka"
    FULL_REPROCESS = "full"


# ============================================================================
# BOOK FETCHING (with OL subjects, like runner)
# ============================================================================


def fetch_book_with_ol_subjects(db: Session, item_idx: int) -> Optional[Dict[str, Any]]:
    """
    Fetch book metadata including OL subjects.
    """
    result = (
        db.query(Book, Author)
        .outerjoin(Author, Book.author_idx == Author.author_idx)
        .filter(Book.item_idx == item_idx)
        .first()
    )

    if not result:
        return None

    book, author = result

    # Fetch OL subjects
    ol_subjects_query = (
        db.query(OLSubject.subject)
        .join(BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx)
        .filter(BookOLSubject.item_idx == item_idx)
        .all()
    )

    ol_subjects = [subj for (subj,) in ol_subjects_query]

    return {
        "item_idx": int(book.item_idx),
        "title": book.title or "",
        "author": author.name if author else "",
        "description": book.description or "",
        "ol_subjects": ol_subjects,
    }


# ============================================================================
# BACKFILL STRATEGIES (difference from runner: selective querying)
# ============================================================================


def iter_failed_books_from_sql(
    db: Session,
    tags_version: str = "v1",
    error_codes: Optional[list[str]] = None,
    hours: Optional[int] = None,
    from_run_id: Optional[str] = None,
) -> Iterable[dict]:
    """
    Query SQL for failures, then fetch FRESH book data.
    This is the PRIMARY backfill path.
    """
    q = db.query(EnrichmentError.item_idx).filter(EnrichmentError.tags_version == tags_version)

    if from_run_id:
        q = q.filter(EnrichmentError.last_run_id == from_run_id)

    if error_codes:
        q = q.filter(EnrichmentError.error_code.in_(error_codes))

    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(EnrichmentError.last_seen_at >= cutoff)

    failed_ids = {row.item_idx for row in q}

    if not failed_ids:
        return []

    logger.info(f"Found {len(failed_ids)} failed book IDs from SQL")

    # Fetch full book data with OL subjects
    for item_idx in failed_ids:
        book_data = fetch_book_with_ol_subjects(db, item_idx)
        if book_data:
            yield book_data


def iter_failed_books_from_kafka_dlq(
    tags_version: str = "v1",
    hours: int = 24,
    max_messages: int = 10000,
    timeout_ms: int = 30000,
    batch_size: int = 100,
) -> Iterable[dict]:
    """
    Replay from Kafka DLQ topic (secondary path).
    Batches DB fetches for efficiency.
    """
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        "enrich.errors.v1",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=timeout_ms,
        group_id=f"backfill-dlq-{int(time.time())}",
    )

    cutoff_ts = int((time.time() - hours * 3600) * 1000)
    seen = set()
    count = 0
    pending_ids = []

    def flush_batch():
        if not pending_ids:
            return []

        with SessionLocal() as db:
            results = []
            for item_idx in pending_ids:
                book_data = fetch_book_with_ol_subjects(db, item_idx)
                if book_data:
                    results.append(book_data)
            return results

    try:
        for msg in consumer:
            count += 1
            if count > max_messages:
                logger.info(f"Reached max_messages limit ({max_messages})")
                break

            error_event = msg.value

            if error_event.get("tags_version") != tags_version:
                continue
            if error_event.get("timestamp", 0) < cutoff_ts:
                continue

            item_idx = error_event["item_idx"]
            if item_idx in seen:
                continue
            seen.add(item_idx)
            pending_ids.append(item_idx)

            if len(pending_ids) >= batch_size:
                for book in flush_batch():
                    yield book
                pending_ids.clear()

        for book in flush_batch():
            yield book

    except Exception as e:
        logger.error(f"Kafka consumer error: {e}")
    finally:
        consumer.close()


def iter_books_for_full_reprocess(
    db: Session, tags_version: str = "v1", exclude_successful: bool = True
) -> Iterable[dict]:
    """
    Full catalog reprocess (nuclear option).
    """
    q = db.query(Book.item_idx)

    if exclude_successful:
        enriched_ids = {
            row.item_idx
            for row in db.query(BookGenre.item_idx).filter(BookGenre.tags_version == tags_version)
        }
        if enriched_ids:
            q = q.filter(~Book.item_idx.in_(enriched_ids))

    for (item_idx,) in q:
        book_data = fetch_book_with_ol_subjects(db, item_idx)
        if book_data:
            yield book_data


def get_or_create_run_id(db: Session, tags_version: str) -> tuple[str, str]:
    """
    Get the previous run_id to backfill, and create a new run_id for this run.

    Returns:
        (previous_run_id, new_run_id)
    """
    # Get the most recent run_id from errors
    latest = (
        db.query(EnrichmentError.last_run_id)
        .filter(EnrichmentError.tags_version == tags_version)
        .order_by(EnrichmentError.last_seen_at.desc())
        .first()
    )

    previous_run_id = latest[0] if latest else None

    # Generate new run_id for this backfill
    new_run_id = f"backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    return previous_run_id, new_run_id


# ============================================================================
# MAIN BACKFILL RUNNER (improved with runner patterns)
# ============================================================================


def main(
    strategy: BackfillStrategy = BackfillStrategy.SQL_ERRORS,
    workers: int = 2,
    dry_run: bool = False,
    sleep_s: float = 0.0,
    **filters,
):
    """
    Main backfill runner with improved error handling and statistics tracking.

    Key improvements from runner_kafka.py:
    - Tier statistics tracking with retry success counters
    - Better error handling and logging
    - Proper handling of enrichment_quality field
    - Better progress reporting
    """
    ensure_enrichment_ready()

    # Load ontologies (shared with runner)
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones(ONTOLOGY_VERSION)
    genre_rows, genre_id_slugs_line, valid_genre_ids, genre_id2slug, genre_slug2id = load_genres()

    # Initialize tier statistics (like runner)
    tier_stats = {
        "RICH": {"ok": 0, "err": 0, "retry_success": 0},
        "SPARSE": {"ok": 0, "err": 0, "retry_success": 0},
        "MINIMAL": {"ok": 0, "err": 0, "retry_success": 0},
        "BASIC": {"ok": 0, "err": 0, "retry_success": 0},
    }

    # Get run IDs
    with SessionLocal() as db:
        previous_run_id, new_run_id = get_or_create_run_id(db, VERSION_TAG)
        logger.info(f"Previous run ID: {previous_run_id}")
        logger.info(f"New run ID: {new_run_id}")

        if strategy == BackfillStrategy.SQL_ERRORS:
            logger.info(f"📋 Strategy: SQL enrichment_errors (from run {previous_run_id})")

            if not previous_run_id:
                logger.info("No previous run found - nothing to backfill")
                return {"ok": 0, "err": 0, "tier_stats": tier_stats}

            books_iter = iter_failed_books_from_sql(
                db,
                tags_version=VERSION_TAG,
                from_run_id=previous_run_id,
                error_codes=filters.get("error_codes"),
                hours=filters.get("hours"),
            )

        elif strategy == BackfillStrategy.KAFKA_DLQ:
            logger.info(f"📨 Strategy: Kafka DLQ replay (last {filters.get('hours', 24)}h)")
            books_iter = iter_failed_books_from_kafka_dlq(
                tags_version=VERSION_TAG, hours=filters.get("hours", 24)
            )

        elif strategy == BackfillStrategy.FULL_REPROCESS:
            logger.info(
                f"🔄 Strategy: Full reprocess (exclude_successful={filters.get('exclude_successful', True)})"
            )
            books_iter = iter_books_for_full_reprocess(
                db,
                tags_version=VERSION_TAG,
                exclude_successful=filters.get("exclude_successful", True),
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        books_list = list(books_iter)
        if not books_list:
            logger.info("No books to backfill.")
            return {"ok": 0, "err": 0, "tier_stats": tier_stats}

        logger.info(f"Found {len(books_list)} books to backfill\n")

    if dry_run:
        logger.info("🔍 DRY RUN MODE")
        logger.info(f"Would process {len(books_list)} books:")
        for book in books_list[:10]:
            logger.info(f"  - {book['item_idx']}: {book['title'][:60]}")
        if len(books_list) > 10:
            logger.info(f"  ... and {len(books_list) - 10} more")
        logger.info("\nTo execute, remove --dry-run flag")
        return {"ok": 0, "err": 0, "dry_run": True, "tier_stats": tier_stats}

    producer = EnrichmentProducer(enable_kafka=True)
    workers = max(1, int(os.getenv("ENRICH_WORKERS", workers)))
    count_ok, count_err = 0, 0
    start_time = time.time()

    def task(rec):
        """
        Single enrichment task with improved error handling (matching runner).
        Returns: ("ok", result) or ("err", error)
        """
        # Use shared enrichment logic with retry
        result, error = enrich_with_retry(
            rec,
            slug2id,
            valid_tone_ids,
            valid_genre_ids,
            genre_id2slug,
            genre_slug2id,
            tone_slugs,
            genre_id_slugs_line,
            ontology_version=ONTOLOGY_VERSION,
            tags_version=VERSION_TAG,
        )

        if result:
            # Add run metadata (matching runner)
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["run_id"] = new_run_id
            result["metadata"]["model"] = os.getenv("DEEPINFRA_MODEL", "unknown")
            result["metadata"]["backfill"] = True

            # Send to Kafka
            success = producer.send_result(
                item_idx=result["item_idx"],
                subjects=result["subjects"],
                tone_ids=result["tone_ids"],
                genre=result["genre"],
                vibe=result["vibe"],
                tags_version=result["tags_version"],
                scores=result["scores"],
                metadata=result.get("metadata", {}),
            )

            if success:
                if sleep_s:
                    time.sleep(sleep_s)
                return ("ok", result)
            return ("err", result)

        elif error:
            # CRITICAL FIX: Add item_idx to error dict (from runner)
            error["item_idx"] = rec["item_idx"]

            # Send to DLQ
            producer.send_error(
                item_idx=rec["item_idx"],
                error_msg=error.get("error_msg", "Unknown error"),
                stage=error.get("stage", "unknown"),
                error_code=error.get("error_code", "UNKNOWN"),
                error_field=error.get("error_field"),
                title=rec["title"][:256],
                author=rec["author"][:256],
                attempted=error.get("attempted"),
                tags_version=VERSION_TAG,
                run_id=new_run_id,
            )
            return ("err", error)
        else:
            # Both result and error are None - shouldn't happen but handle it (from runner)
            logger.error(f"enrich_with_retry returned (None, None) for item_idx={rec['item_idx']}")
            return (
                "err",
                {
                    "item_idx": rec["item_idx"],
                    "error_code": "NULL_RESPONSE",
                    "error_msg": "Enrichment returned no result and no error",
                    "stage": "unknown",
                    "attempted": {},
                },
            )

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, rec) for rec in books_list]

            for fut in as_completed(futures):
                try:
                    result_tuple = fut.result()

                    # Improved error handling from runner
                    if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                        count_err += 1
                        logger.error(f"✗ Task returned invalid result: {result_tuple}")
                        continue

                    status, data = result_tuple

                    if data is None:
                        count_err += 1
                        logger.error("✗ Task returned None data")
                        continue

                    if status == "ok":
                        count_ok += 1
                        tier = data.get("enrichment_quality", "UNKNOWN")
                        retry_count = data.get("metadata", {}).get("retry_count", 0)
                        item_idx = data.get("item_idx", "?")

                        # Update tier stats (matching runner)
                        if tier in tier_stats:
                            tier_stats[tier]["ok"] += 1
                            if retry_count > 0:
                                tier_stats[tier]["retry_success"] += 1

                        retry_msg = " (retry)" if retry_count > 0 else ""
                        logger.info(f"✓ item_idx={item_idx} ({tier}){retry_msg}")

                    else:  # status == "err"
                        count_err += 1

                        # Extract error info safely (improved from runner)
                        item_idx = data.get("item_idx", "?")
                        attempted = data.get("attempted", {}) if isinstance(data, dict) else {}
                        tier = (
                            attempted.get("tier", "UNKNOWN")
                            if isinstance(attempted, dict)
                            else "UNKNOWN"
                        )
                        error_code = data.get("error_code", "UNKNOWN")
                        error_msg = data.get("error_msg", "")[:80]

                        if tier in tier_stats:
                            tier_stats[tier]["err"] += 1

                        logger.error(f"✗ item_idx={item_idx} ({tier}): {error_code} - {error_msg}")

                except Exception as e:
                    count_err += 1
                    logger.error(f"✗ Task failed with exception: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

                # Progress with ETA (matching runner)
                total = count_ok + count_err
                if total % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0
                    remaining = len(books_list) - total
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

                    logger.info(
                        f"Progress: {total}/{len(books_list)} ({count_ok} ✓, {count_err} ✗) | "
                        f"{rate:.1f}/s | ETA: {eta_str}"
                    )

    finally:
        producer.flush()
        producer.close()

    elapsed = time.time() - start_time

    return {"ok": count_ok, "err": count_err, "elapsed_s": elapsed, "tier_stats": tier_stats}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Backfill failed enrichments")
    parser.add_argument(
        "--strategy", choices=["sql", "kafka", "full"], default="sql", help="Backfill source"
    )
    parser.add_argument("--error-codes", nargs="+", help="Filter by error codes")
    parser.add_argument("--hours", type=int, help="Only backfill errors from last N hours")
    parser.add_argument(
        "--exclude-successful",
        action="store_true",
        help="For full reprocess: skip already enriched",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Sleep duration between enrichments in seconds"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--version", default=None, help="Tags version (overrides env var)")
    parser.add_argument("--ontology", default="v2", help="Ontology version (v1 or v2, default: v2)")

    args = parser.parse_args()

    # Override versions if specified
    if args.version:
        os.environ["ENRICHMENT_JOB_TAG_VERSION"] = args.version
        VERSION_TAG = args.version

    if args.ontology:
        os.environ["ENRICHMENT_ONTOLOGY_VERSION"] = args.ontology
        ONTOLOGY_VERSION = args.ontology

    strategy_map = {
        "sql": BackfillStrategy.SQL_ERRORS,
        "kafka": BackfillStrategy.KAFKA_DLQ,
        "full": BackfillStrategy.FULL_REPROCESS,
    }

    filters = {
        "error_codes": args.error_codes,
        "hours": args.hours,
        "exclude_successful": args.exclude_successful,
    }

    logger.info("=" * 70)
    logger.info("BACKFILL RUNNER (QUALITY TIERING)")
    logger.info("=" * 70)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Tags Version: {VERSION_TAG}")
    logger.info(f"Ontology Version: {ONTOLOGY_VERSION}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Sleep: {args.sleep}s")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 70)

    res = main(
        strategy=strategy_map[args.strategy],
        workers=args.workers,
        dry_run=args.dry_run,
        sleep_s=args.sleep,
        **{k: v for k, v in filters.items() if v is not None},
    )

    logger.info("=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Success: {res.get('ok', 0)}")
    logger.info(f"Errors: {res.get('err', 0)}")
    logger.info(f"Elapsed: {res.get('elapsed_s', 0):.1f}s")

    # Show tier breakdown (matching runner)
    logger.info("\nTier Breakdown:")
    for tier, stats in res.get("tier_stats", {}).items():
        total = stats["ok"] + stats["err"]
        if total > 0:
            success_rate = stats["ok"] / total * 100
            retry_success = stats.get("retry_success", 0)
            logger.info(
                f"  {tier:12} {stats['ok']:>6} ok, {stats['err']:>6} err "
                f"({success_rate:.1f}% success, {retry_success} retries)"
            )

    logger.info("=" * 70)

    if res.get("dry_run"):
        logger.info("(Dry run - no actual processing)")
        sys.exit(0)

    # Exit code
    code = 0
    if res.get("ok", 0) == 0 and res.get("err", 0) > 0:
        code = 2

    sys.exit(code)
