# app/enrichment/backfill_parallel.py
"""
Backfill strategy:
  1. SQL enrichment_errors → discovery (which books failed)
  2. SQL books table → source of truth (fetch fresh data)
  3. Kafka DLQ → optional forensics (preserve context)
"""
import os, json, time
from pathlib import Path
from typing import Dict, Any, Set, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author, EnrichmentError, BookGenre

# Enrichment components
from app.enrichment.prompts import SYSTEM, USER_TEMPLATE
from app.enrichment.postprocess import clean_subjects
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import ensure_enrichment_ready, call_enrichment_llm
from app.enrichment.runner import load_tones, load_genres
from app.enrichment.kafka_producer import EnrichmentProducer, KAFKA_BOOTSTRAP

try:
    from app.enrichment.runner_kafka import enrich_one as _enrich_one
except ImportError:
    raise ImportError(
        "enrich_one not found in runner_kafka. "
        "Ensure runner_kafka.py exports this function."
    )

VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v1")
ROOT = Path(__file__).resolve().parents[2]
OUT_JSONL = ROOT / "data" / "enrichment_v1.jsonl"


class BackfillStrategy(Enum):
    """Backfill source selection"""
    SQL_ERRORS = "sql"
    KAFKA_DLQ = "kafka"
    FULL_REPROCESS = "full"


def iter_failed_books_from_sql(
    db: Session,
    tags_version: str = "v1",
    error_codes: Optional[list[str]] = None,
    hours: Optional[int] = None
) -> Iterable[dict]:
    """
    Query SQL for failures, then fetch FRESH book data.
    This is the PRIMARY backfill path.
    """
    q = db.query(EnrichmentError.item_idx).filter(
        EnrichmentError.tags_version == tags_version
    )
    
    if error_codes:
        q = q.filter(EnrichmentError.error_code.in_(error_codes))
    
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(EnrichmentError.last_seen_at >= cutoff)
    
    failed_ids = {row.item_idx for row in q}
    
    if not failed_ids:
        return []
    
    for b, a in (
        db.query(Book, Author)
        .outerjoin(Author, Book.author_idx == Author.author_idx)
        .filter(Book.item_idx.in_(failed_ids))
    ):
        yield {
            "item_idx": int(b.item_idx),
            "title": b.title or "",
            "author": a.name if a else "",
            "description": b.description or "",
        }


def iter_failed_books_from_kafka_dlq(
    tags_version: str = "v1",
    hours: int = 24,
    max_messages: int = 10000,
    timeout_ms: int = 30000,
    batch_size: int = 100
) -> Iterable[dict]:
    """
    Replay from Kafka DLQ topic (secondary path).
    Batches DB fetches for efficiency.
    """
    from kafka import KafkaConsumer
    
    consumer = KafkaConsumer(
        "enrich.errors.v1",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        consumer_timeout_ms=timeout_ms,
        group_id=f'backfill-dlq-{int(time.time())}'
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
            for b, a in (
                db.query(Book, Author)
                .outerjoin(Author, Book.author_idx == Author.author_idx)
                .filter(Book.item_idx.in_(pending_ids))
            ):
                results.append({
                    "item_idx": int(b.item_idx),
                    "title": b.title or "",
                    "author": a.name if a else "",
                    "description": b.description or "",
                })
            return results
    
    try:
        for msg in consumer:
            count += 1
            if count > max_messages:
                print(f"  Reached max_messages limit ({max_messages})")
                break
            
            error_event = msg.value
            
            if error_event.get('tags_version') != tags_version:
                continue
            if error_event.get('timestamp', 0) < cutoff_ts:
                continue
            
            item_idx = error_event['item_idx']
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
        print(f"  Kafka consumer error: {e}")
    finally:
        consumer.close()


def iter_books_for_full_reprocess(
    db: Session,
    tags_version: str = "v1",
    exclude_successful: bool = True
) -> Iterable[dict]:
    """
    Full catalog reprocess (nuclear option).
    """
    q = db.query(Book, Author).outerjoin(Author, Book.author_idx == Author.author_idx)
    
    if exclude_successful:
        enriched_ids = {
            row.item_idx 
            for row in db.query(BookGenre.item_idx)
                .filter(BookGenre.tags_version == tags_version)
        }
        if enriched_ids:
            q = q.filter(~Book.item_idx.in_(enriched_ids))
    
    for b, a in q:
        yield {
            "item_idx": int(b.item_idx),
            "title": b.title or "",
            "author": a.name if a else "",
            "description": b.description or "",
        }


def main(
    strategy: BackfillStrategy = BackfillStrategy.SQL_ERRORS,
    workers: int = 2,
    sleep_s: float = 0.0,
    dry_run: bool = False,
    **filters
):
    """
    Unified backfill entrypoint with strategy selection.
    """
    if not dry_run:
        ensure_enrichment_ready()
    
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones()
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    with SessionLocal() as db:
        if strategy == BackfillStrategy.SQL_ERRORS:
            print(f"📋 Strategy: SQL enrichment_errors (tags_version={VERSION_TAG})")
            books_iter = iter_failed_books_from_sql(
                db,
                tags_version=VERSION_TAG,
                error_codes=filters.get('error_codes'),
                hours=filters.get('hours')
            )
        
        elif strategy == BackfillStrategy.KAFKA_DLQ:
            print(f"📨 Strategy: Kafka DLQ replay (last {filters.get('hours', 24)}h)")
            books_iter = iter_failed_books_from_kafka_dlq(
                tags_version=VERSION_TAG,
                hours=filters.get('hours', 24)
            )
        
        elif strategy == BackfillStrategy.FULL_REPROCESS:
            print(f"🔄 Strategy: Full reprocess (exclude_successful={filters.get('exclude_successful', True)})")
            books_iter = iter_books_for_full_reprocess(
                db,
                tags_version=VERSION_TAG,
                exclude_successful=filters.get('exclude_successful', True)
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        books_list = list(books_iter)
        if not books_list:
            print("No books to backfill.")
            return {"ok": 0, "err": 0}
        
        print(f"Found {len(books_list)} books to backfill\n")
    
    if dry_run:
        print(f"🔍 DRY RUN MODE")
        print(f"Would process {len(books_list)} books:")
        for book in books_list[:10]:
            print(f"  - {book['item_idx']}: {book['title'][:60]}")
        if len(books_list) > 10:
            print(f"  ... and {len(books_list) - 10} more")
        print(f"\nTo execute, remove --dry-run flag")
        return {"ok": 0, "err": 0, "dry_run": True}
    
    producer = EnrichmentProducer(enable_kafka=True)
    workers = max(1, int(os.getenv("ENRICH_WORKERS", workers)))
    ok = err = 0
    start_time = time.time()
    
    def task(rec):
        nonlocal ok, err
        
        result, error = _enrich_one(
            rec, slug2id, valid_tone_ids, valid_genre_slugs,
            tone_slugs, genre_slugs_line
        )
        
        if result:
            success = producer.send_result(**result)
            if success:
                if sleep_s:
                    time.sleep(sleep_s)
                return True
            return False
        
        # Retry once
        result, error = _enrich_one(
            rec, slug2id, valid_tone_ids, valid_genre_slugs,
            tone_slugs, genre_slugs_line
        )
        
        if result:
            return producer.send_result(**result)
        else:
            producer.send_error(
                item_idx=rec["item_idx"],
                error_msg=error["error_msg"],
                stage=error["stage"],
                error_code=error["error_code"],
                error_field=error.get("error_field"),
                title=rec["title"][:256],
                author=rec["author"][:256],
                tags_version=VERSION_TAG,
            )
            return False
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, rec) for rec in books_list]
            
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    err += 1
                
                # Progress with ETA
                total = ok + err
                if total % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0
                    remaining = len(books_list) - total
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    
                    print(f"Progress: {total}/{len(books_list)} ({ok} ✓, {err} ✗) | "
                          f"{rate:.1f}/s | ETA: {eta_str}")
    
    finally:
        producer.flush()
        producer.close()
    
    return {"ok": ok, "err": err}


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill failed enrichments")
    parser.add_argument(
        "--strategy",
        choices=["sql", "kafka", "full"],
        default="sql",
        help="Backfill source"
    )
    parser.add_argument(
        "--error-codes",
        nargs="+",
        help="Filter by error codes"
    )
    parser.add_argument(
        "--hours",
        type=int,
        help="Only backfill errors from last N hours"
    )
    parser.add_argument(
        "--exclude-successful",
        action="store_true",
        help="For full reprocess: skip already enriched"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed"
    )
    
    args = parser.parse_args()
    
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
    
    res = main(
        strategy=strategy_map[args.strategy],
        workers=args.workers,
        dry_run=args.dry_run,
        **{k: v for k, v in filters.items() if v is not None}
    )
    
    print(f"\nBackfill complete: {res}")
    sys.exit(0 if res["err"] == 0 else 2)