#!/usr/bin/env python3
"""
Run the actual enrichment pipeline and analyze results from SQL.

This script:
1. Runs the ACTUAL runner_kafka.py pipeline with a limit
2. Waits for Spark consumer to process the Kafka events into SQL
3. Analyzes results by reading from SQL database

Usage:
    python ops/enrichment/test_enrichment.py --limit 1000

Outputs:
    - ops/enrichment/test_results.csv: Successful enrichments
    - ops/enrichment/test_errors.txt: Grouped error analysis
"""
import sys
import csv
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import argparse

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import actual runner
from app.enrichment import runner_kafka

# Database access
from app.database import SessionLocal
from app.table_models import (
    Book, Author, OLSubject, BookOLSubject,
    LLMSubject, BookLLMSubject, BookTone, BookGenre,
    Vibe, BookVibe, EnrichmentError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent


def truncate_text(text: str, max_len: int = 150) -> str:
    """Truncate text to max_len chars."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def fetch_enrichment_from_sql(db, item_idx: int):
    """Fetch enrichment data from SQL tables."""
    # Get book basics
    result = db.query(Book, Author).outerjoin(
        Author, Book.author_idx == Author.author_idx
    ).filter(Book.item_idx == item_idx).first()
    
    if not result:
        return None
    
    book, author = result
    
    # Get OL subjects
    ol_subjects = [
        subj for (subj,) in 
        db.query(OLSubject.subject)
        .join(BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx)
        .filter(BookOLSubject.item_idx == item_idx)
        .all()
    ]
    
    # Get LLM subjects
    llm_subjects = [
        subj for (subj,) in
        db.query(LLMSubject.subject)
        .join(BookLLMSubject, LLMSubject.llm_subject_idx == BookLLMSubject.llm_subject_idx)
        .filter(BookLLMSubject.item_idx == item_idx)
        .all()
    ]
    
    # Get tones
    tone_ids = [
        tone_id for (tone_id,) in
        db.query(BookTone.tone_id)
        .filter(BookTone.item_idx == item_idx)
        .all()
    ]
    
    # Get genre
    genre_result = db.query(BookGenre.genre_slug).filter(
        BookGenre.item_idx == item_idx
    ).first()
    genre = genre_result[0] if genre_result else None
    
    # Get vibe
    vibe_result = db.query(Vibe.text).join(
        BookVibe, Vibe.vibe_id == BookVibe.vibe_id
    ).filter(BookVibe.item_idx == item_idx).first()
    vibe = vibe_result[0] if vibe_result else ""
    
    return {
        'item_idx': int(book.item_idx),
        'title': book.title or "",
        'author': author.name if author else "",
        'description': book.description or "",
        'ol_subjects': ol_subjects,
        'llm_subjects': llm_subjects,
        'tone_ids': tone_ids,
        'genre': genre or "",
        'vibe': vibe
    }


def get_recent_errors(db, hours: int = 1):
    """Get errors from the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    error_rows = (
        db.query(EnrichmentError)
        .filter(EnrichmentError.last_seen_at >= cutoff)
        .all()
    )
    
    errors = []
    for err in error_rows:
        # Get book data
        book_result = db.query(Book, Author).outerjoin(
            Author, Book.author_idx == Author.author_idx
        ).filter(Book.item_idx == err.item_idx).first()
        
        if book_result:
            book, author = book_result
            description = book.description or ""
        else:
            description = ""
        
        attempted = {}
        if err.attempted:
            try:
                attempted = json.loads(err.attempted) if isinstance(err.attempted, str) else err.attempted
            except:
                attempted = {}
        
        errors.append({
            'item_idx': err.item_idx,
            'title': err.title or "",
            'author': err.author or "",
            'description': description,
            'stage': err.stage,
            'error_code': err.error_code,
            'error_msg': err.error_msg,
            'attempted': attempted,
            'occurrence_count': err.occurrence_count,
            'last_seen_at': err.last_seen_at
        })
    
    return errors


def wait_for_spark_consumer(expected_count: int, timeout_minutes: int = 5):
    """
    Wait for Spark consumer to process Kafka events into SQL.
    Polls the database to see when new enrichments appear.
    """
    logger.info(f"\nWaiting for Spark consumer to process {expected_count} books...")
    logger.info("(Spark consumer should be running with docker-compose)")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    initial_count = 0
    
    with SessionLocal() as db:
        initial_count = db.query(BookLLMSubject.item_idx).distinct().count()
    
    logger.info(f"Initial enriched count: {initial_count}")
    target_count = initial_count + expected_count
    
    while time.time() - start_time < timeout_seconds:
        with SessionLocal() as db:
            current_count = db.query(BookLLMSubject.item_idx).distinct().count()
        
        processed = current_count - initial_count
        
        if processed >= expected_count:
            logger.info(f"✓ Spark consumer processed {processed} books")
            return True
        
        if processed > 0:
            logger.info(f"  Processed: {processed}/{expected_count}")
        
        time.sleep(5)  # Check every 5 seconds
    
    with SessionLocal() as db:
        final_count = db.query(BookLLMSubject.item_idx).distinct().count()
    
    processed = final_count - initial_count
    logger.warning(f"Timeout reached. Processed: {processed}/{expected_count}")
    return False


def run_pipeline_and_analyze(limit: int, wait_for_consumer: bool = True):
    """
    Run the actual enrichment pipeline and analyze results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*80)
    logger.info("RUNNING ENRICHMENT PIPELINE TEST")
    logger.info("="*80)
    logger.info(f"Books to process: {limit}")
    logger.info(f"Timestamp: {timestamp}\n")
    
    # Step 1: Get item_idxs we're going to process
    logger.info("Step 1: Fetching book IDs to process...")
    with SessionLocal() as db:
        book_ids = [
            item_idx for (item_idx,) in 
            db.query(Book.item_idx)
            .filter(Book.item_idx.isnot(None))
            .limit(limit)
            .all()
        ]
    logger.info(f"✓ Will process {len(book_ids)} books\n")
    
    # Step 2: Run the actual pipeline
    logger.info("Step 2: Running actual enrichment pipeline...")
    logger.info("-"*80)
    
    pipeline_start = time.time()
    runner_kafka.main(limit=limit, sleep_s=0.0, workers=3)
    pipeline_elapsed = time.time() - pipeline_start
    
    logger.info("-"*80)
    logger.info(f"✓ Pipeline execution complete ({pipeline_elapsed:.1f}s)\n")
    
    # Step 3: Wait for Spark consumer (optional)
    if wait_for_consumer:
        logger.info("Step 3: Waiting for Spark consumer to process events...")
        wait_for_spark_consumer(limit, timeout_minutes=5)
        logger.info("")
    else:
        logger.info("Step 3: Skipping Spark consumer wait (--no-wait flag)")
        logger.info("Analyzing current state of database...\n")
    
    # Step 4: Analyze results from SQL
    logger.info("Step 4: Analyzing results from SQL database...")
    
    successes = []
    not_enriched = []
    
    with SessionLocal() as db:
        for item_idx in book_ids:
            data = fetch_enrichment_from_sql(db, item_idx)
            if data:
                if data['llm_subjects']:  # Has LLM subjects = enriched
                    successes.append(data)
                else:
                    not_enriched.append(item_idx)
        
        # Get recent errors (last hour)
        errors = get_recent_errors(db, hours=1)
    
    logger.info(f"  Enriched: {len(successes)}")
    logger.info(f"  Not enriched: {len(not_enriched)}")
    logger.info(f"  Errors (last hour): {len(errors)}\n")
    
    # Step 5: Export
    logger.info("Step 5: Exporting results...")
    write_success_csv(successes, timestamp)
    write_error_report(errors, timestamp)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Successfully enriched: {len(successes)}")
    logger.info(f"Not yet enriched: {len(not_enriched)}")
    logger.info(f"Errors found: {len(errors)}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - {OUTPUT_DIR}/test_results_{timestamp}.csv")
    logger.info(f"  - {OUTPUT_DIR}/test_errors_{timestamp}.txt")
    
    if not_enriched:
        logger.info(f"\nNote: {len(not_enriched)} books not yet in SQL.")
        logger.info("This is normal if Spark consumer is still processing.")
        logger.info("Check Spark logs or wait a few minutes and re-run analysis.")


def write_success_csv(successes, timestamp):
    """Write successful enrichments to CSV."""
    output_file = OUTPUT_DIR / f"test_results_{timestamp}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'item_idx',
            'title',
            'author',
            'description',
            'ol_subjects',
            'llm_subjects',
            'tone_ids',
            'genre',
            'vibe'
        ])
        
        # Data rows
        for data in successes:
            writer.writerow([
                data['item_idx'],
                data['title'],
                data['author'],
                truncate_text(data['description'], 150),
                '; '.join(data['ol_subjects'][:5]) + (
                    f" (+{len(data['ol_subjects'])-5} more)" 
                    if len(data['ol_subjects']) > 5 else ""
                ),
                '; '.join(data['llm_subjects']),
                ', '.join(map(str, data['tone_ids'])),
                data['genre'],
                data['vibe']
            ])
    
    logger.info(f"✓ Wrote {len(successes)} results to {output_file}")


def write_error_report(errors, timestamp):
    """Write error analysis to text file."""
    output_file = OUTPUT_DIR / f"test_errors_{timestamp}.txt"
    
    # Group by error code
    by_error_code = defaultdict(list)
    for error in errors:
        error_code = error.get('error_code', 'UNKNOWN')
        by_error_code[error_code].append(error)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Frequency table
        f.write("Error Frequency:\n")
        f.write("-"*80 + "\n")
        sorted_codes = sorted(by_error_code.items(), key=lambda x: len(x[1]), reverse=True)
        for error_code, items in sorted_codes:
            f.write(f"{error_code:30s} | {len(items):4d} occurrences\n")
        f.write("\n\n")
        
        # Detailed breakdowns (ALL error types, not just top 5)
        for error_code, items in sorted_codes:
            f.write("="*80 + "\n")
            f.write(f"ERROR: {error_code} ({len(items)} occurrences)\n")
            f.write("="*80 + "\n\n")
            
            # Show first 10 examples
            for error in items[:10]:
                f.write("-"*80 + "\n")
                f.write(f"item_idx: {error['item_idx']}\n")
                f.write(f"title: {error.get('title', '')}\n")
                f.write(f"author: {error.get('author', '')}\n")
                f.write(f"description: {truncate_text(error.get('description', ''), 200)}\n")
                f.write(f"stage: {error.get('stage', 'unknown')}\n")
                f.write(f"error_msg: {error.get('error_msg', '')}\n")
                
                # Show attempted response if available
                attempted = error.get('attempted', {})
                if attempted:
                    f.write(f"attempted_response:\n{json.dumps(attempted, indent=2)}\n")
                
                f.write("\n")
            
            f.write("\n\n")
    
    logger.info(f"✓ Wrote error analysis to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run actual enrichment pipeline and analyze results from SQL"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=1000, 
        help="Number of books to test"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Spark consumer, analyze current SQL state"
    )
    args = parser.parse_args()
    
    run_pipeline_and_analyze(args.limit, wait_for_consumer=not args.no_wait)
