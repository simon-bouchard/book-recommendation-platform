#!/usr/bin/env python3
"""
Run the actual enrichment pipeline and check basic stats.

This script:
1. Runs the ACTUAL runner_kafka.py pipeline with a limit
2. Waits for Spark consumer to process the Kafka events into SQL
3. Reports basic statistics

Usage:
    python ops/enrichment/test_enrichment.py --limit 1000 --version v2
    
After running, query SQL directly to analyze results:
    SELECT * FROM book_llm_subjects WHERE tags_version = 'v2' LIMIT 100;
    SELECT * FROM enrichment_errors WHERE tags_version = 'v2';
"""
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import os

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import actual runner
from app.enrichment import runner_kafka

# Database access
from app.database import SessionLocal
from app.table_models import (
    Book, BookLLMSubject, EnrichmentError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wait_for_spark_consumer(expected_count: int, tags_version: str, timeout_minutes: int = 5):
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
        initial_count = db.query(BookLLMSubject.item_idx).filter(
            BookLLMSubject.tags_version == tags_version
        ).distinct().count()
    
    logger.info(f"Initial enriched count (v={tags_version}): {initial_count}")
    
    while time.time() - start_time < timeout_seconds:
        with SessionLocal() as db:
            current_count = db.query(BookLLMSubject.item_idx).filter(
                BookLLMSubject.tags_version == tags_version
            ).distinct().count()
        
        processed = current_count - initial_count
        
        if processed >= expected_count:
            logger.info(f"✓ Spark consumer processed {processed} books")
            return True
        
        if processed > 0:
            logger.info(f"  Processed: {processed}/{expected_count}")
        
        time.sleep(5)  # Check every 5 seconds
    
    with SessionLocal() as db:
        final_count = db.query(BookLLMSubject.item_idx).filter(
            BookLLMSubject.tags_version == tags_version
        ).distinct().count()
    
    processed = final_count - initial_count
    logger.warning(f"Timeout reached. Processed: {processed}/{expected_count}")
    return False


def get_stats(tags_version: str, hours: int = 1):
    """Get enrichment statistics for this version."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    with SessionLocal() as db:
        # Count enriched books
        enriched_count = db.query(BookLLMSubject.item_idx).filter(
            BookLLMSubject.tags_version == tags_version
        ).distinct().count()
        
        # Count recent errors
        error_count = db.query(EnrichmentError).filter(
            EnrichmentError.tags_version == tags_version,
            EnrichmentError.last_seen_at >= cutoff
        ).count()
        
        # Error breakdown
        error_codes = {}
        errors = db.query(
            EnrichmentError.error_code,
            EnrichmentError.stage
        ).filter(
            EnrichmentError.tags_version == tags_version,
            EnrichmentError.last_seen_at >= cutoff
        ).all()
        
        for code, stage in errors:
            key = f"{stage}:{code}"
            error_codes[key] = error_codes.get(key, 0) + 1
    
    return {
        'enriched_count': enriched_count,
        'error_count': error_count,
        'error_codes': error_codes
    }


def run_pipeline_and_analyze(limit: int, tags_version: str, wait_for_consumer: bool = True):
    """
    Run the actual enrichment pipeline and report stats.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*80)
    logger.info("RUNNING ENRICHMENT PIPELINE TEST")
    logger.info("="*80)
    logger.info(f"Books to process: {limit}")
    logger.info(f"Tags version: {tags_version}")
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
        wait_for_spark_consumer(limit, tags_version, timeout_minutes=5)
        logger.info("")
    else:
        logger.info("Step 3: Skipping Spark consumer wait (--no-wait flag)")
        logger.info("Analyzing current state of database...\n")
    
    # Step 4: Get statistics
    logger.info("Step 4: Getting statistics from SQL...")
    stats = get_stats(tags_version, hours=1)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Tags version: {tags_version}")
    logger.info(f"Total enriched books: {stats['enriched_count']}")
    logger.info(f"Errors (last hour): {stats['error_count']}")
    
    if stats['error_codes']:
        logger.info("\nError breakdown:")
        for error_type, count in sorted(stats['error_codes'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_type}: {count}")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\nQuery SQL directly to analyze results:")
    logger.info(f"  -- View enriched books")
    logger.info(f"  SELECT b.title, COUNT(s.llm_subject_idx) as subject_count")
    logger.info(f"  FROM books b")
    logger.info(f"  JOIN book_llm_subjects s ON b.item_idx = s.item_idx")
    logger.info(f"  WHERE s.tags_version = '{tags_version}'")
    logger.info(f"  GROUP BY b.item_idx")
    logger.info(f"  LIMIT 100;")
    logger.info(f"")
    logger.info(f"  -- View errors")
    logger.info(f"  SELECT error_code, stage, COUNT(*) as cnt")
    logger.info(f"  FROM enrichment_errors")
    logger.info(f"  WHERE tags_version = '{tags_version}'")
    logger.info(f"  GROUP BY error_code, stage;")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run actual enrichment pipeline and report basic stats"
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
    parser.add_argument(
        "--version",
        default="v2",
        help="Tags version (default: v2)"
    )
    parser.add_argument(
        "--ontology",
        default="v2",
        help="Ontology version (default: v2)"
    )
    args = parser.parse_args()
    
    # Set environment variables for runner
    os.environ["ENRICHMENT_JOB_TAG_VERSION"] = args.version
    os.environ["ENRICHMENT_ONTOLOGY_VERSION"] = args.ontology
    
    run_pipeline_and_analyze(
        args.limit, 
        args.version,
        wait_for_consumer=not args.no_wait
    )
