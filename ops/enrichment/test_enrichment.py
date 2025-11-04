#!/usr/bin/env python3
"""
Run the actual enrichment pipeline and check basic stats.

This script:
1. Runs the ACTUAL runner_kafka.py pipeline with a limit
2. Reports basic statistics from the database
3. Note: Spark consumer runs independently and processes events asynchronously

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
import subprocess
import json

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


def get_kafka_consumer_lag(topic: str = "enrichment-events", group: str = "enrichment-consumer-group"):
    """
    Check Kafka consumer lag using kafka-consumer-groups command.
    Returns the total lag across all partitions.
    """
    try:
        # Run kafka-consumer-groups command
        result = subprocess.run(
            [
                "docker", "exec", "kafka",
                "kafka-consumer-groups.sh",
                "--bootstrap-server", "localhost:9092",
                "--group", group,
                "--describe"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            logger.warning(f"Could not get consumer lag: {result.stderr}")
            return None
        
        # Parse output to get total lag
        total_lag = 0
        for line in result.stdout.split('\n'):
            if topic in line:
                parts = line.split()
                # LAG is typically the last column
                try:
                    lag = int(parts[-1])
                    total_lag += lag
                except (ValueError, IndexError):
                    pass
        
        return total_lag
    except Exception as e:
        logger.warning(f"Error checking consumer lag: {e}")
        return None


def wait_for_kafka_consumption(timeout_minutes: int = 5, check_interval: int = 5):
    """
    Wait for Kafka consumer to catch up (lag = 0).
    Polls consumer group lag until it's zero or timeout is reached.
    """
    logger.info(f"\nWaiting for Kafka consumer to process events...")
    logger.info("(Checking consumer lag every few seconds)")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while time.time() - start_time < timeout_seconds:
        lag = get_kafka_consumer_lag()
        
        if lag is None:
            logger.warning("  Cannot check consumer lag (Kafka might not be accessible)")
            logger.info("  Waiting 30s before checking stats...")
            time.sleep(30)
            return False
        
        if lag == 0:
            logger.info(f"✓ Consumer lag is 0 - all events processed")
            return True
        
        elapsed = int(time.time() - start_time)
        logger.info(f"  Consumer lag: {lag} events (elapsed: {elapsed}s)")
        
        time.sleep(check_interval)
    
    lag = get_kafka_consumer_lag()
    logger.warning(f"Timeout reached. Final lag: {lag} events")
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
    logger.info(f"✓ Pipeline execution complete ({pipeline_elapsed:.1f}s)")
    logger.info(f"  Events published to Kafka topic\n")
    
    # Step 3: Wait for Kafka consumer to process events
    if wait_for_consumer:
        logger.info("Step 3: Waiting for Kafka consumer to process events...")
        wait_for_kafka_consumption(timeout_minutes=5, check_interval=5)
        logger.info("")
    else:
        logger.info("Step 3: Skipping consumer wait (--no-wait flag)")
        logger.info("  Note: Stats may not reflect just-published events\n")
    
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
        help="Don't wait for Kafka consumer, show current SQL state immediately"
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
