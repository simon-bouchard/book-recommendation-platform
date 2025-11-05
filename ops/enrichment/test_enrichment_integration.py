# test_enrichment_integration.py
"""
Integration test for enrichment pipeline: Producer → Kafka → Spark → SQL

Verifies that data sent to Kafka by the producer correctly appears in SQL tables.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import (
    Book, Author, OLSubject, BookOLSubject,
    BookTone, BookGenre, BookVibe, BookLLMSubject,
    LLMSubject, Vibe, EnrichmentError
)
from app.enrichment.core import enrich_with_retry, load_tones, load_genres
from app.enrichment.kafka_producer import EnrichmentProducer
from sqlalchemy import text

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test configuration
TEST_TAGS_VERSION = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ONTOLOGY_VERSION = "v2"
CONSUMER_WAIT_SECONDS = 120  # Max time to wait for Spark consumer


class KafkaEventCapture:
    """Captures events sent to Kafka for verification"""
    
    def __init__(self):
        self.results: Dict[int, dict] = {}
        self.errors: Dict[int, dict] = {}
    
    def capture_result(self, event: dict):
        """Capture a result event"""
        item_idx = event['item_idx']
        self.results[item_idx] = {
            'subjects': event['subjects'],
            'tone_ids': event['tone_ids'],
            'genre': event['genre'],
            'vibe': event['vibe'],
            'tags_version': event['tags_version'],
            'timestamp': event.get('timestamp'),
            'metadata': event.get('metadata', {})
        }
    
    def capture_error(self, event: dict):
        """Capture an error event"""
        item_idx = event['item_idx']
        self.errors[item_idx] = {
            'error_msg': event['error_msg'],
            'stage': event['stage'],
            'error_code': event['error_code'],
            'error_field': event.get('error_field'),
            'tags_version': event['tags_version'],
            'attempted': event.get('attempted', {})
        }


class InterceptingProducer(EnrichmentProducer):
    """Producer that captures events before sending"""
    
    def __init__(self, capture: KafkaEventCapture, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
    
    def send_result(self, item_idx, subjects, tone_ids, genre, vibe, 
                   tags_version, scores=None, metadata=None) -> bool:
        # Capture the event
        event = {
            'item_idx': item_idx,
            'subjects': subjects,
            'tone_ids': tone_ids,
            'genre': genre,
            'vibe': vibe,
            'tags_version': tags_version,
            'scores': scores or {},
            'timestamp': int(time.time() * 1000),
            'metadata': metadata or {}
        }
        self.capture.capture_result(event)
        
        # Send to Kafka
        return super().send_result(
            item_idx, subjects, tone_ids, genre, vibe,
            tags_version, scores, metadata
        )
    
    def send_error(self, item_idx, error_msg, stage, error_code,
                  tags_version="v1", error_field=None, title=None, 
                  author=None, attempted=None, run_metadata=None, run_id=None) -> bool:
        # Capture the event
        event = {
            'item_idx': item_idx,
            'error_msg': error_msg,
            'stage': stage,
            'error_code': error_code,
            'error_field': error_field,
            'tags_version': tags_version,
            'attempted': attempted or {}
        }
        self.capture.capture_error(event)
        
        # Send to Kafka
        return super().send_error(
            item_idx, error_msg, stage, error_code, tags_version,
            error_field, title, author, attempted, run_metadata, run_id
        )


def fetch_book_data(db, item_idx: int) -> Optional[Dict[str, Any]]:
    """Fetch book metadata for enrichment"""
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


def run_enrichment_on_books(
    book_ids: List[int], 
    capture: KafkaEventCapture,
    tags_version: str
) -> Dict[str, int]:
    """Run enrichment on specific books and capture events"""
    
    # Override environment for test
    os.environ["ENRICHMENT_JOB_TAG_VERSION"] = tags_version
    os.environ["ENRICHMENT_ONTOLOGY_VERSION"] = ONTOLOGY_VERSION
    
    # Load ontology
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones(ONTOLOGY_VERSION)
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    # Create intercepting producer
    producer = InterceptingProducer(capture, enable_kafka=True)
    
    stats = {"success": 0, "error": 0, "skipped": 0}
    
    try:
        with SessionLocal() as db:
            for item_idx in book_ids:
                book_data = fetch_book_data(db, item_idx)
                
                if not book_data:
                    logger.warning(f"Book {item_idx} not found, skipping")
                    stats["skipped"] += 1
                    continue
                
                logger.info(f"Enriching book {item_idx}: {book_data['title']}")
                
                # Run enrichment
                result, error = enrich_with_retry(
                    book_data, slug2id, valid_tone_ids, valid_genre_slugs,
                    tone_slugs, genre_slugs_line, ONTOLOGY_VERSION, tags_version
                )
                
                if result:
                    # Send via intercepting producer
                    success = producer.send_result(
                        item_idx=result["item_idx"],
                        subjects=result["subjects"],
                        tone_ids=result["tone_ids"],
                        genre=result["genre"],
                        vibe=result["vibe"],
                        tags_version=result["tags_version"],
                        scores=result.get("scores", {}),
                        metadata=result.get("metadata", {})
                    )
                    
                    if success:
                        stats["success"] += 1
                        logger.info(f"✅ Enriched {item_idx}")
                    else:
                        stats["error"] += 1
                        logger.error(f"❌ Failed to send {item_idx} to Kafka")
                
                else:
                    # Send error
                    producer.send_error(
                        item_idx=book_data["item_idx"],
                        error_msg=error["error_msg"],
                        stage=error["stage"],
                        error_code=error["error_code"],
                        error_field=error.get("error_field"),
                        title=book_data["title"][:256],
                        author=book_data["author"][:256],
                        attempted=error.get("attempted"),
                        tags_version=tags_version
                    )
                    stats["error"] += 1
                    logger.error(f"❌ Enrichment failed for {item_idx}: {error['error_code']}")
    
    finally:
        producer.flush()
        producer.close()
    
    return stats


def wait_for_consumer(tags_version: str, expected_count: int, timeout: int = CONSUMER_WAIT_SECONDS):
    """Wait for Spark consumer to process events"""
    logger.info(f"\nWaiting for Spark consumer to process events (timeout: {timeout}s)...")
    
    start = time.time()
    last_count = 0
    
    with SessionLocal() as db:
        while time.time() - start < timeout:
            # Count total processed records
            total = 0
            total += db.query(BookTone).filter(BookTone.tags_version == tags_version).count()
            total += db.query(BookGenre).filter(BookGenre.tags_version == tags_version).count()
            total += db.query(BookVibe).filter(BookVibe.tags_version == tags_version).count()
            total += db.query(BookLLMSubject).filter(BookLLMSubject.tags_version == tags_version).count()
            total += db.query(EnrichmentError).filter(EnrichmentError.tags_version == tags_version).count()
            
            if total > last_count:
                logger.info(f"  Progress: {total} records in SQL")
                last_count = total
            
            if total >= expected_count:
                elapsed = time.time() - start
                logger.info(f"✅ Consumer processed data in {elapsed:.1f}s")
                return True
            
            time.sleep(5)
    
    logger.warning(f"⚠️  Timeout waiting for consumer (found {last_count} records)")
    return False


def verify_sql_data(capture: KafkaEventCapture, tags_version: str) -> Dict[str, Any]:
    """Verify SQL data matches captured Kafka events"""
    
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION: Comparing Kafka events with SQL data")
    logger.info("="*80)
    
    discrepancies = []
    verified = {"subjects": 0, "tones": 0, "genres": 0, "vibes": 0, "errors": 0}
    
    with SessionLocal() as db:
        # Verify successful enrichments
        for item_idx, kafka_event in capture.results.items():
            logger.info(f"\nVerifying book {item_idx}...")
            
            # 1. Verify subjects
            sql_subjects = db.query(LLMSubject.subject).join(
                BookLLMSubject, LLMSubject.llm_subject_idx == BookLLMSubject.llm_subject_idx
            ).filter(
                BookLLMSubject.item_idx == item_idx,
                BookLLMSubject.tags_version == tags_version
            ).all()
            
            sql_subject_set = {s[0] for s in sql_subjects}
            kafka_subject_set = set(kafka_event['subjects'])
            
            if sql_subject_set != kafka_subject_set:
                discrepancies.append({
                    'item_idx': item_idx,
                    'field': 'subjects',
                    'kafka': kafka_subject_set,
                    'sql': sql_subject_set,
                    'diff': f"Missing in SQL: {kafka_subject_set - sql_subject_set}, Extra in SQL: {sql_subject_set - kafka_subject_set}"
                })
            else:
                verified['subjects'] += 1
                logger.info(f"  ✅ Subjects match ({len(sql_subject_set)} items)")
            
            # 2. Verify tones
            sql_tones = db.query(BookTone.tone_id).filter(
                BookTone.item_idx == item_idx,
                BookTone.tags_version == tags_version
            ).all()
            
            sql_tone_set = {t[0] for t in sql_tones}
            kafka_tone_set = set(kafka_event['tone_ids'])
            
            if sql_tone_set != kafka_tone_set:
                discrepancies.append({
                    'item_idx': item_idx,
                    'field': 'tone_ids',
                    'kafka': kafka_tone_set,
                    'sql': sql_tone_set
                })
            else:
                verified['tones'] += 1
                logger.info(f"  ✅ Tones match ({len(sql_tone_set)} items)")
            
            # 3. Verify genre
            sql_genre = db.query(BookGenre.genre_slug).filter(
                BookGenre.item_idx == item_idx,
                BookGenre.tags_version == tags_version
            ).first()
            
            sql_genre_val = sql_genre[0] if sql_genre else None
            kafka_genre_val = kafka_event['genre']
            
            if sql_genre_val != kafka_genre_val:
                discrepancies.append({
                    'item_idx': item_idx,
                    'field': 'genre',
                    'kafka': kafka_genre_val,
                    'sql': sql_genre_val
                })
            else:
                verified['genres'] += 1
                logger.info(f"  ✅ Genre matches: {sql_genre_val}")
            
            # 4. Verify vibe
            if kafka_event['vibe']:
                sql_vibe = db.query(Vibe.text).join(
                    BookVibe, Vibe.vibe_id == BookVibe.vibe_id
                ).filter(
                    BookVibe.item_idx == item_idx,
                    BookVibe.tags_version == tags_version
                ).first()
                
                sql_vibe_val = sql_vibe[0] if sql_vibe else None
                kafka_vibe_val = kafka_event['vibe']
                
                if sql_vibe_val != kafka_vibe_val:
                    discrepancies.append({
                        'item_idx': item_idx,
                        'field': 'vibe',
                        'kafka': kafka_vibe_val,
                        'sql': sql_vibe_val
                    })
                else:
                    verified['vibes'] += 1
                    logger.info(f"  ✅ Vibe matches: {sql_vibe_val}")
        
        # Verify errors
        for item_idx, kafka_error in capture.errors.items():
            logger.info(f"\nVerifying error for book {item_idx}...")
            
            sql_error = db.query(EnrichmentError).filter(
                EnrichmentError.item_idx == item_idx,
                EnrichmentError.tags_version == tags_version
            ).first()
            
            if not sql_error:
                discrepancies.append({
                    'item_idx': item_idx,
                    'field': 'error',
                    'kafka': kafka_error,
                    'sql': None,
                    'message': 'Error in Kafka but not in SQL'
                })
            else:
                # Check error fields
                mismatches = []
                if sql_error.error_code != kafka_error['error_code']:
                    mismatches.append(f"error_code: {kafka_error['error_code']} vs {sql_error.error_code}")
                if sql_error.stage != kafka_error['stage']:
                    mismatches.append(f"stage: {kafka_error['stage']} vs {sql_error.stage}")
                
                if mismatches:
                    discrepancies.append({
                        'item_idx': item_idx,
                        'field': 'error',
                        'mismatches': mismatches
                    })
                else:
                    verified['errors'] += 1
                    logger.info(f"  ✅ Error matches: {sql_error.error_code}")
    
    return {
        'verified': verified,
        'discrepancies': discrepancies,
        'total_checked': len(capture.results) + len(capture.errors)
    }


def cleanup_test_data(tags_version: str):
    """Clean up test data from SQL"""
    logger.info(f"\nCleaning up test data (tags_version={tags_version})...")
    
    with SessionLocal() as db:
        try:
            # Delete link tables
            db.query(BookTone).filter(BookTone.tags_version == tags_version).delete()
            db.query(BookGenre).filter(BookGenre.tags_version == tags_version).delete()
            db.query(BookVibe).filter(BookVibe.tags_version == tags_version).delete()
            db.query(BookLLMSubject).filter(BookLLMSubject.tags_version == tags_version).delete()
            db.query(EnrichmentError).filter(EnrichmentError.tags_version == tags_version).delete()
            
            db.commit()
            logger.info("✅ Test data cleaned up")
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Cleanup failed: {e}")


def main():
    """Run integration test"""
    
    # Test configuration
    test_book_ids = [1, 2, 18, 19, 50]  # Adjust based on your test database
    
    logger.info("="*80)
    logger.info("ENRICHMENT INTEGRATION TEST")
    logger.info("="*80)
    logger.info(f"Tags Version: {TEST_TAGS_VERSION}")
    logger.info(f"Ontology Version: {ONTOLOGY_VERSION}")
    logger.info(f"Test Books: {test_book_ids}")
    logger.info(f"Consumer Wait: {CONSUMER_WAIT_SECONDS}s")
    logger.info("="*80)
    
    # Create capture
    capture = KafkaEventCapture()
    
    try:
        # Step 1: Run enrichment
        logger.info("\n[Step 1] Running enrichment on test books...")
        stats = run_enrichment_on_books(test_book_ids, capture, TEST_TAGS_VERSION)
        
        logger.info(f"\nEnrichment complete:")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Errors: {stats['error']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        
        if stats['success'] == 0 and stats['error'] == 0:
            logger.error("❌ No books processed - check book IDs")
            return 1
        
        # Step 2: Wait for consumer
        logger.info("\n[Step 2] Waiting for Spark streaming consumer...")
        expected_records = (
            len(capture.results) * 4 +  # Approx records per book (subjects, tones, genre, vibe)
            len(capture.errors)  # Error records
        )
        
        consumer_ok = wait_for_consumer(TEST_TAGS_VERSION, expected_records)
        
        if not consumer_ok:
            logger.warning("⚠️  Consumer may not have finished - verification might show discrepancies")
        
        # Step 3: Verify SQL data
        logger.info("\n[Step 3] Verifying SQL data matches Kafka events...")
        results = verify_sql_data(capture, TEST_TAGS_VERSION)
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Total Checked: {results['total_checked']} books")
        logger.info(f"\nVerified:")
        for field, count in results['verified'].items():
            logger.info(f"  {field}: {count}")
        
        if results['discrepancies']:
            logger.error(f"\n❌ Found {len(results['discrepancies'])} discrepancies:")
            for disc in results['discrepancies']:
                logger.error(f"\n  Item {disc['item_idx']} - {disc['field']}:")
                logger.error(f"    Kafka: {disc.get('kafka')}")
                logger.error(f"    SQL: {disc.get('sql')}")
                if 'diff' in disc:
                    logger.error(f"    Diff: {disc['diff']}")
            
            return 1
        else:
            logger.info("\n✅ All checks passed! SQL data matches Kafka events.")
            return 0
    
    finally:
        # Cleanup
        cleanup = input("\nClean up test data? (yes/no): ")
        if cleanup.lower() == "yes":
            cleanup_test_data(TEST_TAGS_VERSION)


if __name__ == "__main__":
    sys.exit(main())
