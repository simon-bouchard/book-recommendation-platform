# app/enrichment/runner_kafka.py
"""
Enrichment runner with Kafka support.
Sends results to Kafka (and optionally JSONL during dual-write period).
"""
import csv, time, os, sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.enrichment.prompts import SYSTEM, USER_TEMPLATE
from app.enrichment.postprocess import (
    render_tone_slugs, render_genre_slugs, clean_subjects,
)
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm, ensure_enrichment_ready
from app.enrichment.kafka_producer import EnrichmentProducer

from app.database import SessionLocal
from app.table_models import Book, Author
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TONES_CSV = ROOT / "ontology" / "tones_v1.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"

STOP_ON_FIRST_LLM_ERROR = os.getenv("ENRICH_STOP_ON_FIRST_LLM_ERROR", "0") == "1"
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v1")

def load_tones():
    rows = []
    with open(TONES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    slugs = render_tone_slugs(rows)
    valid_ids = {int(r["tone_id"]) for r in rows}
    slug2id = {r["slug"]: int(r["tone_id"]) for r in rows}
    return rows, slugs, valid_ids, slug2id

def load_genres():
    rows = []
    with open(GENRES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    slugs_line = render_genre_slugs(rows)
    valid_slugs = {r["slug"] for r in rows}
    return rows, slugs_line, valid_slugs

def iter_books_from_db(db: Session, limit: int | None = None):
    """
    Query books with item_idx (integers only for Kafka keys).
    """
    q = (db.query(Book, Author)
         .outerjoin(Author, Book.author_idx == Author.author_idx)
         .filter(Book.item_idx.isnot(None)))
    
    if limit:
        q = q.limit(limit)
    
    for b, a in q:
        yield {
            "item_idx": int(b.item_idx),
            "title": b.title or "",
            "author": a.name if a else "",
            "description": b.description or "",
        }

def enrich_one(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str,
) -> tuple[dict | None, dict | None]:
    """
    Enrich a single book record.
    
    Returns:
        (result_dict, error_dict): One will be None
    """
    user = USER_TEMPLATE.format(
        title=rec["title"],
        author=rec["author"],
        description=rec["description"],
        tone_instructions=f"Fixed tones: [{tone_slugs}]",
        genre_instructions=f"Fixed genres: [{genre_slugs_line}]",
        noisy_subjects_block="",
    )
    
    start_time = time.time()
    
    try:
        # LLM call
        raw = call_enrichment_llm(SYSTEM, user)
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
        
        # Validate
        data = validate_payload(raw, valid_tone_ids, valid_genre_slugs)
        data.subjects = clean_subjects(data.subjects)[:8]
        
        result = {
            "item_idx": rec["item_idx"],
            "subjects": data.subjects,
            "tone_ids": data.tone_ids,
            "genre": data.genre,
            "vibe": data.vibe,
            "tags_version": VERSION_TAG,
            "scores": {},
            "metadata": {
                "latency_ms": latency_ms,
                "model": os.getenv("DEEPINFRA_MODEL", "unknown"),
            }
        }
        
        return result, None
        
    except ValueError as e:
        # Validation error
        error = {
            "stage": "validate",
            "error_code": "VALIDATION_FAILED",
            "error_msg": str(e)[:512],
            "error_field": None,
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
        }
        return None, error


def main(limit: int | None = None, sleep_s: float = 0.0):
    """
    Main enrichment runner with Kafka support.
    """
    # Fail fast
    ensure_enrichment_ready()
    
    # Load ontology
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones()
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    # Initialize Kafka producer
    producer = EnrichmentProducer(enable_kafka=True)
    
    count_ok, count_err = 0, 0
    start_time = time.time()
    
    try:
        with SessionLocal() as db:
            for rec in iter_books_from_db(db, limit):
                # First attempt
                result, error = enrich_one(
                    rec, slug2id, valid_tone_ids, valid_genre_slugs,
                    tone_slugs, genre_slugs_line
                )
                
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
                        metadata=result.get("metadata"),
                    )
                    
                    if success:
                        count_ok += 1
                        logger.info(f"✓ Enriched item_idx={result['item_idx']}")
                    else:
                        logger.warning(f"Failed to send result for item_idx={result['item_idx']}")
                        count_err += 1
                    
                    if sleep_s:
                        time.sleep(sleep_s)
                    continue
                
                # First attempt failed - retry once
                logger.warning(f"First attempt failed for item_idx={rec['item_idx']}, retrying...")
                result, error = enrich_one(
                    rec, slug2id, valid_tone_ids, valid_genre_slugs,
                    tone_slugs, genre_slugs_line
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
                        metadata=result.get("metadata"),
                    )
                    
                    if success:
                        count_ok += 1
                        logger.info(f"✓ Enriched item_idx={result['item_idx']} (retry)")
                    else:
                        count_err += 1
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
                    )
                    count_err += 1
                    logger.error(f"✗ Failed item_idx={rec['item_idx']}: {error['error_msg']}")
                    
                    # Abort on first LLM error if configured
                    if STOP_ON_FIRST_LLM_ERROR and error["stage"] in ("llm_invoke", "llm_parse"):
                        logger.error("Aborting due to LLM error (ENRICH_STOP_ON_FIRST_LLM_ERROR=1)")
                        return {"ok": count_ok, "error": count_err, "aborted": True}
                
                # Progress indicator
                total = count_ok + count_err
                if total % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {total} books ({count_ok} ok, {count_err} err) | {rate:.1f}/s")
    
    finally:
        # Ensure all messages are sent before exiting
        logger.info("Flushing Kafka producer...")
        producer.flush()
        producer.close()
    
    elapsed = time.time() - start_time
    return {"ok": count_ok, "error": count_err, "aborted": False, "elapsed_s": elapsed}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrichment runner with Kafka support")
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
        "--version",
        default=None,
        help="Tags version (overrides ENRICHMENT_JOB_TAG_VERSION env var)"
    )
    
    args = parser.parse_args()
    
    # Override version if specified
    if args.version:
        os.environ["ENRICHMENT_JOB_TAG_VERSION"] = args.version
        VERSION_TAG = args.version
    
    logger.info("="*70)
    logger.info("ENRICHMENT RUNNER (KAFKA MODE)")
    logger.info("="*70)
    logger.info(f"Version: {VERSION_TAG}")
    logger.info(f"Limit: {args.limit or 'unlimited'}")
    logger.info(f"Sleep: {args.sleep}s")
    logger.info(f"Stop on LLM error: {STOP_ON_FIRST_LLM_ERROR}")
    logger.info("="*70)
    
    res = main(limit=args.limit, sleep_s=args.sleep)
    
    logger.info("="*70)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"Success: {res['ok']}")
    logger.info(f"Errors: {res['error']}")
    logger.info(f"Elapsed: {res.get('elapsed_s', 0):.1f}s")
    if res.get('aborted'):
        logger.info("Status: ABORTED")
    logger.info("="*70)
    
    # Exit code
    code = 0
    if res.get("aborted") or (res.get("ok", 0) == 0 and res.get("error", 0) > 0):
        code = 2
    
    sys.exit(code)
