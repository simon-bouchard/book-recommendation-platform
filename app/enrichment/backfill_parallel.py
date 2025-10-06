# app/enrichment/backfill_parallel.py
"""
Parallel backfill for failed enrichments.
Phase 1: Uses Kafka producer, only processes books with item_idx.
"""
import os, json, time
from pathlib import Path
from typing import Dict, Any, Set, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author

# Enrichment components
from app.enrichment.prompts import SYSTEM, USER_TEMPLATE
from app.enrichment.postprocess import clean_subjects
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import ensure_enrichment_ready, call_enrichment_llm
from app.enrichment.runner import load_tones, load_genres
from app.enrichment.kafka_producer import EnrichmentProducer

ROOT = Path(__file__).resolve().parents[2]
OUT_JSONL = ROOT / "data" / "enrichment_v1.jsonl"


def _iter_failed_ids(jsonl_path: Path) -> Set[int]:
    """
    Parse JSONL to find failed enrichments.
    ONLY returns integer item_idx values (skips work_id strings).
    """
    needs = set()
    if not jsonl_path.exists():
        return needs
    
    last: Dict[int, Dict[str, Any]] = {}
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            
            bid = obj.get("book_id")
            
            # ONLY process integer IDs (item_idx)
            if not isinstance(bid, int):
                continue
            
            last[bid] = obj
    
    # Find failures
    for bid, obj in last.items():
        # Has error field
        if "error" in obj:
            needs.add(bid)
            continue
        
        # Missing required fields
        if not isinstance(obj.get("subjects", []), list):
            needs.add(bid)
            continue
        if not isinstance(obj.get("tone_ids", []), list):
            needs.add(bid)
            continue
        if not (isinstance(obj.get("genre"), str) and obj["genre"]):
            needs.add(bid)
            continue
    
    return needs


def _iter_books_by_ids(db: Session, ids: Set[int]) -> Iterable[dict]:
    """
    Query books by item_idx only.
    Returns dict with item_idx, title, author, description.
    """
    if not ids:
        return []
    
    for b, a in (
        db.query(Book, Author)
        .outerjoin(Author, Book.author_idx == Author.author_idx)
        .filter(Book.item_idx.in_(ids))
    ):
        yield {
            "item_idx": int(b.item_idx),
            "title": b.title or "",
            "author": (a.name if a else "") or "",
            "description": b.description or "",
        }


def _enrich_one(
    rec: dict,
    slug2id: dict,
    valid_tone_ids: set,
    valid_genre_slugs: set,
    tone_slugs: str,
    genre_slugs_line: str
) -> tuple[dict | None, dict | None]:
    """
    Enrich a single book.
    Returns: (result_dict, error_dict) - one will be None
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
        
        # Normalize tone_ids (slug -> id if needed)
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
            "tags_version": "v1",
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


def main(workers: int = 2, sleep_s: float = 0.0) -> Dict[str, int]:
    """
    Backfill failed enrichments using Kafka producer.
    Only processes books with integer item_idx.
    """
    # Fail fast
    ensure_enrichment_ready()
    
    # Load ontology
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones()
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    
    # Find failed IDs (integers only)
    need_ids = _iter_failed_ids(OUT_JSONL)
    if not need_ids:
        print("No failed enrichments found (or all failures are work_id strings, which we skip).")
        return {"ok": 0, "err": 0, "skipped_work_ids": 0}
    
    print(f"Found {len(need_ids)} failed item_idx values to backfill")
    
    # Configure workers
    workers = max(1, int(os.getenv("ENRICH_WORKERS", workers)))
    
    # Initialize Kafka producer
    producer = EnrichmentProducer(enable_kafka=True)
    
    ok = err = 0
    
    def task(rec):
        nonlocal ok, err
        
        # First attempt
        result, error = _enrich_one(
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
                if sleep_s:
                    time.sleep(sleep_s)
                return True
            else:
                return False
        
        # First attempt failed - retry once
        print(f"  Retry item_idx={rec['item_idx']}...")
        result, error = _enrich_one(
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
            return success
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
                tags_version="v1",
            )
            return False
    
    try:
        with SessionLocal() as db, ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for rec in _iter_books_by_ids(db, need_ids):
                futures.append(ex.submit(task, rec))
            
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    err += 1
                
                # Progress indicator
                total = ok + err
                if total % 100 == 0:
                    print(f"Progress: {total}/{len(need_ids)} ({ok} ok, {err} err)")
    
    finally:
        # Ensure all messages sent
        print("Flushing Kafka producer...")
        producer.flush()
        producer.close()
    
    return {"ok": ok, "err": err}


if __name__ == "__main__":
    import sys
    
    workers = int(os.getenv("ENRICH_WORKERS", "2"))
    sleep_s = float(os.getenv("ENRICH_SLEEP_S", "0.0"))
    
    res = main(workers=workers, sleep_s=sleep_s)
    
    print(f"\nBackfill complete: {res}")
    
    # Exit with error if all failed
    code = 0 if res["err"] == 0 else 2
    sys.exit(code)