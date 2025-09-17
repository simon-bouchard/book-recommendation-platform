# app/enrichment/backfill_parallel.py
import os, json, time
from pathlib import Path
from typing import Dict, Any, Set, Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author

# Reuse your existing enrichment components
from app.enrichment.prompts import SYSTEM, USER_TEMPLATE  # prompt templates 
from app.enrichment.postprocess import clean_subjects      # subject cleanup 
from app.enrichment.validator import validate_payload      # schema/validation   
from app.enrichment.llm_client import ensure_enrichment_ready, call_enrichment_llm  # warmup + LLM call  
from app.enrichment.runner import load_tones, load_genres  # tone/genre loaders  

ROOT = Path(__file__).resolve().parents[2]
OUT_JSONL = ROOT / "data" / "enrichment_v1.jsonl"

# ---- helpers (same logic as your single-thread backfill) ----
def _iter_failed_ids(jsonl_path: Path) -> Set[str | int]:
    needs = set()
    if not jsonl_path.exists():
        return needs
    last: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            bid = obj.get("book_id")
            if bid is None: continue
            last[str(bid)] = obj
    for k, obj in last.items():
        if "error" in obj:
            needs.add(obj["book_id"]); continue
        if not isinstance(obj.get("subjects", []), list):
            needs.add(obj["book_id"]); continue
        if not isinstance(obj.get("tone_ids", []), list):
            needs.add(obj["book_id"]); continue
        if not (isinstance(obj.get("genre"), str) and obj["genre"]):
            needs.add(obj["book_id"]); continue
    return needs

def _iter_books_by_ids(db: Session, ids: Set[str | int]) -> Iterable[dict]:
    if not ids: return []
    # partition by type to keep queries efficient
    str_ids = [str(x) for x in ids if not isinstance(x, int)]
    int_ids = [int(x) for x in ids if isinstance(x, int)]

    if str_ids:
        for b, a in (
            db.query(Book, Author).outerjoin(Author, Book.author_idx == Author.author_idx)
              .filter(Book.work_id.in_(str_ids))
        ):
            yield {"book_id": b.work_id, "title": b.title or "", "author": (a.name if a else "") or "", "description": b.description or ""}

    if int_ids:
        for b, a in (
            db.query(Book, Author).outerjoin(Author, Book.author_idx == Author.author_idx)
              .filter(Book.item_idx.in_(int_ids))
        ):
            yield {"book_id": int(b.item_idx), "title": b.title or "", "author": (a.name if a else "") or "", "description": b.description or ""}

def _enrich_one(rec, slug2id, valid_tone_ids, valid_genre_slugs, tone_slugs, genre_slugs_line):
    user = USER_TEMPLATE.format(
        title=rec["title"],
        author=rec["author"],
        description=rec["description"],
        tone_instructions=f"Fixed tones: [{tone_slugs}]",
        genre_instructions=f"Fixed genres: [{genre_slugs_line}]",
        noisy_subjects_block="",  # no hints for backfill
    )
    raw = call_enrichment_llm(SYSTEM, user)

    # normalize tone_ids like runner
    tone_ids = raw.get("tone_ids")
    if not tone_ids or any(isinstance(t, str) for t in tone_ids):
        mapped = []
        for t in raw.get("tone_ids", []):
            if isinstance(t, int): mapped.append(t)
            elif isinstance(t, str) and t in slug2id: mapped.append(slug2id[t])
        raw["tone_ids"] = mapped

    data = validate_payload(raw, valid_tone_ids, valid_genre_slugs)
    data.subjects = clean_subjects(data.subjects)[:8]

    return {
        "book_id": rec["book_id"],   # stays whatever you selected (int item_idx or string work_id)
        "subjects": data.subjects,
        "tone_ids": data.tone_ids,
        "genre": data.genre,
        "vibe": data.vibe,
        "tags_version": "v1",
        "scores": {}
    }

# ---- writer thread to serialize file appends ----
def _writer_thread(outfile: Path, q: Queue):
    with open(outfile, "a", encoding="utf-8") as f:
        while True:
            item = q.get()
            if item is None:
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            q.task_done()

def main(workers: int = 2, sleep_s: float = 0.0) -> Dict[str, int]:
    # Fail fast (auth/quota/network)
    ensure_enrichment_ready()

    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones()
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()

    need_ids = _iter_failed_ids(OUT_JSONL)
    if not need_ids:
        print("No failed/invalid rows to backfill.")
        return {"ok": 0, "err": 0}

    # Low-impact defaults for a 6 vCPU box that’s also serving web:
    #  - I/O-bound workload → threads are fine
    #  - Start with 2–3 workers; increase carefully if latency stays low
    workers = max(1, int(os.getenv("ENRICH_WORKERS", workers)))

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    ok = err = 0

    # writer
    q = Queue(maxsize=1000)
    wt = Thread(target=_writer_thread, args=(OUT_JSONL, q), daemon=True); wt.start()

    def task(rec):
        try:
            out = _enrich_one(rec, slug2id, valid_tone_ids, valid_genre_slugs, tone_slugs, genre_slugs_line)
            q.put(out)
            if sleep_s: time.sleep(sleep_s)
            return True
        except Exception as e:
            q.put({"book_id": rec["book_id"], "error": str(e), "tags_version": "v1"})
            return False

    with SessionLocal() as db, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for rec in _iter_books_by_ids(db, need_ids):
            futures.append(ex.submit(task, rec))
        for fut in as_completed(futures):
            if fut.result(): ok += 1
            else: err += 1

    # drain writer
    q.join()
    q.put(None); wt.join(timeout=5)

    return {"ok": ok, "err": err}

if __name__ == "__main__":
    # Default: 2 workers, gentle pacing via ENRICH_SLEEP_S if set
    workers = int(os.getenv("ENRICH_WORKERS", "2"))
    sleep_s = float(os.getenv("ENRICH_SLEEP_S", "0.0"))
    res = main(workers=workers, sleep_s=sleep_s)
    print(res)
    import sys; sys.exit(0 if res["err"] == 0 else 2)
