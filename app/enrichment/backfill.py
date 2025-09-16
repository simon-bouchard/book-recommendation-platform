# app/enrichment/backfill.py
import csv, json, os, sys, time
from pathlib import Path
from typing import Iterable, Set, Dict, Any, List

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author

from app.enrichment.prompts import SYSTEM, USER_TEMPLATE
from app.enrichment.postprocess import (
    render_tone_slugs, render_genre_slugs, clean_subjects,
)
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm, ensure_enrichment_ready

ROOT = Path(__file__).resolve().parents[2]
TONES_CSV = ROOT / "ontology" / "tones_v1.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"
OUT_JSONL  = ROOT / "data" / "enrichment_v1.jsonl"

def _load_csv_rows(p: Path) -> List[dict]:
    rows: List[dict] = []
    with open(p, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    return rows

def _load_taxonomies():
    tone_rows = _load_csv_rows(TONES_CSV)
    genre_rows = _load_csv_rows(GENRES_CSV)

    tone_slugs = render_tone_slugs(tone_rows)
    slug2id = {r["slug"]: int(r["tone_id"]) for r in tone_rows}
    valid_tone_ids = {int(r["tone_id"]) for r in tone_rows}

    genre_slugs_line = render_genre_slugs(genre_rows)
    valid_genre_slugs = {r["slug"] for r in genre_rows}
    return tone_slugs, slug2id, valid_tone_ids, genre_slugs_line, valid_genre_slugs

def _iter_failed_work_ids(jsonl_path: Path) -> Set[str]:
    """
    Collect work_ids that need backfill: any last-seen line with "error",
    or missing/invalid required fields. Uses the *last* line per work_id.
    Only returns string work_ids (never item_idx).
    """
    needs: Set[str] = set()
    if not jsonl_path.exists():
        return needs

    last: Dict[str, Dict[str, Any]] = {}
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
            if not isinstance(bid, str) or not bid.strip():
                # If it isn't a non-empty string, ignore it (we only backfill work_id)
                continue
            last[bid] = obj

    for wid, obj in last.items():
        if "error" in obj:
            needs.add(wid); continue
        if not isinstance(obj.get("subjects", []), list):
            needs.add(wid); continue
        if not isinstance(obj.get("tone_ids", []), list):
            needs.add(wid); continue
        if not (isinstance(obj.get("genre"), str) and obj["genre"]):
            needs.add(wid); continue
    return needs

def _iter_books_by_work_ids(db: Session, work_ids: Set[str]) -> Iterable[dict]:
    """
    Yield records strictly by Book.work_id. Any rows with NULL/empty work_id are skipped.
    """
    if not work_ids:
        return []
    q = (
        db.query(Book)
        .join(Author, isouter=True)
        .filter(Book.work_id.in_(work_ids))
    )
    for b in q:
        if not b.work_id:
            continue
        yield {
            "work_id": b.work_id,                  # enforce work_id only
            "title": b.title or "",
            "author": b.author.name if b.author else "",
            "description": b.description or "",
        }

def _enrich_one(rec,
                slug2id, valid_tone_ids, valid_genre_slugs,
                tone_slugs, genre_slugs_line) -> Dict[str, Any]:
    user = USER_TEMPLATE.format(
        title=rec["title"],
        author=rec["author"],
        description=rec["description"],
        tone_instructions=f"Fixed tones: [{tone_slugs}]",
        genre_instructions=f"Fixed genres: [{genre_slugs_line}]",
        noisy_subjects_block="",  # backfill path doesn't inject hints
    )
    raw = call_enrichment_llm(SYSTEM, user)

    # Map tone slugs → ids if necessary (same as main runner)
    tone_ids = raw.get("tone_ids")
    if not tone_ids or any(isinstance(t, str) for t in tone_ids):
        mapped = []
        for t in raw.get("tone_ids", []):
            if isinstance(t, int):
                mapped.append(t)
            elif isinstance(t, str) and t in slug2id:
                mapped.append(slug2id[t])
        raw["tone_ids"] = mapped

    data = validate_payload(raw, valid_tone_ids, valid_genre_slugs)
    data.subjects = clean_subjects(data.subjects)[:8]

    return {
        "book_id": rec["work_id"],   # strictly work_id
        "subjects": data.subjects,
        "tone_ids": data.tone_ids,
        "genre": data.genre,
        "vibe": data.vibe,
        "tags_version": "v1",
        "scores": {}
    }

def main(sleep_s: float = 0.0) -> Dict[str, int]:
    # Fail fast on auth/quota/network before doing any work
    ensure_enrichment_ready()

    tone_slugs, slug2id, valid_tone_ids, genre_slugs_line, valid_genre_slugs = _load_taxonomies()

    need_work_ids = _iter_failed_work_ids(OUT_JSONL)
    if not need_work_ids:
        print("No failed or invalid rows to backfill (by work_id).")
        return {"ok": 0, "err": 0, "skipped": 0}

    ok, err = 0, 0
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db, open(OUT_JSONL, "a", encoding="utf-8") as f_out:
        for rec in _iter_books_by_work_ids(db, need_work_ids):
            try:
                out = _enrich_one(rec, slug2id, valid_tone_ids, valid_genre_slugs, tone_slugs, genre_slugs_line)
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                ok += 1
                if sleep_s:
                    time.sleep(sleep_s)
            except Exception as e:
                f_out.write(json.dumps({
                    "book_id": rec["work_id"],   # still the same work_id key
                    "error": str(e),
                    "tags_version": "v1"
                }, ensure_ascii=False) + "\n")
                err += 1
    return {"ok": ok, "err": err, "skipped": 0}

if __name__ == "__main__":
    sleep_s = float(os.getenv("ENRICH_SLEEP_S", "0.0"))
    res = main(sleep_s=sleep_s)
    print(res)
    sys.exit(0 if res["err"] == 0 else 2)
