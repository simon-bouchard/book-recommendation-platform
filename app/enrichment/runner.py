# app/enrichment/runner.py
import csv, json, time
from pathlib import Path
from sqlalchemy.orm import Session

from app.enrichment.prompts import SYSTEM, USER_TEMPLATE
from app.enrichment.postprocess import (
    render_tone_slugs, render_genre_slugs, clean_subjects,  # clean_subjects is for OUTPUT capping/dedupe only
)
from app.enrichment.validator import validate_payload
from app.enrichment.llm_client import call_enrichment_llm

from app.database import SessionLocal
from app.table_models import Book, Author, Subject, BookSubject

ROOT = Path(__file__).resolve().parents[2]
TONES_CSV = ROOT / "ontology" / "tones_v1.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"
OUT_JSONL = ROOT / "data" / "enrichment_v1.jsonl"

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

def _subjects_for_book(db: Session, item_idx: int):
    """
    Return ALL raw subjects linked to this book, with exact de-duplication only
    (no filtering, no caps). Order preserved on first occurrence.
    """
    rows = (
        db.query(Subject.subject)
          .join(BookSubject, Subject.subject_idx == BookSubject.subject_idx)
          .filter(BookSubject.item_idx == item_idx)
          .all()
    )
    seen = set()
    out = []
    for (s,) in rows:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def iter_books_from_db(db: Session, limit: int | None = None):
    q = db.query(Book).join(Author, isouter=True)
    if limit:
        q = q.limit(limit)
    for b in q:
        yield {
            "work_id": b.work_id or b.item_idx,
            "item_idx": b.item_idx,  # needed to fetch subjects
            "title": b.title or "",
            "author": b.author.name if b.author else "",
            "description": b.description or "",
        }

def main(limit: int | None = None, sleep_s: float = 0.0):
    tone_rows, tone_slugs, valid_tone_ids, slug2id = load_tones()
    genre_rows, genre_slugs_line, valid_genre_slugs = load_genres()
    count_ok, count_err = 0, 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db, open(OUT_JSONL, "a", encoding="utf-8") as f_out:
        for rec in iter_books_from_db(db, limit):
            # Add ALL internal subjects as extra context (no filtering)
            raw_subjects = _subjects_for_book(db, rec["item_idx"])
            if raw_subjects:
                noisy_subjects_block = "Noisy subject hints (may contain errors): " + "; ".join(raw_subjects)
            else:
                noisy_subjects_block = "No noisy subject hints provided."

            user = USER_TEMPLATE.format(
                title=rec["title"],
                author=rec["author"],
                description=rec["description"],
                noisy_subjects_block=noisy_subjects_block,
                tone_instructions=f"Fixed tones: [{tone_slugs}]",
                genre_instructions=f"Fixed genres: [{genre_slugs_line}]",
            )

            def _one_try():
                raw = call_enrichment_llm(SYSTEM, user)

                # --- normalize optional fields before validation ---
                # tones: map slugs → ids if needed
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

                # subjects: OUTPUT cleanup/cap remains (not related to input hints)
                data.subjects = clean_subjects(data.subjects)[:8]

                return {
                    "book_id": rec["work_id"],
                    "subjects": data.subjects,
                    "tone_ids": data.tone_ids,
                    "genre": data.genre,
                    "vibe": data.vibe,
                    "tags_version": "v1",
                    "scores": {}
                }

            try:
                out = _one_try()
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                count_ok += 1
                if sleep_s: time.sleep(sleep_s)
            except Exception:
                # retry once
                try:
                    out = _one_try()
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    count_ok += 1
                except Exception as e2:
                    f_out.write(json.dumps({
                        "book_id": rec["work_id"],
                        "error": str(e2),
                        "tags_version": "v1"
                    }, ensure_ascii=False) + "\n")
                    count_err += 1

    return {"ok": count_ok, "error": count_err}

if __name__ == "__main__":
    print(main())
