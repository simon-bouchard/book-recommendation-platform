# data/import_enrichment_csvs.py
"""
Import raw LLM enrichment tags (genre/tones/llm_subjects/vibe per book) so
the semantic index can be rebuilt with
app/semantic_index/builders/build_enriched_index.py, without re-running the
LLM enrichment pipeline itself.

Expects data/book_enrichment_v2.csv with columns:
    work_id, genre, tones, llm_subjects, vibe
where `tones` and `llm_subjects` are JSON-encoded lists of strings.

Run after data/create_tables.py and data/import_csvs.py:
    python data/import_enrichment_csvs.py
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from app.table_models import Book, BookGenre, BookLLMSubject, BookTone, BookVibe, LLMSubject, Vibe

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "book_enrichment_v2.csv"
TAGS_VERSION = "v2"
GENRE_ONTOLOGY_VERSION = "v2"
TONE_V2_OFFSET = 100  # must match data/seed_ontologies.py
CHUNK = 5000


def load_tone_ids(db) -> dict:
    """
    Ensure every tone slug the export can reference has a `tones` row, and
    return {slug: tone_id}.

    Prefers the v2 ontology id (offset +100, matching data/seed_ontologies.py)
    when a slug exists there; falls back to the v1 file's native id for slugs
    the v2 ontology dropped (the export contains some of these even though
    it's tagged tags_version='v2').
    """
    with open(ROOT / "ontology" / "tones_v1.csv", encoding="utf-8") as f:
        v1_rows = list(csv.DictReader(f))
    with open(ROOT / "ontology" / "tones_v2.csv", encoding="utf-8") as f:
        v2_rows = list(csv.DictReader(f))

    slug_to_id = {}
    upserts = []
    for row in v2_rows:
        tone_id = int(row["tone_id"]) + TONE_V2_OFFSET
        slug_to_id[row["slug"]] = tone_id
        upserts.append((tone_id, row["slug"], row["display_name"], "v2"))
    for row in v1_rows:
        if row["slug"] in slug_to_id:
            continue
        tone_id = int(row["tone_id"])
        slug_to_id[row["slug"]] = tone_id
        upserts.append((tone_id, row["slug"], row["display_name"], "v1"))

    db.execute(
        text("""
            INSERT INTO tones (tone_id, slug, name, ontology_version)
            VALUES (:tone_id, :slug, :name, :ontology_version)
            ON DUPLICATE KEY UPDATE slug = VALUES(slug), name = VALUES(name)
        """),
        [{"tone_id": t, "slug": s, "name": n, "ontology_version": v} for t, s, n, v in upserts],
    )
    db.commit()
    return slug_to_id


def load_genre_slugs(db) -> set:
    """Ensure `genres` has GENRE_ONTOLOGY_VERSION rows (mirrors
    data/seed_ontologies.py so this script works standalone)."""
    with open(ROOT / "ontology" / "genres_v1.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    db.execute(
        text("""
            INSERT INTO genres (slug, name, ontology_version)
            VALUES (:slug, :name, :ontology_version)
            ON DUPLICATE KEY UPDATE name = VALUES(name)
        """),
        [
            {"slug": r["slug"], "name": r["display"], "ontology_version": GENRE_ONTOLOGY_VERSION}
            for r in rows
        ],
    )
    db.commit()
    return {r["slug"] for r in rows}


def parse_list(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def main():
    db = SessionLocal()
    try:
        print("Seeding tone/genre ontology rows referenced by the export...")
        tone_ids = load_tone_ids(db)
        genre_slugs = load_genre_slugs(db)

        print("Loading work_id -> item_idx map...")
        work_to_item_idx = dict(db.query(Book.work_id, Book.item_idx))

        print(f"Reading {CSV_PATH}...")
        with open(CSV_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        subject_cache = {s.subject: s.llm_subject_idx for s in db.query(LLMSubject)}
        vibe_cache = {v.text: v.vibe_id for v in db.query(Vibe)}

        skipped_no_book = 0
        skipped_no_genre = 0

        for i in tqdm(range(0, len(rows), CHUNK), desc="Importing enrichment"):
            chunk = rows[i : i + CHUNK]

            parsed_chunk = []
            new_subjects, new_vibes = set(), set()
            for row in chunk:
                item_idx = work_to_item_idx.get(row["work_id"])
                if item_idx is None:
                    skipped_no_book += 1
                    continue

                subjects = parse_list(row.get("llm_subjects"))
                vibe = (row.get("vibe") or "").strip()
                parsed_chunk.append(
                    {
                        "item_idx": item_idx,
                        "genre": (row.get("genre") or "").strip(),
                        "tones": set(parse_list(row.get("tones"))),
                        "subjects": subjects,
                        "vibe": vibe,
                    }
                )
                new_subjects.update(s for s in subjects if s not in subject_cache)
                if vibe and vibe not in vibe_cache:
                    new_vibes.add(vibe)

            if new_subjects:
                objs = [LLMSubject(subject=s) for s in new_subjects]
                db.add_all(objs)
                db.flush()
                for o in objs:
                    subject_cache[o.subject] = o.llm_subject_idx

            if new_vibes:
                objs = [Vibe(text=v) for v in new_vibes]
                db.add_all(objs)
                db.flush()
                for o in objs:
                    vibe_cache[o.text] = o.vibe_id

            book_genres, book_tones, book_llm_subjects, book_vibes = [], [], [], []
            for item in parsed_chunk:
                if item["genre"]:
                    if item["genre"] in genre_slugs:
                        book_genres.append(
                            BookGenre(
                                item_idx=item["item_idx"],
                                genre_slug=item["genre"],
                                genre_ontology_version=GENRE_ONTOLOGY_VERSION,
                                tags_version=TAGS_VERSION,
                            )
                        )
                    else:
                        skipped_no_genre += 1

                for tone_slug in item["tones"]:
                    tone_id = tone_ids.get(tone_slug)
                    if tone_id:
                        book_tones.append(
                            BookTone(
                                item_idx=item["item_idx"],
                                tone_id=tone_id,
                                tags_version=TAGS_VERSION,
                            )
                        )

                for subject in item["subjects"]:
                    sid = subject_cache.get(subject)
                    if sid:
                        book_llm_subjects.append(
                            BookLLMSubject(
                                item_idx=item["item_idx"],
                                llm_subject_idx=sid,
                                tags_version=TAGS_VERSION,
                            )
                        )

                if item["vibe"]:
                    vid = vibe_cache.get(item["vibe"])
                    if vid:
                        book_vibes.append(
                            BookVibe(
                                item_idx=item["item_idx"], vibe_id=vid, tags_version=TAGS_VERSION
                            )
                        )

            db.add_all(book_genres)
            db.add_all(book_tones)
            db.add_all(book_llm_subjects)
            db.add_all(book_vibes)
            db.flush()

        db.commit()
        print(
            "✅ Enrichment import complete. "
            f"Skipped {skipped_no_book} rows with unknown work_id, "
            f"{skipped_no_genre} with unrecognized genre slug."
        )

    except Exception as e:
        db.rollback()
        print(f"❌ Import failed: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
