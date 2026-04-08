# ops/ol_subjects/ingest_ol_subjects_to_db.py
"""
Ingest cleaned OL subjects from JSONL into database.

Creates and populates:
  - ol_subjects (dictionary table)
  - book_ol_subjects (link table)

Usage:
    python ops/ol_subjects/ingest_ol_subjects_to_db.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.database import SessionLocal
from app.table_models import Book, BookOLSubject, OLSubject

INPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5_combined.jsonl"


def load_jsonl(path: Path):
    """Load all records from JSONL."""
    print(f"Loading {path}...")
    records = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} invalid JSON: {e}")

    print(f"  Loaded {len(records):,} records")
    return records


def map_work_ids_to_item_idx(db, work_ids: set):
    """Query database to map work_id -> item_idx."""
    print(f"\nMapping {len(work_ids):,} work_ids to item_idx...")

    # Query in batches to avoid overwhelming DB
    batch_size = 10000
    work_id_list = list(work_ids)
    mapping = {}

    for i in range(0, len(work_id_list), batch_size):
        batch = work_id_list[i : i + batch_size]
        results = db.query(Book.work_id, Book.item_idx).filter(Book.work_id.in_(batch)).all()

        for work_id, item_idx in results:
            mapping[work_id] = item_idx

    print(f"  Mapped {len(mapping):,} work_ids to item_idx")
    not_found = len(work_ids) - len(mapping)
    if not_found > 0:
        print(f"  Warning: {not_found:,} work_ids not found in books table")

    return mapping


def insert_ol_subjects(db, all_subjects: set):
    """Insert unique subjects into ol_subjects table."""
    print(f"\nInserting {len(all_subjects):,} unique subjects...")

    # Truncate subjects >500 chars (MySQL VARCHAR limit)
    truncated_count = 0
    processed_subjects = set()

    for s in all_subjects:
        if len(s) > 500:
            truncated_count += 1
            processed_subjects.add(s[:500])
        else:
            processed_subjects.add(s)

    if truncated_count > 0:
        print(f"  Warning: Truncated {truncated_count} subjects >500 chars")
        print(f"  Unique subjects after truncation: {len(processed_subjects):,}")

    # Check which subjects already exist
    existing = {s for (s,) in db.query(OLSubject.subject).all()}
    new_subjects = processed_subjects - existing

    if not new_subjects:
        print("  All subjects already exist")
    else:
        print(f"  Found {len(existing):,} existing, inserting {len(new_subjects):,} new...")

        # Bulk insert in batches
        batch_size = 5000
        subjects_list = list(new_subjects)
        inserted = 0

        for i in range(0, len(subjects_list), batch_size):
            batch = subjects_list[i : i + batch_size]
            objects = [OLSubject(subject=s) for s in batch]

            try:
                db.bulk_save_objects(objects)
                db.commit()
                inserted += len(batch)

                if inserted % 10000 == 0:
                    print(f"    Inserted {inserted:,}/{len(new_subjects):,}")

            except Exception as e:
                db.rollback()
                print(f"    Error in batch {i}-{i + batch_size}: {e}")
                # Try one by one for this batch
                for subj in batch:
                    try:
                        db.add(OLSubject(subject=subj))
                        db.commit()
                        inserted += 1
                    except Exception:
                        db.rollback()

        print(f"  Inserted {inserted:,} new subjects")

    # Build mapping subject -> ol_subject_idx
    print("  Building subject -> ol_subject_idx mapping...")
    mapping = {s: idx for s, idx in db.query(OLSubject.subject, OLSubject.ol_subject_idx).all()}
    print(f"  Mapped {len(mapping):,} subjects")

    return mapping, truncated_count


def insert_book_ol_subject_links(db, records, work_id_map, subject_map):
    """Insert book-subject links into book_ol_subjects table."""
    print("\nBuilding book-subject links...")

    links = []
    skipped_work_id = 0
    skipped_subject = 0

    for rec in records:
        work_id = rec.get("work_id")
        subjects = rec.get("subjects", [])

        # Map work_id to item_idx
        if work_id not in work_id_map:
            skipped_work_id += 1
            continue

        item_idx = work_id_map[work_id]

        # Map each subject to ol_subject_idx
        for subj in subjects:
            if not isinstance(subj, str):
                continue

            # Truncate if needed (must match what was done in insert_ol_subjects)
            subj_key = subj[:500] if len(subj) > 500 else subj

            if subj_key not in subject_map:
                skipped_subject += 1
                continue

            ol_subject_idx = subject_map[subj_key]
            links.append({"item_idx": item_idx, "ol_subject_idx": ol_subject_idx})

    print(f"  Built {len(links):,} links")
    if skipped_work_id > 0:
        print(f"  Skipped {skipped_work_id:,} records (work_id not found)")
    if skipped_subject > 0:
        print(f"  Skipped {skipped_subject:,} subject instances (subject not in mapping)")

    # Bulk insert with duplicate handling
    print(f"\nInserting {len(links):,} book-subject links...")

    batch_size = 5000
    inserted = 0
    duplicates = 0

    for i in range(0, len(links), batch_size):
        batch = links[i : i + batch_size]

        # Use raw SQL with INSERT IGNORE for MySQL
        try:
            values = ", ".join(f"({link['item_idx']}, {link['ol_subject_idx']})" for link in batch)

            sql = text(f"""
                INSERT IGNORE INTO book_ol_subjects (item_idx, ol_subject_idx)
                VALUES {values}
            """)

            result = db.execute(sql)
            db.commit()

            rows_inserted = result.rowcount
            inserted += rows_inserted
            duplicates += len(batch) - rows_inserted

            if inserted % 50000 == 0:
                print(
                    f"    Inserted {inserted:,}/{len(links):,} (skipped {duplicates:,} duplicates)"
                )

        except Exception as e:
            db.rollback()
            print(f"    Error in batch {i}-{i + batch_size}: {e}")

            # Fallback: try SQLAlchemy ORM with individual error handling
            for link in batch:
                try:
                    obj = BookOLSubject(
                        item_idx=link["item_idx"], ol_subject_idx=link["ol_subject_idx"]
                    )
                    db.add(obj)
                    db.commit()
                    inserted += 1
                except Exception:
                    db.rollback()
                    duplicates += 1

    print(f"  Inserted {inserted:,} links")
    print(f"  Skipped {duplicates:,} duplicates")

    return {"inserted": inserted, "duplicates": duplicates}


def verify_ingestion(db):
    """Verify data was inserted correctly."""
    print("\nVerifying ingestion...")

    subject_count = db.query(OLSubject).count()
    link_count = db.query(BookOLSubject).count()
    books_with_subjects = db.query(BookOLSubject.item_idx).distinct().count()

    print(f"  ol_subjects: {subject_count:,} rows")
    print(f"  book_ol_subjects: {link_count:,} rows")
    print(f"  Books with OL subjects: {books_with_subjects:,}")

    # Average subjects per book
    if books_with_subjects > 0:
        avg = link_count / books_with_subjects
        print(f"  Avg subjects per book: {avg:.1f}")

    # Sample data
    print("\n  Sample data (first 5 books):")
    sample = (
        db.query(Book.title, OLSubject.subject)
        .join(BookOLSubject, Book.item_idx == BookOLSubject.item_idx)
        .join(OLSubject, BookOLSubject.ol_subject_idx == OLSubject.ol_subject_idx)
        .limit(5)
        .all()
    )

    for title, subject in sample:
        print(f"    {title[:50]} -> {subject[:50]}")


def main():
    """Main ingestion process."""
    print("=" * 80)
    print("OPEN LIBRARY SUBJECTS INGESTION")
    print("=" * 80)

    if not INPUT_JSONL.exists():
        print(f"\nError: Input file not found: {INPUT_JSONL}")
        print("Run the cleaning pipeline first.")
        return 1

    # Load JSONL
    records = load_jsonl(INPUT_JSONL)

    # Collect all work_ids and subjects
    all_work_ids = {rec["work_id"] for rec in records if rec.get("work_id")}
    all_subjects = set()
    for rec in records:
        subjects = rec.get("subjects", [])
        if isinstance(subjects, list):
            all_subjects.update(s for s in subjects if isinstance(s, str))

    print("\nSummary:")
    print(f"  Records: {len(records):,}")
    print(f"  Unique work_ids: {len(all_work_ids):,}")
    print(f"  Unique subjects: {len(all_subjects):,}")

    # Database operations
    db = SessionLocal()

    try:
        # Step 1: Map work_id to item_idx
        work_id_map = map_work_ids_to_item_idx(db, all_work_ids)

        # Step 2: Insert subjects
        subject_map, truncated_count = insert_ol_subjects(db, all_subjects)

        # Step 3: Insert links
        link_stats = insert_book_ol_subject_links(db, records, work_id_map, subject_map)

        # Step 4: Verify
        verify_ingestion(db)

        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print("=" * 80)
        print(f"✓ Subjects: {len(subject_map):,}")
        if truncated_count > 0:
            print(f"  (truncated {truncated_count} subjects >500 chars)")
        print(f"✓ Links: {link_stats['inserted']:,}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
