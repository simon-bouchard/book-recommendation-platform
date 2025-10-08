"""
ops/ol_subjects/clean_subjects_v5.py

Load the v4 cleaned subjects file (JSONL of {"work_id":..., "subjects":[...]})
Remove any record whose work_id is NOT present in the books table of your database,
and write the remaining records to a v5 JSONL file.

Place this file at the repository root (same layout that clean_subjects_v4.py expects),
then run:

    python ops/ol_subjects/clean_subjects_v5.py

This script will try to import your project's DB models the same way your app does,
but has a fallback to import 'database.py' and 'table_models.py' if those imports fail.
Make sure your environment has DATABASE_URL set (or otherwise your app DB config).
"""

import json
from pathlib import Path
import sys

# --- Paths (adjust if your repo layout differs) ---
ROOT = Path(__file__).resolve().parents[2]
INPUT_V4 = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v4.jsonl"
OUTPUT_V5 = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5.jsonl"

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# --- Import DB session and Book model (try app-style import, fall back to local files) ---
try:
    # Preferred: same imports your app uses
    from app.database import SessionLocal
    from app.table_models import Book
except Exception:
    try:
        # Fallback if running from repo root where files are present directly
        # Adjust sys.path so local modules can be imported
        repo_root = str(ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from database import SessionLocal
        from table_models import Book
    except Exception as e:
        raise ImportError(
            "Could not import database/session or Book model. "
            "Ensure this script is run from your project root and that DB modules are importable. "
            f"Original error: {e}"
        )

def load_existing_work_ids():
    """Return a set of work_id strings that exist in the books table."""
    session = SessionLocal()
    try:
        # Query only non-null work_ids
        rows = session.query(Book.work_id).filter(Book.work_id != None).all()
        # rows is list of single-element tuples; convert & normalize to str
        work_ids = {str(r[0]).strip() for r in rows if r[0] is not None}
        return work_ids
    finally:
        session.close()

def filter_v4_to_v5(input_path: Path, output_path: Path, existing_work_ids: set):
    total_in = 0
    total_out = 0
    removed_count = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            total_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # skip bad lines but count them
                removed_count += 1
                continue

            work_id = rec.get("work_id")
            if work_id is None:
                removed_count += 1
                continue

            # normalize to string for comparison
            work_id_str = str(work_id).strip()

            if work_id_str in existing_work_ids:
                # keep
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_out += 1
            else:
                removed_count += 1

    return {
        "total_in": total_in,
        "total_out": total_out,
        "removed": removed_count,
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    if not INPUT_V4.exists():
        raise FileNotFoundError(f"Input v4 file not found: {INPUT_V4}")

    print("Loading existing work_ids from DB...")
    work_ids = load_existing_work_ids()
    print(f"Found {len(work_ids):,} work_ids in DB.")

    print(f"Filtering {INPUT_V4} -> {OUTPUT_V5} ...")
    stats = filter_v4_to_v5(INPUT_V4, OUTPUT_V5, work_ids)

    print("\nDone.")
    print(f"  Records read:    {stats['total_in']:,}")
    print(f"  Records kept:    {stats['total_out']:,}")
    print(f"  Records removed: {stats['removed']:,}")
    print(f"  Output written:  {stats['output_path']}")
