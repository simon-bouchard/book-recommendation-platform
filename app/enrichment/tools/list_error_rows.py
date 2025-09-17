# app/enrichment/tools/list_error_rows_by_workid.py
from pathlib import Path
from collections import OrderedDict
import json, csv

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "enrichment_v1.jsonl"
OUT = ROOT / "data" / "error_rows_by_workid.csv"

def _load_last_per_workid(p: Path):
    last = OrderedDict()
    if not p.exists():
        return last
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            bid = obj.get("book_id")
            # keep only work_id-shaped keys (strings)
            if isinstance(bid, str) and bid:
                last[bid] = obj
    return last

def main():
    last = _load_last_per_workid(SRC)
    rows = []
    with SessionLocal() as db:
        for wid, rec in last.items():
            if "error" not in rec:
                continue
            b_a = (db.query(Book, Author)
                     .outerjoin(Author, Book.author_idx == Author.author_idx)
                     .filter(Book.work_id == wid)
                     .first())
            title = (b_a[0].title if b_a else "") or ""
            author = (b_a[1].name if (b_a and b_a[1]) else "") or ""
            rows.append({"work_id": wid, "title": title, "author": author, "error": rec["error"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["work_id","title","author","error"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
