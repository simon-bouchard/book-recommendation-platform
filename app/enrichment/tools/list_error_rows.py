# app/enrichment/tools/list_last_errors.py
from pathlib import Path
from collections import OrderedDict
import json, csv

from app.database import SessionLocal
from app.table_models import Book, Author

SRC = Path("data/enrichment_v1.jsonl")   # change if your file is elsewhere
OUT = Path("data/last_error_rows.csv")

def load_last_per_work_id(p: Path):
    last = OrderedDict()  # preserves arrival order; we overwrite per work_id
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            wid = obj.get("book_id")
            if isinstance(wid, str) and wid:   # STRICT: work_id strings only
                last[wid] = obj
    return last

def main():
    if not SRC.exists():
        print(f"Missing: {SRC}")
        return 2

    last = load_last_per_work_id(SRC)
    error_wids = [wid for wid, obj in last.items() if isinstance(obj, dict) and ("error" in obj)]

    with SessionLocal() as db:
        q = (db.query(Book, Author)
               .outerjoin(Author, Book.author_idx == Author.author_idx)
               .filter(Book.work_id.in_(error_wids)))
        rows = []
        for b, a in q.all():
            wid = str(b.work_id)
            err = last[wid].get("error", "")
            rows.append({
                "work_id": wid,
                "title": b.title or "",
                "author": (a.name if a else "") or "",
                "error": err
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["work_id","title","author","error"])
        w.writeheader()
        w.writerows(rows)

    print(f"Total work_ids seen: {len(last)}")
    print(f"Errors (last per work_id): {len(error_wids)}")
    print(f"Wrote: {OUT}")

if __name__ == "__main__":
    raise SystemExit(main())

