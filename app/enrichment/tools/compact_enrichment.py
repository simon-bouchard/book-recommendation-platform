# app/compact_enrichment.py
import json, sys
from pathlib import Path
from collections import OrderedDict
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "enrichment_v1.jsonl"
OUT = ROOT / "data" / "enrichment_v1.item_idx.jsonl"

def load_last_per_work_id(p: Path):
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
            wid = obj.get("book_id")
            if not isinstance(wid, str) or not wid:
                continue  # only remap records keyed by work_id (string)
            last[wid] = obj
    return last

def main():
    last = load_last_per_work_id(SRC)
    if not last:
        print("No work_id-keyed records found.")
        return 0

    ok = err = 0
    with SessionLocal() as db, OUT.open("w", encoding="utf-8") as out:
        for wid, rec in last.items():
            # lookup item_idx
            b: Book | None = db.query(Book).filter(Book.work_id == wid).first()
            if not b or b.item_idx is None:
                # preserve an explicit error row so it can be found later
                out.write(json.dumps({"book_id": wid, "error": "Missing item_idx for work_id", "tags_version": rec.get("tags_version","v1")}) + "\n")
                err += 1
                continue
            # rewrite with int book_id
            if "error" in rec:
                out.write(json.dumps({"book_id": int(b.item_idx), "error": rec["error"], "tags_version": rec.get("tags_version","v1")}) + "\n")
            else:
                out.write(json.dumps({
                    "book_id": int(b.item_idx),
                    "subjects": rec.get("subjects", []),
                    "tone_ids": rec.get("tone_ids", []),
                    "genre": rec.get("genre"),
                    "vibe": rec.get("vibe",""),
                    "tags_version": rec.get("tags_version","v1"),
                    "scores": rec.get("scores", {})
                }, ensure_ascii=False) + "\n")
            ok += 1
    print({"rewritten": ok, "missing_item_idx": err, "output": str(OUT)})
    return 0 if err == 0 else 2

if __name__ == "__main__":
    sys.exit(main())
