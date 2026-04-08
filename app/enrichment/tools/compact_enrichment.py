# app/compact_enrichment.py
import json
import sys
from collections import OrderedDict
from pathlib import Path

from app.database import SessionLocal
from app.table_models import Book

ROOT = Path(__file__).resolve().parents[3]
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
    # Load last record per work_id from source once
    last = load_last_per_work_id(SRC)
    if not last:
        print("No work_id-keyed records found.")
        return 0

    # Preload all work_id -> item_idx mappings in one query
    with SessionLocal() as db:
        rows = db.query(Book.work_id, Book.item_idx).all()
        mapping = {wid: idx for (wid, idx) in rows if wid and idx is not None}

    ok = err = 0
    with OUT.open("w", encoding="utf-8") as out:
        for wid, rec in last.items():
            idx = mapping.get(wid)
            if idx is None:
                # preserve an explicit error row so it can be found later
                out.write(
                    json.dumps(
                        {
                            "book_id": wid,
                            "error": "Missing item_idx for work_id",
                            "tags_version": rec.get("tags_version", "v1"),
                        }
                    )
                    + "\n"
                )
                err += 1
                continue

            # rewrite with int book_id
            if "error" in rec:
                out.write(
                    json.dumps(
                        {
                            "book_id": int(idx),
                            "error": rec["error"],
                            "tags_version": rec.get("tags_version", "v1"),
                        }
                    )
                    + "\n"
                )
            else:
                out.write(
                    json.dumps(
                        {
                            "book_id": int(idx),
                            "subjects": rec.get("subjects", []),
                            "tone_ids": rec.get("tone_ids", []),
                            "genre": rec.get("genre"),
                            "vibe": rec.get("vibe", ""),
                            "tags_version": rec.get("tags_version", "v1"),
                            "scores": rec.get("scores", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            ok += 1

    print({"rewritten": ok, "missing_item_idx": err, "output": str(OUT)})
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
