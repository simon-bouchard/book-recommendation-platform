# data/seed_ontologies.py
import csv
from pathlib import Path
from typing import List, Dict

from sqlalchemy import text
from app.database import engine

ROOT = Path(__file__).resolve().parents[1]
TONES_CSV = ROOT / "ontology" / "tones_v1.csv"     # expects columns: tone_id, slug [, name]
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"   # expects columns: slug [, name]

def _read_csv(p: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with p.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({k.strip(): (v.strip() if v is not None else v) for k, v in r.items()})
    return rows

def seed_tones():
    rows = _read_csv(TONES_CSV)
    if not rows:
        print(f"⚠️ No rows found in {TONES_CSV}")
        return
    with engine.begin() as conn:
        # insert with optional `name` if present
        has_name = "name" in rows[0]
        if has_name:
            sql = text("""
                INSERT INTO tones (tone_id, slug, name)
                VALUES (:tone_id, :slug, :name)
                ON DUPLICATE KEY UPDATE slug = VALUES(slug), name = VALUES(name)
            """)
        else:
            sql = text("""
                INSERT INTO tones (tone_id, slug)
                VALUES (:tone_id, :slug)
                ON DUPLICATE KEY UPDATE slug = VALUES(slug)
            """)
        for r in rows:
            params = {
                "tone_id": int(r["tone_id"]),
                "slug": r["slug"],
            }
            if has_name:
                params["name"] = r.get("name")
            conn.execute(sql, params)
    print(f"✅ Seeded tones from {TONES_CSV}")

def seed_genres():
    rows = _read_csv(GENRES_CSV)
    if not rows:
        print(f"⚠️ No rows found in {GENRES_CSV}")
        return
    with engine.begin() as conn:
        has_name = "name" in rows[0]
        if has_name:
            sql = text("""
                INSERT INTO genres (slug, name)
                VALUES (:slug, :name)
                ON DUPLICATE KEY UPDATE name = VALUES(name)
            """)
        else:
            sql = text("""
                INSERT INTO genres (slug)
                VALUES (:slug)
                ON DUPLICATE KEY UPDATE slug = VALUES(slug)
            """)
        for r in rows:
            params = {"slug": r["slug"]}
            if has_name:
                params["name"] = r.get("name")
            conn.execute(sql, params)
    print(f"✅ Seeded genres from {GENRES_CSV}")

if __name__ == "__main__":
    seed_tones()
    seed_genres()

