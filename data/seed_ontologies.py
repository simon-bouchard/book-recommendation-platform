# data/seed_ontologies_v2.py
import csv
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import engine

TONES_CSV = ROOT / "ontology" / "tones_v2.csv"
GENRES_CSV = ROOT / "ontology" / "genres_v1.csv"  # or genres_v2 if you change them

ONTOLOGY_VERSION = "v2"
TONE_ID_OFFSET = 100  # Start v2 tone IDs at 100


def seed_tones_v2():
    """Seed tones_v2 with ID offset to avoid conflicts with v1"""
    rows = []
    with open(TONES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print(f"⚠️ No rows in {TONES_CSV}")
        return

    with engine.begin() as conn:
        sql = text("""
            INSERT INTO tones (tone_id, slug, name, ontology_version)
            VALUES (:tone_id, :slug, :name, :ontology_version)
            ON DUPLICATE KEY UPDATE
                slug = VALUES(slug),
                name = VALUES(name),
                ontology_version = VALUES(ontology_version)
        """)

        for r in rows:
            conn.execute(
                sql,
                {
                    "tone_id": int(r["tone_id"]) + TONE_ID_OFFSET,  # Offset IDs
                    "slug": r["slug"],
                    "name": r.get("display_name", r["slug"]),
                    "ontology_version": ONTOLOGY_VERSION,
                },
            )

    print(
        f"✅ Seeded {len(rows)} tones (v2) with IDs {TONE_ID_OFFSET}-{TONE_ID_OFFSET + len(rows) - 1}"
    )


def seed_genres_v2():
    """Seed genres (if ontology changed)"""
    rows = []
    with open(GENRES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print(f"⚠️ No rows in {GENRES_CSV}")
        return

    with engine.begin() as conn:
        sql = text("""
            INSERT INTO genres (slug, name, ontology_version)
            VALUES (:slug, :name, :ontology_version)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                ontology_version = VALUES(ontology_version)
        """)

        for r in rows:
            conn.execute(
                sql,
                {
                    "slug": r["slug"],
                    "name": r.get("display", r["slug"]),
                    "ontology_version": ONTOLOGY_VERSION,
                },
            )

    print(f"✅ Seeded {len(rows)} genres (v2)")


if __name__ == "__main__":
    seed_tones_v2()
    seed_genres_v2()
