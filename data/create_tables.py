# data/create_tables.py
"""
Create all database tables including Phase 1 staging tables.

Usage:
    python data/create_tables.py              # Create all tables
    python data/create_tables.py --migrate    # Migrate existing enrichment_errors table
"""

import argparse
import os
import sys

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import engine
from app.table_models import Base


def create_all_tables():
    """Create all tables defined in table_models"""
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    # List created tables
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\nTotal tables: {len(tables)}")

    # Group tables by category
    core_tables = [t for t in tables if not t.startswith("tmp_") and "enrichment" not in t]
    enrichment_tables = [
        t
        for t in tables
        if "enrichment" in t
        or t
        in [
            "tones",
            "genres",
            "vibes",
            "llm_subjects",
            "book_tones",
            "book_genres",
            "book_vibes",
            "book_llm_subjects",
        ]
    ]
    staging_tables = [t for t in tables if t.startswith("tmp_")]

    print(f"\nCore tables ({len(core_tables)}):")
    for t in sorted(core_tables):
        print(f"  - {t}")

    print(f"\nEnrichment tables ({len(enrichment_tables)}):")
    for t in sorted(enrichment_tables):
        print(f"  - {t}")

    if staging_tables:
        print(f"\nStaging tables ({len(staging_tables)}):")
        for t in sorted(staging_tables):
            print(f"  - {t}")


def migrate_enrichment_errors():
    """Migrate enrichment_errors to Phase 1 structure"""
    print("Running enrichment_errors migration...\n")

    # Import the migration function
    from data.migrate_enrichment_errors import migrate_enrichment_errors as do_migrate

    do_migrate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create database tables")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate existing enrichment_errors table to Phase 1 structure",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DATABASE SETUP")
    print("=" * 80 + "\n")

    if args.migrate:
        migrate_enrichment_errors()
        print("\n" + "=" * 80)
    else:
        create_all_tables()
        print("\n" + "=" * 80)
        print("\nNext steps:")
        print("  1. If you have existing enrichment_errors data, run:")
        print("     python data/create_tables.py --migrate")
        print("  2. Import your CSV data:")
        print("     python data/import_csvs.py")
        print("  3. Set up Kafka infrastructure")
        print("=" * 80)
