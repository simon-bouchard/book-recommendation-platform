# tests/integration/models/setup_test_data.py
"""
Test Data Configuration for Performance Tests
==============================================

This file helps you identify and configure test data for performance testing.

Run this script to analyze your database and suggest test IDs:
    python tests/integration/models/setup_test_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import User, Book, Interaction, UserFavSubject
from models.shared_utils import ModelStore
import json


def find_warm_users(db: Session, limit: int = 10) -> list[int]:
    """Find users with >=10 ratings for warm user testing."""
    warm_users = (
        db.query(Interaction.user_id, func.count(Interaction.id).label("rating_count"))
        .filter(Interaction.rating.isnot(None))
        .group_by(Interaction.user_id)
        .having(func.count(Interaction.id) >= 10)
        .order_by(desc("rating_count"))
        .limit(limit)
        .all()
    )
    return [user_id for user_id, count in warm_users]


def find_cold_users(db: Session, limit: int = 10) -> list[int]:
    """Find users with 1-9 ratings for cold user testing."""
    cold_users = (
        db.query(Interaction.user_id, func.count(Interaction.id).label("rating_count"))
        .filter(Interaction.rating.isnot(None))
        .group_by(Interaction.user_id)
        .having(func.count(Interaction.id).between(1, 9))
        .order_by(desc("rating_count"))
        .limit(limit)
        .all()
    )
    return [user_id for user_id, count in cold_users]


def find_no_subject_users(db: Session, limit: int = 10) -> list[int]:
    """Find users with ratings but no favorite subjects."""
    users_with_ratings = (
        db.query(Interaction.user_id)
        .filter(Interaction.rating.isnot(None))
        .group_by(Interaction.user_id)
        .having(func.count(Interaction.id) >= 5)  # At least 5 ratings
        .subquery()
    )

    users_with_subjects = db.query(UserFavSubject.user_id).distinct().subquery()

    no_subject_users = (
        db.query(users_with_ratings.c.user_id)
        .outerjoin(
            users_with_subjects, users_with_ratings.c.user_id == users_with_subjects.c.user_id
        )
        .filter(users_with_subjects.c.user_id.is_(None))
        .limit(limit)
        .all()
    )

    return [user_id for (user_id,) in no_subject_users]


def find_test_books(db: Session, limit: int = 10) -> dict[str, list[int]]:
    """
    Find books suitable for similarity testing.
    Returns dict with categories: has_als, popular, niche
    """
    store = ModelStore()

    # Books with ALS data
    als_book_ids = store.get_book_als_id_set()

    # Popular books (high rating count)
    popular_books = (
        db.query(Book.item_idx, func.count(Interaction.id).label("rating_count"))
        .join(Interaction, Interaction.item_idx == Book.item_idx)
        .filter(Interaction.rating.isnot(None))
        .group_by(Book.item_idx)
        .having(func.count(Interaction.id) >= 50)
        .order_by(desc("rating_count"))
        .limit(limit)
        .all()
    )
    popular_ids = [book_id for book_id, _ in popular_books]

    # Niche books (low rating count but not zero)
    niche_books = (
        db.query(Book.item_idx, func.count(Interaction.id).label("rating_count"))
        .join(Interaction, Interaction.item_idx == Book.item_idx)
        .filter(Interaction.rating.isnot(None))
        .group_by(Book.item_idx)
        .having(func.count(Interaction.id).between(5, 15))
        .order_by(desc("rating_count"))
        .limit(limit)
        .all()
    )
    niche_ids = [book_id for book_id, _ in niche_books]

    # Books with ALS data (intersection with popular)
    als_popular = [bid for bid in popular_ids if bid in als_book_ids][:limit]

    return {
        "has_als": als_popular,
        "popular": popular_ids,
        "niche": niche_ids,
    }


def generate_test_config():
    """Generate test configuration and save to file."""
    db = SessionLocal()

    try:
        print("🔍 Analyzing database for test data...")

        warm_users = find_warm_users(db, limit=10)
        cold_users = find_cold_users(db, limit=10)
        no_subject_users = find_no_subject_users(db, limit=5)
        test_books = find_test_books(db, limit=10)

        config = {
            "warm_user_ids": warm_users,
            "cold_user_ids": cold_users,
            "no_subject_user_ids": no_subject_users,
            "test_book_ids": {
                "has_als": test_books["has_als"],
                "popular": test_books["popular"],
                "niche": test_books["niche"],
                "all": list(
                    set(test_books["has_als"] + test_books["popular"] + test_books["niche"])
                ),
            },
        }

        # Save to JSON
        output_file = Path(__file__).parent / "test_data_config.json"
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Test configuration saved to: {output_file}")
        print("\n📊 Summary:")
        print(f"  - Warm users: {len(warm_users)}")
        print(f"  - Cold users: {len(cold_users)}")
        print(f"  - No subject users: {len(no_subject_users)}")
        print(f"  - Books with ALS: {len(test_books['has_als'])}")
        print(f"  - Popular books: {len(test_books['popular'])}")
        print(f"  - Niche books: {len(test_books['niche'])}")

        print("\n📝 Next steps:")
        print("  1. Review test_data_config.json")
        print("  2. Copy IDs to test_models_performance.py")
        print("  3. Run: pytest tests/integration/models/test_models_performance.py -v")

        return config

    finally:
        db.close()


def verify_test_data(config: dict):
    """Verify that test data is suitable for performance testing."""
    db = SessionLocal()
    store = ModelStore()

    try:
        print("\n🔬 Verifying test data...")

        # Verify warm users actually have enough ratings
        for user_id in config["warm_user_ids"][:3]:
            count = (
                db.query(func.count(Interaction.id))
                .filter(Interaction.user_id == user_id, Interaction.rating.isnot(None))
                .scalar()
            )
            print(f"  ✓ User {user_id}: {count} ratings (warm)")

        # Verify cold users
        for user_id in config["cold_user_ids"][:3]:
            count = (
                db.query(func.count(Interaction.id))
                .filter(Interaction.user_id == user_id, Interaction.rating.isnot(None))
                .scalar()
            )
            print(f"  ✓ User {user_id}: {count} ratings (cold)")

        # Verify books with ALS
        als_book_ids = store.get_book_als_id_set()
        for book_id in config["test_book_ids"]["has_als"][:3]:
            has_als = book_id in als_book_ids
            print(f"  ✓ Book {book_id}: ALS data = {has_als}")

        print("\n✅ Test data verification complete")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup performance test data")
    parser.add_argument("--verify", action="store_true", help="Verify existing config")
    args = parser.parse_args()

    if args.verify:
        config_file = Path(__file__).parent / "test_data_config.json"
        if not config_file.exists():
            print("❌ Config file not found. Run without --verify first.")
            sys.exit(1)

        with open(config_file) as f:
            config = json.load(f)
        verify_test_data(config)
    else:
        config = generate_test_config()
        verify_test_data(config)
