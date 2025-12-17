# tests/integration/models/setup_test_data.py
"""
Analyzes database to identify suitable user and book IDs for performance testing.
Categorizes users by rating count and subject preferences to cover all code paths.
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
    """
    Find users with >=10 ratings for warm user testing.
    These users will use ALS-based recommendations in auto/behavioral mode.
    """
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


def find_cold_users_with_subjects(db: Session, limit: int = 10) -> list[int]:
    """
    Find cold users (1-9 ratings) who HAVE favorite subjects.
    Tests similarity + Bayesian blended recommendation path.
    """
    users_with_subjects = db.query(UserFavSubject.user_id).distinct().subquery()

    cold_users = (
        db.query(Interaction.user_id, func.count(Interaction.id).label("rating_count"))
        .join(users_with_subjects, Interaction.user_id == users_with_subjects.c.user_id)
        .filter(Interaction.rating.isnot(None))
        .group_by(Interaction.user_id)
        .having(func.count(Interaction.id).between(1, 9))
        .order_by(desc("rating_count"))
        .limit(limit)
        .all()
    )

    return [user_id for user_id, count in cold_users]


def find_cold_users_without_subjects(db: Session, limit: int = 10) -> list[int]:
    """
    Find cold users (1-9 ratings) with NO favorite subjects.
    Tests pure Bayesian fallback path (fastest cold recommendation).
    """
    users_with_subjects = db.query(UserFavSubject.user_id).distinct().subquery()

    cold_users = (
        db.query(Interaction.user_id, func.count(Interaction.id).label("rating_count"))
        .filter(Interaction.rating.isnot(None))
        .group_by(Interaction.user_id)
        .having(func.count(Interaction.id).between(1, 9))
        .subquery()
    )

    no_subject_cold = (
        db.query(cold_users.c.user_id, cold_users.c.rating_count)
        .outerjoin(users_with_subjects, cold_users.c.user_id == users_with_subjects.c.user_id)
        .filter(users_with_subjects.c.user_id.is_(None))
        .order_by(desc(cold_users.c.rating_count))
        .limit(limit)
        .all()
    )

    return [user_id for user_id, _ in no_subject_cold]


def find_test_books(db: Session, limit: int = 10) -> dict[str, list[int]]:
    """
    Find books suitable for similarity testing.
    Categorizes by ALS availability and popularity to test different code paths.
    """
    store = ModelStore()
    als_book_ids = store.get_book_als_id_set()

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

    als_popular = [bid for bid in popular_ids if bid in als_book_ids][:limit]

    return {
        "has_als": als_popular,
        "popular": popular_ids,
        "niche": niche_ids,
    }


def generate_test_config():
    """
    Generate test configuration and save to file.
    Categorizes users and books to cover all performance-critical code paths.
    """
    db = SessionLocal()

    try:
        print("Analyzing database for test data...")

        warm_users = find_warm_users(db, limit=10)
        cold_with_subjects = find_cold_users_with_subjects(db, limit=10)
        cold_without_subjects = find_cold_users_without_subjects(db, limit=10)
        test_books = find_test_books(db, limit=10)

        config = {
            "warm_user_ids": warm_users,
            "cold_with_subjects_user_ids": cold_with_subjects,
            "cold_without_subjects_user_ids": cold_without_subjects,
            "test_book_ids": {
                "has_als": test_books["has_als"],
                "popular": test_books["popular"],
                "niche": test_books["niche"],
                "all": list(
                    set(test_books["has_als"] + test_books["popular"] + test_books["niche"])
                ),
            },
        }

        output_file = Path(__file__).parent / "test_data_config.json"
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nTest configuration saved to: {output_file}")
        print("\nSummary:")
        print(f"  - Warm users: {len(warm_users)}")
        print(f"  - Cold users (with subjects): {len(cold_with_subjects)}")
        print(f"  - Cold users (without subjects): {len(cold_without_subjects)}")
        print(f"  - Books with ALS: {len(test_books['has_als'])}")
        print(f"  - Popular books: {len(test_books['popular'])}")
        print(f"  - Niche books: {len(test_books['niche'])}")

        print("\nNext steps:")
        print("  1. Review test_data_config.json")
        print("  2. Copy IDs to test_models_performance.py")
        print("  3. Run: pytest tests/integration/models/test_models_performance.py -v")

        return config

    finally:
        db.close()


def verify_test_data(config: dict):
    """
    Verify that test data is suitable for performance testing.
    Checks user rating counts and book ALS availability.
    """
    db = SessionLocal()
    store = ModelStore()

    try:
        print("\nVerifying test data...")

        for user_id in config["warm_user_ids"][:3]:
            count = (
                db.query(func.count(Interaction.id))
                .filter(Interaction.user_id == user_id, Interaction.rating.isnot(None))
                .scalar()
            )
            print(f"  User {user_id}: {count} ratings (warm)")

        for user_id in config["cold_with_subjects_user_ids"][:3]:
            count = (
                db.query(func.count(Interaction.id))
                .filter(Interaction.user_id == user_id, Interaction.rating.isnot(None))
                .scalar()
            )
            has_subjects = (
                db.query(UserFavSubject).filter(UserFavSubject.user_id == user_id).first()
                is not None
            )
            print(f"  User {user_id}: {count} ratings (cold, subjects={has_subjects})")

        for user_id in config["cold_without_subjects_user_ids"][:3]:
            count = (
                db.query(func.count(Interaction.id))
                .filter(Interaction.user_id == user_id, Interaction.rating.isnot(None))
                .scalar()
            )
            has_subjects = (
                db.query(UserFavSubject).filter(UserFavSubject.user_id == user_id).first()
                is not None
            )
            print(f"  User {user_id}: {count} ratings (cold, subjects={has_subjects})")

        als_book_ids = store.get_book_als_id_set()
        for book_id in config["test_book_ids"]["has_als"][:3]:
            has_als = book_id in als_book_ids
            print(f"  Book {book_id}: ALS data = {has_als}")

        print("\nTest data verification complete")

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
            print("Config file not found. Run without --verify first.")
            sys.exit(1)

        with open(config_file) as f:
            config = json.load(f)
        verify_test_data(config)
    else:
        config = generate_test_config()
        verify_test_data(config)
