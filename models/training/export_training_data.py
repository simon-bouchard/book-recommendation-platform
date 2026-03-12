# models/training/export_training_data.py
"""
Exports a consistent snapshot of all training and runtime data from the
database into the artifact staging directory (staging/data/).
"""

import shutil

import pandas as pd

from app.database import SessionLocal
from models.core.paths import PATHS

_EXPORT_SENTINEL = ".export_complete"


def main() -> None:
    """
    Export all required data tables to PATHS.staging_data_dir.

    The staging data directory is cleared before writing so that no stale
    files from a previous training run can survive into the new snapshot.
    Each table is written as a pickle file using the canonical path defined
    in PATHS, ensuring consistency with both training scripts and runtime
    loaders.

    A sentinel file (.export_complete) is written as the final step.
    artifact_registry._assert_staging_complete() checks for this file before
    allowing promotion, so a crash mid-export will be caught at the gate.
    """
    output_dir = PATHS.staging_data_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    db = SessionLocal()
    engine = db.get_bind()

    try:
        print("Exporting interactions...")
        interactions = pd.read_sql(
            "SELECT user_id, item_idx, rating FROM interactions",
            engine,
        )
        interactions.to_pickle(output_dir / "interactions.pkl")

        rated = interactions[interactions["rating"].notnull()].copy()
        global_avg_rating = rated["rating"].mean()

        print("Exporting users...")
        users = pd.read_sql(
            "SELECT user_id, age, filled_age, country FROM users",
            engine,
        )

        user_stats = (
            rated.groupby("user_id")["rating"]
            .agg(
                user_num_ratings="count",
                user_avg_rating="mean",
                user_rating_std="std",
            )
            .reset_index()
        )

        users = users.merge(user_stats, how="left", on="user_id")
        users["user_num_ratings"] = users["user_num_ratings"].fillna(0).astype(int)
        users["user_rating_std"] = users["user_rating_std"].fillna(0.0)
        users["user_avg_rating"] = users["user_avg_rating"].fillna(global_avg_rating)
        users["age"] = users["age"].fillna(users["age"].mean())
        users["country"] = users["country"].fillna("Unknown")
        users.to_pickle(output_dir / "users.pkl")

        print("Exporting books...")
        books = pd.read_sql(
            """
            SELECT
                item_idx, title, author_idx, isbn, cover_id,
                main_subject, year, filled_year,
                num_pages, filled_num_pages, language
            FROM books
            """,
            engine,
        )

        authors = pd.read_sql("SELECT author_idx, name FROM authors", engine)
        books = books.merge(authors, how="left", on="author_idx")
        books = books.drop(columns=["author_idx"])
        books = books.rename(columns={"name": "author"})

        book_stats = (
            rated.groupby("item_idx")["rating"]
            .agg(
                book_num_ratings="count",
                book_avg_rating="mean",
                book_rating_std="std",
            )
            .reset_index()
        )

        books = books.merge(book_stats, how="left", on="item_idx")
        books["book_num_ratings"] = books["book_num_ratings"].fillna(0).astype(int)
        books["book_rating_std"] = books["book_rating_std"].fillna(0.0)
        books["book_avg_rating"] = books["book_avg_rating"].fillna(global_avg_rating)
        books.to_pickle(output_dir / "books.pkl")

        print("Exporting book_subjects...")
        book_subjects = pd.read_sql(
            "SELECT item_idx, subject_idx FROM book_subjects",
            engine,
        )
        book_subjects.to_pickle(output_dir / "book_subjects.pkl")

        print("Exporting user_fav_subjects...")
        user_fav_subjects = pd.read_sql(
            "SELECT user_id, subject_idx FROM user_fav_subjects",
            engine,
        )
        user_fav_subjects.to_pickle(output_dir / "user_fav_subjects.pkl")

        print("Exporting subjects...")
        subjects = pd.read_sql(
            "SELECT subject_idx, subject FROM subjects",
            engine,
        )
        subjects.to_pickle(output_dir / "subjects.pkl")

        (output_dir / _EXPORT_SENTINEL).touch()
        print(f"All data exported to: {output_dir}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
