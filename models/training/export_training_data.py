import pandas as pd
from pathlib import Path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.database import SessionLocal
from app.table_models import (
    Interaction, User, Book, Author,
    BookSubject, UserFavSubject, Subject
)

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    db = SessionLocal()

    try:
        print("🔄 Exporting interactions...")
        interactions = pd.read_sql(
            "SELECT user_id, item_idx, rating FROM interactions",
            db.bind
        )
        interactions.to_pickle(OUTPUT_DIR / "interactions.pkl")

        print("👤 Exporting users...")
        users = pd.read_sql(
            "SELECT user_id, age, age_group, filled_age, country FROM users",
            db.bind
        )
        users.to_pickle(OUTPUT_DIR / "users.pkl")

        print("📚 Exporting books with author name...")
        books = pd.read_sql("SELECT * FROM books", db.bind)
        authors = pd.read_sql("SELECT author_idx, name FROM authors", db.bind)

        books = books.merge(authors, how="left", on="author_idx")
        books = books.drop(columns=["author_idx"])
        books = books.rename(columns={"name": "author_name"})

        rated = interactions[interactions["rating"].notnull()].copy()
        book_stats = rated.groupby("item_idx")["rating"].agg(["count"]).reset_index()
        book_stats.rename(columns={"count": "book_num_ratings"}, inplace=True)
        books = books.merge(book_stats, how="left", on="item_idx")

        books.to_pickle(OUTPUT_DIR / "books.pkl")

        print("📘 Exporting book_subjects...")
        book_subjects = pd.read_sql(
            "SELECT item_idx, subject_idx FROM book_subjects",
            db.bind
        )
        book_subjects.to_pickle(OUTPUT_DIR / "book_subjects.pkl")

        print("🌟 Exporting user_fav_subjects...")
        user_fav_subjects = pd.read_sql(
            "SELECT user_id, subject_idx FROM user_fav_subjects",
            db.bind
        )
        user_fav_subjects.to_pickle(OUTPUT_DIR / "user_fav_subjects.pkl")

        print("🏷️ Exporting subjects...")
        subjects = pd.read_sql(
            "SELECT subject_idx, subject FROM subjects",
            db.bind
        )
        subjects.to_pickle(OUTPUT_DIR / "subjects.pkl")

        print("✅ All data exported to:", OUTPUT_DIR)

    finally:
        db.close()

if __name__ == "__main__":
    main()