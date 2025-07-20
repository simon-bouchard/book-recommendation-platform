import pandas as pd
from app.database import SessionLocal
from app.table_models import Interaction, User, Book, BookSubject, UserFavSubject

from pathlib import Path

OUTPUT_DIR = Path("models/training/exported_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_table_to_csv(query_result, columns, filename):
    df = pd.DataFrame(query_result, columns=columns)
    df.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"✅ Exported: {filename}")

def main():
    db = SessionLocal()
    try:
        print("🔄 Connecting to DB...")

        # Export interactions (filtered to only include rated ones)
        interactions = pd.read_sql(
            "SELECT user_id, item_idx, rating FROM interactions WHERE rating IS NOT NULL",
            db.bind
        )
        interactions.to_csv(OUTPUT_DIR / "interactions.csv", index=False)
        print("✅ Exported: interactions.csv")

        # Export users and books
        users = pd.read_sql("SELECT * FROM users", db.bind)
        users.to_csv(OUTPUT_DIR / "users.csv", index=False)
        print("✅ Exported: users.csv")

        books = pd.read_sql("SELECT * FROM books", db.bind)
        books.to_csv(OUTPUT_DIR / "books.csv", index=False)
        print("✅ Exported: books.csv")

        # Export user favorite subjects
        user_fav_query = db.query(UserFavSubject.user_id, UserFavSubject.subject_idx).all()
        export_table_to_csv(user_fav_query, ["user_id", "subject_idx"], "user_fav_subjects.csv")

        # Export book subjects
        book_subj_query = db.query(BookSubject.item_idx, BookSubject.subject_idx).all()
        export_table_to_csv(book_subj_query, ["item_idx", "subject_idx"], "book_subjects.csv")

        print("🎉 All exports completed.")

    except Exception as e:
        print(f"❌ Error during export: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()