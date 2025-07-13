# data/import_csvs.py
import os
import sys
import pandas as pd
from sqlalchemy import text
import json
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from app.table_models import User, Book, BookSubject, UserFavSubject, Interaction, Author

def import_users():

    df = pd.read_csv("data/users.csv")

    users = []
    for _, row in df.iterrows():
        user = User(
            user_id=int(row["user_id"]),
            username=f"user{int(row['user_id'])}",
            email=f"user{int(row['user_id'])}@example.com",
            password="not_used_but_required",
            created_at=pd.Timestamp.utcnow(),
            age=row["age"],
            age_group=row["age_group"],
            filled_age=row["filled_age"],
            country=row["country"],
        )
        users.append(user)
    return users

def import_authors():
    df = pd.read_csv("data/authors.csv")
    df = df.where(pd.notnull(df), None)

    authors = []
    for _, row in df.iterrows():
        alt_names = None
        try:
            parsed = ast.literal_eval(row["alternate_names"])
            if isinstance(parsed, list):
                alt_names = json.dumps(parsed)
        except Exception as e:
            print(f"⚠️ Skipping bad alternate_names: {row['alternate_names'][:80]} → {e}")

        authors.append(Author(
            author_id=row["author_id"],
            name=row["name"],
            birth_date=str(row["birth_date"]) if row["birth_date"] else None,
            death_date=str(row["death_date"]) if row["death_date"] else None,
            bio=row["bio"] if row["bio"] else None,
            alternate_names=alt_names
        ))

    print(f"✅ Parsed {len(authors)} authors")
    return authors

def import_books():
    df = pd.read_csv("data/books.csv")

    # Drop specifically blacklisted books
    df = df[~df["work_id"].isin([
        "unmapped_055299619X",
        "unmapped_156947012x",
        "unmapped_078688505x",
        "unmapped_039331863x"
    ])]

    # Normalize author_id as string
    df["author_id"] = df["author_id"].astype(str).str.strip()

    # Get valid author_ids from the DB
    valid_author_ids = {row[0] for row in SessionLocal().query(Author.author_id).all()}

    seen = set()
    books = []

    for _, row in df.iterrows():
        work_id = str(row["work_id"]).strip()
        if work_id in seen:
            continue
        seen.add(work_id)

        author_id = row["author_id"] if row["author_id"] in valid_author_ids else None

        books.append(Book(
            work_id=work_id,
            title=row["title"],
            year=row["year"],
            year_bucket=row["year_bucket"],
            filled_year=row["filled_year"],
            description=row["description"],
            cover_id=row["cover_id"],
            language=row["language"],
            num_pages=row["num_pages"],
            filled_num_pages=row["filled_num_pages"],
            author_id=author_id,
            isbn=row["isbn"]
        ))

    return books

def import_interactions():
    df = pd.read_csv("data/interactions.csv")
    return df

def import_book_subjects():
    df = pd.read_csv("data/books_to_subjects.csv")
    return df.dropna(subset=["work_id", "subjects"])

def import_user_fav_subjects():
    df = pd.read_csv("data/users_to_subjects.csv")
    return df.dropna(subset=["user_id", "favorite_subjects"])

def main():
    db = SessionLocal()

    try:
        # --- Step 1: Clear all tables and commit ---
        db.execute(text("DELETE FROM interactions"))
        db.execute(text("DELETE FROM user_fav_subjects"))
        db.execute(text("DELETE FROM book_subjects"))
        db.execute(text("DELETE FROM books"))
        db.execute(text("DELETE FROM authors"))
        db.execute(text("DELETE FROM users"))
        db.commit() 

        # --- Step 2: Import new data ---
        db.bulk_save_objects(import_users())

        authors = import_authors()
        db.add_all(authors)
        db.commit()

        print(f"✅ Inserted {len(authors)} authors into the database.")
        sample = db.query(Author).first()
        print("Sample author from DB:", sample.author_id, "-", sample.name if sample else "❌ None found")

        db.bulk_save_objects(import_books())
        
        from tqdm import tqdm

        inter_df = import_interactions()
        CHUNK = 10000

        for i in tqdm(range(0, len(inter_df), CHUNK), desc="Importing interactions"):
            chunk = inter_df.iloc[i:i+CHUNK]

            interactions = [
                Interaction(
                    user_id=row["user_id"],
                    work_id=row["work_id"],
                    rating=None if pd.isna(row["rating"]) or row["rating"] == 0 else row["rating"]
                    # comment, timestamp, type will be NULL or auto-filled
                )
                for _, row in chunk.iterrows()
            ]

            db.add_all(interactions)
            db.flush()

        # Batched insert for book_subjects
        book_subjects_df = import_book_subjects()

        for i in tqdm(range(0, len(book_subjects_df), CHUNK), desc="Importing book subjects"):
            chunk = book_subjects_df.iloc[i:i+CHUNK]
            subjects = [
                BookSubject(work_id=row.work_id, subject=row.subjects)
                for row in chunk.itertuples(index=False)
            ]
            db.add_all(subjects)
            db.flush()

        # Batched insert for user_fav_subjects
        user_favs_df = import_user_fav_subjects()

        for i in tqdm(range(0, len(user_favs_df), CHUNK), desc="Importing user fav subjects"):
            chunk = user_favs_df.iloc[i:i+CHUNK]
            favs = [
                UserFavSubject(user_id=row.user_id, subject=row.favorite_subjects)
                for row in chunk.itertuples(index=False)
            ]
            db.add_all(favs)
            db.flush()

        db.commit()
        print("✅ All CSVs imported successfully.")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error importing: {e}")

    finally:
        db.close()

if __name__ == "__main__":
    main()
