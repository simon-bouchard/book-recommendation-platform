import os
import sys
import pandas as pd
import json
import ast
from tqdm import tqdm
from sqlalchemy import text
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from app.table_models import (
    User, Book, BookSubject, UserFavSubject, Interaction, Author, Subject
)

CHUNK = 10000

def main():
    db = SessionLocal()

    try:
        # === 1. Clear all tables ===
        for table in ["interactions", "user_fav_subjects", "book_subjects", "subjects", "books", "authors", "users"]:
            db.execute(text(f"DELETE FROM {table}"))
        db.commit()

        # === 2. Import Users ===
        user_df = pd.read_csv("data/users.csv")
        for i in tqdm(range(0, len(user_df), CHUNK), desc="Importing users"):
            chunk = user_df.iloc[i:i+CHUNK]
            db.add_all([
                User(
                    user_id=int(row["user_id"]),
                    username=f"user{int(row['user_id'])}",
                    email=f"user{int(row['user_id'])}@example.com",
                    password="not_used_but_required",
                    created_at=pd.Timestamp.utcnow(),
                    age=row["age"],
                    age_group=row["age_group"],
                    filled_age=row["filled_age"],
                    country=row["country"],
                ) for _, row in chunk.iterrows()
            ])
            db.flush()

        # === 3. Import Authors ===
        author_df = pd.read_csv("data/authors.csv").where(pd.notnull, None)
        for i in tqdm(range(0, len(author_df), CHUNK), desc="Importing authors"):
            chunk = author_df.iloc[i:i+CHUNK]
            authors = []
            for _, row in chunk.iterrows():
                alt_names = None
                try:
                    parsed = ast.literal_eval(row["alternate_names"])
                    if isinstance(parsed, list):
                        alt_names = json.dumps(parsed)
                except:
                    alt_names = None
                authors.append(Author(
                    external_id=str(row["author_id"]).strip(),
                    name=row["name"],
                    birth_date=str(row["birth_date"]) if row["birth_date"] else None,
                    death_date=str(row["death_date"]) if row["death_date"] else None,
                    bio=row["bio"] if row["bio"] else None,
                    alternate_names=alt_names
                ))
            db.add_all(authors)
            db.flush()
        db.commit()

        # Build external_id → author_idx map
        author_map = {a.external_id: a.author_idx for a in db.query(Author)}

        # === 4. Import Subjects ===
        book_sub_df = pd.read_csv("data/books_to_subjects.csv").dropna(subset=["work_id", "subjects"])
        user_sub_df = pd.read_csv("data/users_to_subjects.csv").dropna(subset=["user_id", "favorite_subjects"])

        all_subjects = sorted(set(book_sub_df["subjects"]) | set(user_sub_df["favorite_subjects"]))
        db.bulk_save_objects([Subject(subject=s) for s in all_subjects])
        db.commit()
        subject_map = {s.subject: s.subject_idx for s in db.query(Subject)}

        # === 5. Import Books and build work_id → item_idx map ===
        book_df = pd.read_csv("data/books.csv").dropna(subset=["work_id"])
        
        book_df = book_df.drop_duplicates(subset=["work_id"])
        book_df = book_df.reset_index(drop=True)  # force consistent row order
        book_df["item_idx"] = book_df.index  # reliable, unique, no collisions

        # Then build mapping
        work_to_item_idx = dict(zip(book_df["work_id"], book_df["item_idx"]))
        books = []

        # Count global subject frequency
        excluded_subjects = {"Fiction", "General"}
        subject_counts = Counter(
            s for s in book_sub_df["subjects"] if s not in excluded_subjects
        )

        for _, row in book_df.iterrows():
            external_author_id = str(row["author_id"]).strip()
            author_idx = author_map.get(external_author_id)

            # Lookup subject list for this book
            book_subjs = book_sub_df.loc[book_sub_df["work_id"] == row["work_id"], "subjects"].tolist()
            excluded_subjects = {"Fiction", "General"}
            filtered_subjs = [s for s in book_subjs if s not in excluded_subjects]

            if not filtered_subjs:
                main_subject = "[NO_SUBJECT]"
            else:
                main_subject = max(filtered_subjs, key=lambda s: subject_counts.get(s, 0))

            book = Book(
                work_id=row["work_id"],
                title=row["title"],
                year=row["year"],
                year_bucket=row["year_bucket"],
                filled_year=row["filled_year"],
                description=row["description"],
                cover_id=row["cover_id"],
                language=row["language"],
                num_pages=row["num_pages"],
                filled_num_pages=row["filled_num_pages"],
                author_idx=author_idx,
                isbn=row["isbn"],
                main_subject=main_subject  
            )

            books.append(book)

        for i in tqdm(range(0, len(books), CHUNK), desc="Importing books"):
            batch = books[i:i+CHUNK]
            db.add_all(batch)
            db.flush()  # this assigns item_idx via AUTO_INCREMENT

            for book in batch:
                work_to_item_idx[book.work_id] = book.item_idx

        # === 6. Import Interactions ===
        inter_df = pd.read_csv("data/interactions.csv").dropna(subset=["work_id"])
        for i in tqdm(range(0, len(inter_df), CHUNK), desc="Importing interactions"):
            chunk = inter_df.iloc[i:i+CHUNK]
            interactions = [
                Interaction(
                    user_id=row["user_id"],
                    item_idx=work_to_item_idx.get(row["work_id"]),
                    rating=None if pd.isna(row["rating"]) or row["rating"] == 0 else row["rating"]
                )
                for _, row in chunk.iterrows()
                if row["work_id"] in work_to_item_idx
            ]
            db.add_all(interactions)
            db.flush()

        # === 7. Import Book Subjects ===
        for i in tqdm(range(0, len(book_sub_df), CHUNK), desc="Importing book subjects"):
            chunk = book_sub_df.iloc[i:i+CHUNK]
            subjects = [
                BookSubject(
                    item_idx=work_to_item_idx.get(row.work_id),
                    subject_idx=subject_map.get(row.subjects)
                )
                for row in chunk.itertuples(index=False)
                if row.work_id in work_to_item_idx and row.subjects in subject_map
            ]
            db.add_all(subjects)
            db.flush()

        # === 8. Import User Favorite Subjects ===
        for i in tqdm(range(0, len(user_sub_df), CHUNK), desc="Importing user fav subjects"):
            chunk = user_sub_df.iloc[i:i+CHUNK]
            favs = [
                UserFavSubject(
                    user_id=row.user_id,
                    subject_idx=subject_map.get(row.favorite_subjects)
                )
                for row in chunk.itertuples(index=False)
                if row.favorite_subjects in subject_map
            ]
            db.add_all(favs)
            db.flush()

        db.commit()
        print("✅ All CSVs imported successfully with clean internal indexing.")

    except Exception as e:
        db.rollback()
        print(f"❌ Import failed: {e}")

    finally:
        db.close()

if __name__ == "__main__":
    main()
