import os
import sys
import pandas as pd
import json
import ast
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import SessionLocal
from app.table_models import Author

def load_and_insert_authors():
    path = "data/authors.csv"
    if not os.path.exists(path):
        print("❌ File not found:", path)
        return

    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), None)

    authors = []
    for _, row in df.iterrows():
        alt_names = None
        try:
            parsed = ast.literal_eval(row["alternate_names"])
            if isinstance(parsed, list):
                alt_names = json.dumps(parsed)
        except Exception as e:
            print(f"⚠️ Failed to parse alternate_names: {row['alternate_names'][:80]} → {e}")

        authors.append(Author(
            author_id=row["author_id"],
            name=row["name"],
            birth_date=str(row["birth_date"]) if row["birth_date"] else None,
            death_date=str(row["death_date"]) if row["death_date"] else None,
            bio=row["bio"] if row["bio"] else None,
            alternate_names=alt_names
        ))

    if not authors:
        print("❌ No authors parsed. Check your CSV.")
        return

    db = SessionLocal()
    try:
        #db.execute(text("DELETE FROM authors"))
        db.add_all(authors)
        db.commit()
        print(f"✅ Inserted {len(authors)} authors.")

        sample = db.query(Author).first()
        print("Sample from DB:", sample.author_id, "-", sample.name)

    except Exception as e:
        db.rollback()
        print("❌ Error inserting authors:", e)

    finally:
        db.close()

if __name__ == "__main__":
    load_and_insert_authors()
