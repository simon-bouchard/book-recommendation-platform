import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fuzzywuzzy import process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import DATABASE_URL
from app.table_models import Rating

df = pd.read_csv('BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=';')

df['Book-Rating'] = df['Book-Rating'].astype(int)
df = df.replace({np.nan: None})

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

skipped_ratings = 0
for _, row in df.iterrows():
    isbn = str(row["ISBN"]).strip()

    book_exists = db.execute(
        text("SELECT COUNT(*) FROM books WHERE isbn = :isbn"),
        {"isbn": isbn}
    ).scalar()


    if book_exists:
        rating = Rating(
            user_id=row['User-ID'],
            book_isbn=isbn,
            rating=row['Book-Rating'],
        )
        db.merge(rating)
    else: 
        skipped_ratings += 1
db.commit()
db.close()

print(f'Ratings imported successfully, skppied {skipped_ratings} ratings')
