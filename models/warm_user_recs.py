import numpy as np
import json
import math
import pandas as pd
from app.database import SessionLocal
from app.table_models import User, Book
from models.shared_utils import (
    BOOK_META, get_read_books,
    user_als_embs, book_als_embs,
    user_id_to_als_row, book_row_to_item_idx,
)


def get_als_candidates(user_id, top_k=100):
    if user_id not in user_id_to_als_row:
        return []

    user_vec = user_als_embs[user_id_to_als_row[user_id]]
    scores = book_als_embs @ user_vec
    top_indices = np.argsort(-scores)[:top_k]
    return [book_row_to_item_idx[i] for i in top_indices]

def recommend_books_for_warm_user(user_id: int, top_k: int = 10):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            print('no user')
            return []

        candidate_ids = get_als_candidates(user.user_id, top_k=200)
        if not candidate_ids:
            print('no candidates from ALS')
            return []

        read_books = get_read_books(user.user_id, db)
        candidate_ids = [bid for bid in candidate_ids if bid not in read_books]
        if not candidate_ids:
            print('no candidates after filtering read books')
            return []

        candidate_books = BOOK_META.loc[BOOK_META.index.intersection(candidate_ids)].copy()
        candidate_books = candidate_books.reset_index()

        top_books = candidate_books.head(top_k)
        cols = ["item_idx", "title", "book_avg_rating", "book_num_ratings", "cover_id", "author", "year", "isbn"]
        df = top_books[cols].copy()

        def clean_row(row):
            return {
                k: None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
                for k, v in row.items()
            }

        return [clean_row(row) for row in df.to_dict(orient="records")]
    finally:
        db.close()
