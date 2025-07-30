import numpy as np
import json
import math
import pandas as pd
from app.database import SessionLocal
from app.table_models import User, UserFavSubject
from models.shared_utils import (
    PAD_IDX, attention_pool, decompose_embeddings,
    get_read_books, ModelStore
)

store = ModelStore()

BOOK_META = store.get_book_meta()
USER_META = store.get_user_meta()
book_embs, _ = store.get_book_embeddings()
item_idx_to_row = store.get_item_idx_to_row()
subject_emb, attn_weight, attn_bias = store.get_attention_components()
warm_gbt_model = store.get_warm_gbt_model()
user_als_embs, book_als_embs, user_id_to_als_row, book_row_to_item_idx = store.get_als_embeddings()

def get_als_candidates(user_id, top_k=100):
    if user_id not in user_id_to_als_row:
        return []

    user_vec = user_als_embs[user_id_to_als_row[user_id]]
    scores = book_als_embs @ user_vec
    top_indices = np.argsort(-scores)[:top_k]
    return [book_row_to_item_idx[i] for i in top_indices]

def score_candidates_with_warm_gbt(candidate_books: pd.DataFrame, user, user_emb: np.ndarray) -> pd.DataFrame:
    """Add GBT score to candidate books using warm model"""

    # Add user embedding
    for i, val in enumerate(user_emb):
        candidate_books[f"user_emb_{i}"] = val

    # Add user metadata
    candidate_books["country"] = user.country
    candidate_books["filled_age"] = user.filled_age
    candidate_books["age"] = user.age

    # Set feature columns
    cat_cols = ["country"]
    cont_cols = [
        "age", "year", "num_pages",
        "book_num_ratings", "book_rating_std",
        "user_num_ratings", "user_rating_std", "user_avg_rating"
    ]
    emb_cols = [c for c in candidate_books.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
    features = cont_cols + cat_cols + emb_cols

    # Type enforcement
    for col in cat_cols:
        candidate_books[col] = candidate_books[col].astype("category")
    for col in cont_cols:
        candidate_books[col] = candidate_books[col].astype(np.float32)

    candidate_books["score"] = warm_gbt_model.predict(candidate_books[features])
    return candidate_books

def recommend_books_for_warm_user(user_id: int, top_k: int = 300):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user or user_id not in USER_META.index:
            return []

        # Get favorite subjects for user
        fav_subs = db.query(UserFavSubject.subject_idx).filter(UserFavSubject.user_id == user_id).all()
        fav_subjects_idxs = [r.subject_idx for r in fav_subs] or [PAD_IDX]

        # Get user embedding
        user_emb = attention_pool([fav_subjects_idxs], subject_emb, attn_weight, attn_bias)[0].cpu()

        # Get candidate IDs from ALS
        user_vec = user_als_embs[user_id_to_als_row[user_id]]
        scores = book_als_embs @ user_vec
        top_indices = np.argsort(-scores)[:200]
        candidate_ids = [book_row_to_item_idx[i] for i in top_indices]

        # Filter out already read books
        read_books = get_read_books(user.user_id, db)
        candidate_ids = [bid for bid in candidate_ids if bid not in read_books]
        if not candidate_ids:
            return []

        # Book metadata
        candidate_books = BOOK_META.loc[BOOK_META.index.intersection(candidate_ids)].copy()
        candidate_books = candidate_books.reset_index()

        # Add book embeddings
        dim = book_embs.shape[1]
        book_emb_data = []
        for idx in candidate_books["item_idx"]:
            row_idx = item_idx_to_row.get(idx)
            if row_idx is not None:
                book_emb_data.append(book_embs[row_idx])
            else:
                book_emb_data.append(np.zeros(dim))
        book_emb_df = pd.DataFrame(book_emb_data, columns=[f"book_emb_{i}" for i in range(dim)])
        candidate_books = pd.concat([candidate_books.reset_index(drop=True), book_emb_df], axis=1)

        # Add decomposed user embedding
        user_emb_dict = decompose_embeddings(user_emb.unsqueeze(0), prefix="user_emb")
        for k, v in user_emb_dict.items():
            candidate_books[k] = v

        # Add user-level metadata
        user_row = USER_META.loc[user_id]
        candidate_books["country"] = user.country
        candidate_books["filled_age"] = user.filled_age
        candidate_books["age"] = user.age
        candidate_books["user_num_ratings"] = user_row["user_num_ratings"]
        candidate_books["user_avg_rating"] = user_row["user_avg_rating"]
        candidate_books["user_rating_std"] = user_row["user_rating_std"]

        # Set dtypes for model
        cat_cols = ["country"]
        cont_cols = [
            "age", "year", "num_pages",
            "book_num_ratings", "book_rating_std",
            "user_num_ratings", "user_rating_std", "user_avg_rating"
        ]
        emb_cols = [c for c in candidate_books.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
        features = cont_cols + cat_cols + emb_cols

        for col in cat_cols:
            candidate_books[col] = candidate_books[col].astype("category")
        for col in cont_cols:
            candidate_books[col] = candidate_books[col].astype(np.float32)

        # Predict
        candidate_books["score"] = warm_gbt_model.predict(candidate_books[features])
        top_books = candidate_books.sort_values("score", ascending=False).head(top_k)

        cols = ["item_idx", "title", "score", "book_avg_rating", "book_num_ratings", "cover_id", "author", "year", "isbn"]
        df = top_books[cols].copy()

        def clean_row(row):
            return {
                k: None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
                for k, v in row.items()
            }

        return [clean_row(row) for row in df.to_dict(orient="records")]
    finally:
        db.close()