import numpy as np
import pandas as pd
import torch
import pickle
from collections import defaultdict
import os
import math

from app.database import SessionLocal
from app.table_models import User, UserFavSubject, BookSubject, Book
from models.shared_utils import (
    PAD_IDX, attention_pool, decompose_embeddings,
    compute_subject_overlap, get_read_books, ModelStore
)

store = ModelStore()

subject_emb, attn_weight, attn_bias = store.get_attention_components()
cold_gbt_model = store.get_cold_gbt_model()
book_embs, book_ids = store.get_book_embeddings()
item_idx_to_row = store.get_item_idx_to_row()
bayesian_tensor = store.get_bayesian_tensor()
BOOK_META = store.get_book_meta()
BOOK_TO_SUBJ = store.get_book_to_subj()

# ----------------------------
# Main functions
# ----------------------------

def get_user_embedding(fav_subjects_idxs):
    is_fallback = not fav_subjects_idxs or all(s == PAD_IDX for s in fav_subjects_idxs)

    with torch.no_grad():
        if is_fallback:
            emb = attention_pool([[PAD_IDX]], subject_emb, attn_weight, attn_bias)[0].cpu()
        else:
            emb = attention_pool([fav_subjects_idxs], subject_emb, attn_weight, attn_bias)[0].cpu()

    return emb, is_fallback

def get_tiered_candidates(user_emb, use_only_bayesian=False, top_k_bayes=0, top_k_sim=50, top_k_mixed=150, scale_sim=10.0, w=0.2):
    if use_only_bayesian:
        idx_bayes = torch.topk(torch.tensor(bayesian_tensor), top_k_bayes).indices
        print('bayesian fallback')
        return [book_ids[i] for i in idx_bayes.tolist()]

    user_emb_tensor = torch.tensor(user_emb.numpy())
    sim_scores = scale_sim * torch.matmul(torch.tensor(book_embs), user_emb_tensor)
    final_scores = w * sim_scores + (1 - w) * torch.tensor(bayesian_tensor)

    idx_mixed = torch.topk(final_scores, top_k_mixed).indices
    idx_sim = torch.topk(sim_scores, top_k_sim).indices
    idx_bayes = torch.topk(torch.tensor(bayesian_tensor), top_k_bayes).indices

    all_indices = torch.cat([idx_mixed, idx_sim, idx_bayes])
    unique_indices = torch.unique(all_indices)

    return [book_ids[i] for i in unique_indices.tolist()]

def recommend_books_for_cold_user(user_id: int, top_k: int = 10):
    db = SessionLocal()

    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        fav_subs = db.query(UserFavSubject.subject_idx).filter(UserFavSubject.user_id == user_id).all()
        fav_subjects_idxs = [r.subject_idx for r in fav_subs] or [PAD_IDX]

        user_emb, use_only_bayesian = get_user_embedding(fav_subjects_idxs)
        candidate_ids = get_tiered_candidates(user_emb, use_only_bayesian=use_only_bayesian)

        if not candidate_ids:
            return []

        candidate_books = BOOK_META.loc[BOOK_META.index.intersection(candidate_ids)].copy()
        candidate_books = candidate_books.reset_index()

        # Filter out already seen books
        read_books = get_read_books(user.user_id, db)
        candidate_books = candidate_books[~candidate_books["item_idx"].isin(read_books)]

        # Merge in book embeddings using item_idx → row mapping
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

        candidate_books["subject_overlap"] = candidate_books["item_idx"].apply(
            lambda idx: compute_subject_overlap(fav_subjects_idxs, BOOK_TO_SUBJ.get(idx, []))
        )

        # Fill user-level metadata
        candidate_books["country"] = user.country
        candidate_books["filled_age"] = user.filled_age
        candidate_books["age"] = user.age

        if user_emb is not None:
            user_emb_dict = decompose_embeddings(user_emb.unsqueeze(0), "user_emb")
            for k, v in user_emb_dict.items():
                candidate_books[k] = v

        # Set correct types
        cat_cols = ["country", "filled_year", "filled_age", "main_subject"]
        for col in cat_cols:
            candidate_books[col] = candidate_books[col].astype("category")

        cont_cols = ["age", "year", "num_pages", "subject_overlap", "book_num_ratings"]
        emb_cols = [c for c in candidate_books.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
        features = emb_cols + cont_cols + cat_cols 
        
        candidate_books["score"] = cold_gbt_model.predict(candidate_books[features])

        top_books = candidate_books.sort_values("score", ascending=False).head(top_k)

        cols = ["item_idx", "title", "score", "book_avg_rating", "book_num_ratings",
                "cover_id", "author", "year", "isbn"]
        df = top_books[cols].copy()

        def clean_row(row):
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            return cleaned

        records = [clean_row(row) for row in df.to_dict(orient="records")]
        return records
    finally:
        db.close()
