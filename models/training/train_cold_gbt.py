import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.shared_utils import (
    load_attention_components,
    attention_pool,
    batched_attention_pool,
    load_book_embeddings,
    get_item_idx_to_row,
    compute_subject_overlap,
    decompose_embeddings,
    PAD_IDX
)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent / "data"

def load_data_from_pickle():
    print("📦 Loading .pkl data from:", DATA_DIR)
    interactions = pd.read_pickle(DATA_DIR / "interactions.pkl")
    users = pd.read_pickle(DATA_DIR / "users.pkl")
    books = pd.read_pickle(DATA_DIR / "books.pkl")

    user_fav_df = pd.read_pickle(DATA_DIR / "user_fav_subjects.pkl")
    book_subj_df = pd.read_pickle(DATA_DIR / "book_subjects.pkl")

    user_fav = defaultdict(list)
    for row in user_fav_df.itertuples(index=False):
        user_fav[row.user_id].append(row.subject_idx)

    book_subj = defaultdict(list)
    for row in book_subj_df.itertuples(index=False):
        book_subj[row.item_idx].append(row.subject_idx)

    return interactions, users, books, user_fav, book_subj

def main():
    interactions, users, books, user_fav, book_subj = load_data_from_pickle()

    print("🧠 Loading subject attention components and book embeddings...")
    subject_emb, attn_weight, attn_bias = load_attention_components()
    subject_emb, attn_weight, attn_bias = subject_emb.to(DEVICE), attn_weight.to(DEVICE), attn_bias.to(DEVICE)

    book_embs, book_ids = load_book_embeddings()
    item_idx_to_row = get_item_idx_to_row(book_ids)

    print("🧹 Filtering valid interactions...")
    interactions = interactions[interactions["rating"].notnull()].copy()
    rating_counts = interactions["user_id"].value_counts()
    interactions["is_warm"] = interactions["user_id"].map(lambda uid: rating_counts.get(uid, 0) >= 10)

    valid_user_ids = set(user_fav)
    valid_item_ids = set(book_subj) & set(item_idx_to_row)

    interactions = interactions[
        interactions["user_id"].isin(valid_user_ids) &
        interactions["item_idx"].isin(valid_item_ids)
    ].copy()

    print("📚 Mapping book embeddings...")
    interactions["book_emb_row"] = interactions["item_idx"].map(item_idx_to_row)
    book_emb_matrix = book_embs[interactions["book_emb_row"].values]
    book_emb_df = pd.DataFrame(book_emb_matrix, columns=[f"book_emb_{i}" for i in range(book_emb_matrix.shape[1])])

    print("🧠 Computing user embeddings...")
    fav_subjects_list = [user_fav[uid] for uid in interactions["user_id"]]
    user_emb_matrix = batched_attention_pool(fav_subjects_list, subject_emb, attn_weight, attn_bias, batch_size=2048)
    user_emb_df = pd.DataFrame(user_emb_matrix, columns=[f"user_emb_{i}" for i in range(user_emb_matrix.shape[1])])

    print("🔗 Computing subject overlap...")
    interactions["subject_overlap"] = [
        compute_subject_overlap(user_fav[uid], book_subj[iid])
        for uid, iid in zip(interactions["user_id"], interactions["item_idx"])
    ]

    print("🧱 Merging everything into final dataset...")
    df = pd.concat([interactions.reset_index(drop=True), user_emb_df, book_emb_df], axis=1)

    df = df.merge(users, on="user_id", how="left")
    df = df.merge(
        books[["item_idx", "main_subject", "year", "filled_year", "language", "num_pages", "filled_num_pages", "book_num_ratings"]],
        on="item_idx", how="left"
    )

    cat_cols = ["country", "filled_year", "filled_age", "main_subject"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    cont_cols = ["age", "year", "num_pages", "subject_overlap", "book_num_ratings"]
    emb_cols = [c for c in df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
    features = emb_cols + cont_cols + cat_cols

    train = df[df["is_warm"] == True]
    val = df[df["is_warm"] == False]

    X_train, X_val = train[features].copy(), val[features].copy()
    y_train, y_val = train["rating"], val["rating"]

    X_train[cont_cols] = X_train[cont_cols].astype(np.float32)
    X_val[cont_cols] = X_val[cont_cols].astype(np.float32)

    print("🚀 Training LightGBM model...")
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse', 'mae'],
        callbacks=[early_stopping(50), log_evaluation(50)]
    )

    os.makedirs("models", exist_ok=True)
    with open("models/data/gbt_cold.pickle", "wb") as f:
        pickle.dump(model, f)

    print("✅ Saved: models/data/gbt_cold.pickle")

if __name__ == "__main__":
    main()

