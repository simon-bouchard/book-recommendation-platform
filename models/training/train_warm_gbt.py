import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.shared_utils import (
    load_attention_components,
    attention_pool,
    PAD_IDX,
)

# -------------------------------
# Paths and Constants
# -------------------------------
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path("models/gbt_warm.pickle")

BOOK_EMBS_PATH = "models/data/book_embs.npy"
BOOK_IDS_PATH = "models/data/book_ids.json"

print("📦 Loading data...")
interactions = pd.read_pickle(DATA_DIR / "interactions.pkl")
users = pd.read_pickle(DATA_DIR / "users.pkl")
books = pd.read_pickle(DATA_DIR / "books.pkl")
user_fav_df = pd.read_pickle(DATA_DIR / "user_fav_subjects.pkl")

book_embs = np.load(BOOK_EMBS_PATH)
with open(BOOK_IDS_PATH, "r") as f:
    book_ids = json.load(f)
item_idx_to_row = {idx: i for i, idx in enumerate(book_ids)}

subject_emb, attn_weight, attn_bias = load_attention_components()
subject_emb = subject_emb.to("cpu")
attn_weight = attn_weight.to("cpu")
attn_bias = attn_bias.to("cpu")

# -------------------------------
# Filter Warm Users
# -------------------------------
print("🧹 Filtering warm users...")
interactions = interactions[interactions["rating"].notnull()].copy()
rating_counts = interactions["user_id"].value_counts()
warm_user_ids = rating_counts[rating_counts >= 10].index
interactions = interactions[interactions["user_id"].isin(warm_user_ids)]

# -------------------------------
# Recompute Aggregates (Train Only)
# -------------------------------
print("📊 Recomputing aggregates...")
user_stats = interactions.groupby("user_id")["rating"].agg(
    user_num_ratings="count",
    user_avg_rating="mean",
    user_rating_std="std"
).reset_index()

book_stats = interactions.groupby("item_idx")["rating"].agg(
    book_num_ratings="count",
    book_avg_rating="mean",
    book_rating_std="std"
).reset_index()

# Drop old columns to avoid leakage
for col in ["user_num_ratings", "user_avg_rating", "user_rating_std"]:
    if col in users.columns:
        users = users.drop(columns=[col])

for col in ["book_num_ratings", "book_avg_rating", "book_rating_std"]:
    if col in books.columns:
        books = books.drop(columns=[col])

users = users.merge(user_stats, on="user_id", how="left")
books = books.merge(book_stats, on="item_idx", how="left")

users["user_rating_std"] = users["user_rating_std"].fillna(0.0)
users["user_avg_rating"] = users["user_avg_rating"].fillna(interactions["rating"].mean())
users["user_num_ratings"] = users["user_num_ratings"].fillna(0).astype(int)

books["book_rating_std"] = books["book_rating_std"].fillna(0.0)
books["book_avg_rating"] = books["book_avg_rating"].fillna(interactions["rating"].mean())
books["book_num_ratings"] = books["book_num_ratings"].fillna(0).astype(int)

# -------------------------------
# Build user → fav_subjects map
# -------------------------------
print("📘 Building subject index map...")
user_fav = defaultdict(list)
for row in user_fav_df.itertuples(index=False):
    user_fav[row.user_id].append(row.subject_idx)

# -------------------------------
# Build Feature Rows
# -------------------------------
print("🧠 Building feature table...")
rows = []

for row in interactions.itertuples(index=False):
    uid, iid, rating = row.user_id, row.item_idx, row.rating
    if iid not in item_idx_to_row:
        continue

    book_row = books[books["item_idx"] == iid]
    user_row = users[users["user_id"] == uid]

    if book_row.empty or user_row.empty:
        continue

    book_emb = book_embs[item_idx_to_row[iid]]
    fav_subjs = user_fav.get(uid, [PAD_IDX])
    user_emb = attention_pool([fav_subjs], subject_emb, attn_weight, attn_bias)[0].numpy()

    rows.append({
        "user_id": uid,
        "item_idx": iid,
        "rating": rating,
        **book_row.iloc[0][[
            "main_subject", "year", "filled_year", "language", "num_pages", "filled_num_pages",
            "book_num_ratings", "book_avg_rating", "book_rating_std"
        ]],
        **user_row.iloc[0][[
            "country", "age", "filled_age",
            "user_num_ratings", "user_avg_rating", "user_rating_std"
        ]],
        **{f"user_emb_{i}": user_emb[i] for i in range(len(user_emb))},
        **{f"book_emb_{i}": book_emb[i] for i in range(len(book_emb))}
    })

df = pd.DataFrame(rows)

# -------------------------------
# Prepare Features and Train
# -------------------------------
print("🎯 Preparing training features...")
cat_cols = ["country", "filled_year", "filled_age", "main_subject"]
for col in cat_cols:
    df[col] = df[col].astype("category")

cont_cols = [
    "age", "year", "num_pages",
    "book_num_ratings", "book_avg_rating", "book_rating_std",
    "user_num_ratings", "user_avg_rating", "user_rating_std"
]
emb_cols = [c for c in df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
features = emb_cols + cont_cols + cat_cols

X = df[features]
y = df["rating"]

print("🚀 Training GBT model...")
model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.6,
    random_state=42
)

model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[
        early_stopping(50),
        log_evaluation(50)
    ]
)

os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved to: {MODEL_PATH}")
