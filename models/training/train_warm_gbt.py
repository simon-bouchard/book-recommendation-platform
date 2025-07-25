import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.shared_utils import (
    load_attention_components,
    attention_pool,
    batched_attention_pool,
    PAD_IDX,
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path("models/data/gbt_warm.pickle")

BOOK_EMBS_PATH = "models/data/book_embs.npy"
BOOK_IDS_PATH = "models/data/book_ids.json"

# -------------------------------
# Load base data
# -------------------------------
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
# Filter warm users and split
# -------------------------------
print("🧹 Filtering warm users...")
interactions = interactions[interactions["rating"].notnull()].copy()
rating_counts = interactions["user_id"].value_counts()
warm_user_ids = rating_counts[rating_counts >= 10].index
interactions = interactions[interactions["user_id"].isin(warm_user_ids)]

train_inter, val_inter = train_test_split(
    interactions,
    test_size=0.1,
    stratify=interactions["user_id"],
    random_state=42
)

# -------------------------------
# Aggregates from train only
# -------------------------------
print("📊 Computing aggregates (train only)...")

users = users.drop(columns=[
    "user_num_ratings",
    "user_avg_rating",
    "user_rating_std"
], errors="ignore")

books = books.drop(columns=[
    "book_num_ratings",
    "book_avg_rating",
    "book_rating_std"
], errors="ignore")

user_stats = train_inter.groupby("user_id")["rating"].agg(
    user_num_ratings="count",
    user_avg_rating="mean",
    user_rating_std="std"
).reset_index()

book_stats = train_inter.groupby("item_idx")["rating"].agg(
    book_num_ratings="count",
    book_avg_rating="mean",
    book_rating_std="std"
).reset_index()

users = users.merge(user_stats, on="user_id", how="left")
books = books.merge(book_stats, on="item_idx", how="left")

global_avg = train_inter["rating"].mean()
users["user_rating_std"] = users["user_rating_std"].fillna(0.0)
users["user_avg_rating"] = users["user_avg_rating"].fillna(global_avg)
users["user_num_ratings"] = users["user_num_ratings"].fillna(0).astype(int)

books["book_rating_std"] = books["book_rating_std"].fillna(0.0)
books["book_avg_rating"] = books["book_avg_rating"].fillna(global_avg)
books["book_num_ratings"] = books["book_num_ratings"].fillna(0).astype(int)

users = users.set_index("user_id")
books = books.set_index("item_idx")

# -------------------------------
# Build user → fav_subjects map
# -------------------------------
user_fav = defaultdict(list)
for row in user_fav_df.itertuples(index=False):
    user_fav[row.user_id].append(row.subject_idx)

# -------------------------------
# Feature builder
# -------------------------------
from models.shared_utils import batched_attention_pool

def build_rows(inter_df):
    print("🔍 Merging metadata...")

    # Only keep valid user_id and item_idx
    valid_users = set(users.index)
    valid_books = set(books.index).intersection(item_idx_to_row.keys())
    inter_df = inter_df[
        inter_df["user_id"].isin(valid_users) &
        inter_df["item_idx"].isin(valid_books)
    ].copy()

    # Merge metadata
    inter_df = inter_df.merge(users, on="user_id", how="left")
    inter_df = inter_df.merge(books, on="item_idx", how="left")

    print("📚 Preparing book embeddings...")
    # Map item_idx → row in book_embs
    inter_df["book_emb_row"] = inter_df["item_idx"].map(item_idx_to_row)
    book_emb_matrix = book_embs[inter_df["book_emb_row"].values]

    book_emb_df = pd.DataFrame(
        book_emb_matrix,
        columns=[f"book_emb_{i}" for i in range(book_emb_matrix.shape[1])]
    )
    inter_df = pd.concat([inter_df.reset_index(drop=True), book_emb_df], axis=1)

    print("🧠 Computing user embeddings with batching...")
    # Build list of favorite subject indices
    fav_subjects_list = [user_fav.get(uid, [PAD_IDX]) for uid in inter_df["user_id"]]

    user_emb_matrix = batched_attention_pool(
        fav_subjects_list,
        subject_emb,
        attn_weight,
        attn_bias,
        batch_size=2048  # adjustable based on RAM
    )

    user_emb_df = pd.DataFrame(
        user_emb_matrix,
        columns=[f"user_emb_{i}" for i in range(user_emb_matrix.shape[1])]
    )
    inter_df = pd.concat([inter_df.reset_index(drop=True), user_emb_df], axis=1)

    print("✅ Feature matrix complete.")
    return inter_df[[
        "user_id", "item_idx", "rating",
        "main_subject", "year", "filled_year", "language", "num_pages", "filled_num_pages",
        "book_num_ratings", "book_avg_rating", "book_rating_std",
        "country", "age", "filled_age",
        "user_num_ratings", "user_avg_rating", "user_rating_std"
    ] + list(user_emb_df.columns) + list(book_emb_df.columns)]

print("🧠 Building features...")
train_df = build_rows(train_inter)
val_df = build_rows(val_inter)

# -------------------------------
# Prepare Features and Train
# -------------------------------
cat_cols = ["country", 'language', 'main_subject', 'filled_year']
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    val_df[col] = val_df[col].astype("category")

cont_cols = [
    "age", "year", "num_pages", 
    #"book_num_ratings", "book_avg_rating", "book_rating_std",
    #"user_num_ratings", "user_avg_rating", "user_rating_std"
]
emb_cols = [c for c in train_df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
features = cont_cols + cat_cols + emb_cols

X_train, y_train = train_df[features], train_df["rating"]
X_val, y_val = val_df[features], val_df["rating"]

print("🚀 Training GBT model...")
model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.6,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(50),
        log_evaluation(50)
    ]
)

os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved to: {MODEL_PATH}")

