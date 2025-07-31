import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os
from collections import Counter
from implicit.als import AlternatingLeastSquares

DATA_DIR = "models/training/data"
MODEL_DIR = "models/data"
os.makedirs(MODEL_DIR, exist_ok=True)

ALPHA = 40
FACTORS = 32
ITERATIONS = 15
REGULARIZATION = 0.1
RANDOM_STATE = 42

def main():
    print("📦 Loading interactions...")
    interactions = pd.read_pickle(f"{DATA_DIR}/interactions.pkl")
    interactions = interactions[interactions["rating"].notnull()]
    
    # Warm users only
    rating_counts = interactions["user_id"].value_counts()
    warm_users = rating_counts[rating_counts >= 10].index
    warm_df = interactions[interactions["user_id"].isin(warm_users)]

    print(f"👥 Warm users: {len(warm_users)}")
    print(f"📊 Warm interactions: {len(warm_df)}")

    # Map user/item to index
    user2idx = {u: i for i, u in enumerate(sorted(warm_df["user_id"].unique()))}
    item2idx = {i: j for j, i in enumerate(sorted(warm_df["item_idx"].unique()))}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}

    warm_df["user_idx"] = warm_df["user_id"].map(user2idx)
    warm_df["item_idx"] = warm_df["item_idx"].map(item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)

    print("🔢 Building sparse matrix...")
    values = [1.0 * ALPHA] * len(warm_df)
    user_items = sp.coo_matrix(
        (values, (warm_df["user_idx"], warm_df["item_idx"])),
        shape=(num_users, num_items)
    ).tocsr()

    print("🧠 Training ALS model...")
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        use_gpu=False,
        show_progress=False
    )
    model.fit(user_items)

    print("💾 Saving outputs...")
    np.save(f"{MODEL_DIR}/user_als_emb.npy", model.user_factors)
    np.save(f"{MODEL_DIR}/book_als_emb.npy", model.item_factors)
    
    with open(f"{MODEL_DIR}/user_als_ids.json", "w") as f:
        json.dump([int(idx2user[i]) for i in range(num_users)], f)
    with open(f"{MODEL_DIR}/book_als_ids.json", "w") as f:
        json.dump([int(idx2item[i]) for i in range(num_items)], f)

    print("✅ ALS training complete.")

if __name__ == "__main__":
    main()
