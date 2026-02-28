import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os, sys
from implicit.als import AlternatingLeastSquares
from pathlib import Path

import logging, time

logging.getLogger("implicit").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "models" / "training" / "data"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.core import PATHS
from models.training.metrics import record_training_metrics

ALPHA = 40
FACTORS = 64
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
        (values, (warm_df["user_idx"], warm_df["item_idx"])), shape=(num_users, num_items)
    ).tocsr()

    print("🧠 Training ALS model...")
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        use_gpu=False,
    )
    t0 = time.time()
    model.fit(user_items)
    train_duration = time.time() - t0

    print("💾 Saving outputs...")
    PATHS.ensure_staging_dirs()
    np.save(PATHS.staging_dir / "embeddings" / "user_als_factors.npy", model.user_factors)
    np.save(PATHS.staging_dir / "embeddings" / "book_als_factors.npy", model.item_factors)

    with open(PATHS.staging_dir / "embeddings" / "user_als_ids.json", "w") as f:
        json.dump([int(idx2user[i]) for i in range(num_users)], f)
    with open(PATHS.staging_dir / "embeddings" / "book_als_ids.json", "w") as f:
        json.dump([int(idx2item[i]) for i in range(num_items)], f)

    print("Evaluating Recall@30...")
    user_item_csr = user_items.tocsr()
    per_user_recalls = []
    EVAL_BATCH_SIZE = 512

    for batch_start in range(0, num_users, EVAL_BATCH_SIZE):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, num_users)
        batch_scores = model.user_factors[batch_start:batch_end] @ model.item_factors.T

        for i, user_row in enumerate(range(batch_start, batch_end)):
            user_items_row = user_item_csr[user_row].indices
            if len(user_items_row) == 0:
                continue

            top_30 = np.argpartition(batch_scores[i], -30)[-30:]
            hits = len(set(user_items_row) & set(top_30))
            per_user_recalls.append(hits / len(user_items_row))

    recall_at_30 = float(np.mean(per_user_recalls))

    print(f"Recall@30: {recall_at_30:.4f}  (n_users={len(per_user_recalls)})")

    record_training_metrics(
        "als",
        {
            "recall_at_30": round(recall_at_30, 6),
            "recall_at_30_p25": round(float(np.percentile(per_user_recalls, 25)), 6),
            "recall_at_30_p50": round(float(np.percentile(per_user_recalls, 50)), 6),
            "recall_at_30_p75": round(float(np.percentile(per_user_recalls, 75)), 6),
            "n_users": len(per_user_recalls),
            "n_items": num_items,
            "n_interactions": int(len(warm_df)),
            "factors": FACTORS,
            "iterations": ITERATIONS,
        },
        duration_s=train_duration,
    )

    print("✅ ALS training complete.")


if __name__ == "__main__":
    main()
