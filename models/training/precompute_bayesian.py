import pandas as pd
import numpy as np
import json
from pathlib import Path
from pathlib import Path

# Config
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "models" / "training" / "data"
MODELS_DIR = ROOT / "models" / "data"

INTERACTIONS_PATH = DATA_DIR / "interactions.pkl"
BOOKS_PATH = DATA_DIR / "books.pkl"
BOOK_IDS_PATH = MODELS_DIR / "book_ids.json"
OUTPUT_PATH = MODELS_DIR / "bayesian_tensor.npy"

# Smoothing parameter
m = 30

def main():
    print("📄 Loading files ...")
    interactions = pd.read_pickle(INTERACTIONS_PATH)
    books = pd.read_pickle(BOOKS_PATH)
    with open(BOOK_IDS_PATH) as f:
        book_ids = json.load(f)

    rated = interactions[interactions["rating"].notna()]

    print("📊 Computing book aggregates...")
    book_stats = rated.groupby("item_idx")["rating"].agg(["count", "mean"]).rename(
        columns={"count": "book_num_ratings", "mean": "book_avg_rating"}
    )

    global_avg = book_stats["book_avg_rating"].mean()
    print(f"🌐 Global average rating: {global_avg:.4f}")

    print("⚙️ Computing Bayesian scores...")
    scores = []
    for item_idx in book_ids:
        if item_idx not in book_stats.index:
            scores.append(0.0)
        else:
            n = book_stats.at[item_idx, "book_num_ratings"]
            avg = book_stats.at[item_idx, "book_avg_rating"]
            score = (n / (n + m)) * avg + (m / (n + m)) * global_avg
            scores.append(score)

    score_df = pd.DataFrame({
        "item_idx": book_ids,
        "score": scores
    })
    score_df = score_df.set_index("item_idx")
    score_df["title"] = books["title"]
    score_df = score_df.sort_values("score", ascending=False)

    print("🏆 Top 5 books by Bayesian score:")
    for title, score in score_df.head(5)[["title", "score"]].values:
        print(f"{title} ({score:.6f})")

    bayesian_tensor = score_df.loc[book_ids]["score"].fillna(0).values.astype(np.float32)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, bayesian_tensor)

if __name__ == "__main__":
    main()
