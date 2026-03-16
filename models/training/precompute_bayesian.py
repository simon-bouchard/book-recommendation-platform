import pandas as pd
import numpy as np
from pathlib import Path
import os, sys

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(REPO_ROOT)))

from models.core.paths import PATHS
from models.data.loaders import load_book_subject_embeddings

# Smoothing parameter
m = 30


def main():
    print("📄 Loading files ...")
    interactions = pd.read_pickle(PATHS.staging_data_dir / "interactions.pkl")

    books = pd.read_pickle(PATHS.staging_data_dir / "books.pkl").set_index("item_idx")

    _, book_ids = load_book_subject_embeddings(normalized=False, use_cache=False)

    rated = interactions[interactions["rating"].notna()]

    print("📊 Computing book aggregates...")
    book_stats = (
        rated.groupby("item_idx")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "book_num_ratings", "mean": "book_avg_rating"})
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

    score_df = pd.DataFrame({"item_idx": book_ids, "score": scores})
    score_df = score_df.set_index("item_idx")
    score_df["title"] = books["title"]
    score_df = score_df.sort_values("score", ascending=False)

    print("🏆 Top 5 books by Bayesian score:")
    for title, score in score_df.head(5)[["title", "score"]].values:
        print(f"{title} ({score:.6f})")

    bayesian_tensor = score_df.loc[book_ids]["score"].fillna(0).values.astype(np.float32)
    PATHS.ensure_staging_dirs()
    np.save(PATHS.staging_dir / "scoring" / "bayesian_scores.npy", bayesian_tensor)
    print(
        f"✅ Saved: {PATHS.staging_dir}/scoring/bayesian_scores.npy (shape: {bayesian_tensor.shape})"
    )


if __name__ == "__main__":
    main()
