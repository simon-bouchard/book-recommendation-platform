import pandas as pd
import numpy as np
import json
from pathlib import Path

# Config
BOOKS_PATH = Path("models/training/data/books.pkl")
BOOK_IDS_PATH = Path("models/book_ids.json")
OUTPUT_PATH = Path("models/bayesian_tensor.npy")

# Smoothing parameter
m = 20

def main():
    print("📦 Loading books.pkl...")
    books = pd.read_pickle(BOOKS_PATH)

    print("📄 Loading book_ids.json...")
    with open(BOOK_IDS_PATH, "r") as f:
        book_ids = json.load(f)

    # Safety check
    available_ids = set(books["item_idx"])
    missing = [bid for bid in book_ids if bid not in available_ids]
    if missing:
        print(f"⚠️ Warning: {len(missing)} book_ids missing in books.pkl. They will receive score 0.")
    
    # Index books by item_idx
    books_by_idx = books.set_index("item_idx")

    # Global average rating for smoothing
    global_avg = books_by_idx["book_avg_rating"].mean()
    print(f"🌐 Global average rating: {global_avg:.4f}")

    # Compute Bayesian score aligned to book_ids
    scores = []
    for item_idx in book_ids:
        if item_idx not in books_by_idx.index:
            scores.append(0.0)
            continue
        row = books_by_idx.loc[item_idx]
        n = row["book_num_ratings"]
        avg = row["book_avg_rating"]
        score = (n / (n + m)) * avg + (m / (n + m)) * global_avg
        scores.append(score)

    # Save tensor
    bayesian_tensor = np.array(scores, dtype=np.float32)
    np.save(OUTPUT_PATH, bayesian_tensor)
    print(f"✅ Saved: {OUTPUT_PATH} (shape: {bayesian_tensor.shape})")

if __name__ == "__main__":
    main()