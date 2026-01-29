# models/training/precompute_embs.py
"""
Precompute book embeddings with configurable PAD_IDX.
"""

import os
import argparse
import pandas as pd
import json
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "models" / "training" / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute book subject embeddings")
    parser.add_argument(
        "--pad-idx",
        type=int,
        default=None,
        help="Padding index for invalid subjects (default: from PAD_IDX env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use provided PAD_IDX or fall back to environment variable
    pad_idx = args.pad_idx if args.pad_idx is not None else int(os.getenv("PAD_IDX", "0"))

    print(f"Using PAD_IDX = {pad_idx}")

    ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar")

    print("Loading book and subject mappings...")
    books = pd.read_pickle(DATA_DIR / "books.pkl")
    book_subjects = pd.read_pickle(DATA_DIR / "book_subjects.pkl")

    book_to_subjects = defaultdict(list)
    for row in book_subjects.itertuples():
        book_to_subjects[row.item_idx].append(row.subject_idx)

    book_ids = []
    subject_lists = []

    for row in books.itertuples():
        subjects = book_to_subjects.get(row.item_idx, [])
        if not subjects or all(s == pad_idx for s in subjects):
            continue
        book_ids.append(row.item_idx)
        subject_lists.append(subjects)

    print(f"Books with valid subjects: {len(book_ids)}")

    # Load pooler (need to pass pad_idx to it)
    from models.data.loaders import load_attention_strategy

    pooler = load_attention_strategy(strategy=ATTN_STRATEGY, use_cache=False)

    # Update pooler's padding index
    if hasattr(pooler, "subject_emb"):
        pooler.subject_emb.padding_idx = pad_idx
    elif hasattr(pooler, "shared_subj_emb"):
        pooler.shared_subj_emb.padding_idx = pad_idx

    print("Computing pooled subject embeddings...")
    BATCH_SIZE = 1024
    all_embs = []

    with torch.no_grad():
        for start in range(0, len(subject_lists), BATCH_SIZE):
            batch = subject_lists[start : start + BATCH_SIZE]
            embs = pooler(batch).cpu().numpy()
            all_embs.append(embs)

    pooled_embs = np.vstack(all_embs)
    print(f"Shape: {pooled_embs.shape}")

    # Save
    os.makedirs(REPO_ROOT / "models" / "data", exist_ok=True)
    np.save(REPO_ROOT / "models/artifacts/embeddings/book_subject_embeddings.npy", pooled_embs)

    with open(REPO_ROOT / "models/artifacts/book_subject_ids.json", "w") as f:
        json.dump(book_ids, f)

    print("Saved:")
    print(f"   - {REPO_ROOT}/models/artifacts/embeddings/book_subject_embeddings.npy")
    print(f"   - {REPO_ROOT}/models/artifacts/embeddings/book_subject_ids.json")


if __name__ == "__main__":
    main()
