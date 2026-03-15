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

from models.core import PATHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "perdim")

    print("Loading book and subject mappings...")
    books = pd.read_pickle(PATHS.staging_data_dir / "books.pkl")
    book_subjects = pd.read_pickle(PATHS.staging_data_dir / "book_subjects.pkl")

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
    PATHS.ensure_staging_dirs()
    np.save(PATHS.staging_dir / "embeddings" / "book_subject_embeddings.npy", pooled_embs)

    with open(PATHS.staging_dir / "embeddings" / "book_subject_ids.json", "w") as f:
        json.dump(book_ids, f)

    print("Saved:")
    print(f"   - {PATHS.staging_dir}/embeddings/book_subject_embeddings.npy")
    print(f"   - {PATHS.staging_dir}/embeddings/book_subject_ids.json")


if __name__ == "__main__":
    main()
