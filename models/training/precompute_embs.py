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

from models.core.paths import PATHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PATHS.staging_data_dir


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

    from models.subject_attention_strategy import STRATEGY_REGISTRY

    if ATTN_STRATEGY not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown attention strategy: '{ATTN_STRATEGY}'")

    attn_path = PATHS.staging_dir / "attention" / f"subject_attention_{ATTN_STRATEGY}.pth"
    print(f"Loading attention weights from: {attn_path}")
    pooler = STRATEGY_REGISTRY[ATTN_STRATEGY](path=str(attn_path))

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
    print(f"   - {REPO_ROOT}/models/artifacts/embeddings/book_subject_embeddings.npy")
    print(f"   - {REPO_ROOT}/models/artifacts/embeddings/book_subject_ids.json")


if __name__ == "__main__":
    main()
