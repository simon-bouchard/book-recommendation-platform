# models/training/data_loader.py
"""
Data loader for training scripts with configurable PAD_IDX.
"""

from pathlib import Path
from typing import Dict, List, Literal

import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

# Repo-relative data dir
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "models" / "training" / "data"

DatasetMode = Literal["supervised", "contrastive"]


def load_training_rows_from_pickle(pad_to: int = 5, pad_idx: int = 0) -> List[Dict]:
    """
    Load interactions + subject mappings from pickle files and build row dicts.

    Args:
        pad_to: Number of favorite subjects to pad to
        pad_idx: Padding index for invalid/missing subjects

    Returns:
        List of dicts with keys: user_idx, item_idx, rating, fav_subjects, book_subjects
    """
    print(f"Loading .pkl data from: {DATA_DIR}")
    print(f"Using PAD_IDX = {pad_idx}")

    interactions = pd.read_pickle(DATA_DIR / "interactions.pkl")
    user_fav_df = pd.read_pickle(DATA_DIR / "user_fav_subjects.pkl")
    book_subj_df = pd.read_pickle(DATA_DIR / "book_subjects.pkl")

    interactions = interactions[interactions["rating"].notnull()].copy()

    # Warm-user filter: keep users with >= 10 ratings
    rating_counts = interactions["user_id"].value_counts()
    interactions["is_warm"] = interactions["user_id"].map(
        lambda uid: rating_counts.get(uid, 0) >= 10
    )

    # Build maps
    user_fav = defaultdict(list)
    for row in user_fav_df.itertuples(index=False):
        user_fav[row.user_id].append(row.subject_idx)

    book_subj = defaultdict(list)
    for row in book_subj_df.itertuples(index=False):
        book_subj[row.item_idx].append(row.subject_idx)

    rows: List[Dict] = []
    for row in interactions.itertuples(index=False):
        if not row.is_warm:
            continue

        book_subjs = book_subj.get(row.item_idx, [])
        if not book_subjs or all(s == pad_idx for s in book_subjs):
            continue

        fav_subjs = user_fav.get(row.user_id, [])
        if not fav_subjs:
            fav_subjs_padded = [pad_idx] * pad_to
        else:
            fav_subjs_padded = fav_subjs[:pad_to] + [pad_idx] * max(0, pad_to - len(fav_subjs))

        rows.append(
            {
                "user_idx": row.user_id,
                "item_idx": row.item_idx,
                "rating": float(row.rating),
                "fav_subjects": fav_subjs_padded,
                "book_subjects": book_subjs,
            }
        )

    if not rows:
        print("No valid training rows constructed.")
        return []

    # Pad book_subjects to common length
    max_len = max(len(r["book_subjects"]) for r in rows)
    for r in rows:
        r["book_subjects"] = r["book_subjects"] + [pad_idx] * (max_len - len(r["book_subjects"]))

    return rows


class SupervisedSubjectDataset(Dataset):
    """Dataset for supervised RMSE training."""

    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x = {
            "user_idx": torch.tensor(r["user_idx"], dtype=torch.long),
            "item_idx": torch.tensor(r["item_idx"], dtype=torch.long),
            "book_subjects": torch.tensor(r["book_subjects"], dtype=torch.long),
            "fav_subjects": torch.tensor(r["fav_subjects"], dtype=torch.long),
        }
        y = torch.tensor(r["rating"], dtype=torch.float32)
        return x, y


class ContrastiveSubjectDataset(Dataset):
    """Dataset for contrastive training."""

    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return {
            "user_idx": torch.tensor(r["user_idx"], dtype=torch.long),
            "item_idx": torch.tensor(r["item_idx"], dtype=torch.long),
            "book_subjects": torch.tensor(r["book_subjects"], dtype=torch.long),
            "fav_subjects": torch.tensor(r["fav_subjects"], dtype=torch.long),
            "rating": torch.tensor(r["rating"], dtype=torch.float32),
        }


def load_rows_and_dataset(mode: DatasetMode = "supervised", pad_to: int = 5, pad_idx: int = 0):
    """
    Load rows and return (rows, dataset, n_users, n_items, n_subjects).

    Args:
        mode: "supervised" or "contrastive"
        pad_to: Number of favorite subjects to pad to
        pad_idx: Padding index for invalid/missing subjects

    Returns:
        Tuple of (rows, dataset, n_users, n_items, n_subjects)
    """
    rows = load_training_rows_from_pickle(pad_to=pad_to, pad_idx=pad_idx)
    if not rows:
        return [], None, 0, 0, 0

    n_users = max(r["user_idx"] for r in rows) + 1
    n_items = max(r["item_idx"] for r in rows) + 1
    all_subjs = set(s for r in rows for s in r["book_subjects"] + r["fav_subjects"])
    n_subjects = max(all_subjs) + 1

    if mode == "supervised":
        ds = SupervisedSubjectDataset(rows)
    elif mode == "contrastive":
        ds = ContrastiveSubjectDataset(rows)
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    return rows, ds, n_users, n_items, n_subjects
