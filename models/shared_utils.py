import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sqlalchemy.orm import Session

try:
    from app.table_models import Interaction
    from sqlalchemy.orm import Session
except Exception:
    Interaction = None
    SessionLocal = None
    Session = None

import pandas as pd
import pickle
import json
from collections import defaultdict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

PAD_IDX = 3520

def load_book_embeddings(emb_path="models/data/book_embs.npy", id_path="models/data/book_ids.json"):
    """Load precomputed book embeddings and item_idx list"""
    embs = np.load(emb_path)
    with open(id_path, "r") as f:
        book_ids = json.load(f)

    assert embs.shape[0] == len(book_ids), "Mismatch between book embs and IDs"
    return embs, book_ids

def get_item_idx_to_row(book_ids):
    """Returns: item_idx → row index map"""
    return {idx: i for i, idx in enumerate(book_ids)}

def normalize_embeddings(embs):
    """L2 normalize embeddings along last axis"""
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

def normalize_vector(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm() if x.norm() > 0 else x

def compute_subject_overlap(fav_subjects, book_subjects):
    return len(set(fav_subjects) & set(book_subjects))

def decompose_embeddings(tensor, prefix):
    arr = tensor.detach().cpu().numpy().flatten()
    return {f"{prefix}_{i}": arr[i] for i in range(arr.shape[0])}

def get_read_books(user_id: int, db: Session):
    """Returns a set of item_idx the user has already read/rated."""
    return {
        row.item_idx for row in db.query(Interaction.item_idx).filter(
            Interaction.user_id == user_id
        ).all()
    }

def get_user_embedding(fav_subjects_idxs: list[int], strategy: str = "") -> tuple[torch.Tensor, bool]:
    store = ModelStore()
    is_fallback = not fav_subjects_idxs or all(s == PAD_IDX for s in fav_subjects_idxs)
    pooler = store.get_attention_strategy(strategy)

    with torch.no_grad():
        pooled = pooler([[PAD_IDX]] if is_fallback else [fav_subjects_idxs])[0].cpu()

    return pooled, is_fallback

def get_candidate_book_df(candidate_ids: list[int]) -> pd.DataFrame:
    BOOK_META = ModelStore().get_book_meta()
    # Select and preserve order
    df = BOOK_META.loc[BOOK_META.index.intersection(candidate_ids)].copy()
    df["item_idx"] = df.index  
    df["__sort"] = df["item_idx"].map({idx: i for i, idx in enumerate(candidate_ids)})
    return df.sort_values("__sort").drop(columns="__sort").reset_index(drop=True)

def old_get_candidate_book_df(candidate_ids: list[int]) -> pd.DataFrame:
    """Subset BOOK_META to candidate book rows"""
    BOOK_META = ModelStore().get_book_meta()
    return BOOK_META.loc[BOOK_META.index.intersection(candidate_ids)].copy().reset_index()

def filter_read_books(df: pd.DataFrame, user_id: int, db: Session) -> pd.DataFrame:
    """Remove books the user has already read"""
    read = get_read_books(user_id, db)
    return df[~df["item_idx"].isin(read)]


def add_book_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach book embeddings to a candidate DataFrame"""
    store = ModelStore()
    book_embs, _ = store.get_book_embeddings()
    item_idx_to_row = store.get_item_idx_to_row()
    dim = book_embs.shape[1]

    book_emb_data = []
    for idx in df["item_idx"]:
        row_idx = item_idx_to_row.get(idx)
        if row_idx is not None:
            book_emb_data.append(book_embs[row_idx])
        else:
            book_emb_data.append(np.zeros(dim))

    book_emb_df = pd.DataFrame(book_emb_data, columns=[f"book_emb_{i}" for i in range(dim)])
    return pd.concat([df.reset_index(drop=True), book_emb_df], axis=1)

# -------------------------------
# Singleton Model Loader
# -------------------------------
class ModelStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelStore, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self._subject_emb = None
        self._book_embs = None
        self._book_ids = None
        self._item_idx_to_row = None
        self._bayesian_tensor = None
        self._book_meta = None
        self._user_meta = None
        self._book_to_subj = None
        self._cold_gbt = None
        self._warm_gbt = None
        self._user_als_embs = None
        self._book_als_embs = None
        self._user_id_to_als_row = None
        self._book_row_to_item_idx = None

    def get_book_embeddings(self):
        if self._book_embs is None:
            self._book_embs, self._book_ids = load_book_embeddings()
            self._item_idx_to_row = get_item_idx_to_row(self._book_ids)
        return self._book_embs, self._book_ids

    def get_item_idx_to_row(self):
        if self._item_idx_to_row is None:
            self.get_book_embeddings()
        return self._item_idx_to_row

    def get_bayesian_tensor(self):
        if self._bayesian_tensor is None:
            arr = np.load("models/data/bayesian_tensor.npy")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self._bayesian_tensor = arr
        return self._bayesian_tensor

    def get_book_meta(self):
        if self._book_meta is None:
            self._book_meta = pd.read_pickle("models/training/data/books.pkl").set_index("item_idx")
        return self._book_meta

    def get_user_meta(self):
        if self._user_meta is None:
            self._user_meta = pd.read_pickle("models/training/data/users.pkl").set_index("user_id")
        return self._user_meta

    def get_book_to_subj(self):
        if self._book_to_subj is None:
            self._book_to_subj = defaultdict(list)
            path = "models/training/data/book_subjects.pkl"
            if os.path.exists(path):
                df = pd.read_pickle(path)
                for row in df.itertuples(index=False):
                    self._book_to_subj[row.item_idx].append(row.subject_idx)
        return self._book_to_subj

    def get_cold_gbt_model(self):
        if self._cold_gbt is None:
            with open("models/data/gbt_cold.pickle", "rb") as f:
                self._cold_gbt = pickle.load(f)
        return self._cold_gbt

    def get_warm_gbt_model(self):
        if self._warm_gbt is None:
            with open("models/data/gbt_warm.pickle", "rb") as f:
                self._warm_gbt = pickle.load(f)
        return self._warm_gbt

    def get_als_embeddings(self):
        if self._user_als_embs is None:
            with open("models/data/user_als_ids.json") as f:
                als_user_ids = json.load(f)
            with open("models/data/book_als_ids.json") as f:
                als_book_ids = json.load(f)

            self._user_als_embs = np.load("models/data/user_als_emb.npy")
            self._book_als_embs = np.load("models/data/book_als_emb.npy")

            self._user_id_to_als_row = {uid: i for i, uid in enumerate(als_user_ids)}
            self._book_row_to_item_idx = {i: iid for i, iid in enumerate(als_book_ids)}

        return (self._user_als_embs, self._book_als_embs,
                self._user_id_to_als_row, self._book_row_to_item_idx)

    def get_attention_strategy(self, name=None):
        name = name or os.getenv("ATTN_STRATEGY", "scalar")

        if not hasattr(self, "_attn_strategy") or self._attn_strategy_name != name:
            from models.subject_attention_strategy import STRATEGY_REGISTRY
            if name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown attention strategy: {name}")

            strategy_class = STRATEGY_REGISTRY[name]
            
            path = f"models/data/subject_attention_components_{name}.pth" if name != "scalar" else "models/data/subject_attention_components.pth"
            self._attn_strategy = strategy_class(path=path)

            self._attn_strategy_name = name

        return self._attn_strategy
