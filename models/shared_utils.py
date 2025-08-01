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

def load_attention_components(path="models/data/subject_attention_components.pth"):
    """Load subject embedding weights + attention weights"""
    state = torch.load(path, map_location="cpu")

    emb_weight = state['subject_embs']['weight']
    num_embeddings, embedding_dim = emb_weight.shape

    subject_embs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    subject_embs.load_state_dict({'weight': emb_weight})
    subject_embs.weight.requires_grad_(False)

    attn_weight = state['attn_weight']
    attn_bias = state['attn_bias']

    return subject_embs, attn_weight, attn_bias

def attention_pool(indices_list, emb_layer, weight, bias):
    """Pooled vector from a list of subject indices using attention"""
    from models.shared_utils import PAD_IDX
    
    device = emb_layer.weight.device
    batch_size = len(indices_list)
    max_len = max((len(lst) for lst in indices_list), default=1)

    padded = [lst + [0] * (max_len - len(lst)) if len(lst) > 0 else [0]*max_len for lst in indices_list]
    idx_tensor = torch.tensor(padded, device=device)
    mask = (idx_tensor != PAD_IDX)

    # Safety fix: unmask 1st position if all are PADs
    has_real_subjects = mask.any(dim=1)
    for i in range(len(mask)):
        if not has_real_subjects[i]:
            mask[i, 0] = True

    embs = emb_layer(idx_tensor)
    scores = (embs @ weight.T) + bias
    scores = scores.squeeze(-1).masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1).unsqueeze(-1)

    pooled = (embs * attn).sum(dim=1)
    pooled = pooled.nan_to_num(0.0)
    return pooled


def batched_attention_pool(indices_list, emb_layer, weight, bias, batch_size=1024):
    all_outputs = []
    for i in range(0, len(indices_list), batch_size):
        batch = indices_list[i:i+batch_size]
        pooled = attention_pool(batch, emb_layer, weight, bias)
        all_outputs.append(pooled.detach().cpu().numpy())
    return np.concatenate(all_outputs, axis=0)

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

def get_user_embedding(fav_subjects_idxs: list[int]) -> tuple[torch.Tensor, bool]:
    """
    Returns:
    - pooled embedding (torch.Tensor)
    - is_fallback: True if [PAD_IDX] was used
    """
    from models.shared_utils import PAD_IDX

    store = ModelStore()
    subject_emb, attn_weight, attn_bias = store.get_attention_components()

    is_fallback = not fav_subjects_idxs or all(s == PAD_IDX for s in fav_subjects_idxs)

    with torch.no_grad():
        if is_fallback:
            emb = attention_pool([[PAD_IDX]], subject_emb, attn_weight, attn_bias)[0].cpu()
        else:
            emb = attention_pool([fav_subjects_idxs], subject_emb, attn_weight, attn_bias)[0].cpu()

    return emb, is_fallback

def get_candidate_book_df(candidate_ids: list[int]) -> pd.DataFrame:
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
        self._attn_weight = None
        self._attn_bias = None
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

    def get_attention_components(self):
        if self._subject_emb is None:
            self._subject_emb, self._attn_weight, self._attn_bias = load_attention_components()
            self._subject_emb = self._subject_emb.to("cpu")
            self._attn_weight = self._attn_weight.to("cpu")
            self._attn_bias = self._attn_bias.to("cpu")
        return self._subject_emb, self._attn_weight, self._attn_bias

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