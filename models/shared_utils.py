import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

# -----------------------
# Loaded Static Components
# -----------------------
subject_emb, attn_weight, attn_bias = load_attention_components("models/data/subject_attention_components.pth")
subject_emb = subject_emb.to("cpu")
attn_weight = attn_weight.to("cpu")
attn_bias = attn_bias.to("cpu")

book_embs, book_ids = load_book_embeddings("models/data/book_embs.npy", "models/data/book_ids.json")
item_idx_to_row = get_item_idx_to_row(book_ids)

bayesian_tensor = np.load("models/data/bayesian_tensor.npy")
BOOK_META = pd.read_pickle("models/training/data/books.pkl").set_index("item_idx")
USER_META = pd.read_pickle("models/training/data/users.pkl").set_index("user_id")

BOOK_SUBJ_PATH = "models/training/data/book_subjects.pkl"
BOOK_TO_SUBJ = defaultdict(list)
if os.path.exists(BOOK_SUBJ_PATH):
    book_subj_df = pd.read_pickle(BOOK_SUBJ_PATH)
    for row in book_subj_df.itertuples(index=False):
        BOOK_TO_SUBJ[row.item_idx].append(row.subject_idx)

with open("models/data/gbt_cold.pickle", "rb") as f:
    cold_gbt_model = pickle.load(f)
with open("models/data/gbt_warm.pickle", "rb") as f:
    warm_gbt_model = pickle.load(f)

with open("models/data/user_als_ids.json") as f:
    als_user_ids = json.load(f)
with open("models/data/book_als_ids.json") as f:
    als_book_ids = json.load(f)

user_als_embs = np.load("models/data/user_als_emb.npy")
book_als_embs = np.load("models/data/book_als_emb.npy")

user_id_to_als_row = {uid: i for i, uid in enumerate(als_user_ids)}
book_row_to_item_idx = {i: iid for i, iid in enumerate(als_book_ids)}
