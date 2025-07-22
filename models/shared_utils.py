import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PAD_IDX = 3520

def load_attention_components(path="models/subject_attention_components.pth"):
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

def load_book_embeddings(emb_path="models/book_embs.npy", id_path="models/book_ids.json"):
    """Load precomputed book embeddings and item_idx list"""
    embs = np.load(emb_path)
    with open(id_path, "r") as f:
        book_ids = json.load(f)

    assert embs.shape[0] == len(book_ids), "Mismatch between book embs and IDs"
    return embs, book_ids

def normalize_embeddings(embs):
    """L2 normalize embeddings along last axis"""
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

def get_item_idx_to_row(book_ids):
    """Returns: item_idx → row index map"""
    return {idx: i for i, idx in enumerate(book_ids)}

def compute_subject_overlap(fav_subjects, book_subjects):
    return len(set(fav_subjects) & set(book_subjects))

def decompose_embeddings(tensor, prefix):
    arr = tensor.detach().cpu().numpy().flatten()
    return {f"{prefix}_{i}": arr[i] for i in range(arr.shape[0])}
