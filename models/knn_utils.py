import os
import json
import numpy as np
import torch
import faiss
from sklearn.neighbors import NearestNeighbors
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.shared_utils import load_book_embeddings, normalize_embeddings, get_item_idx_to_row

# ------------------------------
# Load precomputed embeddings
# ------------------------------
print("📦 Loading book embeddings and item_idx list...")
pooled_embs, book_ids = load_book_embeddings()
item_idx_to_row = get_item_idx_to_row(book_ids)
norm_embs = normalize_embeddings(pooled_embs)

assert pooled_embs.shape[0] == len(book_ids), "Embeddings and ID list are misaligned"

# Create mapping: item_idx → row index
item_idx_to_row = {idx: i for i, idx in enumerate(book_ids)}
book_ids_array = np.array(book_ids)

# Normalize for cosine similarity
norm_embs = pooled_embs / np.linalg.norm(pooled_embs, axis=1, keepdims=True)

# ------------------------------
# Build Sklearn KNN index
# ------------------------------
print("🔧 Building Sklearn KNN index...")
knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
knn_model.fit(norm_embs)

# ------------------------------
# Build FAISS index
# ------------------------------
print("🚀 Building FAISS index...")
faiss_index = faiss.IndexFlatIP(norm_embs.shape[1])
faiss_index.add(norm_embs.astype(np.float32))

# ------------------------------
# Book metadata (optional to preload externally)
# ------------------------------
book_meta = {}  # Fill this externally: item_idx → (title, subjects)

def set_book_meta(meta_dict):
    global book_meta
    book_meta = meta_dict

# ------------------------------
# Shared retrieval function
# ------------------------------

def get_similar_books(item_idx, top_k=10, method="faiss"):
    if item_idx not in item_idx_to_row:
        raise ValueError(f"item_idx {item_idx} not found")

    row = item_idx_to_row[item_idx]
    query = norm_embs[row].reshape(1, -1).astype(np.float32)

    if method == "sklearn":
        distances, indices = knn_model.kneighbors(query, n_neighbors=top_k + 1)
        sim_scores = 1 - distances[0]
        result_rows = indices[0]
    elif method == "faiss":
        sim_scores, result_rows = faiss_index.search(query, top_k + 1)
        sim_scores = sim_scores[0]
        result_rows = result_rows[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    results = []
    for i, row_idx in enumerate(result_rows):
        sim_id = book_ids[row_idx]
        if sim_id == item_idx:
            continue  # skip self

        title, subjects = book_meta.get(sim_id, ("[Unknown Title]", []))
        results.append({
            "item_idx": sim_id,
            "title": title,
            "subjects": subjects,
            "score": round(float(sim_scores[i]), 4)
        })

        if len(results) == top_k:
            break

    return results