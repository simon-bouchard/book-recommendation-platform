import os
import json
import pandas as pd
import numpy as np
import torch
import faiss
from sklearn.neighbors import NearestNeighbors
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.shared_utils import normalize_embeddings, ModelStore

store = ModelStore()
book_embs, book_ids = store.get_book_embeddings()
item_idx_to_row = store.get_item_idx_to_row()
BOOK_META = store.get_book_meta()

# Load precomputed embeddings
print("📦 Using preloaded book embeddings from shared_utils...")

norm_embs = normalize_embeddings(book_embs)
book_ids_array = np.array(book_ids)

# ------------------------------
# Build Sklearn KNN index and FAISS
# ------------------------------
print("🔧 Building Sklearn KNN index...")
knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
knn_model.fit(norm_embs)

print("🚀 Building FAISS index...")
faiss_index = faiss.IndexFlatIP(norm_embs.shape[1])
faiss_index.add(norm_embs.astype(np.float32))

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

        if sim_id in BOOK_META.index:
            row = BOOK_META.loc[sim_id]
            results.append({
                "item_idx": int(sim_id),  
                "title": str(row["title"]),
                "cover_id": str(row["cover_id"]) if pd.notnull(row["cover_id"]) else None,
                "author": str(row["author"]) if pd.notnull(row["author"]) else None,
                "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                "isbn": str(row["isbn"]) if pd.notnull(row["isbn"]) else None,
                "score": float(sim_scores[i])  
            })

        if len(results) == top_k:
            break

    return results