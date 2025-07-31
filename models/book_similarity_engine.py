import numpy as np
import pandas as pd
import faiss
from abc import ABC, abstractmethod
from models.shared_utils import (
    normalize_embeddings, ModelStore
)

# ------------------------------
# Strategy base class
# ------------------------------
class SimilarityStrategy(ABC):
    @abstractmethod
    def get_similar_books(self, item_idx: int, top_k: int = 10) -> list[dict]:
        pass

# ------------------------------
# Subject-based similarity (attention pooled)
# ------------------------------
class SubjectSimilarityStrategy(SimilarityStrategy):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # This will still be called on every construction attempt, so guard it
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.store = ModelStore()
        self.embs, self.book_ids = self.store.get_book_embeddings()
        self.item_idx_to_row = self.store.get_item_idx_to_row()
        self.BOOK_META = self.store.get_book_meta()

        self.norm_embs = normalize_embeddings(self.embs)
        self.index = faiss.IndexFlatIP(self.norm_embs.shape[1])
        self.index.add(self.norm_embs.astype(np.float32))

    @classmethod
    def reset(cls):
        cls._instance = None

    def get_similar_books(self, item_idx, top_k=10):
        if item_idx not in self.item_idx_to_row:
            return []

        row = self.item_idx_to_row[item_idx]
        query = self.norm_embs[row].reshape(1, -1).astype(np.float32)
        sim_scores, result_rows = self.index.search(query, top_k + 1)

        return self._format_results(result_rows[0], sim_scores[0], item_idx, top_k)

    def _format_results(self, result_rows, scores, original_idx, top_k):
        results = []
        for i, row_idx in enumerate(result_rows):
            sim_id = self.book_ids[row_idx]
            if sim_id == original_idx:
                continue
            if sim_id in self.BOOK_META.index:
                row = self.BOOK_META.loc[sim_id]
                results.append({
                    "item_idx": int(sim_id),
                    "title": str(row["title"]),
                    "cover_id": str(row["cover_id"]) if pd.notnull(row["cover_id"]) else None,
                    "author": str(row["author"]) if pd.notnull(row["author"]) else None,
                    "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                    "isbn": str(row["isbn"]) if pd.notnull(row["isbn"]) else None,
                    "score": float(scores[i])
                })
            if len(results) == top_k:
                break
        return results

# ------------------------------
# ALS-based similarity
# ------------------------------
class ALSSimilarityStrategy(SimilarityStrategy):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.store = ModelStore()
        _, self.embs, _, self.row_to_idx = self.store.get_als_embeddings()
        self.BOOK_META = self.store.get_book_meta()

        self.book_ids = [self.row_to_idx[i] for i in range(self.embs.shape[0])]
        self.item_idx_to_row = {v: k for k, v in enumerate(self.book_ids)}

        self.norm_embs = normalize_embeddings(self.embs)
        self.index = faiss.IndexFlatIP(self.norm_embs.shape[1])
        self.index.add(self.norm_embs.astype(np.float32))

    @classmethod
    def reset(cls):
        cls._instance = None
        
    def get_similar_books(self, item_idx, top_k=10):
        if item_idx not in self.item_idx_to_row:
            return []

        row = self.item_idx_to_row[item_idx]
        query = self.norm_embs[row].reshape(1, -1).astype(np.float32)
        sim_scores, result_rows = self.index.search(query, top_k + 1)

        return self._format_results(result_rows[0], sim_scores[0], item_idx, top_k)

    def _format_results(self, result_rows, scores, original_idx, top_k):
        results = []
        for i, row_idx in enumerate(result_rows):
            sim_id = self.book_ids[row_idx]
            if sim_id == original_idx:
                continue
            if sim_id in self.BOOK_META.index:
                row = self.BOOK_META.loc[sim_id]
                results.append({
                    "item_idx": int(sim_id),
                    "title": str(row["title"]),
                    "cover_id": str(row["cover_id"]) if pd.notnull(row["cover_id"]) else None,
                    "author": str(row["author"]) if pd.notnull(row["author"]) else None,
                    "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                    "isbn": str(row["isbn"]) if pd.notnull(row["isbn"]) else None,
                    "score": float(scores[i])
                })
            if len(results) == top_k:
                break
        return results

# ------------------------------
# Hybrid strategy (weighted combination)
# ------------------------------
class HybridSimilarityStrategy(SimilarityStrategy):
    _instance = None

    def __init__(self, alpha=0.5):
        self.subject = SubjectSimilarityStrategy()
        self.als = ALSSimilarityStrategy()
        self.alpha = alpha

    def get_similar_books(self, item_idx, top_k=10):
        subj = self.subject.get_similar_books(item_idx, top_k=50)
        als = self.als.get_similar_books(item_idx, top_k=50)

        # Merge by item_idx
        combined_scores = {}
        for r in subj:
            combined_scores[r["item_idx"]] = self.alpha * r["score"]
        for r in als:
            combined_scores[r["item_idx"]] = combined_scores.get(r["item_idx"], 0.0) + (1 - self.alpha) * r["score"]

        merged = []
        meta = self.subject.BOOK_META  # same across both
        for item_idx, score in combined_scores.items():
            if item_idx in meta.index:
                row = meta.loc[item_idx]
                merged.append({
                    "item_idx": int(item_idx),
                    "title": str(row["title"]),
                    "cover_id": str(row["cover_id"]) if pd.notnull(row["cover_id"]) else None,
                    "author": str(row["author"]) if pd.notnull(row["author"]) else None,
                    "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                    "isbn": str(row["isbn"]) if pd.notnull(row["isbn"]) else None,
                    "score": float(score)
                })

        return sorted(merged, key=lambda x: -x["score"])[:top_k]
    
    @classmethod
    def reset(cls):
        SubjectSimilarityStrategy.reset()
        ALSSimilarityStrategy.reset()

# ------------------------------
# Strategy selector
# ------------------------------
def get_similarity_strategy(mode="subject", alpha=0.5) -> SimilarityStrategy:
    if mode == "subject":
        return SubjectSimilarityStrategy()
    elif mode == "als":
        return ALSSimilarityStrategy()
    elif mode == "hybrid":
        return HybridSimilarityStrategy(alpha=alpha)
    else:
        raise ValueError(f"Unknown strategy: {mode}")