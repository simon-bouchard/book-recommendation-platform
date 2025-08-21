import numpy as np
import pandas as pd
import faiss
from abc import ABC, abstractmethod
from typing import Any
from models.shared_utils import ModelStore

# ------------------------------
# Strategy base class
# ------------------------------
class SimilarityStrategy(ABC):
    @abstractmethod
    def get_similar_books(self, item_idx: int, top_k: int = 10, **kwargs:Any) -> list[dict]:
        pass

    def _format_from_rows(
        self,
        result_rows: np.ndarray,
        scores: np.ndarray,
        original_idx: int,
        top_k: int,
        book_ids: list[int],
        BOOK_META: pd.DataFrame,
    ) -> list[dict]:
        results: list[dict] = []
        for i, row_idx in enumerate(result_rows):
            sim_id = int(book_ids[row_idx])
            if sim_id == original_idx:
                continue
            if sim_id in BOOK_META.index:
                row = BOOK_META.loc[sim_id]
                results.append({
                    "item_idx": sim_id,
                    "title": str(row["title"]),
                    "cover_id": str(row["cover_id"]) if pd.notnull(row["cover_id"]) else None,
                    "author": str(row["author"]) if pd.notnull(row["author"]) else None,
                    "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                    "isbn": str(row["isbn"]) if pd.notnull(row["isbn"]) else None,
                    "score": float(scores[i]),
                })
            if len(results) == top_k:
                break
        return results

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
        # Guard re-init
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.store = ModelStore()
        # Use pre-normalized embeddings from ModelStore
        self.norm_embs, self.book_ids = self.store.get_book_embeddings(normalized=True)
        self.item_idx_to_row = self.store.get_item_idx_to_row()
        self.BOOK_META = self.store.get_book_meta()

        self.index = faiss.IndexFlatIP(self.norm_embs.shape[1])
        self.index.add(self.norm_embs.astype(np.float32))

    @classmethod
    def reset(cls):
        cls._instance = None

    def get_similar_books(self, item_idx, top_k=10, **kwargs):
        if item_idx not in self.item_idx_to_row:
            return []

        row = self.item_idx_to_row[item_idx]
        query = self.norm_embs[row].reshape(1, -1).astype(np.float32)
        sim_scores, result_rows = self.index.search(query, top_k + 1)

        return self._format_from_rows(
            result_rows=result_rows[0],
            scores=sim_scores[0],
            original_idx=item_idx,
            top_k=top_k,
            book_ids=self.book_ids,
            BOOK_META=self.BOOK_META,
        )
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
        self.norm_embs, self.row_to_idx = self.store.get_als_book_embeddings(normalized=True)
        self.BOOK_META = self.store.get_book_meta()

        self.book_ids = [self.row_to_idx[i] for i in range(self.norm_embs.shape[0])]
        self.item_idx_to_row = {v: k for k, v in enumerate(self.book_ids)}

        self.index = faiss.IndexFlatIP(self.norm_embs.shape[1])
        self.index.add(self.norm_embs)

    @classmethod
    def reset(cls):
        cls._instance = None
        
    def get_similar_books(self, item_idx, top_k=10, **kwargs):
        if item_idx not in self.item_idx_to_row:
            return []

        row = self.item_idx_to_row[item_idx]
        query = self.norm_embs[row].reshape(1, -1).astype(np.float32)
        sim_scores, result_rows = self.index.search(query, top_k + 1)

        return self._format_from_rows(
            result_rows=result_rows[0],
            scores=sim_scores[0],
            original_idx=item_idx,
            top_k=top_k,
            book_ids=self.book_ids,
            BOOK_META=self.BOOK_META,
        )
    
# ------------------------------
# Hybrid strategy (full cosine/IP then blend)
# ------------------------------
class HybridSimilarityStrategy(SimilarityStrategy):
    _instance = None

    def __init__(self, alpha: float = 0.5):
        # Reuse initialized singletons
        self.subject = SubjectSimilarityStrategy()
        self.als = ALSSimilarityStrategy()
        self.alpha = float(alpha)

        # Map: subject-row -> ALS-row (or -1 if not present)
        als_item_to_row = {iid: row for row, iid in self.als.row_to_idx.items()}
        n_subj = len(self.subject.book_ids)
        self._als_row_for_subj_row = np.full(n_subj, -1, dtype=np.int32)
        for i, item_idx in enumerate(self.subject.book_ids):
            self._als_row_for_subj_row[i] = als_item_to_row.get(int(item_idx), -1)

        # Convenience refs (subject order is our master order)
        self.BOOK_META = self.subject.BOOK_META
        self.book_ids = self.subject.book_ids
        self.item_idx_to_row = self.subject.item_idx_to_row

    @classmethod
    def reset(cls):
        SubjectSimilarityStrategy.reset()
        ALSSimilarityStrategy.reset()

    def get_similar_books(self, item_idx: int, top_k: int = 10, **kwargs: Any) -> list[dict]:
        # API already guards: if mode in ("als","hybrid") and not has_book_als(item_idx) → 422
        # (Behavioral similarity unavailable. Try Subject.) 
        if item_idx not in self.item_idx_to_row:
            return []

        alpha = float(kwargs.get("alpha", self.alpha))

        # 1) Query vectors (subject is guaranteed; ALS present due to API guard)
        subj_row = self.item_idx_to_row[item_idx]
        q_subj = self.subject.norm_embs[subj_row].astype(np.float32, copy=False)

        als_row = self._als_row_for_subj_row[subj_row]
        # Defensive: if somehow missing, treat ALS query as zeros (shouldn’t happen thanks to guard)
        if als_row >= 0:
            q_als = self.als.norm_embs[als_row].astype(np.float32, copy=False)
            has_als = True
        else:
            q_als = np.zeros((self.als.norm_embs.shape[1],), dtype=np.float32)
            has_als = False

        # 2) Full cosine/IP across all items in each space
        subj_scores = self.subject.norm_embs @ q_subj  # [N_subj]

        if has_als:
            als_scores_all = self.als.norm_embs @ q_als  # [N_als]
            als_scores_subj = np.zeros_like(subj_scores, dtype=np.float32)
            valid = self._als_row_for_subj_row >= 0
            als_scores_subj[valid] = als_scores_all[self._als_row_for_subj_row[valid]]
        else:
            als_scores_subj = np.zeros_like(subj_scores, dtype=np.float32)

        # 3) Blend
        final_scores = (1.0 - alpha) * subj_scores + alpha * als_scores_subj

        # 4) Top‑k (skip the query item in formatting)
        k = min(top_k + 1, final_scores.shape[0])
        idx_part = np.argpartition(-final_scores, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-final_scores[idx_part])]

        # 5) Unified formatter
        return self._format_from_rows(
            result_rows=idx_sorted,
            scores=final_scores[idx_sorted],
            original_idx=item_idx,
            top_k=top_k,
            book_ids=self.book_ids,
            BOOK_META=self.BOOK_META,
        )

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