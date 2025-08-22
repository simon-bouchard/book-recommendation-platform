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
    MIN_COUNT = 10

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.store = ModelStore()

        # full matrix for queries (do not filter here)
        self.norm_embs, self.row_to_idx = self.store.get_als_book_embeddings(normalized=True)
        self.book_ids_full = [self.row_to_idx[i] for i in range(self.norm_embs.shape[0])]
        self.item_idx_to_fullrow = {iid: row for row, iid in enumerate(self.book_ids_full)}

        # counts from BOOK_META
        self.BOOK_META = self.store.get_book_meta()
        counts = (self.BOOK_META["book_num_ratings"]
                  .astype("int32", copy=False))

        # trusted candidate pool: items with count >= MIN_COUNT
        full_ids = np.array(self.book_ids_full, dtype=np.int64)
        cvals = counts.reindex(full_ids, fill_value=0).to_numpy(dtype=np.int32)
        trusted_mask = cvals >= int(self.MIN_COUNT)

        self.trusted_rows_full = np.flatnonzero(trusted_mask)
        self.book_ids = full_ids[trusted_mask].tolist()              # candidates’ item_idx
        self.cand_embs = self.norm_embs[trusted_mask].astype(np.float32)

        dim = self.cand_embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        if self.cand_embs.shape[0] > 0:
            self.index.add(self.cand_embs)

    @classmethod
    def set_min_count(cls, n: int):
        cls.MIN_COUNT = int(n)
        cls.reset()

    @classmethod
    def reset(cls):
        cls._instance = None
        
    def get_similar_books(self, item_idx: int, top_k: int = 10, **kwargs: Any) -> list[dict]:
        # No ALS vector for this book? (not in the factorization)
        fullrow = self.item_idx_to_fullrow.get(item_idx)
        if fullrow is None:
            return []

        # Empty trusted pool? (e.g., extreme thresholds)
        if self.index.ntotal == 0:
            return []

        # Build query from the FULL matrix (even if this book itself is low-count)
        query = self.norm_embs[fullrow].reshape(1, -1).astype(np.float32)

        # Search only within the trusted candidate pool
        k = min(top_k + 1, max(1, self.index.ntotal))
        sim_scores, result_rows = self.index.search(query, k)

        # Format with the candidate-book_ids list
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
    MIN_COUNT = 5

    def __init__(self, alpha: float = 0.5):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.subject = SubjectSimilarityStrategy()
        self.als = ALSSimilarityStrategy()
        self.alpha = float(alpha)

        self.BOOK_META = self.subject.BOOK_META
        self.book_ids = self.subject.book_ids
        self.item_idx_to_row = self.subject.item_idx_to_row

        # map ALS rows to subject rows (so we can align ALS scores to subject space)
        als_item_to_row = {iid: row for row, iid in enumerate(self.als.book_ids_full)}
        n_subj = len(self.book_ids)
        self._als_row_for_subj_row = np.full(n_subj, -1, dtype=np.int32)
        for i, iid in enumerate(self.book_ids):
            self._als_row_for_subj_row[i] = als_item_to_row.get(int(iid), -1)

        # counts aligned to subject order
        self._counts_subj = (self.BOOK_META
                             .loc[self.book_ids, "book_num_ratings"]
                             .fillna(0).astype("int32").to_numpy())
        
    @classmethod
    def reset(cls):
        SubjectSimilarityStrategy.reset()
        ALSSimilarityStrategy.reset()

    def get_similar_books(self, item_idx: int, top_k: int = 10, **kwargs: Any) -> list[dict]:
        if item_idx not in self.item_idx_to_row:
            return []

        alpha = float(kwargs.get("alpha", self.alpha))
        min_count = int(kwargs.get("min_count", self.MIN_COUNT))
        filter_low = bool(kwargs.get("filter_low_count", True))

        # --- build queries & compute subj + ALS scores (your existing logic) ---
        subj_row = self.item_idx_to_row[item_idx]
        q_subj = self.subject.norm_embs[subj_row].astype(np.float32, copy=False)
        subj_scores = self.subject.norm_embs @ q_subj  # [N_subj]

        als_row = self._als_row_for_subj_row[subj_row]
        if als_row >= 0:
            q_als = self.als.norm_embs[als_row].astype(np.float32, copy=False)
            als_scores_all = self.als.norm_embs @ q_als  # [N_als]
            als_scores_subj = np.zeros_like(subj_scores, dtype=np.float32)
            valid = self._als_row_for_subj_row >= 0
            als_scores_subj[valid] = als_scores_all[self._als_row_for_subj_row[valid]]
        else:
            als_scores_subj = np.zeros_like(subj_scores, dtype=np.float32)

        final_scores = (1.0 - alpha) * subj_scores + alpha * als_scores_subj

        # --- mask out low-count items in Hybrid only ---
        cand = (np.flatnonzero(self._counts_subj >= min_count)
                if filter_low else np.arange(final_scores.shape[0], dtype=np.int64))
        if cand.size == 0:
            return []

        k = min(top_k + 1, cand.size)
        scores_c = final_scores[cand]
        idx_part = np.argpartition(-scores_c, k - 1)[:k]
        idx_sorted = cand[idx_part[np.argsort(-scores_c[idx_part])]]

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
