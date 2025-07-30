from sqlalchemy.orm import Session
from abc import ABC, abstractmethod
import numpy as np
import torch
from models.shared_utils import ModelStore

store = ModelStore()
book_embs, book_ids = store.get_book_embeddings()
bayesian_tensor = store.get_bayesian_tensor()


class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, user_id: int, user_emb: np.ndarray, **kwargs) -> list[int]:
        """
        Return a list of item_idx candidates for a given user.
        """
        pass


class ColdHybridCandidateGenerator(CandidateGenerator):
    def generate(
        self,
        user_id: int,
        user_emb: np.ndarray,
        use_only_bayesian=False,
        top_k_bayes=0,
        top_k_sim=50,
        top_k_mixed=150,
        scale_sim=10.0,
        w=0.5,
        db: Session = None
    ) -> list[int]:
        if use_only_bayesian:
            top_k = top_k_bayes + top_k_sim + top_k_mixed
            idx_bayes = torch.topk(torch.tensor(bayesian_tensor), top_k).indices
            return [book_ids[i] for i in idx_bayes.tolist()]

        user_emb_tensor = torch.tensor(user_emb.numpy())
        sim_scores = scale_sim * torch.matmul(torch.tensor(book_embs), user_emb_tensor)
        final_scores = w * sim_scores + (1 - w) * torch.tensor(bayesian_tensor)

        idx_mixed = torch.topk(final_scores, top_k_mixed).indices
        idx_sim = torch.topk(sim_scores, top_k_sim).indices
        idx_bayes = torch.topk(torch.tensor(bayesian_tensor), top_k_bayes).indices

        all_indices = torch.cat([idx_mixed, idx_sim, idx_bayes])
        unique_indices = torch.unique(all_indices)

        return [book_ids[i] for i in unique_indices.tolist()]


class ALSCandidateGenerator(CandidateGenerator):
    def generate(self, user_id: int, user_emb: np.ndarray = None, top_k: int = 500, db: Session = None, **kwargs) -> list[int]:
        user_als_embs, book_als_embs, user_id_to_als_row, book_row_to_item_idx = store.get_als_embeddings()

        if user_id not in user_id_to_als_row:
            return []

        user_vec = user_als_embs[user_id_to_als_row[user_id]]
        scores = book_als_embs @ user_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [book_row_to_item_idx[i] for i in top_indices]