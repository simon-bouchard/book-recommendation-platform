from sqlalchemy.orm import Session
from abc import ABC, abstractmethod
import numpy as np
import torch
from models.shared_utils import ModelStore, normalize_vector

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
        top_k_sim=00,
        top_k_mixed=200,
        scale_sim=10.0,
        w=0.7,
        db: Session = None
    ) -> list[int]:
        if use_only_bayesian:
            print("Using only Bayesian candidates")
            top_k = top_k_bayes + top_k_sim + top_k_mixed
            idx_bayes = torch.topk(torch.tensor(bayesian_tensor), top_k).indices
            return [book_ids[i] for i in idx_bayes.tolist()]

        print("Using hybrid candidates")
        # Compute similarity and hybrid scores
        user_emb_tensor = normalize_vector(torch.tensor(user_emb.numpy()))
        sim_scores = scale_sim * torch.matmul(torch.tensor(book_embs), user_emb_tensor)
        bayes_scores = torch.tensor(bayesian_tensor)
        final_scores = w * sim_scores + (1 - w) * bayes_scores
        
        # Top from each tier
        idx_mixed = torch.topk(final_scores, top_k_mixed).indices.tolist()
        idx_sim = torch.topk(sim_scores, top_k_sim).indices.tolist()
        idx_bayes = torch.topk(bayes_scores, top_k_bayes).indices.tolist()

        # Build final list with priority and uniqueness
        seen = set()
        final_ranked = []

        def add_ranked(indices, scores, top_k):
            added = 0
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    final_ranked.append((idx, scores[idx].item()))
                    added += 1
                    if added == top_k:
                        break

        add_ranked(idx_mixed, final_scores, top_k_mixed)
        add_ranked(idx_sim, sim_scores, top_k_sim)
        add_ranked(idx_bayes, bayes_scores, top_k_bayes)

        # Final output ordered by score descending
        final_ranked.sort(key=lambda x: -x[1])
        return [book_ids[i] for i, _ in final_ranked]

class ALSCandidateGenerator(CandidateGenerator):
    def generate(self, user_id: int, user_emb: np.ndarray = None, top_k: int = 500, db: Session = None, **kwargs) -> list[int]:
        print("Using ALS candidate generator")
        user_als_embs, book_als_embs, user_id_to_als_row, book_row_to_item_idx = store.get_als_embeddings()

        if user_id not in user_id_to_als_row:
            return []

        user_vec = user_als_embs[user_id_to_als_row[user_id]]
        scores = book_als_embs @ user_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [book_row_to_item_idx[i] for i in top_indices]