from sqlalchemy.orm import Session
from models.candidate_generators import CandidateGenerator
from models.rerankers import Reranker
from models.shared_utils import get_user_embedding

class RecommendationEngine:
    def __init__(self, candidate_strategy: CandidateGenerator, reranker: Reranker):
        self.generator = candidate_strategy
        self.reranker = reranker

    def recommend(self, user, top_k: int = 100, db: Session = None):
        """
        Main recommendation pipeline.
        Args:
            user: User SQLAlchemy object (must have user_id, country, filled_age, age, fav_subjects_idxs)
            top_k: how many books to return
        Returns:
            List of dicts: top-k scored book metadata
        """
        # Get subject-based user embedding
        fav_subjects = getattr(user, "fav_subjects_idxs", None)
        user_emb, is_fallback = get_user_embedding(fav_subjects or [])

        # Generate candidates
        candidate_ids = self.generator.generate(
            user_id=user.user_id,
            user_emb=user_emb,
            use_only_bayesian=is_fallback,
            db=db
        )

        if not candidate_ids:
            return []

        # Score and rank
        df = self.reranker.score(user=user, candidate_ids=candidate_ids, user_emb=user_emb, db=db)
        top_df = df.sort_values("score", ascending=False).head(top_k)

        # Clean output
        cols = ["item_idx", "title", "score", "book_avg_rating", "book_num_ratings",
                "cover_id", "author", "year", "isbn"]
        df = top_df[cols].copy()

        def clean_row(row):
            return {
                k: None if (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))) else v
                for k, v in row.items()
            }

        return [clean_row(row) for row in df.to_dict(orient="records")]