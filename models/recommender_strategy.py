from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from models.recommendation_engine import RecommendationEngine
from models.rerankers import GBTWarmReranker, GBTColdReranker, NoOpReranker
from models.candidate_generators import ALSCandidateGenerator, ColdHybridCandidateGenerator
from typing import Any

# --------------------------------
# Base Strategy Interface + Factory
# --------------------------------
class RecommenderStrategy(ABC):
    @abstractmethod
    def recommend(self, user, db: Session, top_k: int, **kwargs: Any) -> list[dict]:
        pass

    @staticmethod
    def get_strategy(num_ratings: int) -> "RecommenderStrategy":
        if num_ratings >= 10:
            return WarmRecommender()
        else:
            return ColdRecommender()
        
    @staticmethod
    def reset_singleton():
        from models.shared_utils import ModelStore
        ModelStore._instance = None  # clear all loaded models
        RecommenderStrategy._singleton = None  # force engine rebuild


# --------------------------------
# Concrete Strategies
# --------------------------------
class WarmRecommender(RecommenderStrategy):
    def __init__(self):
        self.engine = RecommendationEngine(ALSCandidateGenerator(), NoOpReranker())

    def recommend(self, user, db: Session, top_k: int, **kwargs) -> list[dict]:
        return self.engine.recommend(user, top_k=top_k, db=db, **kwargs)

class ColdRecommender(RecommenderStrategy):
    def __init__(self):
        self.engine = RecommendationEngine(ColdHybridCandidateGenerator(), NoOpReranker())

    def recommend(self, user, db: Session, top_k: int, **kwargs) -> list[dict]:
        return self.engine.recommend(user, top_k=top_k, db=db, **kwargs)
