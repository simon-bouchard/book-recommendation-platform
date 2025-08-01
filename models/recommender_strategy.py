from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from models.recommendation_engine import RecommendationEngine
from models.rerankers import GBTWarmReranker, GBTColdReranker
from models.candidate_generators import ALSCandidateGenerator, ColdHybridCandidateGenerator

# --------------------------------
# Base Strategy Interface + Factory
# --------------------------------
class RecommenderStrategy(ABC):
    @abstractmethod
    def recommend(self, user_id: int) -> list[int]:
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
        self.engine = RecommendationEngine(ALSCandidateGenerator(), GBTWarmReranker())

    def recommend(self, user, db: Session) -> list[dict]:
        print('warm')
        return self.engine.recommend(user, db=db)

class ColdRecommender(RecommenderStrategy):
    def __init__(self):
        self.engine = RecommendationEngine(ColdHybridCandidateGenerator(), GBTColdReranker())

    def recommend(self, user: int, db: Session) -> list[int]:
        print('cold')
        return self.engine.recommend(user, db=db)