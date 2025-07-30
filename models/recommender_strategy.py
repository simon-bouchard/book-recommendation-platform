from abc import ABC, abstractmethod

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
        return ColdRecommender()


# --------------------------------
# Concrete Strategies
# --------------------------------
class WarmRecommender(RecommenderStrategy):
    def recommend(self, user_id: int) -> list[int]:
        from .warm_user_recs import recommend_books_for_warm_user
        return recommend_books_for_warm_user(user_id)


class ColdRecommender(RecommenderStrategy):
    def recommend(self, user_id: int) -> list[int]:
        from .cold_user_recs import recommend_books_for_cold_user
        return recommend_books_for_cold_user(user_id)