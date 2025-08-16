def reload_all_models():
    from models.recommender_strategy import RecommenderStrategy
    from models.shared_utils import ModelStore
    from models.book_similarity_engine import (
        SubjectSimilarityStrategy,
        ALSSimilarityStrategy,
        HybridSimilarityStrategy
    )

    RecommenderStrategy.reset_singleton()
    SubjectSimilarityStrategy.reset()
    ALSSimilarityStrategy.reset()

    ModelStore.reset()
    ModelStore().preload()

    HybridSimilarityStrategy.reset()
