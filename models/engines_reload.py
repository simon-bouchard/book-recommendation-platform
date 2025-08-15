def reload_all_models():
    from models.recommender_strategy import RecommenderStrategy
    from models.book_similarity_engine import (
        SubjectSimilarityStrategy,
        ALSSimilarityStrategy,
        HybridSimilarityStrategy
    )

    SubjectSimilarityStrategy.reset()
    ALSSimilarityStrategy.reset()
    HybridSimilarityStrategy.reset()

    ModelStore.reset()
    ModelStore().preload()

    RecommenderStrategy.reset_singleton()
