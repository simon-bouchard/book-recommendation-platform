def reload_all_models():
    from models.shared_utils import ModelStore
    from models.book_similarity_engine import (
        SubjectSimilarityStrategy,
        ALSSimilarityStrategy,
        HybridSimilarityStrategy,
    )

    SubjectSimilarityStrategy.reset()
    ALSSimilarityStrategy.reset()
    HybridSimilarityStrategy.reset()

    ModelStore.reset()
    ModelStore().preload()
