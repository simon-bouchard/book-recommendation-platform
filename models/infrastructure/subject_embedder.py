# models/infrastructure/subject_embedder.py
"""
Subject embedder wrapper providing clean interface to attention pooling strategies.
"""

from typing import Optional, List

import numpy as np
import torch

from models.core.config import Config


class SubjectEmbedder:
    """
    Wrapper around subject attention pooling strategies.

    Provides clean interface for computing user/book embeddings from subject lists.
    Uses singleton pattern for performance (caching loaded models) but supports
    injection for testing.

    Example:
        embedder = SubjectEmbedder()

        # Embed single user's favorite subjects
        user_emb = embedder.embed([5, 12, 23])

        # Embed batch of book subject lists
        book_embs = embedder.embed_batch([
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ])
    """

    _instance: Optional["SubjectEmbedder"] = None

    def __new__(cls, strategy: Optional[str] = None, pooler=None):
        """
        Singleton instantiation with optional injection.

        Args:
            strategy: Attention strategy name (e.g., "scalar", "perdim").
                     If None, uses ATTN_STRATEGY from environment.
            pooler: Injected pooler for testing. If provided, bypasses loading.
        """
        if pooler is not None:
            # Injection mode - create new instance without singleton
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self, strategy: Optional[str] = None, pooler=None):
        """
        Initialize subject embedder.

        Args:
            strategy: Attention strategy name. If None, uses Config.
            pooler: Injected pooler for testing.
        """
        # Guard against re-initialization of singleton
        if self._initialized:
            return

        self._initialized = True

        if pooler is not None:
            # Use injected pooler (for testing)
            self.pooler = pooler
            self.strategy_name = "injected"
        else:
            # Load strategy from disk
            if strategy is None:
                strategy = Config.get_attention_strategy()

            from models.data.loaders import load_attention_strategy

            self.pooler = load_attention_strategy(strategy=strategy, use_cache=True)
            self.strategy_name = strategy

    def embed(self, subjects: List[int]) -> np.ndarray:
        """
        Compute embedding for a single subject list.

        Args:
            subjects: List of subject indices

        Returns:
            1D embedding array of shape (D,)
        """
        with torch.no_grad():
            embedding = self.pooler([subjects])[0].cpu().numpy()

        return embedding

    def embed_batch(self, subjects_list: List[List[int]]) -> np.ndarray:
        """
        Compute embeddings for multiple subject lists.

        Args:
            subjects_list: List of subject lists

        Returns:
            2D embedding array of shape (N, D)
        """
        with torch.no_grad():
            embeddings = self.pooler(subjects_list).cpu().numpy()

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get dimensionality of embeddings produced by this embedder."""
        return self.pooler.get_embedding_dim()

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing or reloading models)."""
        cls._instance = None
