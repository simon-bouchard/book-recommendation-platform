# models/infrastructure/als_model.py
"""
ALS collaborative filtering model wrapper with clean interface.
"""

from typing import Optional, List, Tuple

import numpy as np


class ALSModel:
    """
    Wrapper around ALS collaborative filtering model.

    Provides clean interface for ALS-based recommendations. Uses singleton pattern
    for performance (caching loaded factors) but supports injection for testing.

    Example:
        als = ALSModel()

        # Check if user/book in model
        if als.has_user(123):
            recommendations = als.recommend(user_id=123, k=20)

        # Check if book has ALS factors
        if als.has_book(456):
            print("Book has behavioral data")
    """

    _instance: Optional["ALSModel"] = None

    def __new__(
        cls,
        user_factors: Optional[np.ndarray] = None,
        book_factors: Optional[np.ndarray] = None,
        user_ids: Optional[List[int]] = None,
        book_ids: Optional[List[int]] = None,
    ):
        """
        Singleton instantiation with optional injection.

        Args:
            user_factors: Injected user factors for testing
            book_factors: Injected book factors for testing
            user_ids: Injected user IDs for testing
            book_ids: Injected book IDs for testing
        """
        if any(x is not None for x in [user_factors, book_factors, user_ids, book_ids]):
            # Injection mode - create new instance without singleton
            instance = super().__new__(cls)
            instance._initialized = False
            return instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        user_factors: Optional[np.ndarray] = None,
        book_factors: Optional[np.ndarray] = None,
        user_ids: Optional[List[int]] = None,
        book_ids: Optional[List[int]] = None,
    ):
        """
        Initialize ALS model.

        Args:
            user_factors: User latent factors (N_users x D)
            book_factors: Book latent factors (N_books x D)
            user_ids: User IDs corresponding to rows in user_factors
            book_ids: Book IDs corresponding to rows in book_factors
        """
        # Guard against re-initialization of singleton
        if self._initialized:
            return

        self._initialized = True

        if user_factors is not None:
            # Use injected data (for testing)
            self.user_factors = user_factors
            self.book_factors = book_factors
            self.user_ids = user_ids
            self.book_ids = book_ids

            self.user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
            self.book_row_to_id = {i: bid for i, bid in enumerate(book_ids)}
            self.book_id_to_row = {bid: i for i, bid in enumerate(book_ids)}
        else:
            # Load from disk
            from models.data.loaders import load_als_factors

            (self.user_factors, self.book_factors, user_id_map, book_row_map) = load_als_factors(
                normalized=False, use_cache=True
            )

            # user_id_map: user_id -> row
            # book_row_map: row -> item_idx
            self.user_id_to_row = user_id_map
            self.book_row_to_id = book_row_map

            # Create reverse mapping for has_book()
            self.book_id_to_row = {bid: row for row, bid in book_row_map.items()}

            # Extract ID lists
            self.user_ids = sorted(user_id_map.keys())
            self.book_ids = sorted(self.book_id_to_row.keys())

    def has_user(self, user_id: int) -> bool:
        """
        Check if user exists in ALS model.

        Args:
            user_id: User ID to check

        Returns:
            True if user has ALS factors (is warm)
        """
        return int(user_id) in self.user_id_to_row

    def has_book(self, item_idx: int) -> bool:
        """
        Check if book exists in ALS model.

        Args:
            item_idx: Book item index to check

        Returns:
            True if book has ALS factors
        """
        return int(item_idx) in self.book_id_to_row

    def recommend(self, user_id: int, k: int) -> List[int]:
        """
        Generate ALS-based recommendations for a user.

        Args:
            user_id: User to generate recommendations for
            k: Number of recommendations to return

        Returns:
            List of item_idx sorted by predicted score (descending)
            Returns empty list if user not in model or k <= 0.
        """
        if not self.has_user(user_id):
            return []

        if k <= 0:
            return []

        # Get user latent factors
        user_row = self.user_id_to_row[user_id]
        user_vec = self.user_factors[user_row]

        # Compute scores: book_factors @ user_vec
        scores = self.book_factors @ user_vec

        # Get top k indices
        top_indices = np.argsort(-scores)[:k]

        # Map row indices to item_idx
        recommendations = [self.book_row_to_id[int(i)] for i in top_indices]

        return recommendations

    def get_user_factors(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get latent factors for a specific user.

        Args:
            user_id: User ID

        Returns:
            1D array of user factors, or None if user not in model
        """
        if not self.has_user(user_id):
            return None

        user_row = self.user_id_to_row[user_id]
        return self.user_factors[user_row]

    def get_book_factors(self, item_idx: int) -> Optional[np.ndarray]:
        """
        Get latent factors for a specific book.

        Args:
            item_idx: Book item index

        Returns:
            1D array of book factors, or None if book not in model
        """
        if not self.has_book(item_idx):
            return None

        book_row = self.book_id_to_row[item_idx]
        return self.book_factors[book_row]

    @property
    def num_users(self) -> int:
        """Number of users in the model."""
        return self.user_factors.shape[0]

    @property
    def num_books(self) -> int:
        """Number of books in the model."""
        return self.book_factors.shape[0]

    @property
    def num_factors(self) -> int:
        """Dimensionality of latent factors."""
        return self.user_factors.shape[1]

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing or reloading models)."""
        cls._instance = None
