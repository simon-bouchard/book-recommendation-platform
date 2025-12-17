# models/core/paths.py
"""
Centralized path definitions for all model artifacts and data files.
Provides a single source of truth for file locations throughout the codebase.
"""

from pathlib import Path
from typing import Literal

AttentionStrategy = Literal["scalar", "perdim", "selfattn", "selfattn_perdim"]


class ModelPaths:
    """
    Centralized registry of all model artifact paths.

    Directory structure:
        models/
        ├── artifacts/
        │   ├── embeddings/     # Vector representations
        │   ├── attention/      # Attention pooling components
        │   └── scoring/        # Ranking/scoring models
        └── training/
            └── data/           # Training input data
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize path registry.

        Args:
            project_root: Root directory of the project. If None, auto-detects from file location.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent

        self.project_root = project_root
        self.models_root = project_root / "models"

        # Top-level directories
        self.artifacts_dir = self.models_root / "artifacts"
        self.training_root = self.models_root / "training"

        # Artifact subdirectories
        self.embeddings_dir = self.artifacts_dir / "embeddings"
        self.attention_dir = self.artifacts_dir / "attention"
        self.scoring_dir = self.artifacts_dir / "scoring"

        # Training data directory
        self.training_data_dir = self.training_root / "data"

    # -------------------------------------------------------------------------
    # Embeddings (vector representations)
    # -------------------------------------------------------------------------

    @property
    def book_subject_embeddings(self) -> Path:
        """Attention-pooled book embeddings derived from subjects."""
        return self.embeddings_dir / "book_subject_embeddings.npy"

    @property
    def book_subject_ids(self) -> Path:
        """Item indices corresponding to book_subject_embeddings."""
        return self.embeddings_dir / "book_subject_ids.json"

    @property
    def book_als_factors(self) -> Path:
        """ALS collaborative filtering latent factors for books."""
        return self.embeddings_dir / "book_als_factors.npy"

    @property
    def book_als_ids(self) -> Path:
        """Item indices corresponding to book_als_factors."""
        return self.embeddings_dir / "book_als_ids.json"

    @property
    def user_als_factors(self) -> Path:
        """ALS collaborative filtering latent factors for users."""
        return self.embeddings_dir / "user_als_factors.npy"

    @property
    def user_als_ids(self) -> Path:
        """User IDs corresponding to user_als_factors."""
        return self.embeddings_dir / "user_als_ids.json"

    # -------------------------------------------------------------------------
    # Attention (attention pooling components)
    # -------------------------------------------------------------------------

    def get_attention_path(self, strategy: AttentionStrategy) -> Path:
        """
        Get path for attention model components by strategy name.

        Args:
            strategy: One of 'scalar', 'perdim', 'selfattn', 'selfattn_perdim'

        Returns:
            Path to the attention model file

        Raises:
            ValueError: If strategy is not recognized
        """
        valid_strategies = {"scalar", "perdim", "selfattn", "selfattn_perdim"}

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown attention strategy: '{strategy}'. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )

        return self.attention_dir / f"subject_attention_{strategy}.pth"

    @property
    def subject_attention_scalar(self) -> Path:
        """Scalar attention pooling components."""
        return self.attention_dir / "subject_attention_scalar.pth"

    @property
    def subject_attention_perdim(self) -> Path:
        """Per-dimension attention pooling components."""
        return self.attention_dir / "subject_attention_perdim.pth"

    @property
    def subject_attention_selfattn(self) -> Path:
        """Self-attention pooling components."""
        return self.attention_dir / "subject_attention_selfattn.pth"

    @property
    def subject_attention_selfattn_perdim(self) -> Path:
        """Self-attention with per-dimension pooling components."""
        return self.attention_dir / "subject_attention_selfattn_perdim.pth"

    # -------------------------------------------------------------------------
    # Scoring (ranking and scoring models)
    # -------------------------------------------------------------------------

    @property
    def bayesian_scores(self) -> Path:
        """Precomputed Bayesian popularity scores for books."""
        return self.scoring_dir / "bayesian_scores.npy"

    @property
    def gbt_cold(self) -> Path:
        """Gradient boosted tree model for cold-start users."""
        return self.scoring_dir / "gbt_cold.pickle"

    @property
    def gbt_warm(self) -> Path:
        """Gradient boosted tree model for warm-start users."""
        return self.scoring_dir / "gbt_warm.pickle"

    # -------------------------------------------------------------------------
    # Training data
    # -------------------------------------------------------------------------

    @property
    def training_interactions(self) -> Path:
        """Training data: user-book interactions."""
        return self.training_data_dir / "interactions.pkl"

    @property
    def training_users(self) -> Path:
        """Training data: user metadata."""
        return self.training_data_dir / "users.pkl"

    @property
    def training_books(self) -> Path:
        """Training data: book metadata."""
        return self.training_data_dir / "books.pkl"

    @property
    def training_book_subjects(self) -> Path:
        """Training data: book-to-subject mappings."""
        return self.training_data_dir / "book_subjects.pkl"

    @property
    def training_user_fav_subjects(self) -> Path:
        """Training data: user favorite subjects."""
        return self.training_data_dir / "user_fav_subjects.pkl"

    @property
    def training_subjects(self) -> Path:
        """Training data: subject metadata."""
        return self.training_data_dir / "subjects.pkl"

    # -------------------------------------------------------------------------
    # Directory creation helpers
    # -------------------------------------------------------------------------

    def ensure_artifact_dirs(self) -> None:
        """Create all artifact directories if they don't exist."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        self.scoring_dir.mkdir(parents=True, exist_ok=True)

    def ensure_training_dirs(self) -> None:
        """Create training data directory if it doesn't exist."""
        self.training_data_dir.mkdir(parents=True, exist_ok=True)


# Global singleton instance
PATHS = ModelPaths()
